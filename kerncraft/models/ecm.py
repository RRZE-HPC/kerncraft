#!/usr/bin/env python
"""Execution-Cache-Memory model class and helper functions."""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import copy
import sys
import math
from pprint import pformat

import six
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_support = True
except ImportError:
    plot_support = False

from kerncraft.prefixedunit import PrefixedUnit
from kerncraft.cacheprediction import LayerConditionPredictor, CacheSimulationPredictor


def round_to_next(x, base):
    """Round float to next multiple of base."""
    # Based on: http://stackoverflow.com/a/2272174
    return int(base * math.ceil(float(x)/base))


def blocking(indices, block_size, initial_boundary=0):
    """
    Split list of integers into blocks of block_size and return block indices.

    First block element will be located at initial_boundary (default 0).

    >>> blocking([0, -1, -2, -3, -4, -5, -6, -7, -8, -9], 8)
    [0,-1]
    >>> blocking([0], 8)
    [0]
    >>> blocking([0], 8, initial_boundary=32)
    [-4]
    """
    blocks = []

    for idx in indices:
        bl_idx = (idx-initial_boundary)//float(block_size)
        if bl_idx not in blocks:
            blocks.append(bl_idx)
    blocks.sort()

    return blocks


class ECMData(object):
    """Representation of Data portion of the Execution-Cache-Memory Model."""

    name = "Execution-Cache-Memory (data transfers only)"

    @classmethod
    def configure_arggroup(cls, parser):
        """Configure argument group of parser."""
        pass

    def __init__(self, kernel, machine, args=None, parser=None, cores=1,
                 cache_predictor=CacheSimulationPredictor, verbose=0):
        """
        Create Execcution-Cache-Memory data model from kernel and machine objects.

        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line

        If *args* is None, *cores*, *cache_predictor* and *verbose* are taken into account,
        otherwise *args* takes precedence.
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args
        self._parser = parser

        if args:
            self.verbose = self._args.verbose
            self.cores = self._args.cores
            if self._args.cache_predictor == 'SIM':
                self.predictor = CacheSimulationPredictor(self.kernel, self.machine, self.cores)
            elif self._args.cache_predictor == 'LC':
                self.predictor = LayerConditionPredictor(self.kernel, self.machine, self.cores)
            else:
                raise NotImplementedError("Unknown cache predictor, only LC (layer condition) and "
                                          "SIM (cache simulation with pycachesim) is supported.")
        else:
            self.cores = cores
            self.predictor = cache_predictor(self.kernel, self.machine, self.cores)
            self.verbose = verbose

    def calculate_cache_access(self):
        """Dispatch to cache predictor to get cache stats."""
        self.results = {'cycles': [],  # will be filled by caclculate_cycles()
                        'misses': self.predictor.get_misses(),
                        'hits': self.predictor.get_hits(),
                        'evicts': self.predictor.get_evicts(),
                        'verbose infos': self.predictor.get_infos()}  # only for verbose outputs

    def calculate_cycles(self):
        """
        Calculate performance model cycles from cache stats.

        calculate_cache_access() needs to have been execute before.
        """
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        elements_per_cacheline = float(self.machine['cacheline size']) // element_size

        misses, evicts = (self.predictor.get_misses(), self.predictor.get_evicts())

        for cache_level, cache_info in list(enumerate(self.machine['memory hierarchy']))[:-1]:
            cache_cycles = cache_info['cycles per cacheline transfer']

            if cache_cycles is not None:
                # only cache cycles count
                cycles = (misses[cache_level] + evicts[cache_level]) * cache_cycles
            else:
                # Memory transfer
                # we use bandwidth to calculate cycles and then add panalty cycles (if given)

                # choose bw according to cache level and problem
                # first, compile stream counts at current cache level
                # write-allocate is allready resolved in cache predictor
                read_streams = misses[cache_level]
                write_streams = evicts[cache_level]
                # second, try to find best fitting kernel (closest to stream seen stream counts):
                threads_per_core = 1
                bw, measurement_kernel = self.machine.get_bandwidth(
                    cache_level + 1, read_streams, write_streams, threads_per_core)

                # calculate cycles
                cycles = float(misses[cache_level] + evicts[cache_level]) * \
                    float(elements_per_cacheline) * float(element_size) * \
                    float(self.machine['clock']) / float(bw)
                # add penalty cycles for each read stream
                if 'penalty cycles per read stream' in cache_info:
                    cycles += misses[cache_level] * \
                              cache_info['penalty cycles per read stream']

            if cache_cycles is None:
                self.results.update({
                    'memory bandwidth kernel': measurement_kernel,
                    'memory bandwidth': bw})

            self.results['cycles'].append((
                '{}-{}'.format(
                    cache_info['level'],
                    self.machine['memory hierarchy'][cache_level + 1]['level']),
                cycles))

            # TODO remove the following by makeing testcases more versatile:
            self.results['{}-{}'.format(
                cache_info['level'], self.machine['memory hierarchy'][cache_level + 1]['level'])
            ] = cycles

        return self.results

    def analyze(self):
        """Run complete anaylysis and return results."""
        self.calculate_cache_access()
        self.calculate_cycles()

        return self.results

    def conv_cy(self, cy_cl, unit, default='cy/CL'):
        """Convert cycles (cy/CL) to other units, such as FLOP/s or It/s."""
        if not isinstance(cy_cl, PrefixedUnit):
            cy_cl = PrefixedUnit(cy_cl, '', 'cy/CL')
        if not unit:
            unit = default

        clock = self.machine['clock']
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        elements_per_cacheline = int(self.machine['cacheline size']) // element_size
        if cy_cl != 0:
            it_s = clock/cy_cl*elements_per_cacheline
            it_s.unit = 'It/s'
        else:
            it_s = PrefixedUnit('inf It/S')
        flops_per_it = sum(self.kernel._flops.values())
        performance = it_s*flops_per_it
        performance.unit = 'FLOP/s'
        cy_it = cy_cl*elements_per_cacheline
        cy_it.unit = 'cy/It'

        return {'It/s': it_s,
                'cy/CL': cy_cl,
                'cy/It': cy_it,
                'FLOP/s': performance}[unit]

    def report(self, output_file=sys.stdout):
        """Print generated model data in human readable format."""
        if self.verbose > 1:
            print('{}'.format(pformat(self.results['verbose infos'])), file=output_file)

        for level, cycles in self.results['cycles']:
            print('{} = {}'.format(
                level, self.conv_cy(float(cycles), self._args.unit)), file=output_file)

        if self.verbose > 1:
            if 'memory bandwidth kernel' in self.results:
                print('memory cycles based on {} kernel with {}'.format(
                          self.results['memory bandwidth kernel'],
                          self.results['memory bandwidth']),
                      file=output_file)


class ECMCPU(object):
    """Representation of the In-core execution part of the Execution-Cache-Memory model."""

    name = "Execution-Cache-Memory (CPU operations only)"

    @classmethod
    def configure_arggroup(cls, parser):
        """Configure argument parser."""
        pass

    def __init__(self, kernel, machine, args=None, parser=None, asm_block='auto',
                 pointer_increment='auto', verbose=0):
        """
        Create Execution-Cache-Memory model from kernel and machine objects.

        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        if *args* is given also *parser* has to be provided

        If *args* is None, *asm_block*, *pointer_increment* and *verbose* will be used, otherwise
        *args* takes precedence.
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args
        self._parser = parser

        if args:
            # handle CLI info
            self.asm_block = self._args.asm_block
            self.pointer_increment = self._args.pointer_increment
            self.verbose = self._args.verbose
        else:
            self.asm_block = asm_block
            self.pointer_increment = pointer_increment
            self.verbose = verbose

        # Validate arguments
        if self.asm_block not in ['auto', 'manual']:
            try:
                self.asm_block = int(args.asm_block)
            except ValueError:
                parser.error('asm_block can only be "auto", "manual" or an integer')
        if self.pointer_increment not in ['auto', 'auto_with_manual_fallback', 'manual']:
            try:
                self.pointer_increment = int(args.pointer_increment)
            except ValueError:
                parser.error('pointer_increment can only be "auto", '
                             '"auto_with_manual_fallback", "manual" or an integer')

    def analyze(self):
        """
        Run complete analysis and return results.
        """
        try:
            iaca_analysis, asm_block = self.kernel.iaca_analysis(
                micro_architecture=self.machine['micro-architecture'],
                asm_block=self.asm_block,
                pointer_increment=self.pointer_increment,
                verbose=self.verbose > 2)
        except RuntimeError as e:
            print("IACA analysis failed: " + str(e))
            sys.exit(1)

        block_throughput = iaca_analysis['throughput']
        port_cycles = iaca_analysis['port cycles']
        uops = iaca_analysis['uops']

        # Normalize to cycles per cacheline
        elements_per_block = abs(asm_block['pointer_increment']
                                 // self.kernel.datatypes_size[self.kernel.datatype])
        block_size = elements_per_block*self.kernel.datatypes_size[self.kernel.datatype]
        try:
            block_to_cl_ratio = float(self.machine['cacheline size'])/block_size
        except ZeroDivisionError as e:
            print("Too small block_size / pointer_increment:", e, file=sys.stderr)
            sys.exit(1)

        port_cycles = dict([(i[0], i[1]*block_to_cl_ratio) for i in list(port_cycles.items())])
        uops = uops*block_to_cl_ratio
        cl_throughput = block_throughput*block_to_cl_ratio

        # Compile most relevant information
        T_OL = max([v for k, v in list(port_cycles.items())
                    if k in self.machine['overlapping model']['ports']])
        T_nOL = max([v for k, v in list(port_cycles.items())
                     if k in self.machine['non-overlapping model']['ports']])

        # Use IACA throughput prediction if it is slower then T_nOL
        if T_nOL < cl_throughput:
            T_OL = cl_throughput

        # Create result dictionary
        self.results = {
            'port cycles': port_cycles,
            'cl throughput': cl_throughput,
            'uops': uops,
            'T_nOL': T_nOL,
            'T_OL': T_OL,
            'IACA output': iaca_analysis['output']}
        return self.results

    def conv_cy(self, cy_cl, unit, default='cy/CL'):
        """Convert cycles (cy/CL) to other units, such as FLOP/s or It/s."""
        if not isinstance(cy_cl, PrefixedUnit):
            cy_cl = PrefixedUnit(cy_cl, '', 'cy/CL')
        if not unit:
            unit = default

        clock = self.machine['clock']
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        elements_per_cacheline = int(self.machine['cacheline size']) // element_size
        it_s = clock/cy_cl*elements_per_cacheline
        it_s.unit = 'It/s'
        flops_per_it = sum(self.kernel._flops.values())
        performance = it_s*flops_per_it
        performance.unit = 'FLOP/s'
        cy_it = cy_cl*elements_per_cacheline
        cy_it.unit = 'cy/It'

        return {'It/s': it_s,
                'cy/CL': cy_cl,
                'cy/It': cy_it,
                'FLOP/s': performance}[unit]

    def report(self, output_file=sys.stdout):
        """Print generated model data in human readable format."""
        if self.verbose > 2:
            print("IACA Output:", file=output_file)
            print(self.results['IACA output'], file=output_file)
            print('', file=output_file)

        if self.verbose > 1:
            print('Ports and cycles:', six.text_type(self.results['port cycles']), file=output_file)
            print('Uops:', six.text_type(self.results['uops']), file=output_file)

            print('Throughput: {}'.format(
                      self.conv_cy(self.results['cl throughput'], self._args.unit)),
                  file=output_file)

        print('T_nOL = {:.1f} cy/CL'.format(self.results['T_nOL']), file=output_file)
        print('T_OL = {:.1f} cy/CL'.format(self.results['T_OL']), file=output_file)


class ECM(object):
    """
    Complete representation of the Execution-Cache-Memory Model (data and operations).

    more info to follow...
    """

    name = "Execution-Cache-Memory"

    @classmethod
    def configure_arggroup(cls, parser):
        """Configure argument parser."""
        # others are being configured in ECMData and ECMCPU
        parser.add_argument(
            '--ecm-plot',
            help='Filename to save ECM plot to (supported extensions: pdf, png, svg and eps)')

    def __init__(self, kernel, machine, args=None, parser=None, asm_block="auto",
                 pointer_increment="auto", cores=1, cache_predictor=CacheSimulationPredictor,
                 verbose=0):
        """
        Create complete Execution-Cache-Memory model from kernel and machine objects.

        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line

        If *args* is None, *asm_block*, *pointer_increment*, *cores*, *cache_predictor* and
        *verbose* will be used, otherwise *args* takes precedence.
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args
        self.verbose = verbose

        if args:
            self.verbose = self._args.verbose

        self._CPU = ECMCPU(kernel, machine, args, parser, asm_block=asm_block,
                           pointer_increment=pointer_increment, verbose=verbose)
        self._data = ECMData(kernel, machine, args, parser,
                             cache_predictor=CacheSimulationPredictor, cores=1, verbose=verbose)

    def analyze(self):
        """Run complete analysis."""
        self._CPU.analyze()
        self._data.analyze()
        self.results = copy.deepcopy(self._CPU.results)
        self.results.update(copy.deepcopy(self._data.results))

        # Saturation/multi-core scaling analysis
        # very simple approach. Assumptions are:
        #  - bottleneck is always LLC-MEM
        #  - all caches scale with number of cores (bw AND size(WRONG!))
        if self.results['cycles'][-1][1] == 0.0:
            # Full caching in higher cache level
            self.results['scaling cores'] = float('inf')
        else:
            self.results['scaling cores'] = (
                max(self.results['T_OL'],
                    self.results['T_nOL'] + sum([c[1] for c in self.results['cycles']])) /
                self.results['cycles'][-1][1])

    def report(self, output_file=sys.stdout):
        """Print generated model data in human readable format."""
        report = ''
        if self.verbose > 1:
            self._CPU.report()
            self._data.report()

        total_cycles = max(
            self.results['T_OL'],
            sum([self.results['T_nOL']]+[i[1] for i in self.results['cycles']]))
        report += '{{ {:.1f} || {:.1f} | {} }} cy/CL'.format(
            self.results['T_OL'],
            self.results['T_nOL'],
            ' | '.join(['{:.1f}'.format(i[1]) for i in self.results['cycles']]))

        if self._args.unit:
            report += ' = {}'.format(self._CPU.conv_cy(total_cycles, self._args.unit))

        report += '\n{{ {} \ {} }} cy/CL'.format(
            max(self.results['T_OL'], self.results['T_nOL']),
            ' \ '.join(['{:.1f}'.format(max(sum([x[1] for x in self.results['cycles'][:i+1]]) +
                                            self.results['T_nOL'], self.results['T_OL']))
                        for i in range(len(self.results['cycles']))]))

        if self.verbose > 1:
            if 'memory bandwidth kernel' in self.results:
                report += '\nmemory cycles based on {} kernel with {}\n'.format(
                    self.results['memory bandwidth kernel'],
                    self.results['memory bandwidth'])

        report += '\nsaturating at {:.1f} cores'.format(self.results['scaling cores'])

        print(report, file=output_file)

        if self._args and self._args.ecm_plot:
            assert plot_support, "matplotlib couldn't be imported. Plotting is not supported."
            fig = plt.figure(frameon=False)
            self.plot(fig)

    def plot(self, fig=None):
        """Plot visualization of model prediction."""
        if not fig:
            fig = plt.gcf()

        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
        ax = fig.add_subplot(1, 1, 1)

        sorted_overlapping_ports = sorted(
            [(p, self.results['port cycles'][p]) for p in self.machine['overlapping ports']],
            key=lambda x: x[1])

        yticks_labels = []
        yticks = []
        xticks_labels = []
        xticks = []

        # Plot configuration
        height = 0.9

        i = 0
        # T_OL
        colors = ([(254. / 255, 177. / 255., 178. / 255.)] +
                  [(255. / 255., 255. / 255., 255. / 255.)] * (len(sorted_overlapping_ports) - 1))
        for p, c in sorted_overlapping_ports:
            ax.barh(i, c, height, align='center', color=colors.pop(),
                    edgecolor=(0.5, 0.5, 0.5), linestyle='dashed')
            if i == len(sorted_overlapping_ports) - 1:
                ax.text(c / 2.0, i, '$T_\mathrm{OL}$', ha='center', va='center')
            yticks_labels.append(p)
            yticks.append(i)
            i += 1
        xticks.append(sorted_overlapping_ports[-1][1])
        xticks_labels.append('{:.1f}'.format(sorted_overlapping_ports[-1][1]))

        # T_nOL + memory transfers
        y = 0
        colors = [(187. / 255., 255 / 255., 188. / 255.)] * (len(self.results['cycles'])) + \
                 [(119. / 255, 194. / 255., 255. / 255.)]
        for k, v in [('nOL', self.results['T_nOL'])] + self.results['cycles']:
            ax.barh(i, v, height, y, align='center', color=colors.pop())
            ax.text(y + v / 2.0, i, '$T_\mathrm{' + k + '}$', ha='center', va='center')
            xticks.append(y + v)
            xticks_labels.append('{:.1f}'.format(y + v))
            y += v
        yticks_labels.append('LD')
        yticks.append(i)

        ax.tick_params(axis='y', which='both', left='off', right='off')
        ax.tick_params(axis='x', which='both', top='off')
        ax.set_xlabel('t [cy]')
        ax.set_ylabel('execution port')
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks_labels)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labels, rotation='vertical')
        ax.xaxis.grid(alpha=0.7, linestyle='--')
        fig.savefig(self._args.ecm_plot)
