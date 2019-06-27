#!/usr/bin/env python3
"""Execution-Cache-Memory model class and helper functions."""
import copy
import sys
import math
import pprint

import sympy
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_support = True
except ImportError:
    plot_support = False

from kerncraft.prefixedunit import PrefixedUnit
from kerncraft.cacheprediction import LayerConditionPredictor, CacheSimulationPredictor
from .base import PerformanceModel


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


class ECMData(PerformanceModel):
    """Representation of Data portion of the Execution-Cache-Memory Model."""

    name = "Execution-Cache-Memory (data transfers only)"

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
        self.results = {}

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
        self.results.update({
                        'cycles': [],  # will be filled by caclculate_cycles()
                        'misses': self.predictor.get_misses(),
                        'hits': self.predictor.get_hits(),
                        'evicts': self.predictor.get_evicts(),
                        'verbose infos': self.predictor.get_infos()})  # only for verbose outputs

    def calculate_cycles(self):
        """
        Calculate performance model cycles from cache stats.

        calculate_cache_access() needs to have been execute before.
        """
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        elements_per_cacheline = float(self.machine['cacheline size']) // element_size
        iterations_per_cacheline = (sympy.Integer(self.machine['cacheline size']) /
                                    sympy.Integer(self.kernel.bytes_per_iteration))
        self.results['iterations per cacheline'] = iterations_per_cacheline
        cacheline_size = float(self.machine['cacheline size'])

        loads, stores = (self.predictor.get_loads(), self.predictor.get_stores())

        for cache_level, cache_info in list(enumerate(self.machine['memory hierarchy']))[1:]:
            throughput, duplexness = cache_info['upstream throughput']

            if type(throughput) is str and throughput == 'full socket memory bandwidth':
                # Memory transfer
                # we use bandwidth to calculate cycles and then add panalty cycles (if given)

                # choose bw according to cache level and problem
                # first, compile stream counts at current cache level
                # write-allocate is allready resolved in cache predictor
                read_streams = loads[cache_level]
                write_streams = stores[cache_level]
                # second, try to find best fitting kernel (closest to stream seen stream counts):
                threads_per_core = 1
                bw, measurement_kernel = self.machine.get_bandwidth(
                    cache_level, read_streams, write_streams, threads_per_core)

                # calculate cycles
                if duplexness == 'half-duplex':
                    cycles = float(loads[cache_level] + stores[cache_level]) * \
                             float(elements_per_cacheline) * float(element_size) * \
                             float(self.machine['clock']) / float(bw)
                else:  # full-duplex
                    raise NotImplementedError(
                        "full-duplex mode is not (yet) supported for memory transfers.")
                # add penalty cycles for each read stream
                if 'penalty cycles per read stream' in cache_info:
                    cycles += stores[cache_level] * \
                              cache_info['penalty cycles per read stream']

                self.results.update({
                    'memory bandwidth kernel': measurement_kernel,
                    'memory bandwidth': bw})
            else:
                # since throughput is given in B/cy, and we need CL/cy:
                throughput = float(throughput) / cacheline_size
                # only cache cycles count
                if duplexness == 'half-duplex':
                    cycles = (loads[cache_level] + stores[cache_level]) / float(throughput)
                elif duplexness == 'full-duplex':
                    cycles = max(loads[cache_level] / float(throughput),
                                 stores[cache_level] / float(throughput))
                else:
                    raise ValueError("Duplexness of cache throughput may only be 'half-duplex'"
                                     "or 'full-duplex', found {} in {}.".format(
                        duplexness, cache_info['name']))

            self.results['cycles'].append((cache_info['level'], cycles))

            self.results[cache_info['level']] = cycles

        return self.results

    def analyze(self):
        """Run complete anaylysis and return results."""
        self.calculate_cache_access()
        self.calculate_cycles()
        self.results['flops per iteration'] = sum(self.kernel._flops.values())

        return self.results

    def conv_cy(self, cy_cl):
        """Convert cycles (cy/CL) to other units, such as FLOP/s or It/s."""
        if not isinstance(cy_cl, PrefixedUnit):
            cy_cl = PrefixedUnit(cy_cl, '', 'cy/CL')

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
                'FLOP/s': performance}

    def report_data_transfers(self):
        cacheline_size = float(self.machine['cacheline size'])
        r = "Data Transfers:\nLevel   | Loads    | Store    |\n"
        loads, stores = (self.predictor.get_loads(), self.predictor.get_stores())
        for cache_level, cache_info in list(enumerate(self.machine['memory hierarchy']))[1:]:
            r += ("{:>7} | {:>3.0f} B/CL | {:>3.0f} B/CL |\n".format(
                self.machine['memory hierarchy'][cache_level-1]['level']+'-'+cache_info['level'],
                loads[cache_level] * cacheline_size,
                stores[cache_level] * cacheline_size))
        return r

    def report(self, output_file=sys.stdout):
        """Print generated model data in human readable format."""
        if self.verbose > 1:
            print('{}'.format(pprint.pformat(self.results['verbose infos'])), file=output_file)

        for level, cycles in self.results['cycles']:
            print('{} = {}'.format(
                level, self.conv_cy(cycles)[self._args.unit]), file=output_file)

        if self.verbose > 1:
            if 'memory bandwidth kernel' in self.results:
                print('memory cycles based on {} kernel with {}'.format(
                          self.results['memory bandwidth kernel'],
                          self.results['memory bandwidth']),
                      file=output_file)

        if self.verbose > 1:
            print(file=output_file)
            print(self.report_data_transfers(), file=output_file)

        if any(['_Complex' in var_info[0] for var_info in self.kernel.variables.values()]) and \
                self._args.unit == 'FLOP/s':
            print("WARNING: FLOP counts are probably wrong, because complex flops are counted\n"
                  "         as single flops. All other units should not be affected.\n",
                  file=sys.stderr)


class ECMCPU(PerformanceModel):
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
        self.results = {}

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
            incore_analysis, asm_block = self.kernel.incore_analysis(
                asm_block=self.asm_block,
                pointer_increment=self.pointer_increment,
                verbose=self.verbose > 2)
        except RuntimeError as e:
            print("In-core analysis failed: " + str(e))
            sys.exit(1)

        block_throughput = incore_analysis['throughput']
        port_cycles = incore_analysis['port cycles']
        uops = incore_analysis['uops']

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
        if uops is not None:
            uops = uops*block_to_cl_ratio
        cl_throughput = block_throughput*block_to_cl_ratio

        # Compile most relevant information
        T_comp = float(max([v for k, v in list(port_cycles.items())
                            if k in self.machine['overlapping model']['ports']] + [0]))
        T_RegL1 = float(max([v for k, v in list(port_cycles.items())
                             if k in self.machine['non-overlapping model']['ports']] + [0]))

        # Use IACA throughput prediction if it is slower then T_RegL1
        if T_RegL1 < cl_throughput:
            T_comp = cl_throughput

        # Create result dictionary
        self.results = {
            'port cycles': port_cycles,
            'cl throughput': self.conv_cy(cl_throughput),
            'uops': uops,
            'T_comp': T_comp,
            'T_RegL1': T_RegL1,
            'IACA output': incore_analysis['output'],
            'elements_per_block': elements_per_block,
            'pointer_increment': asm_block['pointer_increment'],
            'flops per iteration': sum(self.kernel._flops.values())}
        return self.results

    def conv_cy(self, cy_cl):
        """Convert cycles (cy/CL) to other units, such as FLOP/s or It/s."""
        if not isinstance(cy_cl, PrefixedUnit):
            cy_cl = PrefixedUnit(cy_cl, '', 'cy/CL')
        clock = self.machine['clock']
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        elements_per_cacheline = int(self.machine['cacheline size']) // element_size
        if cy_cl == 0.0:
            it_s = PrefixedUnit(float('inf'))
        else:
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
                'FLOP/s': performance}

    def report(self, output_file=sys.stdout):
        """Print generated model data in human readable format."""
        if self.verbose > 2:
            print("IACA Output:", file=output_file)
            print(self.results['IACA output'], file=output_file)
            print('', file=output_file)

        if self.verbose > 1:
            print('Detected pointer increment: {}'.format(self.results['pointer_increment']),
                  file=output_file)
            print('Derived elements stored to per asm block iteration: {}'.format(
                  self.results['elements_per_block']), file=output_file)
            print('Ports and cycles:', str(self.results['port cycles']), file=output_file)
            print('Uops:', str(self.results['uops']), file=output_file)

            print('Throughput: {}'.format(self.results['cl throughput'][self._args.unit]),
                  file=output_file)

        print('T_comp = {:.1f} cy/CL'.format(self.results['T_comp']), file=output_file)
        print('T_RegL1 = {:.1f} cy/CL'.format(self.results['T_RegL1']), file=output_file)

        if any(['_Complex' in var_info[0] for var_info in self.kernel.variables.values()]) and \
                self._args.unit == 'FLOP/s':
            print("WARNING: FLOP counts are probably wrong, because complex flops are counted \n"
                  "         as single flops. All other units should not be affected.\n",
                  file=sys.stderr)


class ECM(PerformanceModel):
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
        self.results = None

        if args:
            self.verbose = self._args.verbose

        self._CPU = ECMCPU(kernel, machine, args, parser, asm_block=asm_block,
                           pointer_increment=pointer_increment, verbose=verbose)
        self._data = ECMData(kernel, machine, args, parser,
                             cache_predictor=CacheSimulationPredictor, cores=cores, verbose=verbose)

    def analyze(self):
        """Run complete analysis."""
        self._CPU.analyze()
        self._data.analyze()
        self.results = copy.deepcopy(self._CPU.results)
        self.results.update(copy.deepcopy(self._data.results))

        cores_per_numa_domain = self.machine['cores per NUMA domain']

        # Compile ECM model
        ECM_OL, ECM_OL_construction = [self.results['T_comp']], ['T_comp']
        ECM_nOL, ECM_nOL_construction = [], []
        if self.machine['memory hierarchy'][0]['transfers overlap']:
            nonoverlap_region = False
            ECM_OL.append(self.results['T_RegL1'])
            ECM_OL_construction.append('T_RegL1')
        else:
            nonoverlap_region = True
            ECM_nOL.append(self.results['T_RegL1'])
            ECM_nOL_construction.append('T_RegL1')

        for cache_level, cache_info in list(enumerate(self.machine['memory hierarchy']))[1:]:
            cycles = self.results['cycles'][cache_level-1][1]
            if cache_info['transfers overlap']:
                if nonoverlap_region:
                    raise ValueError("Overlapping changes back and forth between levels, this is "
                                     "currently not supported.")
                ECM_OL.append(cycles)
                ECM_OL_construction.append(
                    'T_' + self.machine['memory hierarchy'][cache_level-1]['level'] +
                    cache_info['level'])
            else:
                nonoverlap_region = True
                ECM_nOL.append(cycles)
                ECM_nOL_construction.append(
                    'T_' + self.machine['memory hierarchy'][cache_level-1]['level'] +
                    cache_info['level'])
        # TODO consider multiple paths per cache level with victim caches
        self.results['ECM'] = tuple(ECM_OL + [tuple(ECM_nOL)])
        self.results['ECM Model Construction'] = tuple(ECM_OL_construction +
                                                       [tuple(ECM_nOL_construction)])

        # Compile total single-core prediction
        self.results['total cycles'] = self._CPU.conv_cy(max(sum(ECM_nOL), *ECM_OL))
        T_ECM = float(self.results['total cycles']['cy/CL'])
        # T_MEM is the cycles accounted to memory transfers
        T_MEM = self.results['cycles'][-1][1]

        # Simple scaling prediction:
        # Assumptions are:
        #  - bottleneck is always LLC-MEM
        #  - all caches scale with number of cores (bw AND size(WRONG!))

        # Full caching in higher cache level
        self.results['scaling cores'] = float('inf')
        # Not full caching:
        if self.results['cycles'][-1][1] != 0.0:
            # Considering memory bus utilization
            utilization = [0]
            self.results['scaling cores'] = float('inf')
            for c in range(1, cores_per_numa_domain + 1):
                if c * T_MEM > (T_ECM + utilization[c - 1] * (c - 1) * T_MEM / 2):
                    utilization.append(1.0)
                    self.results['scaling cores'] = min(self.results['scaling cores'], c)
                else:
                    utilization.append(c * T_MEM /
                                       (T_ECM + utilization[c - 1] * (c - 1) * T_MEM / 2))
            utilization = utilization[1:]

            # scaling code
            scaling_predictions = []
            for cores in range(1, self.machine['cores per socket'] + 1):
                scaling = {'cores': cores, 'notes': [], 'performance': None,
                           'in-NUMA performance': None}
                # Detailed scaling:
                if cores <= self.results['scaling cores']:
                    # Is it purely in-cache?
                    innuma_rectp = PrefixedUnit(T_ECM / (T_ECM/T_MEM),
                                                "cy/CL")
                    scaling['notes'].append("memory-interface not saturated")
                else:
                    innuma_rectp = PrefixedUnit(self.results['cycles'][-1][1], 'cy/CL')
                    scaling['notes'].append("memory-interface saturated on first NUMA domain")
                # Include NUMA-local performance in results dict
                scaling['in-NUMA performance'] = innuma_rectp

                if 0 < cores <= cores_per_numa_domain:
                    # only in-numa scaling to consider
                    scaling['performance'] = self._CPU.conv_cy(
                        innuma_rectp / utilization[cores - 1])
                    scaling['notes'].append("in-NUMA-domain scaling")
                elif cores <= self.machine['cores per socket'] * self.machine['sockets']:
                    # out-of-numa scaling behavior
                    scaling['performance'] = self._CPU.conv_cy(
                        innuma_rectp * cores_per_numa_domain / cores)
                    scaling['notes'].append("out-of-NUMA-domain scaling")
                else:
                    raise ValueError("Number of cores must be greater than zero and upto the max. "
                                     "number of cores defined by cores per socket and sockets in"
                                     "machine file.")
                scaling_predictions.append(scaling)
        else:
            # pure in-cache performace (perfect scaling)
            scaling_predictions = [
                {'cores': cores, 'notes': ['pure in-cache'],
                 'performance': self._CPU.conv_cy(T_ECM/cores),
                 'in-NUMA performance': self._CPU.conv_cy(T_ECM/cores_per_numa_domain)}
                for cores in range(1, self.machine['cores per socket'] + 1)]

        # Also include prediction for all in-NUMA core counts in results
        self.results['scaling prediction'] = scaling_predictions
        if self._args.cores:
            self.results['multi-core'] = scaling_predictions[self._args.cores - 1]
        else:
            self.results['multi-core'] = None

    def report(self, output_file=sys.stdout):
        """Print generated model data in human readable format."""
        report = ''
        if self.verbose > 1:
            self._CPU.report()
            self._data.report()

        model_construction = 'max({}, sum({})) cy/CL'.format(
            ', '.join(self.results['ECM Model Construction'][:-1]),
            ', '.join(self.results['ECM Model Construction'][-1]))
        ecm_string = 'max({}, sum({})) cy/CL'.format(
            ', '.join(['{:.1f}'.format(c) for c in self.results['ECM'][:-1]]),
            ', '.join(['{:.1f}'.format(c) for c in self.results['ECM'][-1]]))
        report += 'T_ECM = ' + model_construction + '\n' + \
                  '      = ' + ecm_string + '\n' + \
                  '      = {}'.format(self.results['total cycles'][self._args.unit])

        if self._args.cores > 1:
            report += " (single core)"

        report += '\nsaturating at {:.0f} cores'.format(self.results['scaling cores'])

        if self.results['multi-core']:
            report += "\nprediction for {} cores,".format(self.results['multi-core']['cores']) + \
                      " assuming static scheduling: "
            report += "{} ({})\n".format(
                self.results['multi-core']['performance'][self._args.unit],
                ', '.join(self.results['multi-core']['notes']))

        if self.results['scaling prediction']:
            report += "\nScaling prediction, considering memory bus utilization penalty and " \
                "assuming all scalable caches:\n"
            if self.machine['cores per socket'] > self.machine['cores per NUMA domain']:
                report += "1st NUMA dom." + (len(self._args.unit) - 4) * ' ' + '||' + \
                    '--------' * (self.machine['cores per NUMA domain']-1) + '-------|\n'

            report +=  "cores " + (len(self._args.unit)+2)*' ' + " || " + ' | '.join(
                ['{:<5}'.format(s['cores']) for s in self.results['scaling prediction']]) + '\n'
            report +=  "perf. ({}) || ".format(self._args.unit) + ' | '.join(
                ['{:<5.1f}'.format(float(s['performance'][self._args.unit]))
                 for s in self.results['scaling prediction']]) + '\n'

        print(report, file=output_file)

        if self._args and self._args.ecm_plot:
            assert plot_support, "matplotlib couldn't be imported. Plotting is not supported."
            fig = plt.figure(frameon=False)
            self.plot(fig)

        if any(['_Complex' in var_info[0] for var_info in self.kernel.variables.values()]) and \
                self._args.unit == 'FLOP/s':
            print("WARNING: FLOP counts are probably wrong, because complex flops are counted\n"
                  "         as single flops. All other units should not be affected.\n",
                  file=sys.stderr)

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
        # T_comp
        colors = ([(254. / 255, 177. / 255., 178. / 255.)] +
                  [(255. / 255., 255. / 255., 255. / 255.)] * (len(sorted_overlapping_ports) - 1))
        for p, c in sorted_overlapping_ports:
            ax.barh(i, c, height, align='center', color=colors.pop(),
                    edgecolor=(0.5, 0.5, 0.5), linestyle='dashed')
            if i == len(sorted_overlapping_ports) - 1:
                ax.text(c / 2.0, i, '$T_\mathrm{comp}$', ha='center', va='center')
            yticks_labels.append(p)
            yticks.append(i)
            i += 1
        xticks.append(sorted_overlapping_ports[-1][1])
        xticks_labels.append('{:.1f}'.format(sorted_overlapping_ports[-1][1]))

        # T_RegL1 + memory transfers
        y = 0
        colors = [(187. / 255., 255 / 255., 188. / 255.)] * (len(self.results['cycles'])) + \
                 [(119. / 255, 194. / 255., 255. / 255.)]
        for k, v in [('RegL1', self.results['T_RegL1'])] + self.results['cycles']:
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
