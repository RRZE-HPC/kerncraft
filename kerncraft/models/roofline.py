#!/usr/bin/env python

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from functools import reduce
import operator
import subprocess
import re
import sys
from distutils.spawn import find_executable

import sympy

from kerncraft.prefixedunit import PrefixedUnit
from kerncraft.kernel import KernelCode


class Roofline(object):
    """
    class representation of the Roofline Model

    more info to follow...
    """

    name = "Roofline"

    @classmethod
    def configure_arggroup(cls, parser):
        pass

    def __init__(self, kernel, machine, args=None, parser=None):
        """
        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args
        self._parser = parser

        if args:
            # handle CLI info
            pass

    def calculate_cache_access(self, CPUL1=True):
        # FIXME handle multiple datatypes
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        cacheline_size = self.machine['cacheline size']
        elements_per_cacheline = int(cacheline_size // element_size)

        # Get the machine's cache model and simulator
        csim = self.machine.get_cachesim()

        # Calculate the number of iterations necessary for warm-up
        max_cache_size = max(map(lambda c: c.size(), csim.levels(with_mem=False)))
        max_array_size = max(self.kernel.array_sizes(in_bytes=True, subs_consts=True).values())

        offsets = []
        if max_array_size < max_cache_size:
            # Full caching possible, go through all itreration before actual initialization
            warmup_iteration_count = self.kernel.iteration_length()//3
            offsets = list(self.kernel.compile_global_offsets(
                iteration=range(0, self.kernel.iteration_length())))

        # Regular Initialization
        warmup_indices = {
            sympy.Symbol(l['index'], positive=True): ((l['stop']-l['start'])//l['increment'])//3
            for l in self.kernel.get_loop_stack(subs_consts=True)}
        warmup_iteration_count = self.kernel.indices_to_global_iterator(warmup_indices)

        # Align iteration count with cachelines
        # do this by aligning either writes (preferred) or reads:
        # Assumption: writes (and reads) increase linearly
        inner_loop = list(self.kernel.get_loop_stack(subs_consts=True))[-1]
        inner_increment = inner_loop['increment']
        o = list(self.kernel.compile_global_offsets(iteration=warmup_iteration_count))[0]
        if o[1]:
            # we have a write to work with:
            first_offset = min(o[1])
        else:
            # we use reads
            first_offset = min(o[0])
        # Distance from cacheline boundary (in bytes)
        diff = first_offset - \
               (int(first_offset)>>csim.first_level.cl_bits<<csim.first_level.cl_bits)
        warmup_iteration_count -= (diff//element_size)//inner_increment
        warmup_indices = self.kernel.global_iterator_to_indices(warmup_iteration_count)

        offsets += list(self.kernel.compile_global_offsets(
            iteration=range(0, warmup_iteration_count)))

        # Do the warm-up
        csim.loadstore(offsets, length=element_size)
        # FIXME compile_global_offsets should already expand to element_size

        # Force write-back on all cache levels
        csim.force_write_back()

        # Reset stats to conclude warm-up phase
        csim.reset_stats()

        # Benchmark iterations:
        inner_index = sympy.Symbol(inner_loop['index'], positive=True)
        # Strting point is one past the last warmup element
        bench_iteration_start = warmup_iteration_count
        # End point is the end of the current dimension (cacheline alligned)
        first_dim_factor = int((inner_loop['stop'] - warmup_indices[inner_index] - 1) 
                               // (elements_per_cacheline//inner_increment))
        bench_iteration_end = (bench_iteration_start + 
                               elements_per_cacheline*inner_increment*first_dim_factor)

        # compile access needed for one cache-line
        offsets = list(self.kernel.compile_global_offsets(
            iteration=range(bench_iteration_start, bench_iteration_end)))

        # simulate
        csim.loadstore(offsets, length=element_size)
        # FIXME compile_global_offsets should already expand to element_size

        # Force write-back on all cache levels
        csim.force_write_back()

        # use stats to build results
        stats = list(csim.stats())

        total_flops = sum(self.kernel._flops.values())*elements_per_cacheline

        results = {'bottleneck level': 0,
                   'mem bottlenecks': [],
                   'cachelines in stats': first_dim_factor}

        # TODO let user choose threads_per_core:
        threads_per_core = 1

        # Compile relevant information
        # TODO unite CPU-L1 and other level handling

        # CPU-L1 stats (in bytes!)
        total_loads = stats[0]['LOAD_byte']/first_dim_factor
        total_evicts = stats[0]['STORE_byte']/first_dim_factor
        read_streams = stats[0]['LOAD_count']/first_dim_factor
        write_streams = stats[1]['STORE_count']/first_dim_factor
        bw, measurement_kernel = self.machine.get_bandwidth(
            0, read_streams, write_streams,
            threads_per_core, cores=self._args.cores)

        # Calculate performance (arithmetic intensity * bandwidth with
        # arithmetic intensity = flops / bytes transfered)
        bytes_transfered = total_loads + total_evicts

        if bytes_transfered == 0:
            # This happens in case of full-caching
            arith_intens = None
            performance = None
        else:
            arith_intens = float(total_flops)/bytes_transfered
            performance = arith_intens * float(bw)

        results['mem bottlenecks'].append({
            'performance': PrefixedUnit(performance, 'FLOP/s'),
            'level': ('CPU-' +
                      self.machine['memory hierarchy'][0]['level']),
            'arithmetic intensity': arith_intens,
            'bw kernel': measurement_kernel,
            'bandwidth': bw})
        if performance <= results.get('min performance', performance):
            results['bottleneck level'] = len(results['mem bottlenecks'])-1
            results['min performance'] = performance

        # for other cache and memory levels:
        for cache_level, cache_info in list(enumerate(self.machine['memory hierarchy']))[:-1]:
            cache_stats = stats[cache_level]

            # Compiling stats (in bytes!)
            total_misses = stats[cache_level+1]['LOAD_byte']/first_dim_factor
            total_evicts = cache_stats['STORE_byte']/first_dim_factor

            # choose bw according to cache level and problem
            # first, compile stream counts at current cache level
            # write-allocate is allready resolved above
            read_streams = cache_stats['MISS_count']/first_dim_factor
            write_streams = stats[cache_level-1]['STORE_count']/first_dim_factor
            # second, try to find best fitting kernel (closest to stream seen stream counts):
            bw, measurement_kernel = self.machine.get_bandwidth(
                cache_level+1, read_streams, write_streams, threads_per_core,
                cores=self._args.cores)

            # Calculate performance (arithmetic intensity * bandwidth with
            # arithmetic intensity = flops / bytes transfered)
            bytes_transfered = total_misses + total_evicts

            if bytes_transfered == 0:
                # This happens in case of full-caching
                arith_intens = None
                performance = None
            else:
                arith_intens = float(total_flops)/bytes_transfered
                performance = arith_intens * float(bw)

            results['mem bottlenecks'].append({
                'performance': PrefixedUnit(performance, 'FLOP/s'),
                'level': (cache_info['level'] + '-' +
                          self.machine['memory hierarchy'][cache_level+1]['level']),
                'arithmetic intensity': arith_intens,
                'bw kernel': measurement_kernel,
                'bandwidth': bw})
            if performance <= results.get('min performance', performance):
                results['bottleneck level'] = len(results['mem bottlenecks'])-1
                results['min performance'] = performance

        return results

    def analyze(self):
        self.results = self.calculate_cache_access()

    def conv_perf(self, performance, unit, default='FLOP/s'):
        '''Convert performance (FLOP/s) to other units, such as It/s or cy/CL'''
        if not unit:
            unit = default

        clock = self.machine['clock']
        flops_per_it = sum(self.kernel._flops.values())
        it_s = performance/flops_per_it
        it_s.unit = 'It/s'
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        elements_per_cacheline = int(float(self.machine['cacheline size'])) / element_size
        cy_cl = clock/it_s*elements_per_cacheline
        cy_cl.unit = 'cy/CL'
        cy_it = clock/it_s
        cy_it.unit = 'cy/It'

        return {'It/s': it_s,
                'cy/CL': cy_cl,
                'cy/It': cy_it,
                'FLOP/s': performance}[unit]

    def report(self, output_file=sys.stdout):
        precision = 'DP' if self.kernel.datatype == 'double' else 'SP'
        max_flops = self.machine['clock']*self._args.cores*sum(
            self.machine['FLOPs per cycle'][precision].values())
        max_flops.unit = "FLOP/s"
        if self._args and self._args.verbose >= 1:
            print('Cachelines in stats: {}'.format(self.results['cachelines in stats']),
                  file=output_file)
            print('Bottlnecks:', file=output_file)
            print('  level | a. intensity |   performance   |   bandwidth  | bandwidth kernel',
                  file=output_file)
            print('--------+--------------+-----------------+--------------+-----------------',
                  file=output_file)
            print('    CPU |              | {!s:>15} |              |'.format(
                      self.conv_perf(max_flops, self._args.unit)),
                  file=output_file)
            for b in self.results['mem bottlenecks']:
                print('{level:>7} | {arithmetic intensity:>5.2} FLOP/B | {!s:>15} |'
                      ' {bandwidth!s:>12} | {bw kernel:<8}'.format(
                          self.conv_perf(b['performance'], self._args.unit), **b),
                      file=output_file)
            print('', file=output_file)

        if self.results['min performance'] > max_flops:
            # CPU bound
            print('CPU bound with {} cores(s)'.format(self._args.cores), file=output_file)
            print('{!s} due to CPU max. FLOP/s'.format(max_flops), file=output_file)
        else:
            # Cache or mem bound
            print('Cache or mem bound with {} core(s)'.format(self._args.cores), file=output_file)

            bottleneck = self.results['mem bottlenecks'][self.results['bottleneck level']]
            print('{!s} due to {} transfer bottleneck (bw with from {} benchmark)'.format(
                    self.conv_perf(bottleneck['performance'], self._args.unit),
                    bottleneck['level'],
                    bottleneck['bw kernel']),
                  file=output_file)
            print('Arithmetic Intensity: {:.2f} FLOP/B'.format(bottleneck['arithmetic intensity']),
                  file=output_file)


class RooflineIACA(Roofline):
    """
    class representation of the Roofline Model (with IACA throughput analysis)

    more info to follow...
    """

    name = "Roofline (with IACA throughput)"

    @classmethod
    def configure_arggroup(cls, parser):
        pass

    def __init__(self, kernel, machine, args=None, parser=None):
        """
        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        if *args* is given also *parser* has to be provided
        """
        if not isinstance(kernel, KernelCode):
            raise ValueError("Kernel was not derived from code, can not perform RooflineIACA "
                             "analysis. Try Roofline.")
        Roofline.__init__(self, kernel, machine, args, parser)

    def analyze(self):
        self.results = self.calculate_cache_access(CPUL1=False)

        # For the IACA/CPU analysis we need to compile and assemble
        asm_name = self.kernel.compile(
            self.machine['compiler'], compiler_args=self.machine['compiler flags'])
        bin_name = self.kernel.assemble(
           self.machine['compiler'], asm_name, iaca_markers=True, asm_block=self._args.asm_block,
           asm_increment=self._args.asm_increment)

        # Making sure iaca.sh is available:
        if find_executable('iaca.sh') is None:
            print("iaca.sh was not found. Make sure it is found in PATH.", file=sys.stderr)
            sys.exit(1)

        # Get total cycles per loop iteration
        try:
            cmd = ['iaca.sh', '-64', '-arch', self.machine['micro-architecture'], bin_name]
            iaca_output = subprocess.check_output(cmd).decode('utf-8')
        except OSError as e:
            print("IACA execution failed:", ' '.join(cmd), file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print("IACA throughput analysis failed:", e, file=sys.stderr)
            sys.exit(1)

        match = re.search(
            r'^Block Throughput: ([0-9\.]+) Cycles', iaca_output, re.MULTILINE)
        assert match, "Could not find Block Throughput in IACA output."
        block_throughput = float(match.groups()[0])

        # Find ports and cyles per port
        ports = [l for l in iaca_output.split('\n') if l.startswith('|  Port  |')]
        cycles = [l for l in iaca_output.split('\n') if l.startswith('| Cycles |')]
        assert ports and cycles, "Could not find ports/cylces lines in IACA output."
        ports = [p.strip() for p in ports[0].split('|')][2:]
        cycles = [p.strip() for p in cycles[0].split('|')][2:]
        port_cycles = []
        for i in range(len(ports)):
            if '-' in ports[i] and ' ' in cycles[i]:
                subports = [p.strip() for p in ports[i].split('-')]
                subcycles = [c for c in cycles[i].split(' ') if bool(c)]
                port_cycles.append((subports[0], float(subcycles[0])))
                port_cycles.append((subports[0]+subports[1], float(subcycles[1])))
            elif ports[i] and cycles[i]:
                port_cycles.append((ports[i], float(cycles[i])))
        port_cycles = dict(port_cycles)

        match = re.search(r'^Total Num Of Uops: ([0-9]+)', iaca_output, re.MULTILINE)
        assert match, "Could not find Uops in IACA output."
        uops = float(match.groups()[0])

        # Get latency prediction from IACA
        try:
            iaca_latency_output = subprocess.check_output(
                ['iaca.sh', '-64', '-analysis', 'LATENCY', '-arch',
                 self.machine['micro-architecture'], bin_name]).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print("IACA latency analysis failed:", e, file=sys.stderr)
            sys.exit(1)

        # Get predicted latency
        match = re.search(
            r'^Latency: ([0-9\.]+) Cycles', iaca_latency_output, re.MULTILINE)
        assert match, "Could not find Latency in IACA latency analysis output."
        block_latency = float(match.groups()[0])

        # Normalize to cycles per cacheline
        elements_per_block = abs(self.kernel.asm_block['pointer_increment']
                                 / self.kernel.datatypes_size[self.kernel.datatype])
        block_size = elements_per_block*self.kernel.datatypes_size[self.kernel.datatype]
        try:
            block_to_cl_ratio = float(self.machine['cacheline size'])/block_size
        except ZeroDivisionError as e:
            print("Too small block_size / pointer_increment:", e, file=sys.stderr)
            sys.exit(1)

        port_cycles = dict([(i[0], i[1]*block_to_cl_ratio) for i in list(port_cycles.items())])
        uops = uops*block_to_cl_ratio
        cl_throughput = block_throughput*block_to_cl_ratio
        cl_latency = block_latency*block_to_cl_ratio
        flops_per_element = sum(self.kernel._flops.values())

        # Create result dictionary
        self.results.update({
            'cpu bottleneck': {
                'port cycles': port_cycles,
                'cl throughput': cl_throughput,
                'cl latency': cl_latency,
                'uops': uops,
                'performance throughput':
                    self.machine['clock']/block_throughput*elements_per_block*flops_per_element
                    *self._args.cores,
                'performance latency':
                    self.machine['clock']/block_latency*elements_per_block*flops_per_element
                    *self._args.cores,
                'IACA output': iaca_output,
                'IACA latency output': iaca_latency_output}})
        self.results['cpu bottleneck']['performance throughput'].unit = 'FLOP/s'
        self.results['cpu bottleneck']['performance latency'].unit = 'FLOP/s'

    def report(self, output_file=sys.stdout):
        if not self._args.latency:
            cpu_flops = PrefixedUnit(
                self.results['cpu bottleneck']['performance throughput'], "FLOP/s")
        else:
            cpu_flops = PrefixedUnit(
                self.results['cpu bottleneck']['performance latency'], "FLOP/s")
        if self._args and self._args.verbose >= 1:
            print('Bottlnecks:', file=output_file)
            print('  level | a. intensity |   performance   |   bandwidth  | bandwidth kernel',
                  file=output_file)
            print('--------+--------------+-----------------+--------------+-----------------',
                  file=output_file)
            print('    CPU |              | {!s:>15} |              |'.format(
                      self.conv_perf(cpu_flops, self._args.unit)),
                  file=output_file)
            for b in self.results['mem bottlenecks']:
                print('{level:>7} | {arithmetic intensity:>5.2} FLOP/B | {!s:>15} |'
                      ' {bandwidth!s:>12} | {bw kernel:<8}'.format(
                          self.conv_perf(b['performance'], self._args.unit), **b),
                      file=output_file)
            print('', file=output_file)
            print('IACA analisys:', file=output_file)
            if self._args.verbose >= 3:
                print(self.results['cpu bottleneck']['IACA output'], file=output_file)
                print(self.results['cpu bottleneck']['IACA latency output'], file=output_file)
            print('{!s}'.format(
                {k: v
                 for k, v in list(self.results['cpu bottleneck'].items())
                 if k not in['IACA output', 'IACA latency output']}),
                file=output_file)

        if float(self.results['min performance']) > float(cpu_flops):
            # CPU bound
            print('CPU bound with {} core(s)'.format(self._args.cores), file=output_file)
            print('{!s} due to CPU bottleneck'.format(self.conv_perf(cpu_flops, self._args.unit)),
                  file=output_file)
        else:
            # Cache or mem bound
            print('Cache or mem bound with {} core(s)'.format(self._args.cores), file=output_file)

            bottleneck = self.results['mem bottlenecks'][self.results['bottleneck level']]
            print('{!s} due to {} transfer bottleneck (bw with from {} benchmark)'.format(
                      self.conv_perf(bottleneck['performance'], self._args.unit),
                      bottleneck['level'],
                      bottleneck['bw kernel']),
                  file=output_file)
            print('Arithmetic Intensity: {:.2f} FLOP/B'.format(bottleneck['arithmetic intensity']),
                  file=output_file)
