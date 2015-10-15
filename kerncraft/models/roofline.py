#!/usr/bin/env python

from __future__ import print_function

from pprint import pprint
from functools import reduce
import operator
import subprocess
import re
from copy import deepcopy
import sys

from kerncraft.intervals import Intervals
from kerncraft.prefixedunit import PrefixedUnit


def blocking(indices, block_size, initial_boundary=0):
    '''
    splits list of integers into blocks of block_size. returns block indices.

    first block element is located at initial_boundary (default 0).

    >>> blocking([0, -1, -2, -3, -4, -5, -6, -7, -8, -9], 8)
    [0,-1]
    >>> blocking([0], 8)
    [0]
    '''
    blocks = []

    for idx in indices:
        bl_idx = (idx-initial_boundary)/block_size
        if bl_idx not in blocks:
            blocks.append(bl_idx)
    blocks.sort()

    return blocks


def flatten_dict(d):
    '''
    transforms 2d-dict d[i][k] into a new 1d-dict e[(i,k)] with 2-tuple keys
    '''
    e = {}
    for k in d.keys():
        for l in d[k].keys():
            e[(k, l)] = d[k][l]
    return e


class Roofline:
    """
    class representation of the Roofline Model

    more info to follow...
    """

    name = "Roofline"
    _expand_to_cacheline_blocks_cache = {}

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

    def _calculate_relative_offset(self, name, access_dimensions):
        '''
        returns the offset from the iteration center in number of elements and the order of indices
        used in access.
        '''
        offset = 0
        base_dims = self.kernel._variables[name][1]

        for dim, offset_info in enumerate(access_dimensions):
            offset_type, idx_name, dim_offset = offset_info
            assert offset_type == 'rel', 'Only relative access to arrays is supported at the moment'

            if offset_type == 'rel':
                offset += dim_offset*reduce(operator.mul, base_dims[dim+1:], 1)
            else:
                # should not happen
                pass

        return offset

    def _calculate_iteration_offset(self, name, index_order, loop_index):
        '''
        returns the offset from one to the next iteration using *loop_index*.
        *index_order* is the order used by the access dimensions e.g. 'ijk' corresponse to [i][j][k]
        *loop_index* specifies the loop to be used for iterations (this is typically the inner
        moste one)
        '''
        offset = 0
        base_dims = self.kernel._variables[name][1]

        for dim, index_name in enumerate(index_order):
            if loop_index == index_name:
                offset += reduce(operator.mul, base_dims[dim+1:], 1)

        return offset

    def _get_index_order(self, access_dimensions):
        '''Returns the order of indices used in *access_dimensions*.'''
        return ''.join(map(lambda d: d[1], access_dimensions))

    def _expand_to_cacheline_blocks(self, first, last):
        '''
        Returns first and last values wich align with cacheline blocks, by increasing range.
        '''
        if (first,last) not in self._expand_to_cacheline_blocks_cache:
            # handle multiple datatypes
            element_size = self.kernel.datatypes_size[self.kernel.datatype]
            elements_per_cacheline = int(float(self.machine['cacheline size'])) / element_size

            self._expand_to_cacheline_blocks_cache[(first,last)] = [
                first - first % elements_per_cacheline,
                last - last % elements_per_cacheline + elements_per_cacheline - 1]

        return self._expand_to_cacheline_blocks_cache[(first,last)]

    def calculate_cache_access(self, CPUL1=True):
        results = {'bottleneck level': None, 'mem bottlenecks': []}

        read_offsets = {var_name: dict() for var_name in self.kernel._variables.keys()}
        write_offsets = {var_name: dict() for var_name in self.kernel._variables.keys()}
        iteration_offsets = {var_name: dict() for var_name in self.kernel._variables.keys()}

        # handle multiple datatypes
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        elements_per_cacheline = int(float(self.machine['cacheline size'])) / element_size

        loop_order = ''.join(map(lambda l: l[0], self.kernel._loop_stack))

        for var_name in self.kernel._variables.keys():
            var_type, var_dims = self.kernel._variables[var_name]

            # Skip the following access: (they are hopefully kept in registers)
            #   - scalar values
            if var_dims is None:
                continue
            #   - access does not change with inner-most loop index (they are hopefully kept in 
            #     registers)
            writes = filter(
                lambda acs: loop_order[-1] in map(lambda a: a[1], acs),
                self.kernel._destinations.get(var_name, []))
            reads = filter(
                lambda acs: loop_order[-1] in map(lambda a: a[1], acs),
                self.kernel._sources.get(var_name, []))

            # Compile access pattern
            for r in reads:
                offset = self._calculate_relative_offset(var_name, r)
                idx_order = self._get_index_order(r)
                read_offsets[var_name].setdefault(idx_order, [])
                read_offsets[var_name][idx_order].append(offset)
            for w in writes:
                offset = self._calculate_relative_offset(var_name, w)
                idx_order = self._get_index_order(w)
                write_offsets[var_name].setdefault(idx_order, [])
                write_offsets[var_name][idx_order].append(offset)

            # With ECM we would do unrolling, but not with roofline

        # initialize misses and hits
        misses = {}
        hits = {}
        evicts = {}
        total_misses = {}
        total_hits = {}
        total_evicts = {}
        
        memory_hierarchy = deepcopy(self.machine['memory hierarchy'])
        
        # L1-CPU level is special, because everything is a miss here
        if CPUL1:
            memory_hierarchy.insert(0, {
                'cores per group': 1,
                'cycles per cacheline transfer': None,
                'groups': 16,
                'level': 'CPU',
                'bandwidth': None,
                'size per group': 0,
                'threads per group': 2,
            })

        # Check for layer condition towards all cache levels
        for cache_level, cache_info in list(enumerate(memory_hierarchy))[:-1]:
            cache_size = int(float(cache_info['size per group']))
            cache_cycles = cache_info['cycles per cacheline transfer']

            trace_length = 0
            updated_length = True
            x = 0
            while updated_length:
                x += 1
                updated_length = False

                # Initialize cache, misses, hits and evicts for current level
                cache = {}
                misses[cache_level] = {}
                hits[cache_level] = {}
                evicts[cache_level] = {}

                # We consider everythin a miss in the beginning, unless it is completly cached
                # TODO here read and writes are treated the same, this implies write-allocate
                #      to support nontemporal stores, this needs to be changed
                for name in read_offsets.keys()+write_offsets.keys():
                    cache[name] = {}
                    misses[cache_level][name] = {}
                    hits[cache_level][name] = {}

                    for idx_order in read_offsets[name].keys()+write_offsets[name].keys():
                        cache[name][idx_order] = Intervals()
                        
                        # Check for complete caching/in-cache
                        # TODO change from pessimistic to more realistic approach (different 
                        #      indexes are treasted as individual arrays)
                        total_array_size = reduce(
                            operator.mul, self.kernel._variables[name][1])*element_size
                        if total_array_size < trace_length:
                            # all hits no misses
                            misses[cache_level][name][idx_order] = []
                            if cache_level-1 not in misses:
                                hits[cache_level][name][idx_order] = sorted(
                                    read_offsets.get(name, {}).get(idx_order, []) +
                                    write_offsets.get(name, {}).get(idx_order, []),
                                    reverse=True)
                            else:
                                hits[cache_level][name][idx_order] = list(
                                    misses[cache_level-1][name][idx_order])
                          
                        # partial caching (default case) 
                        else:
                            if cache_level-1 not in misses:
                                misses[cache_level][name][idx_order] = sorted(
                                    read_offsets.get(name, {}).get(idx_order, []) +
                                    write_offsets.get(name, {}).get(idx_order, []),
                                    reverse=True)
                            else:
                                misses[cache_level][name][idx_order] = list(
                                    misses[cache_level-1][name][idx_order])
                            hits[cache_level][name][idx_order] = []

                # Caches are still empty (thus only misses)
                trace_count = 0
                cache_used_size = 0

                # Now we trace the cache access backwards (in time/iterations) and check for hits
                for var_name in misses[cache_level].keys():
                    for idx_order in misses[cache_level][var_name].keys():
                        iter_offset = self._calculate_iteration_offset(
                            var_name, idx_order, loop_order[-1])

                        # Add cache trace
                        for offset in list(misses[cache_level][var_name][idx_order]):
                            # If already present in cache add to hits
                            if offset in cache[var_name][idx_order]:
                                misses[cache_level][var_name][idx_order].remove(offset)

                                # We might have multiple hits on the same offset (e.g in DAXPY)
                                if offset not in hits[cache_level][var_name][idx_order]:
                                    hits[cache_level][var_name][idx_order].append(offset)

                            # Add cache, we can do this since misses are sorted in reverse order of
                            # access and we assume LRU cache replacement policy
                            if iter_offset <= elements_per_cacheline:
                                # iterations overlap, thus we can savely add the whole range
                                cached_first, cached_last = self._expand_to_cacheline_blocks(
                                    offset-iter_offset*trace_length, offset+1)
                                cache[var_name][idx_order] &= Intervals(
                                    [cached_first, cached_last+1], sane=True)
                            else:
                                # There is no overlap, we can append the ranges onto one another
                                # TODO optimize this code section (and maybe merge with above)
                                new_cache = [self._expand_to_cacheline_blocks(o, o) for o in range(
                                    offset-iter_offset*trace_length, offset+1, iter_offset)]
                                new_cache = Intervals(*new_cache, sane=True)
                                cache[var_name][idx_order] &= new_cache

                        trace_count += len(cache[var_name][idx_order]._data)
                        cache_used_size += len(cache[var_name][idx_order])*element_size
                
                # Calculate new possible trace_length according to free space in cache
                # TODO take CL blocked access into account
                # TODO make /2 customizable
                #new_trace_length = trace_length + \
                #    ((cache_size/2 - cache_used_size)/trace_count)/element_size
                if trace_count > 0:  # to catch complete caching
                    new_trace_length = trace_length + \
                        ((cache_size - cache_used_size)/trace_count)/element_size

                if new_trace_length > trace_length:
                    trace_length = new_trace_length
                    updated_length = True

                # All writes to require the data to be evicted eventually
                evicts[cache_level] = {
                    var_name: dict() for var_name in self.kernel._variables.keys()}
                for name in write_offsets.keys():
                    for idx_order in write_offsets[name].keys():
                        evicts[cache_level][name][idx_order] = list(write_offsets[name][idx_order])
            
            # Compiling stats
            total_misses[cache_level] = sum(map(
                lambda l: sum(map(len, l.values())), misses[cache_level].values()))
            total_hits[cache_level] = sum(map(
                lambda l: sum(map(len, l.values())), hits[cache_level].values()))
            total_evicts[cache_level] = sum(map(
                lambda l: sum(map(len, l.values())), evicts[cache_level].values()))

            # Calculate performance (arithmetic intensity * bandwidth with
            # arithmetic intensity = flops / bytes transfered)
            bytes_transfered = (total_misses[cache_level]+total_evicts[cache_level])*element_size
            total_flops = sum(self.kernel._flops.values())
            arith_intens = float(total_flops)/float(bytes_transfered)

            # choose bw according to cache level and problem
            # first, compile stream counts at current cache level
            # write-allocate is allready resolved above
            read_streams = 0
            for var_name in misses[cache_level].keys():
                for idx_order in misses[cache_level][var_name]:
                    read_streams += len(misses[cache_level][var_name][idx_order])
            write_streams = 0
            for var_name in evicts[cache_level].keys():
                for idx_order in evicts[cache_level][var_name]:
                    write_streams += len(evicts[cache_level][var_name][idx_order])
            # second, try to find best fitting kernel (closest to stream seen stream counts):
            # write allocate has to be handled in kernel information (all writes are also reads)
            # TODO support for non-write-allocate architectures
            measurement_kernel = 'load'
            measurement_kernel_info = self.machine['benchmarks']['kernels'][measurement_kernel]
            for kernel_name, kernel_info in self.machine['benchmarks']['kernels'].items():
                if (read_streams >= (kernel_info['read streams']['streams'] +
                                     kernel_info['write streams']['streams'] -
                                     kernel_info['read+write streams']['streams']) >
                        measurement_kernel_info['read streams']['streams'] +
                        measurement_kernel_info['write streams']['streams'] -
                        measurement_kernel_info['read+write streams']['streams'] and
                        write_streams >= kernel_info['write streams']['streams'] >
                        measurement_kernel_info['write streams']['streams']):
                    measurement_kernel = kernel_name
                    measurement_kernel_info = kernel_info

            # TODO choose smt and cores:
            threads_per_core, cores = 1, self._args.cores
            bw_level = memory_hierarchy[cache_level+1]['level']
            bw_measurements = \
                self.machine['benchmarks']['measurements'][bw_level][threads_per_core]
            assert threads_per_core == bw_measurements['threads per core'], \
                'malformed measurement dictionary in machine file.'
            run_index = bw_measurements['cores'].index(cores)
            bw = bw_measurements['results'][measurement_kernel][run_index]

            # Correct bandwidth due to miss-measurement of write allocation
            # TODO support non-temporal stores and non-write-allocate architectures
            measurement_kernel_info = self.machine['benchmarks']['kernels'][measurement_kernel]
            factor = (float(measurement_kernel_info['read streams']['bytes']) +
                      2.0*float(measurement_kernel_info['write streams']['bytes']) -
                      float(measurement_kernel_info['read+write streams']['bytes'])) / \
                     (float(measurement_kernel_info['read streams']['bytes']) +
                      float(measurement_kernel_info['write streams']['bytes']))
            bw = bw * factor

            performance = arith_intens * float(bw)
            results['mem bottlenecks'].append({
                'performance': PrefixedUnit(performance, 'FLOP/s'),
                'level': (memory_hierarchy[cache_level]['level'] + '-' +
                          memory_hierarchy[cache_level+1]['level']),
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
        
        return {'It/s': it_s,
                'cy/CL': cy_cl,
                'FLOP/s': performance}[unit]

    def report(self):
        precision = 'DP' if self.kernel.datatype == 'double' else 'SP'
        max_flops = self.machine['clock']*self._args.cores*sum(
            self.machine['FLOPs per cycle'][precision].values())
        max_flops.unit = "FLOP/s"
        if self._args and self._args.verbose >= 1:
            print('Bottlnecks:')
            print('  level | a. intensity |   performance   |   bandwidth  | bandwidth kernel')
            print('--------+--------------+-----------------+--------------+-----------------')
            print('    CPU |              | {:>15} |              |'.format(
                self.conv_perf(max_flops, self._args.unit)))
            for b in self.results['mem bottlenecks']:
                print('{level:>7} | {arithmetic intensity:>5.2} FLOP/b | {:>15} |'
                      ' {bandwidth:>12} | {bw kernel:<8}'.format(
                          self.conv_perf(b['performance'], self._args.unit), **b))
            print()

        if self.results['min performance'] > max_flops:
            # CPU bound
            print('CPU bound with', self._args.cores, 'core(s)')
            print('{!s} due to CPU max. FLOP/s'.format(max_flops))
        else:
            # Cache or mem bound
            print('Cache or mem bound with', self._args.cores, 'core(s)')

            bottleneck = self.results['mem bottlenecks'][self.results['bottleneck level']]
            print('{!s} due to {} transfer bottleneck (bw with from {} benchmark)'.format(
                self.conv_perf(bottleneck['performance'], self._args.unit),
                bottleneck['level'],
                bottleneck['bw kernel']))
            print('Arithmetic Intensity: {:.2f} FLOP/b'.format(bottleneck['arithmetic intensity']))

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
        Roofline.__init__(self, kernel, machine, args, parser)

    def analyze(self):
        self.results = self.calculate_cache_access(CPUL1=False)
        
        # For the IACA/CPU analysis we need to compile and assemble
        asm_name = self.kernel.compile(
            self.machine['compiler'], compiler_args=self.machine['compiler flags'])
        bin_name = self.kernel.assemble(
           self.machine['compiler'], asm_name, iaca_markers=True, asm_block=self._args.asm_block)

        # Get total cycles per loop iteration
        try:
            cmd = ['iaca.sh', '-64', '-arch', self.machine['micro-architecture'], bin_name]
            iaca_output = subprocess.check_output(cmd)
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
        ports = filter(lambda l: l.startswith('|  Port  |'), iaca_output.split('\n'))
        cycles = filter(lambda l: l.startswith('| Cycles |'), iaca_output.split('\n'))
        assert ports and cycles, "Could not find ports/cylces lines in IACA output."
        ports = map(str.strip, ports[0].split('|'))[2:]
        cycles = map(str.strip, cycles[0].split('|'))[2:]
        port_cycles = []
        for i in range(len(ports)):
            if '-' in ports[i] and ' ' in cycles[i]:
                subports = map(str.strip, ports[i].split('-'))
                subcycles = filter(bool, cycles[i].split(' '))
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
                 self.machine['micro-architecture'], bin_name])
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

        port_cycles = dict(map(lambda i: (i[0], i[1]*block_to_cl_ratio), port_cycles.items()))
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

    def report(self):
        if not self._args.latency:
            cpu_flops = PrefixedUnit(
                self.results['cpu bottleneck']['performance throughput'], "FLOP/s")
        else:
            cpu_flops = PrefixedUnit(
                self.results['cpu bottleneck']['performance latency'], "FLOP/s")
        if self._args and self._args.verbose >= 1:
            print('Bottlnecks:')
            print('  level | a. intensity |   performance   |   bandwidth  | bandwidth kernel')
            print('--------+--------------+-----------------+--------------+-----------------')
            print('    CPU |              | {:>15} |              |'.format(
                self.conv_perf(cpu_flops, self._args.unit)))
            for b in self.results['mem bottlenecks']:
                print('{level:>7} | {arithmetic intensity:>5.2} FLOP/b | {:>15} |'
                      ' {bandwidth:>12} | {bw kernel:<8}'.format(
                          self.conv_perf(b['performance'], self._args.unit), **b))
            print()
            print('IACA analisys:')
            if self._args.verbose >= 3:
                print(self.results['cpu bottleneck']['IACA output'])
                print(self.results['cpu bottleneck']['IACA latency output'])
            print({k: v for k, v in self.results['cpu bottleneck'].items() if k not in 
                ['IACA output', 'IACA latency output']})

        if float(self.results['min performance']) > float(cpu_flops):
            # CPU bound
            print('CPU bound with', self._args.cores, 'core(s)')
            print('{!s} due to CPU bottleneck'.format(self.conv_perf(cpu_flops, self._args.unit)))
        else:
            # Cache or mem bound
            print('Cache or mem bound with', self._args.cores, 'core(s)')

            bottleneck = self.results['mem bottlenecks'][self.results['bottleneck level']]
            print('{!s} due to {} transfer bottleneck (bw with from {} benchmark)'.format(
                self.conv_perf(bottleneck['performance'], self._args.unit),
                bottleneck['level'],
                bottleneck['bw kernel']))
            print('Arithmetic Intensity: {:.2f} FLOP/b'.format(bottleneck['arithmetic intensity']))
