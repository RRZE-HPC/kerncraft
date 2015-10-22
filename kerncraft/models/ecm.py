#!/usr/bin/env python
# pylint: disable=W0142

from __future__ import print_function

from pprint import pprint
from functools import reduce as reduce_
import operator
import copy
import sys
import subprocess
import re
import imp
import math
import sys


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_support = True
except ImportError:
    plot_support = False

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


class ECMData:
    """
    class representation of the Execution-Cache-Memory Model (only the data part)

    more info to follow...
    """

    name = "Execution-Cache-Memory (data transfers only)"
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
                offset += dim_offset*reduce_(operator.mul, base_dims[dim+1:], 1)
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
                offset += reduce_(operator.mul, base_dims[dim+1:], 1)

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

    def calculate_cache_access(self):
        results = {}

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

            # Do unrolling so that one iteration equals one cacheline worth of workload:
            # unrolling is done on inner-most loop only!
            for i in range(1, elements_per_cacheline):
                for r in reads:
                    idx_order = self._get_index_order(r)
                    offset = self._calculate_relative_offset(var_name, r)
                    offset += i * self._calculate_iteration_offset(
                        var_name, idx_order, loop_order[-1])
                    read_offsets[var_name][idx_order].append(offset)

                    # Remove multiple access to same offsets
                    read_offsets[var_name][idx_order] = \
                        sorted(list(set(read_offsets[var_name][idx_order])), reverse=True)

                for w in writes:
                    idx_order = self._get_index_order(w)
                    offset = self._calculate_relative_offset(var_name, w)
                    offset += i * self._calculate_iteration_offset(
                        var_name, idx_order, loop_order[-1])
                    write_offsets[var_name][idx_order].append(offset)

                    # Remove multiple access to same offsets
                    write_offsets[var_name][idx_order] = \
                        sorted(list(set(write_offsets[var_name][idx_order])), reverse=True)

        # initialize misses and hits
        misses = {}
        hits = {}
        evicts = {}
        total_misses = {}
        total_hits = {}
        total_evicts = {}
        total_lines_misses = {}
        total_lines_hits = {}
        total_lines_evicts = {}

        self.results = {'memory hierarchy': [], 'cycles': []}

        # Check for layer condition towards all cache levels (except main memory/last level)
        for cache_level, cache_info in list(enumerate(self.machine['memory hierarchy']))[:-1]:
            cache_size = int(float(cache_info['size per group']))
            # reduce cache size in parallel execution
            if self._args.cores > 1 and cache_info['cores per group'] is not None and \
                    cache_info['cores per group'] > 1:
                if self._args.cores < cache_info['cores per group']:
                    cache_size /= self._args.cores
                else:
                    cache_size /= cache_info['cores per group']
            cache_cycles = cache_info['cycles per cacheline transfer']
            bandwidth = cache_info['bandwidth']

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

            total_lines_misses[cache_level] = sum(map(
                lambda o: sum(map(lambda n: len(blocking(n, elements_per_cacheline)), o.values())),
                misses[cache_level].values()))
            total_lines_hits[cache_level] = sum(map(lambda o: sum(map(
                lambda n: len(blocking(n, elements_per_cacheline)), o.values())),
                hits[cache_level].values()))
            total_lines_evicts[cache_level] = sum(map(lambda o: sum(map(
                lambda n: len(blocking(n, elements_per_cacheline)), o.values())),
                evicts[cache_level].values()))

            if not bandwidth:
                # only cache cycles count
                cycles = (total_lines_misses[cache_level] + total_lines_evicts[cache_level]) * \
                    cache_cycles
            else:
                # Memory transfer
                # we use bandwidth to calculate cycles and then add panalty cycles (if given)
                
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

                # choose smt, and then use max/saturation bw
                threads_per_core = 1
                bw_level = self.machine['memory hierarchy'][cache_level+1]['level']
                bw_measurements = \
                    self.machine['benchmarks']['measurements'][bw_level][threads_per_core]
                assert threads_per_core == bw_measurements['threads per core'], \
                    'malformed measurement dictionary in machine file.'
                bw = max(bw_measurements['results'][measurement_kernel])

                # Correct bandwidth due to miss-measurement of write allocation
                # TODO support non-temporal stores and non-write-allocate architectures
                measurement_kernel_info = self.machine['benchmarks']['kernels'][measurement_kernel]
                factor = (float(measurement_kernel_info['read streams']['bytes']) +
                          2.0*float(measurement_kernel_info['write streams']['bytes']) -
                          float(measurement_kernel_info['read+write streams']['bytes'])) / \
                         (float(measurement_kernel_info['read streams']['bytes']) +
                          float(measurement_kernel_info['write streams']['bytes']))
                bw = bw * factor
                
                # calculate cycles
                cycles = (total_lines_misses[cache_level] + total_lines_evicts[cache_level]) * \
                    elements_per_cacheline * element_size * \
                    float(self.machine['clock']) / float(bw)
                # add penalty cycles for each read stream
                if cache_cycles:
                    cycles += total_lines_misses[cache_level]*cache_cycles

            self.results['memory hierarchy'].append({
                'index': i,
                'level': '{}'.format(cache_info['level']),
                'total misses': total_misses[cache_level],
                'total hits': total_hits[cache_level],
                'total evicts': total_evicts[cache_level],
                'total lines misses': total_lines_misses[cache_level],
                'total lines hits': total_lines_hits[cache_level],
                'total lines evicts': total_lines_evicts[cache_level],
                'trace length': trace_length,
                'misses': misses[cache_level],
                'hits': hits[cache_level],
                'evicts': evicts[cache_level],
                'cycles': cycles})
            if bandwidth:
                self.results['memory hierarchy'][-1].update({
                    'memory bandwidth kernel': measurement_kernel,
                    'memory bandwidth': bw})
            self.results['cycles'].append((
                '{}-{}'.format(cache_info['level'], self.machine['memory hierarchy'][cache_level+1]['level']),
                cycles))

            # TODO remove the following by makeing testcases more versatile:
            self.results[
                '{}-{}'.format(cache_info['level'], self.machine['memory hierarchy'][cache_level+1]['level'])
                ] = cycles

        return results

    def analyze(self):
        self._results = self.calculate_cache_access()

    def conv_cy(self, cy_cl, unit, default='cy/CL'):
        '''Convert cycles (cy/CL) to other units, such as FLOP/s or It/s'''
        if not isinstance(cy_cl, PrefixedUnit):
            cy_cl = PrefixedUnit(cy_cl, '', 'cy/CL')
        if not unit:
            unit = default
        
        clock = self.machine['clock']
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        elements_per_cacheline = int(float(self.machine['cacheline size'])) / element_size
        it_s = clock/cy_cl*elements_per_cacheline
        it_s.unit = 'It/s'
        flops_per_it = sum(self.kernel._flops.values())
        performance = it_s*flops_per_it
        performance.unit = 'FLOP/s'
        
        return {'It/s': it_s,
                'cy/CL': cy_cl,
                'FLOP/s': performance}[unit]

    def report(self):
        if self._args and self._args.verbose > 1:
            for r in self.results['memory hierarchy']:
                print('Trace legth per access in {}:'.format(r['level']), r['trace length'])
                print('Hits in {}:'.format(r['level']), r['total hits'], r['hits'])
                print('Misses in {}: {} ({}CL):'.format(
                    r['level'], r['total misses'], r['total lines misses']),
                    r['misses'])
                print('Evicts from {} {} ({}CL):'.format(
                    r['level'], r['total evicts'], r['total lines evicts']),
                    r['evicts'])
                if 'memory bandwidth' in r:
                    print('memory bandwidth: {} (from {} kernel benchmark)'.format(
                        r['memory bandwidth'], r['memory bandwidth kernel']))

        for level, cycles in self.results['cycles']:
            print('{} = {} cy/CL'.format(level, cycles))


class ECMCPU:
    """
    class representation of the Execution-Cache-Memory Model (only the operation part)

    more info to follow...
    """

    name = "Execution-Cache-Memory (CPU operations only)"

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
        self.kernel = kernel
        self.machine = machine
        self._args = args
        self._parser = parser

        if args:
            # handle CLI info
            if self._args.asm_block not in ['auto', 'manual']:
                try:
                    self._args.asm_block = int(args.asm_block)
                except ValueError:
                    parser.error('--asm-block can only be "auto", "manual" or an integer')

    def analyze(self):
        # For the IACA/CPU analysis we need to compile and assemble
        asm_name = self.kernel.compile(
            self.machine['compiler'], compiler_args=self.machine['compiler flags'])
        bin_name = self.kernel.assemble(
            self.machine['compiler'], asm_name, iaca_markers=True, asm_block=self._args.asm_block)

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

        # Get total cycles per loop iteration
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

        # Compile most relevant information
        T_OL = max(
            [v for k, v in port_cycles.items() if k in self.machine['overlapping ports']])
        T_nOL = max(
            [v for k, v in port_cycles.items() if k in self.machine['non-overlapping ports']])
        
        # Use IACA throughput prediction if it is slower then T_nOL
        if T_nOL < cl_throughput:
            T_OL = cl_throughput
        
        # Use latency if requested
        if self._args.latency:
            T_OL = cl_latency
        
        # Create result dictionary
        self.results = {
            'port cycles': port_cycles,
            'cl throughput': cl_throughput,
            'cl latency': cl_latency,
            'uops': uops,
            'T_nOL': T_nOL,
            'T_OL': T_OL,
            'IACA output': iaca_output,
            'IACA latency output': iaca_latency_output}


    def conv_cy(self, cy_cl, unit, default='cy/CL'):
        '''Convert cycles (cy/CL) to other units, such as FLOP/s or It/s'''
        if not isinstance(cy_cl, PrefixedUnit):
            cy_cl = PrefixedUnit(cy_cl, '', 'cy/CL')
        if not unit:
            unit = default
        
        clock = self.machine['clock']
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        elements_per_cacheline = int(float(self.machine['cacheline size'])) / element_size
        it_s = clock/cy_cl*elements_per_cacheline
        it_s.unit = 'It/s'
        flops_per_it = sum(self.kernel._flops.values())
        performance = it_s*flops_per_it
        performance.unit = 'FLOP/s'
        
        return {'It/s': it_s,
                'cy/CL': cy_cl,
                'FLOP/s': performance}[unit]

    def report(self):
        if self._args and self._args.verbose > 2:
            print("IACA Output:")
            print(self.results['IACA output'])
            print(self.results['IACA latency output'])
            print()
        
        if self._args and self._args.verbose > 1:
            print('Ports and cycles:', self.results['port cycles'])
            print('Uops:', self.results['uops'])
            
            print('Throughput: {}'.format(
                self.conv_cy(self.results['cl throughput'], self._args.unit)))
            
            print('Latency: {}'.format(
                self.conv_cy(self.results['cl latency'], self._args.unit)))
        
        print('T_nOL = {} cy/CL'.format(self.results['T_nOL']))
        print('T_OL = {} cy/CL'.format(self.results['T_OL']))


class ECM:
    """
    class representation of the Execution-Cache-Memory Model (data and operations)

    more info to follow...
    """

    name = "Execution-Cache-Memory"

    @classmethod
    def configure_arggroup(cls, parser):
        # they are being configured in ECMData and ECMCPU
        parser.add_argument(
            '--ecm-plot',
            help='Filename to save ECM plot to (supported extensions: pdf, png, svg and eps)')

    def __init__(self, kernel, machine, args=None, parser=None):
        """
        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args

        if args:
            # handle CLI info
            pass

        self._CPU = ECMCPU(kernel, machine, args, parser)
        self._data = ECMData(kernel, machine, args, parser)

    def analyze(self):
        self._CPU.analyze()
        self._data.analyze()
        self.results = copy.deepcopy(self._CPU.results)
        self.results.update(copy.deepcopy(self._data.results))
        
        # Saturation/multi-core scaling analysis
        # very simple approach. Assumptions are:
        #  - bottleneck is always LLC-MEM
        #  - all caches scale with number of cores (bw AND size(WRONG!))
        self.results['scaling cores'] = int(math.ceil(
                max(self.results['T_OL'],
                self.results['T_nOL']+sum([c[1] for c in self.results['cycles']])) / \
            self.results['cycles'][-1][1]))

    def report(self):
        report = ''
        if self._args and self._args.verbose > 1:
            self._CPU.report()
            self._data.report()
        
        total_cycles = max(
            self.results['T_OL'],
            sum([self.results['T_nOL']]+[i[1] for i in self.results['cycles']]))
        report += '{{ {} || {} | {} }} = {:.2f} cy/CL'.format(
            self.results['T_OL'],
            self.results['T_nOL'],
            ' | '.join([str(i[1]) for i in self.results['cycles']]),
            total_cycles)
        
        if self._args.unit:
            report += ' = {}'.format(self._CPU.conv_cy(total_cycles, self._args.unit))
        
        report += '\n{{ {} \ {} }} cy/CL'.format(
            max(self.results['T_OL'], self.results['T_nOL']),
            ' \ '.join(['{:.2f}'.format(max(sum(map(lambda x: x[1], self.results['cycles'][:i+1])) +
            self.results['T_nOL'], self.results['T_OL']))
                for i in range(len(self.results['cycles']))]),
            total_cycles)

        report += '\nsaturating at {} cores'.format(self.results['scaling cores'])

        print(report)

        if self._args and self._args.ecm_plot:
            assert plot_support, "matplotlib couldn't be imported. Plotting is not supported."

            fig = plt.figure(frameon=False)
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
            ax = fig.add_subplot(1, 1, 1)

            sorted_overlapping_ports = sorted(
                map(lambda p: (p, self.results['port cycles'][p]),
                    self.machine['overlapping ports']),
                key=lambda x: x[1])

            yticks_labels = []
            yticks = []
            xticks_labels = []
            xticks = []

            # Plot configuration
            height = 0.9

            i = 0
            # T_OL
            colors = [(254./255, 177./255., 178./255.)] + [(255./255., 255./255., 255./255.)] * \
                (len(sorted_overlapping_ports) - 1)
            for p, c in sorted_overlapping_ports:
                ax.barh(i, c, height, align='center', color=colors.pop())
                if i == len(sorted_overlapping_ports)-1:
                    ax.text(c/2.0, i, '$T_\mathrm{OL}$', ha='center', va='center')
                yticks_labels.append(p)
                yticks.append(i)
                i += 1
            xticks.append(sorted_overlapping_ports[-1][1])
            xticks_labels.append('{:.1f}'.format(sorted_overlapping_ports[-1][1]))

            # T_nOL + memory transfers
            y = 0
            colors = [(187./255., 255/255., 188./255.)] * (len(self.results['cycles'])) + \
                [(119./255, 194./255., 255./255.)]
            for k, v in [('nOL', self.results['T_nOL'])]+self.results['cycles']:
                ax.barh(i, v, height, y, align='center', color=colors.pop())
                ax.text(y+v/2.0, i, '$T_\mathrm{'+k+'}$', ha='center', va='center')
                xticks.append(y+v)
                xticks_labels.append('{:.1f}'.format(y+v))
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

