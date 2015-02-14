#!/usr/bin/env python

from __future__ import print_function
from textwrap import dedent
from pprint import pprint
from functools import reduce
import operator
import math
import copy
import sys

from kerncraft import intervals
from kerncraft.kernel import Kernel
from kerncraft.machinemodel import MachineModel
from kerncraft.prefixedunit import PrefixedUnit

# Datatype sizes in bytes
datatype_size = {'double': 8, 'float': 4}


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


def find(f, seq):
  """Return first item in sequence where f(item) == True."""
  for item in seq:
    if f(item): 
      return item
    

def flatten_dict(d):
    '''
    transforms 2d-dict d[i][k] into a new 1d-dict e[(i,k)] with 2-tuple keys
    '''
    e = {}
    for k in d.keys():
        for l in d[k].keys():
            e[(k,l)] = d[k][l]
    return e


class Roofline:
    """
    class representation of the Roofline Model
    
    more info to follow...
    """
    
    name = "Roofline"
    
    @classmethod
    def configure_arggroup(cls, parser):
        pass
    
    def __init__(self, kernel, machine, args=None):
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
        # TODO how to handle multiple datatypes (with different size)?
        element_size = datatype_size['double']
        elements_per_cacheline = int(float(self.machine['cacheline size'])) / element_size
        
        first = first - first%elements_per_cacheline
        last = last - last%elements_per_cacheline + elements_per_cacheline - 1
        
        return [first, last]
    
    def calculate_cache_access(self):
        results = {'bottleneck level': None, 'mem bottlenecks': []}
        
        read_offsets = {var_name: dict() for var_name in self.kernel._variables.keys()}
        write_offsets = {var_name: dict() for var_name in self.kernel._variables.keys()}
        iteration_offsets = {var_name: dict() for var_name in self.kernel._variables.keys()}
        
        # TODO how to handle multiple datatypes (with different size)?
        element_size = datatype_size['double']
        elements_per_cacheline = int(float(self.machine['cacheline size'])) / element_size
        
        loop_order = ''.join(map(lambda l: l[0], self.kernel._loop_stack))
        
        for var_name in self.kernel._variables.keys():
            var_type, var_dims = self.kernel._variables[var_name]
            
            # Skip the following access: (they are hopefully kept in registers)
            #   - scalar values
            if var_dims is None: continue
            #   - access does not change with inner-most loop index
            writes = filter(lambda acs: loop_order[-1] in map(lambda a: a[1], acs),
                self.kernel._destinations.get(var_name, []))
            reads = filter(lambda acs: loop_order[-1] in map(lambda a: a[1], acs),
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
        total_lines_misses = {}
        total_lines_hits = {}
        total_lines_evicts = {}
        
        # Check for layer condition towards all cache levels (except main memory/last level)
        for cache_level, cache_info in list(enumerate(self.machine['memory hierarchy']))[:-1]:
            cache_size = int(float(cache_info['size per group']))
            cache_cycles = cache_info['cycles per cacheline transfer']
            
            trace_length = 0
            updated_length = True
            while updated_length:
                updated_length = False
                
                # Initialize cache, misses, hits and evicts for current level
                cache = {}
                misses[cache_level] = {}
                hits[cache_level] = {}
                evicts[cache_level] = {}
                
                # We consider everythin a miss in the beginning
                # TODO here read and writes are treated the same, this implies write-allocate
                #      to support nontemporal stores, this needs to be changed
                for name in read_offsets.keys()+write_offsets.keys():
                    cache[name] = {}
                    misses[cache_level][name] = {}
                    hits[cache_level][name] = {}

                    for idx_order in read_offsets[name].keys()+write_offsets[name].keys():
                        cache[name][idx_order] = intervals.Intervals()
                        if cache_level-1 not in misses:
                            misses[cache_level][name][idx_order] = sorted(
                                read_offsets.get(name, {}).get(idx_order, []) + 
                                write_offsets.get(name, {}).get(idx_order, []),
                                reverse=True)
                        else:
                            misses[cache_level][name][idx_order] = \
                                 list(misses[cache_level-1][name][idx_order])
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
                                cache[var_name][idx_order] &= intervals.Intervals(
                                    [cached_first, cached_last+1], sane=True)
                            else:
                                # There is no overlap, we can append the ranges onto one another
                                # TODO optimize this code section (and maybe merge with above)
                                new_cache = [self._expand_to_cacheline_blocks(o, o) for o in range(
                                    offset-iter_offset*trace_length, offset+1, iter_offset)]
                                new_cache = intervals.Intervals(*new_cache, sane=True)
                                cache[var_name][idx_order] &= new_cache
                                
                        trace_count += len(cache[var_name][idx_order]._data)
                        cache_used_size += len(cache[var_name][idx_order])*element_size
                
                # Calculate new possible trace_length according to free space in cache
                # TODO take CL blocked access into account
                # TODO make /2 customizable
                new_trace_length = trace_length + \
                    ((cache_size/2 - cache_used_size)/trace_count)/element_size
                
                if new_trace_length > trace_length:
                    trace_length = new_trace_length
                    updated_length = True
                
                # All writes to require the data to be evicted eventually
                evicts[cache_level] = \
                    {var_name: dict() for var_name in self.kernel._variables.keys()}
                for name in write_offsets.keys():
                    for idx_order in write_offsets[name].keys():
                        evicts[cache_level][name][idx_order] = list(write_offsets[name][idx_order])
            
            # Compiling stats
            total_misses[cache_level] = sum(map(lambda l: sum(map(len, l.values())),
                misses[cache_level].values()))
            total_hits[cache_level] = sum(map(lambda l: sum(map(len, l.values())),
                hits[cache_level].values()))
            total_evicts[cache_level] = sum(map(lambda l: sum(map(len, l.values())),
                evicts[cache_level].values()))
            
            total_lines_misses[cache_level] = sum(map(
                lambda o: sum(map(lambda n: len(blocking(n, elements_per_cacheline)), o.values())),
                misses[cache_level].values()))
            total_lines_hits[cache_level] = sum(map(lambda o: sum(map(lambda n:
                len(blocking(n, elements_per_cacheline)), o.values())),
                hits[cache_level].values()))
            total_lines_evicts[cache_level] = sum(map(lambda o: sum(map(lambda n: 
                len(blocking(n,elements_per_cacheline)), o.values())),
                evicts[cache_level].values()))
        
            # Calculate performance (arithmetic intensity * bandwidth with 
            # arithmetic intensity = flops / bytes transfered)
            bytes_transfered = (total_misses[cache_level]+total_evicts[cache_level])*element_size
            total_flops = sum(self.kernel._flops.values())
            arith_intens = float(total_flops)/float(bytes_transfered)
            
            # choose bw according to cache level and problem
            # first, compile stream counts at current cache level
            read_streams = 0
            for var_name in misses[cache_level].keys():
                for idx_order in misses[cache_level][var_name]:
                    read_streams += len(misses[cache_level][var_name][idx_order])
            write_streams = 0
            for var_name in evicts[cache_level].keys():
                for idx_order in evicts[cache_level][var_name]:
                    write_streams += len(evicts[cache_level][var_name][idx_order])
            read_write_streams = 0
            for var_name in set(evicts[cache_level].keys()) & set(misses[cache_level].keys()):
                for idx_order in set(evicts[cache_level][var_name].keys()) & \
                        set(misses[cache_level][var_name].keys()):
                    read_write_streams += len(set(evicts[cache_level][var_name][idx_order]) & 
                                              set(misses[cache_level][var_name][idx_order]))
            # second, try to find best fitting kernel (closest to stream seen stream counts):
            measurement_kernel = 'load'
            measurement_kernel_info = self.machine['benchmarks']['kernels'][measurement_kernel]
            for kernel_name, kernel_info in self.machine['benchmarks']['kernels'].items():
                if (read_streams >= kernel_info['read streams']['streams'] > 
                        measurement_kernel_info['read streams']['streams'] and
                        write_streams >= kernel_info['write streams']['streams'] >
                        measurement_kernel_info['write streams']['streams'] and
                        read_write_streams >= kernel_info['read+write streams']['streams'] > 
                        measurement_kernel_info['read+write streams']['streams']):
                    measurement_kernel = kernel_name
                    measurement_kernel_info = kernel_info
            
            # TODO choose smt and cores:
            threads_per_core, cores = 1, 1
            bw_measurements = self.machine['benchmarks']['measurements'][cache_info['level']]
            bw = bw_measurements[threads_per_core]['results'][measurement_kernel][cores]
            
            performance = arith_intens * float(bw)
            results['mem bottlenecks'].append({
                'performance': PrefixedUnit(performance, 'FLOP/s'),
                'level': self.machine['memory hierarchy'][cache_level]['level'] + '-' + \
                    self.machine['memory hierarchy'][cache_level+1]['level'],
                'arithmetic intensity': arith_intens,
                'bw kernel': measurement_kernel,
                'bandwidth': bw})
            if performance <= results.get('min performance', performance):
                results['bottleneck level'] = len(results['mem bottlenecks'])-1
                results['min performance'] = performance
        return results

    def analyze(self):
        self._results = self.calculate_cache_access()
    
    def report(self):
        max_flops = self.machine['clock']*sum(self.machine['FLOPs per cycle']['DP'].values())
        max_flops.unit = "FLOP/s"
        if self._args and self._args.verbose >= 1:
            print('Bottlnecks:')
            print('  level |   performance   |  bandwidth | bandwidth kernel')
            print('--------+-----------------+------------+-----------------')
            print('    CPU | {:>15} |            |'.format(max_flops))
            for b in self._results['mem bottlenecks']:
                print('{level:>7} | {performance:>15} | {bandwidth:>10} | {bw kernel:<8}'.format(
                    **b))
            print()
        
        # TODO support SP
        if self._results['min performance'] > max_flops:
            # CPU bound
            print('CPU bound')
            print('{!s} due to CPU max. FLOP/s'.format(max_flops))
        else:
            # Cache or mem bound
            print('Cache or mem bound')
            
            bottleneck = self._results['mem bottlenecks'][self._results['bottleneck level']]
            print('{!s} due to {} transfer bottleneck (bw with from {} benchmark)'.format(
                bottleneck['performance'],
                bottleneck['level'],
                bottleneck['bw kernel']))
        print('Arithmetic Intensity: {:.2f}'.format(bottleneck['arithmetic intensity']))
