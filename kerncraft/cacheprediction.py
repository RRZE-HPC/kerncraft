#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from itertools import chain
from collections import defaultdict
from pprint import pprint

import sympy


# Not useing functools.cmp_to_key, because it does not exit in python 2.x
def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


class CachePredictor(object):
    '''
    Predictor class used to interface LayerCondition and CacheSimulation with model classes.
    
    It's goal is to predict the amount of hits and misses it takes to process one cache line worth
    of work under a steady state assumption.
    
    
    Only stubs here.
    '''
    def __init__(self, kernel, machine):
        self.kernel = kernel
        self.machine = machine

    def get_hits(self):
        '''Returns a list with cache lines of hits per cache level'''
        raise NotImplementedError("CachePredictor should only be used as a base class.")

    def get_misses(self):
        '''Returns a list with cache lines of misses per cache level'''
        raise NotImplementedError("CachePredictor should only be used as a base class.")

    def get_evicts(self):
        '''Returns a list with cache lines of misses per cache level'''
        raise NotImplementedError("CachePredictor should only be used as a base class.")

    def get_infos(self):
        '''Returns verbose information about the predictor'''
        raise NotImplementedError("CachePredictor should only be used as a base class.")

class LayerConditionPredictor(CachePredictor):
    '''
    Predictor class based on layer condition analysis.
    '''
    def __init__(self, kernel, machine):
        CachePredictor.__init__(self, kernel, machine)
        
        # check that layer conditions can be applied on this kernel:
        # 1. All iterations may only have a step width of 1
        loop_stack = list(self.kernel.get_loop_stack())
        if any([l['increment'] != 1 for l in loop_stack]):
            raise ValueError("Can not apply layer-condition, since not all loops are of step "
                             "length 1.")
        
        # 2. The order of iterations must be reflected in the order of indices in all array 
        #    references containing the inner loop index. If the inner loop index is not part of the
        #    reference, the reference is simply ignored
        # TODO support flattend array indexes
        references = list(self.kernel.index_order())
        for aref in references:
            for i, idx_names in enumerate(aref):
                if any([loop_stack[i]['index'] != idx.name for idx in idx_names]):
                    raise ValueError("Can not apply layer-condition, order of indices in array "
                                     "does not follow order of loop indices. Single-dimension is "
                                     "currently not supported.")
        
        # 3. Indices may only increase with one
        # TODO use a public interface, not self.kernel._*
        for arefs in chain(chain(*self.kernel._sources.values()),
                           chain(*self.kernel._destinations.values())):
            if arefs is None:
                continue
            for i, expr in enumerate(arefs):
                diff = sympy.diff(expr, sympy.Symbol(loop_stack[i]['index']))
                if diff != 0 and diff != 1:
                    # TODO support -1 aswell
                    raise ValueError("Can not apply layer-condition, array references may not "
                                     "increment more then one per iteration.")
                
        # FIXME handle multiple datatypes
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        
        accesses = {}
        destinations = set()
        distances = []
        results = {'accesses': accesses,
                   'distances': distances,
                   'destinations': destinations}
        for var_name in self.kernel.variables:
            # Gather all access to current variable/array
            accesses[var_name] = self.kernel._sources.get(var_name, []) + \
                                 self.kernel._destinations.get(var_name, [])
            # Skip non-variable offsets (acs is [None, None, None] or the like)
            if not any(accesses[var_name]):
                continue
            destinations.update(
                [(var_name, tuple(r)) for r in self.kernel._destinations.get(var_name, [])])
            acs = accesses[var_name]
            # Transform them into sympy expressions
            acs = [self.kernel.access_to_sympy(var_name, r) for r in acs]
            # Replace constants with their integer counter parts, to make the entries sortable
            acs = [self.kernel.subs_consts(e) for e in acs]
            # Sort accesses by decreasing order
            acs.sort(reverse=True)

            # Create reuse distances by substracting accesses pairwise in decreasing order
            distances += [(acs[i-1]-acs[i]).simplify() for i in range(1,len(acs))]
            # Add infinity for each array
            distances.append(sympy.oo)
            
        # Sort distances by decreasing order
        distances.sort(reverse=True)
        # Create copy of distances in bytes:
        distances_bytes = [d*element_size for d in distances]
        # CAREFUL! From here on we are working in byte offsets and not in indices anymore.
        
        results['distances_bytes'] = distances_bytes
        results['cache'] = []
        
        sum_array_sizes = sum(self.kernel.array_sizes(in_bytes=True, subs_consts=True).values())
        
        for c in  self.machine.get_cachesim().levels(with_mem=False):
            # Assuming increasing order of cache sizes
            hits = 0
            misses = len(distances_bytes)
            cache_requirement = 0
            
            # Test for full caching
            if c.size() > sum_array_sizes:
                hits = misses
                misses = 0
                cache_requirement = sum_array_sizes
            else:
                for tail in sorted(set(distances_bytes), reverse=True):
                    # Assuming decreasing order of tails
                    # Ignoring infinity tail:
                    if tail is sympy.oo:
                        continue
                    cache_requirement = (
                        sum([d for d in distances_bytes if d<=tail]) +  # Sum of inter-access caches
                        tail*len([d for d in distances_bytes if d>tail]))  # Tails
                    
                    if cache_requirement <= c.size():
                        # If we found a tail that fits into our available cache size
                        # note hits and misses and break
                        hits = len([d for d in distances_bytes if d<=tail])
                        misses = len([d for d in distances_bytes if d>tail])
                        break
            
            # Resulting analysis for current cache level
            results['cache'].append({
                'name': c.name,
                'hits': hits,
                'misses': misses,
                'evicts': len(destinations),
                'requirement': cache_requirement,
                'tail': tail})
        
        self.results = results

    def get_hits(self):
        '''Returns a list with cache lines of hits per cache level'''
        return [c['hits'] for c in self.results['cache']]

    def get_misses(self):
        '''Returns a list with cache lines of misses per cache level'''
        return [c['misses'] for c in self.results['cache']]

    def get_evicts(self):
        '''Returns a list with cache lines of misses per cache level'''
        return [c['evicts'] for c in self.results['cache']]

    def get_infos(self):
        '''Returns verbose information about the predictor'''
        return self.results


class CacheSimulationPredictor(CachePredictor):
    '''
    Predictor class based on layer condition analysis.
    '''
    def __init__(self, kernel, machine):
        CachePredictor.__init__(self, kernel, machine)
        # Get the machine's cache model and simulator
        csim = self.machine.get_cachesim()
        
        # FIXME handle multiple datatypes
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        cacheline_size = self.machine['cacheline size']
        elements_per_cacheline = int(cacheline_size // element_size)
        
        # Gathering some loop information:
        inner_loop = list(self.kernel.get_loop_stack(subs_consts=True))[-1]
        inner_index = sympy.Symbol(inner_loop['index'], positive=True)
        inner_increment = inner_loop['increment']# Calculate the number of iterations for warm-up
        max_cache_size = max(map(lambda c: c.size(), csim.levels(with_mem=False)))
        max_array_size = max(self.kernel.array_sizes(in_bytes=True, subs_consts=True).values())

        offsets = []
        if max_array_size < max_cache_size:
            # Full caching possible, go through all itreration before actual initialization
            offsets = list(self.kernel.compile_global_offsets(
                iteration=range(0, self.kernel.iteration_length())))

        # Regular Initialization
        warmup_indices = {
            sympy.Symbol(l['index'], positive=True): ((l['stop']-l['start'])//l['increment'])//3
            for l in self.kernel.get_loop_stack(subs_consts=True)}
        warmup_iteration_count = self.kernel.indices_to_global_iterator(warmup_indices)
        
        # Make sure we are not handeling gigabytes of data, but 1.5x the maximum cache size
        while warmup_iteration_count*element_size > max_cache_size*1.5:
            for index in [sympy.Symbol(l['index'], positive=True)
                          for l in self.kernel.get_loop_stack()]:
                if warmup_indices[index] > 1:
                    warmup_indices[index] -= 1
                    break
            warmup_iteration_count = self.kernel.indices_to_global_iterator(warmup_indices)
        
        # Align iteration count with cachelines
        # do this by aligning either writes (preferred) or reads:
        # Assumption: writes (and reads) increase linearly
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
        self.stats = list(csim.stats())
        self.first_dim_factor = first_dim_factor

    def get_hits(self):
        '''Returns a list with cache lines of hits per cache level'''
        return [self.stats[cache_level]['HIT_count']/self.first_dim_factor
                for cache_level in range(len(self.machine['memory hierarchy'][:-1]))]

    def get_misses(self):
        '''Returns a list with cache lines of misses per cache level'''
        return [self.stats[cache_level]['MISS_count']/self.first_dim_factor
                for cache_level in range(len(self.machine['memory hierarchy'][:-1]))]

    def get_evicts(self):
        '''Returns a list with cache lines of misses per cache level'''
        return [self.stats[cache_level+1]['STORE_count']/self.first_dim_factor
                for cache_level in range(len(self.machine['memory hierarchy'][:-1]))]

    def get_infos(self):
        '''Returns verbose information about the predictor'''
        first_dim_factor = self.first_dim_factor
        infos = {'memory hierarchy': [], 'cache stats': self.stats,
                        'cachelines in stats': first_dim_factor}
        for cache_level, cache_info in list(enumerate(self.machine['memory hierarchy']))[:-1]:
            infos['memory hierarchy'].append({
                'index': len(infos['memory hierarchy']),
                'level': '{}'.format(cache_info['level']),
                'total misses': self.stats[cache_level]['MISS_byte']/first_dim_factor,
                'total hits': self.stats[cache_level]['HIT_byte']/first_dim_factor,
                'total evicts': self.stats[cache_level]['STORE_byte']/first_dim_factor,
                'total lines misses': self.stats[cache_level]['MISS_count']/first_dim_factor,
                'total lines hits': self.stats[cache_level]['HIT_count']/first_dim_factor,
                # FIXME assumption for line evicts: all stores are consecutive
                'total lines evicts': self.stats[cache_level+1]['STORE_count']/first_dim_factor,
                'cycles': None})
        return infos
