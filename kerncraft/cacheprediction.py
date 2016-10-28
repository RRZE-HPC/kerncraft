#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import sympy


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
    Predictor classed based on layer condition analysis.
    '''
    def __init__(self, kernel, machine):
        CachePredictor.__init__(self, kernel, machine)
        
        

    def get_hits(self):
        '''Returns a list with cache lines of hits per cache level'''
        #TODO
        return [1, 1, 1]

    def get_misses(self):
        '''Returns a list with cache lines of misses per cache level'''
        #TODO
        return [4, 4, 4]

    def get_evicts(self):
        '''Returns a list with cache lines of misses per cache level'''
        #TODO
        return [1,1,1]

    def get_infos(self):
        '''Returns verbose information about the predictor'''
        return {}


class CacheSimulationPredictor(CachePredictor):
    '''
    Predictor classed based on layer condition analysis.
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
        inner_increment = inner_loop['increment']# Calculate the number of iterations necessary for warm-up
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
