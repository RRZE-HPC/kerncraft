#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from itertools import chain
from collections import defaultdict

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
    Predictor classed based on layer condition analysis.
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
        
        # calculate_cache_access()
        
        # FIXME handle multiple datatypes
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        
        results = {'dimensions': {}}
        
        def sympy_compare(a,b):
            c = 0
            for i in range(min(len(a), len(b))):
                s = a[i] - b[i]
                if sympy.simplify(s > 0):
                    c = -1
                elif sympy.simplify(s == 0):
                    c = 0
                else:
                    c = 1
                if c != 0: break
            return c
        
        accesses = defaultdict(list)
        sympy_accesses = defaultdict(list)
        for var_name in self.kernel.variables:
            for r in self.kernel._sources.get(var_name, []):
                if r is None: continue
                accesses[var_name].append(r)
                sympy_accesses[var_name].append(self.kernel.access_to_sympy(var_name, r))
            for w in self.kernel._destinations.get(var_name, []):
                if w is None: continue
                accesses[var_name].append(w)
                sympy_accesses[var_name].append(self.kernel.access_to_sympy(var_name, w))
            # order accesses by increasing order
            accesses[var_name].sort(key=cmp_to_key(sympy_compare))#cmp=sympy_compare)
        
        results['accesses'] = accesses
        results['sympy_accesses'] = sympy_accesses
        
        # For each dimension (1D, 2D, 3D ... nD)
        for dimension in range(1, len(list(self.kernel.get_loop_stack()))+1):
            results['dimensions'][dimension] = {}
            
            slices = defaultdict(list)
            slices_accesses = defaultdict(list)
            for var_name in accesses:
                for a in accesses[var_name]:
                    # slices are identified by the tuple of indices of higher dimensions
                    slice_id = tuple([var_name, tuple(a[:-dimension])])
                    slices[slice_id].append(a)
                    slices_accesses[slice_id].append(self.kernel.access_to_sympy(var_name, a))
            results['dimensions'][dimension]['slices'] = slices
            results['dimensions'][dimension]['slices_accesses'] = slices_accesses
            
            slices_distances = defaultdict(list)
            for k,v in slices_accesses.items():
                for i in range(1, len(v)):
                    slices_distances[k].append((v[i-1] - v[i]).simplify())
            results['dimensions'][dimension]['slices_distances'] = slices_distances
            
            # Check that distances contain only free_symbols based on constants
            for dist in chain(*slices_distances.values()):
                if any([s not in self.kernel.constants.keys() for s in dist.free_symbols]):
                    raise ValueError("Some distances are not based on non-constants: "+str(dist))
            
            # Sum of lengths between relative distances
            slices_sum = sum([sum(dists) for dists in slices_distances.values()])
            results['dimensions'][dimension]['slices_sum'] = slices_sum
            
            # Max of lengths between relative distances
            # Work-around, the arguments with the most symbols get to stay
            # FIXME, may not be correct in all cases. e.g., N+M vs. N*M
            def FuckedUpMax(*args):
                if len(args) == 1:
                    return args[0]
                # expand all expressions:
                args = [a.expand() for a in args]
                # Filter expressions with less than the maximum number of symbols
                max_symbols = max([len(a.free_symbols) for a in args])
                args = list(filter(lambda a: len(a.free_symbols) == max_symbols, args))
                if max_symbols == 0:
                    return sympy.Max(*args)
                # Filter symbols with lower exponent
                max_coeffs = 0
                for a in args:
                    for s in a.free_symbols:
                        max_coeffs = max(max_coeffs, len(sympy.Poly(a, s).all_coeffs()))
                def coeff_filter(a):
                    return max(
                        0, 0,
                        *[len(sympy.Poly(a, s).all_coeffs()) for s in a.free_symbols]) == max_coeffs
                args = list(filter(coeff_filter, args))

                m = sympy.Max(*args)
                #if m.is_Function:
                #    raise ValueError("Could not resolve {} to maximum.".format(m))
                return m
            
            slices_max = FuckedUpMax(*[FuckedUpMax(*dists) for dists in slices_distances.values()])
            results['dimensions'][dimension]['slices_sum'] = slices_sum
            
            # Nmber of slices
            slices_count = len(slices_accesses)
            results['dimensions'][dimension]['slices_count'] = slices_count
            
            # Cache requirement expression
            cache_requirement_bytes = (slices_sum + slices_max*slices_count)*element_size
            results['dimensions'][dimension]['cache_requirement_bytes'] = cache_requirement_bytes
            
            # Apply to all cache sizes
            csim = self.machine.get_cachesim()
            results['dimensions'][dimension]['caches'] = {}
            for cl in  csim.levels(with_mem=False):
                cache_equation = sympy.Eq(cache_requirement_bytes, cl.size())
                if len(self.kernel.constants.keys()) <= 1:
                    inequality = sympy.solve(sympy.LessThan(cache_requirement_bytes, cl.size()),
                                             *self.kernel.constants.keys())
                else:
                    # Sympy does not solve for multiple constants
                    inequality = sympy.LessThan(cache_requirement_bytes, cl.size())
                results['dimensions'][dimension]['caches'][cl.name] = {
                    'cache_size': cl.size(),
                    'equation': cache_equation,
                    'lt': inequality,
                    'eq': sympy.solve(cache_equation, *self.kernel.constants.keys())
                }
        
        self.results = results
        

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
