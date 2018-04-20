#!/usr/bin/env python3
"""Cache prediction interface classes are gathered in this module."""
from itertools import chain

import sympy

from kerncraft.kernel import symbol_pos_int


class CachePredictor(object):
    """
    Predictor class used to interface LayerCondition and CacheSimulation with model classes.

    It's goal is to predict the amount of hits and misses it takes to process one cache line worth
    of work under a steady state assumption.

    Only stubs here.
    """

    def __init__(self, kernel, machine, cores=1):
        """Initialize cache predictor."""
        self.kernel = kernel
        self.machine = machine
        self.cores = cores

    def get_loads(self):
        """Return a list with number of loaded cache lines per memory hierarchy level."""
        raise NotImplementedError("CachePredictor should only be used as a base class.")

    def get_hits(self):
        """Return a list with number of hit cache lines per memory hierarchy level."""
        raise NotImplementedError("CachePredictor should only be used as a base class.")

    def get_misses(self):
        """Return a list with number of missed cache lines per memory hierarchy level."""
        raise NotImplementedError("CachePredictor should only be used as a base class.")

    def get_stores(self):
        """Return a list with number of stored cache lines per memory hierarchy level."""
        raise NotImplementedError("CachePredictor should only be used as a base class.")

    def get_evicts(self):
        """Return a list with number of evicted cache lines per memory hierarchy level."""
        raise NotImplementedError("CachePredictor should only be used as a base class.")

    def get_infos(self):
        """Return verbose information about the predictor."""
        raise NotImplementedError("CachePredictor should only be used as a base class.")


class LayerConditionPredictor(CachePredictor):
    """Predictor class based on layer condition analysis."""

    def __init__(self, kernel, machine, cores=1):
        """Initialize layer condition based predictor from kernel and machine object."""
        CachePredictor.__init__(self, kernel, machine, cores=cores)

        # check that layer conditions can be applied on this kernel:
        # 1. All iterations may only have a step width of 1
        loop_stack = list(self.kernel.get_loop_stack())
        if any([l['increment'] != 1 for l in loop_stack]):
            raise ValueError("Can not apply layer condition, since not all loops are of step "
                             "length 1.")

        # 2. The order of iterations must be reflected in the order of indices in all array
        #    references containing the inner loop index. If the inner loop index is not part of the
        #    reference, the reference is simply ignored
        index_order = [symbol_pos_int(l['index']) for l in loop_stack]
        for var_name, arefs in chain(self.kernel.sources.items(), self.kernel.destinations.items()):
            if next(iter(arefs)) is None:
                # Anything that is a scalar may be ignored
                continue
            for a in [self.kernel.access_to_sympy(var_name, a) for a in arefs]:
                for t in a.expand().as_ordered_terms():
                    # Check each and every term if they are valid according to loop order and array
                    # initialization
                    idx = t.free_symbols.intersection(index_order)

                    # Terms without any indices can be treat as constant offsets and are acceptable
                    if not idx:
                        continue

                    if len(idx) != 1:
                        raise ValueError("Only one loop counter may appear per term. "
                                         "Problematic term: {}.".format(t))
                    else:  # len(idx) == 1
                        idx = idx.pop()
                        # Check that number of multiplication match access order of iterator
                        pow_dict = {k: v for k, v in t.as_powers_dict().items()
                                    if k != idx}
                        stride_dim = sum(pow_dict.values())
                        error = False
                        try:
                            if loop_stack[-stride_dim-1]['index'] != idx.name:
                                error = True
                        except IndexError:
                            error = True
                        if error:
                            raise ValueError("Number of multiplications in index term does not "
                                             "match loop counter order. "
                                             "Problematic term: {}.".format(t))

        # 3. Indices may only increase with one
        inner_index = symbol_pos_int(loop_stack[-1]['index'])
        inner_increment = loop_stack[-1]['increment']
        # TODO use a public interface, not self.kernel._*
        for arefs in chain(chain(*self.kernel.sources.values()),
                           chain(*self.kernel.destinations.values())):
            if arefs is None:
                continue
            for expr in arefs:
                diff = expr.subs(inner_index, 1+inner_increment) - expr.subs(inner_index, 1)
                if diff != 0 and diff != 1:
                    # TODO support -1 aswell
                    raise ValueError("Can not apply layer condition, array references may not "
                                     "increment more then one per iteration.")
        # FIXME handle multiple datatypes
        element_size = self.kernel.datatypes_size[self.kernel.datatype]

        accesses = {}
        destinations = set()
        distances = []
        for var_name in self.kernel.variables:
            # Gather all access to current variable/array
            accesses[var_name] = self.kernel.sources.get(var_name, set()).union(
                                    self.kernel.destinations.get(var_name, set()))
            # Skip non-variable offsets, where acs is [None, None, None] (or similar) or only made
            # up from constant offsets
            if not any(accesses[var_name]) or not any(
                    [a == inner_index or a.coeff(inner_index) != 0
                     for a in chain.from_iterable(accesses[var_name])]):
                continue
            destinations.update(
                [(var_name, tuple(r)) for r in self.kernel.destinations.get(var_name, [])])
            acs = accesses[var_name]
            # Transform them into sympy expressions
            acs = [self.kernel.access_to_sympy(var_name, r) for r in acs]
            # Replace constants with their integer counter parts, to make the entries sortable
            acs = [self.kernel.subs_consts(e) for e in acs]
            # Sort accesses by decreasing order
            acs.sort(reverse=True)

            # Create reuse distances by substracting accesses pairwise in decreasing order
            distances += [(acs[i-1]-acs[i]).simplify() for i in range(1, len(acs))]
            # Add infinity for each array
            distances.append(sympy.oo)

        # Sort distances by decreasing order
        distances.sort(reverse=True)
        # Create copy of distances in bytes:
        distances_bytes = [d*element_size for d in distances]
        # CAREFUL! From here on we are working in byte offsets and not in indices anymore.

        # converting access sets to lists, otherwise pprint will fail during obligatory sorting step
        results = {'accesses': {k: list(v) for k,v in accesses.items()},
                   'distances': distances,
                   'destinations': destinations,
                   'distances_bytes': distances_bytes,
                   'cache': []}

        sum_array_sizes = sum(self.kernel.array_sizes(in_bytes=True, subs_consts=True).values())

        for c in self.machine.get_cachesim(self.cores).levels(with_mem=False):
            # Assuming increasing order of cache sizes
            hits = 0
            misses = len(distances_bytes)
            cache_requirement = 0

            tail = 0
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
                        # Sum of inter-access caches
                        sum([d for d in distances_bytes if d <= tail]) +
                        # Tails
                        tail*len([d for d in distances_bytes if d > tail]))

                    if cache_requirement <= c.size():
                        # If we found a tail that fits into our available cache size
                        # note hits and misses and break
                        hits = len([d for d in distances_bytes if d <= tail])
                        misses = len([d for d in distances_bytes if d > tail])
                        break

            # Resulting analysis for current cache level
            # TODO include loads and stores
            results['cache'].append({
                'name': c.name,
                'hits': hits,
                'misses': misses,
                'evicts': len(destinations) if c.size() < sum_array_sizes else 0,
                'requirement': cache_requirement,
                'tail': tail})

        self.results = results

    def get_loads(self):
        """Return a list with number of loaded cache lines per memory hierarchy level."""
        # TODO FIXME L1 loads need to be derived from accesses
        return [0]+[c['misses'] for c in self.results['cache']]

    def get_hits(self):
        """Return a list with number of hit cache lines per memory hierarchy level."""
        # At last level, all previous misses are hits
        return [c['hits'] for c in self.results['cache']]+[self.results['cache'][-1]['misses']]

    def get_misses(self):
        """Return a list with number of missed cache lines per memory hierarchy level."""
        # At last level, there are no misses
        return [c['misses'] for c in self.results['cache']]+[0]

    def get_stores(self):
        """Return a list with number of stored cache lines per memory hierarchy level."""
        # TODO FIXME L1 stores need to be derived from accesses
        return [0]+[c['evicts'] for c in self.results['cache']]

    def get_evicts(self):
        """Return a list with number of evicted cache lines per memory hierarchy level."""
        # At last level, there are no evicts
        return [c['evicts'] for c in self.results['cache']]+[0]

    def get_infos(self):
        """Return verbose information about the predictor."""
        return self.results


class CacheSimulationPredictor(CachePredictor):
    """Predictor class based on layer condition analysis."""

    def __init__(self, kernel, machine, cores=1):
        """Initialize cache simulation based predictor from kernel and machine object."""
        CachePredictor.__init__(self, kernel, machine, cores)
        # Get the machine's cache model and simulator
        csim = self.machine.get_cachesim(self.cores)

        # FIXME handle multiple datatypes
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        cacheline_size = self.machine['cacheline size']
        elements_per_cacheline = int(cacheline_size // element_size)

        # Gathering some loop information:
        inner_loop = list(self.kernel.get_loop_stack(subs_consts=True))[-1]
        inner_index = symbol_pos_int(inner_loop['index'])
        inner_increment = inner_loop['increment']  # Calculate the number of iterations for warm-up
        max_cache_size = max(map(lambda c: c.size(), csim.levels(with_mem=False)))
        max_array_size = max(self.kernel.array_sizes(in_bytes=True, subs_consts=True).values())

        # Regular Initialization
        warmup_indices = {
            symbol_pos_int(l['index']): ((l['stop']-l['start'])//l['increment'])//3
            for l in self.kernel.get_loop_stack(subs_consts=True)}
        warmup_iteration_count = self.kernel.indices_to_global_iterator(warmup_indices)
        # Make sure we are not handeling gigabytes of data, but 1.5x the maximum cache size
        while warmup_iteration_count * element_size > max_cache_size*1.5:
            # Decreasing indices (starting from outer), until total size is small enough
            for l in self.kernel.get_loop_stack():
                index = symbol_pos_int(l['index'])
                if warmup_indices[index] > l['start']:
                    warmup_indices[index] -= 1
                    # get offset in iterations and size that a change on this level provokes
                    diff_iterations = (warmup_iteration_count -
                                       self.kernel.indices_to_global_iterator(warmup_indices))
                    diff_size = diff_iterations*element_size
                    warmup_indices[index] = max(
                        l['start'],
                        (warmup_iteration_count - (max_cache_size*1.5)// element_size) // diff_size
                    )
                    break
            warmup_iteration_count = self.kernel.indices_to_global_iterator(warmup_indices)

        offsets = []
        if warmup_iteration_count*element_size < max_cache_size or max_array_size < max_cache_size:
            # Full caching possible, go through all itreration before actual initialization
            offsets = list(self.kernel.compile_global_offsets(
                iteration=range(0, self.kernel.iteration_length())))

        # Align iteration count with cachelines
        # do this by aligning either writes (preferred) or reads
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
            (int(first_offset) >> csim.first_level.cl_bits << csim.first_level.cl_bits)
        warmup_iteration_count -= (diff//element_size)//inner_increment
        warmup_indices = self.kernel.global_iterator_to_indices(warmup_iteration_count)

        offsets += list(self.kernel.compile_global_offsets(
            iteration=range(0, warmup_iteration_count)))

        # Do the warm-up
        csim.loadstore(offsets, length=element_size)
        # FIXME compile_global_offsets should already expand to element_size

        # Reset stats to conclude warm-up phase
        csim.reset_stats()

        # Benchmark iterations:
        # Starting point is one past the last warmup element
        bench_iteration_start = warmup_iteration_count
        # End point is the end of the current dimension (cacheline alligned)
        first_dim_factor = int((inner_loop['stop'] - warmup_indices[inner_index] - 1)
                               // (elements_per_cacheline//inner_increment))
        # If end point is less than one cacheline away, go beyond for 100 cachelines and
        # warn user of potentially inaccurate results
        if first_dim_factor == 0:
            # TODO a nicer solution would be to do less warmup iterations to select a
            # cacheline within a first dimension, if possible
            print('Warning: (automatic) warmup vs benchmark iteration choice was not perfect '
                  'and may lead to inaccurate cache miss predictions. This is most likely the '
                  'result of too few inner loop iterations ({} from {} to {}).'.format(
                      inner_loop['index'], inner_loop['start'], inner_loop['stop']
                  ))
            first_dim_factor = 100
        bench_iteration_end = (bench_iteration_start +
                               elements_per_cacheline*inner_increment*first_dim_factor)

        # compile access needed for one cache-line
        offsets = list(self.kernel.compile_global_offsets(
            iteration=range(bench_iteration_start, bench_iteration_end)))
        # simulate
        csim.loadstore(offsets, length=element_size)
        # FIXME compile_global_offsets should already expand to element_size

        # use stats to build results
        self.stats = list(csim.stats())
        self.first_dim_factor = first_dim_factor

    def get_loads(self):
        """Return a list with number of loaded cache lines per memory hierarchy level."""
        return [self.stats[cache_level]['LOAD_count'] / self.first_dim_factor
                for cache_level in range(len(self.machine['memory hierarchy']))]

    def get_hits(self):
        """Return a list with number of hit cache lines per memory hierarchy level."""
        return [self.stats[cache_level]['HIT_count']/self.first_dim_factor
                for cache_level in range(len(self.machine['memory hierarchy']))]

    def get_misses(self):
        """Return a list with number of missed cache lines per memory hierarchy level."""
        return [self.stats[cache_level]['MISS_count']/self.first_dim_factor
                for cache_level in range(len(self.machine['memory hierarchy']))]
    
    def get_stores(self):
        """Return a list with number of stored cache lines per memory hierarchy level."""
        return [self.stats[cache_level]['STORE_count']/self.first_dim_factor
                for cache_level in range(len(self.machine['memory hierarchy']))]

    def get_evicts(self):
        """Return a list with number of evicted cache lines per memory hierarchy level."""
        return [self.stats[cache_level]['EVICT_count']/self.first_dim_factor
                for cache_level in range(len(self.machine['memory hierarchy']))]

    def get_infos(self):
        """Return verbose information about the predictor."""
        first_dim_factor = self.first_dim_factor
        infos = {'memory hierarchy': [], 'cache stats': self.stats,
                 'cachelines in stats': first_dim_factor}
        for cache_level, cache_info in list(enumerate(self.machine['memory hierarchy'])):
            infos['memory hierarchy'].append({
                'index': len(infos['memory hierarchy']),
                'level': '{}'.format(cache_info['level']),
                'total loads': self.stats[cache_level]['LOAD_byte']/first_dim_factor,
                'total misses': self.stats[cache_level]['MISS_byte']/first_dim_factor,
                'total hits': self.stats[cache_level]['HIT_byte']/first_dim_factor,
                'total stores': self.stats[cache_level]['STORE_byte']/first_dim_factor,
                'total evicts': self.stats[cache_level]['EVICT_byte']/first_dim_factor,
                'total lines load': self.stats[cache_level]['LOAD_count']/first_dim_factor,
                'total lines misses': self.stats[cache_level]['MISS_count']/first_dim_factor,
                'total lines hits': self.stats[cache_level]['HIT_count']/first_dim_factor,
                'total lines stores': self.stats[cache_level]['STORE_count']/first_dim_factor,
                'total lines evicts': self.stats[cache_level]['EVICT_count']/first_dim_factor,
                'cycles': None})
        return infos
