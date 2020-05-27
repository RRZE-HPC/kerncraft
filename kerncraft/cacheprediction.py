#!/usr/bin/env python3
"""Cache prediction interface classes are gathered in this module."""
from itertools import chain
from functools import cmp_to_key, reduce
from copy import copy
import fcntl
import compress_pickle

import sympy
from sympy.logic.boolalg import BooleanTrue, BooleanFalse
import numpy as np

from kerncraft.kernel import symbol_pos_int, KernelCode


# From https://stackoverflow.com/a/17511341 by dlitz
def ceildiv(a: int, b: int) -> object:
    return -(-a // b)


def uneven_tuple_cmp(a, b):
    length_diff = max(len(a), len(b)) - min(len(a), len(b))
    if len(a) < len(b):
        a = (0,)*length_diff + a
    elif len(b) < len(a):
        b = (0,)*length_diff + b
    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0


def sympy_expr_abs_distance_key(e):
    """
    Transform expression into tuple for sorting and comparison.

    e.g., sympy_expr_abs_distance_key(N**2 + 23) -> ((2, 0.0), (1, 0.0), (0, 23.0))

    Multiple variables are treated equal (N*M == N**2).
    """
    # Integers
    if type(e) is int or e.is_Integer:
        return ((0, abs(e)),)

    # Infinity
    if abs(e) is sympy.oo:
        return ((sympy.oo, sympy.oo),)

    # Expressions, replace all free_symbols with one
    first_s = None
    for i, s in enumerate(e.free_symbols):
        # Skip and remember first symbol
        if i == 0:
            first_s = s
            continue
        e = e.subs(s, first_s)
    e = e.expand()

    key = []
    # split into terms
    terms, gens = e.as_terms()
    assert gens == [first_s] or first_s is None and gens == [], \
        "Expression was split into unusable terms: {}, expected.".format(gens, first_s)
    # extract exponent and coefficient
    for term, (coeff, cpart, ncpart) in terms:
        coeff_real, coeff_imag = coeff
        assert coeff_imag == 0, "Not supporting imaginary coefficients."
        # Sort order: exponent (cpart), factor
        key.append(cpart + (coeff_real,))
    key[0] = (key[0][0], key[0][1])
    # build key
    key.sort(reverse=True)
    # add missing exponent, coefficient tuples
    i = 0
    for exponent in reversed(range(key[0][0]+1)):
        if len(key) > i and key[i][0] == exponent:
            i += 1
            continue
        else:
            key[i:i] = [(exponent, 0.0)]
            i += 1
    key = tuple(key)
    return key


def dimension_from_factor(dimension_factor):
    """
    Extract dimension from sympy factor

    >>> M = sympy.Symbol(name='M', positive=True, integer=True)
    >>> N = sympy.Symbol(name='N', positive=True, integer=True)
    >>> dimension_from_factor(M*N)
    2
    >>> dimension_from_factor(N**2)
    2
    >>> dimension_from_factor(23)
    0
    >>> dimension_from_factor(N)
    1

    :param dimension_factor: sympy expression, integer or symbol
    :return: dimension integer
    """
    if isinstance(dimension_factor, (sympy.Number, int)):
        return 0
    if isinstance(dimension_factor, sympy.Symbol):
        return 1
    # Replace all free symbols with one:
    if not dimension_factor.free_symbols:
        raise ValueError("dimension_factor is neither a number, a symbol nor an expression based "
                         "on symbols.")
    free_symbols = list(dimension_factor.free_symbols)
    for s in free_symbols[1:]:
        dimension_factor = dimension_factor.subs(s, free_symbols[0])
    if isinstance(dimension_factor, sympy.Pow):
        return dimension_factor.as_base_exp()[1]


# TODO support this delinearization in KernelCode?
def split_sympy_access_in_dim_offset_and_factor(expr, indices):
    """
    Extract dimensional offsets and factors from single-dimension sympy array index expression.

    :param expr: sympy expression
    :param indices: list of index symbols
    :return tuple of offsets with indices and dimension factors

    >>> M = sympy.Symbol(name='M', positive=True, integer=True)
    >>> N = sympy.Symbol(name='N', positive=True, integer=True)
    >>> L = sympy.Symbol(name='L', positive=True, integer=True)
    >>> i = sympy.Symbol(name='i', positive=True, integer=True)
    >>> j = sympy.Symbol(name='j', positive=True, integer=True)
    >>> k = sympy.Symbol(name='k', positive=True, integer=True)
    >>> split_sympy_access_in_dim_offset_and_factor(M*N*k + N*(j+1) + (i-1), [i, j, k])
    ((k, j + 1, i - 1), (M*N, N, 1))
    >>> split_sympy_access_in_dim_offset_and_factor(M*N*k - M*N + N*j+ N*2 + i, [i, j, k])
    ((k - 1, j + 2, i), (M*N, N, 1))
    >>> split_sympy_access_in_dim_offset_and_factor(N**2*(k-2) + N*j+ N*2 + i, [i, j, k])
    ((k - 2, j + 2, i), (N**2, N, 1))
    >>> split_sympy_access_in_dim_offset_and_factor(2*L*N*M + N*M*(k+1)+ N*(2+j) + i, [i, j, k])
    ((2, k + 1, j + 2, i), (L*M*N, M*N, N, 1))
    >>> split_sympy_access_in_dim_offset_and_factor(N*N*k + N*k + i + j, [i, j, k])
    ((k, k, i + j), (N**2, N, 1))
    >>> split_sympy_access_in_dim_offset_and_factor(N*N*k + N*k + i + j + 2, [i, j, k])
    ((k, k, i + j + 2), (N**2, N, 1))
    >>> split_sympy_access_in_dim_offset_and_factor(sympy.Integer(2), [i, j, k])
    ((2,), (1,))
    >>> split_sympy_access_in_dim_offset_and_factor(N*N*k, [i, j, k])
    Traceback (most recent call last):
        ...
    ValueError: Invalid expression. Some dimension terms seem to be missing.
    >>> split_sympy_access_in_dim_offset_and_factor(N*N*k + M*k + i + j, [i, j, k])
    Traceback (most recent call last):
        ...
    ValueError: Invalid expression. Dimensions do not seem to be coefficients of one another. M*k + N**2*k + i + j
    >>> split_sympy_access_in_dim_offset_and_factor(N*N*k + i + j, [i, j, k])
    Traceback (most recent call last):
        ...
    ValueError: Invalid expression. Some dimension terms seem to be missing.
    """
    # Expand polynomial expressions, e.g., N*(j+1) -> N*j + N
    eexpr = expr.expand()
    terms = eexpr.as_ordered_terms()

    # Find dimension factors belonging to indices and remember unmatched terms
    one = sympy.Integer(1)
    dimension_factors = set([one])
    terms_without_index = []
    for term in terms:
        for index in indices:
            c = term.coeff(index)
            if c:
                dimension_factors.add(c)
                break
        else:
            terms_without_index.append(term)

    # Find additional dimension factors, not based on indices, e.g., L*M*N
    for term in terms_without_index:
        for dim_factor in dimension_factors:
            c = term.coeff(dim_factor)
            if isinstance(c, sympy.Mul):
                dimension_factors.add(reduce(sympy.Mul, c.free_symbols) * dim_factor)
                break

    # Sort dimension factors by dimension (highest to lowest)
    dimension_factors = sorted(dimension_factors, key=dimension_from_factor, reverse=True)
    # Check that dimension_factors include preceding dim. factor
    for d1, d2 in zip(dimension_factors, dimension_factors[1:]):
        # Check if d2 is contained in d1 AND d2 is part of a power in d1
        if d1.as_coefficient(d2) is None and d1.as_coeff_exponent(d2)[1] < 1:
            raise ValueError("Invalid expression. Dimensions do not seem to be coefficients of one "
                             "another. {!r}".format(expr))

    # Check that all intermediate dimensions are represented
    if dimension_from_factor(dimension_factors[0]) + 1 != len(dimension_factors):
        raise ValueError("Invalid expression. Some dimension terms seem to be missing.")

    # Find offsets associated with dimension factors
    offsets = []
    for dim_factor in dimension_factors:
        offsets.append(sympy.Integer(0))
        for term in list(terms):
            c = term.as_coefficient(dim_factor)
            if c is not None:
                offsets[-1] += c
                terms.remove(term)
            # because 1.as_coefficient(1) is None, we need to consider integers manually
            elif isinstance(term, sympy.Integer) and dim_factor == 1:
                offsets[-1] += term
                terms.remove(term)

    # Reassemble and check for equality
    assert eexpr == reduce(
        sympy.Add, [o * df for o, df in zip(offsets, dimension_factors)]).expand(), \
        "Reassembly of expression from offsets and dimension_factors did not succeed. May be a bug."

    return tuple(offsets), tuple(dimension_factors)


def canonical_relational(rel):
    """
    Make relational canonical.

    Positive integer on rhs.
    Minimum integer factors on lhs.
    """
    if isinstance(rel, (BooleanTrue, BooleanFalse)):
        # Nothing to do
        return rel
    rel = rel.canonical.simplify().expand()
    lhs = rel.lhs
    rhs = rel.rhs
    rel_op = rel.rel_op

    # Move integer from lhs to rhs
    remainder = lhs.as_coeff_add()[0]
    lhs -= remainder
    rhs -= remainder

    # Find common divider and divide
    gcd = (lhs - rhs).factor().as_coeff_mul()[0]
    if gcd != 1:
        lhs /= max(gcd, -gcd)
        rhs /= max(gcd, -gcd)

    rel = sympy.relational.Relational(lhs, rhs, rel_op)
    if rhs < 0:
        rel = rel.reversedsign
    return rel


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

    def __init__(self, kernel, machine, cores=1, symbolic=False):
        """Initialize layer condition based predictor from kernel and machine object."""
        CachePredictor.__init__(self, kernel, machine, cores=cores)
        if isinstance(kernel, KernelCode):
            # Make use of caching for symbolic LC representation:
            file_name = 'LC_analysis.pickle.lzma'
            file_path = kernel.get_intermediate_location(
                file_name, machine_and_compiler_dependent=False, other_dependencies=[str(cores)])
            lock_mode, lock_fp = kernel.lock_intermediate(file_path)
            if lock_mode == fcntl.LOCK_SH:
                # use cache
                self.results = compress_pickle.load(file_path)
                lock_fp.close()  # release lock
            else:  # lock_mode == fcntl.LOCK_EX
                # needs update
                self.build_symbolic_LCs()
                compress_pickle.dump(self.results, file_path)
                lock_fp.close()  # release lock
        else:
            # No caching support without filename for kernel code
            self.build_symbolic_LCs()

        if not symbolic:
            self.desymbolize()

    def desymbolize(self):
        """Evaluate LCs and remove symbols"""
        for i, options in enumerate(self.results['cache']):
            for o in options:
                if self.kernel.subs_consts(o['condition']):
                    self.results['cache'][i] = o
                    break

    def build_symbolic_LCs(self):
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
        for aref in chain(chain(*self.kernel.sources.values()),
                          chain(*self.kernel.destinations.values())):
            if aref is None:
                continue

            for expr in aref:
                diff = expr.subs(inner_index, 1+inner_increment) - expr.subs(inner_index, 1)
                if diff != 0 and diff != 1:
                    # TODO support -1 aswell
                    raise ValueError("Can not apply layer condition, array references may not "
                                     "increment more then one per iteration.")
        # FIXME handle multiple datatypes
        element_size = self.kernel.datatypes_size[self.kernel.datatype]

        indices = list([symbol_pos_int(l[0]) for l in self.kernel._loop_stack])
        sympy_accesses = self.kernel.compile_sympy_accesses()
        accesses = {}
        destinations = set()
        distances = []
        for var_name in sorted(self.kernel.variables):
            # Gather all access to current variable/array and delinearize accesses
            accesses[var_name] = []
            dimension_factors = None
            for a in sympy_accesses[var_name]:
                o, df = split_sympy_access_in_dim_offset_and_factor(a, indices)
                accesses[var_name].append(o)
                if dimension_factors is None:
                    dimension_factors = df
                elif dimension_factors[-len(indices):] != df[-len(indices):]:
                    raise ValueError("Extracted dimension factors are different within one "
                                     "variable. Must be a bug. {!r} != {!r}".format(
                        df, dimension_factors))
                elif len(dimension_factors) < len(df):
                    dimension_factors = df
            # Skip non-variable offsets, where acs is [None, None, None] (or similar) or only made
            # up from constant offsets
            if not any(accesses[var_name]) or not any(
                    [a == inner_index or a.coeff(inner_index) != 0
                     for a in chain.from_iterable(accesses[var_name])]):
                continue
            destinations.update(
                [(var_name, tuple(r)) for r in self.kernel.destinations.get(var_name, [])])
            acs = list(accesses[var_name])
            # If accesses are of unequal length, pad with leading zero elements
            max_dims = max(map(len, acs))
            for i in range(len(acs)):
                if len(acs[i]) < max_dims:
                    acs[i] = (sympy.Integer(0),)*(max_dims-len(acs[i])) + acs[i]
            # Sort accesses by decreasing order
            acs.sort(reverse=True)
            # Transform back into sympy expressions
            for i in range(len(acs)):
                acs[i] = reduce(sympy.Add, [f*df for f, df in zip(acs[i], dimension_factors)])
            # Create reuse distances by substracting accesses pairwise in decreasing order
            distances += [(acs[i-1]-acs[i]).simplify() for i in range(1, len(acs))]
            # Add infinity for each array
            distances.append(sympy.oo)

        # Sort distances by decreasing order
        distances.sort(reverse=True, key=sympy_expr_abs_distance_key)
        # Create copy of distances in bytes:
        distances_bytes = [d*element_size for d in distances]
        # CAREFUL! From here on we are working in byte offsets and not in indices anymore.

        # converting access sets to lists, otherwise pprint will fail during obligatory sorting step
        results = {'accesses': {k: sorted(list(v), key=cmp_to_key(uneven_tuple_cmp))
                                for k,v in accesses.items()},
                   'distances': distances,
                   'destinations': destinations,
                   'distances_bytes': distances_bytes,
                   'cache': []}

        sum_array_sizes = sum(self.kernel.array_sizes(in_bytes=True, subs_consts=False).values())

        for c in self.machine.get_cachesim(self.cores).levels(with_mem=False):
            # Assuming increasing order of cache sizes
            options = []
            # Full caching
            options.append({
                'condition': canonical_relational(c.size() > sum_array_sizes),
                'hits': len(distances),
                'misses': 0,
                'evicts': 0,
                'tail': sympy.oo,
            })

            for tail in sorted(set([d.simplify().expand() for d in distances_bytes]), reverse=True,
                               key=sympy_expr_abs_distance_key):
                # Assuming decreasing order of tails
                # Ignoring infinity tail:
                if tail is sympy.oo:
                    continue
                cache_requirement = (
                    # Sum of inter-access caches
                    sum([d for d in distances_bytes
                         if sympy_expr_abs_distance_key(d) <= sympy_expr_abs_distance_key(tail)]
                        ) +
                    # Tails
                    tail*len([d for d in distances_bytes
                              if sympy_expr_abs_distance_key(d) >
                                 sympy_expr_abs_distance_key(tail)]))
                condition = canonical_relational(cache_requirement <= c.size())

                hits = len(
                    [d for d in distances_bytes
                     if sympy_expr_abs_distance_key(d) <= sympy_expr_abs_distance_key(tail)])
                misses = len(
                    [d for d in distances_bytes
                     if sympy_expr_abs_distance_key(d) > sympy_expr_abs_distance_key(tail)])

                # Resulting analysis
                options.append({
                    'condition': condition,
                    'hits': hits,
                    'misses': misses,
                    'evicts': len(destinations),
                    'tail': tail})

                # If we encountered a True condition, break to not include multiple such.
                if isinstance(condition, BooleanTrue):
                    break
            if not isinstance(options[-1]['condition'], BooleanTrue):
                # Fallback: no condition matched
                options.append({
                    'condition': True,
                    'hits': 0,
                    'misses': len(distances),
                    'evicts': len(destinations),
                    'tail': 0
                })

            results['cache'].append(options)

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
        if isinstance(kernel, KernelCode):
            # Make use of caching for symbolic LC representation:
            file_name = 'CSIM_analysis.pickle.lzma'
            file_path = kernel.get_intermediate_location(
                file_name, machine_and_compiler_dependent=False,
                other_dependencies=[str(cores)]+[str(t) for t in self.kernel.constants.items()])
            lock_mode, lock_fp = kernel.lock_intermediate(file_path)
            if lock_mode == fcntl.LOCK_SH:
                # use cache
                cache = compress_pickle.load(file_path)
                lock_fp.close()  # release lock
                self.first_dim_factor = cache['first_dim_factor']
                self.stats = cache['stats']
            else:  # lock_mode == fcntl.LOCK_EX
                # needs update
                self.simulate()
                compress_pickle.dump(
                    {'first_dim_factor': self.first_dim_factor, 'stats': self.stats},
                    file_path)
                lock_fp.close()  # release lock
        else:
            # No caching support without filename for kernel code
            self.simulate()

    def simulate(self):
        """Execute simulation"""
        # Get the machine's cache model and simulator
        self.csim = self.machine.get_cachesim(self.cores)

        # FIXME handle multiple datatypes
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        cacheline_size = self.machine['cacheline size']
        elements_per_cacheline = int(cacheline_size // element_size)
        iterations_per_cacheline = (sympy.Integer(self.machine['cacheline size']) /
                                    sympy.Integer(self.kernel.bytes_per_iteration))

        # Gathering some loop information:
        inner_loop = list(self.kernel.get_loop_stack(subs_consts=True))[-1]
        inner_index = symbol_pos_int(inner_loop['index'])
        inner_increment = inner_loop['increment']  # Calculate the number of iterations for warm-up
        total_length = self.kernel.iteration_length()
        max_iterations = self.kernel.subs_consts(total_length)
        max_cache_size = sum([c.size() for c in self.csim.levels(with_mem=False)])


        # Warmup
        # Phase 1:
        # define warmup interval boundaries
        max_steps = 100
        warmup_increment = ceildiv(max_cache_size // element_size, max_steps // 2)
        invalid_entries = self.csim.count_invalid_entries()
        step = 0
        warmup_iteration = 0
        complete_sweep = False
        while invalid_entries > 0 and step < max_steps and not complete_sweep:
            prev_warmup_iteration = warmup_iteration
            warmup_iteration = warmup_iteration + warmup_increment
            if warmup_iteration > max_iterations:
                warmup_iteration = max_iterations
                complete_sweep = True

            # print("warmup_iteration1", warmup_iteration)
            offsets = self.kernel.compile_global_offsets(
                iteration=range(prev_warmup_iteration, warmup_iteration))
            self.csim.loadstore(offsets, length=element_size)
            invalid_entries = self.csim.count_invalid_entries()
            # TODO more intelligent break criteria based on change of invalid entries might be
            #      useful for early termination.
            # print("invalid_entries", invalid_entries)

            step += 1

        # Phase 2:
        # Check that there is enough space left for benchmarking
        if complete_sweep:
            warmup_iteration = 0
        else:
            # Are we far away from max_iterations?
            # print("max_iterations", max_iterations)
            if warmup_iteration > max_iterations - 100000:
                # To close to end, need to complete sweep
                complete_sweep = True
                prev_warmup_iteration = warmup_iteration
                warmup_iteration = max_iterations
                # print("warmup_iteration2", warmup_iteration, end="; ")
                offsets = self.kernel.compile_global_offsets(
                    iteration=range(prev_warmup_iteration, warmup_iteration))
                self.csim.loadstore(offsets, length=element_size)
                warmup_iteration = 0
            if not complete_sweep and invalid_entries > 0:
                print("Warning: Unable to perform complete sweep nor initialize cache completely. "
                      "This might introduce inaccuracies (additional cache misses) in the cache "
                      "prediction.")

        # Phase 3:
        # Iterate to safe handover point
        prev_warmup_iteration = warmup_iteration
        warmup_iteration = self._align_iteration_with_cl_boundary(warmup_iteration, subtract=False)
        if warmup_iteration != prev_warmup_iteration:
            # print("warmup_iteration3", warmup_iteration)
            offsets = self.kernel.compile_global_offsets(
                iteration=range(prev_warmup_iteration, warmup_iteration))
            self.csim.loadstore(offsets, length=element_size)

        # Reset stats to conclude warm-up phase
        self.csim.reset_stats()

        # Benchmark
        bench_iteration = self._align_iteration_with_cl_boundary(min(
            warmup_iteration + 100000, max_iterations - 1))
        # print("bench_iteration", bench_iteration)
        first_dim_factor = float((bench_iteration - warmup_iteration) / iterations_per_cacheline)
        # If end point is less than 100 cacheline away, warn user of inaccuracy
        if first_dim_factor < 1000:
            print("Warning: benchmark iterations are very low ({} CL). This may lead to inaccurate "
                  "cache predictions.".format(first_dim_factor))

        # Compile access needed for one cache-line
        offsets = self.kernel.compile_global_offsets(
            iteration=range(warmup_iteration, bench_iteration))
        # Run cache simulation
        self.csim.loadstore(offsets, length=element_size)
        # FIXME compile_global_offsets should already expand to element_size

        # use stats to build results
        self.stats = list(self.csim.stats())
        self.first_dim_factor = first_dim_factor

    def _align_iteration_with_cl_boundary(self, iteration, subtract=True):
        """Align iteration with cacheline boundary."""
        # FIXME handle multiple datatypes
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        cacheline_size = self.machine['cacheline size']
        elements_per_cacheline = int(cacheline_size // element_size)

        # Gathering some loop information:
        inner_loop = list(self.kernel.get_loop_stack(subs_consts=True))[-1]
        inner_increment = inner_loop['increment']

        # do this by aligning either writes (preferred) or reads
        # Assumption: writes (and reads) increase linearly
        o = self.kernel.compile_global_offsets(iteration=iteration)[0]
        if len(o[1]):
            # we have a write to work with:
            first_offset = min(o[1])
        else:
            # we use reads
            first_offset = min(o[0])

        diff = first_offset - \
               (int(first_offset) >> self.csim.first_level.cl_bits << self.csim.first_level.cl_bits)
        if diff == 0:
            return iteration
        elif subtract:
            return iteration - (diff // element_size) // inner_increment
        else:
            return iteration + (elements_per_cacheline - diff // element_size) // inner_increment

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
