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
from itertools import chain
from pprint import pprint
from copy import deepcopy
from collections import defaultdict

import sympy

from kerncraft.prefixedunit import PrefixedUnit
from kerncraft.kernel import KernelCode


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


class LC(object):
    """
    class representation of the Layer-Condition Model

    See https://rrze-hpc.github.io/layer-condition/ for information about this analytical model.
    """

    name = "Layer-Condition"

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
    
    def calculate_cache_access(self):
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

            slices_max = FuckedUpMax(sympy.Integer(0),
                                     *[FuckedUpMax(*dists) for dists in slices_distances.values()])
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
                    'eq': sympy.solve(cache_equation, *self.kernel.constants.keys(), dict=True)
                }
        
        return results

    def analyze(self):
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
        
        self.results = self.calculate_cache_access()

    def report(self, output_file=sys.stdout):
        if self._args and self._args.verbose > 2:
            pprint(self.results)
        
        for dimension, lc_info in self.results['dimensions'].items():
            print("{}D Layer-Condition:".format(dimension), file=output_file)
            for cache, lc_solution in sorted(lc_info['caches'].items()):
                print(cache+": ", end='', file=output_file)
                if lc_solution['lt'] is sympy.true:
                    print("unconditionally fulfilled", file=output_file)
                else:
                    for solu in lc_solution['eq']:
                        for s, v in solu.items():
                            if v.n().is_Float:
                                print("{} <= {:.0f}".format(s, v.n()), file=output_file)
                            else:
                                print("{} <= {}".format(s, v), file=output_file)
