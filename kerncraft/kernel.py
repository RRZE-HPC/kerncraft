#!/usr/bin/env python3
"""Representation of computational kernel for performance model analysis and helper functions."""
import shutil
import textwrap
from copy import deepcopy
import operator
import tempfile
import subprocess
import os
import os.path
import sys
import numbers
import collections
from datetime import datetime
from functools import reduce, lru_cache
import string
from collections import defaultdict
from itertools import chain
import random
import atexit
import re
from contextlib import contextmanager
import fcntl

from math import floor

import sympy
from sympy.utilities.lambdify import implemented_function
from sympy.parsing.sympy_parser import parse_expr
import numpy
import compress_pickle

from pycparser import CParser, c_ast, plyparser
from pycparser.c_generator import CGenerator

from . import kerncraft
from . import incore_model
from .pycparser_utils import clean_code, replace_id


class LessParanthesizingCGenerator(CGenerator):
    def get_BinaryOp_precedence(self, n):
        """
        Gives precedence for op of n, otherwise -1.
        
        Lower number have precedence over higher numbers.
        """
        binary_op_precedence = {
            # based on https://en.cppreference.com/w/c/language/operator_precedence
            '*': 3, '%': 3,
            '-': 4, '+': 4,
            '<<': 5, '>>': 5,
            '<': 6, '<=': 6, '>': 6, '>=': 6,
            '==': 7, '!=': 7,
            '&': 8,
            '^': 9,
            '|': 10,
            '&&': 11,
            '||': 12
        }
        if not isinstance(n, c_ast.BinaryOp):
            return -1
        else:
            return binary_op_precedence[n.op]

    def visit_BinaryOp(self, n):
        p = self.get_BinaryOp_precedence(n)
        lval_str = self._parenthesize_if(n.left,
                            lambda d: not self._is_simple_node(d) and 
                                      self.get_BinaryOp_precedence(d) > p)
        rval_str = self._parenthesize_if(n.right,
                            lambda d: not self._is_simple_node(d) and
                                      self.get_BinaryOp_precedence(d) >= p)
        return '%s %s %s' % (lval_str, n.op, rval_str)
    

@contextmanager
def set_recursionlimit(new_limit):
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(new_limit)
    try:
        yield new_limit
    finally:
        sys.setrecursionlimit(old_limit)


@lru_cache()
def symbol_pos_int(*args, **kwargs):
    """Create a sympy.Symbol with positive and integer assumptions."""
    kwargs.update({'positive': True,
                   'integer': True})
    return sympy.Symbol(*args, **kwargs)


def string_to_sympy(s):
    """Convert any string to a sympy object or None."""
    if isinstance(s, int):
        return sympy.Integer(s)
    elif isinstance(s, list):
        return tuple([string_to_sympy(e) for e in s])
    elif s is None:
        return None
    else:
        # Step 1 build expression with the whole alphabet redefined:
        local_dict = {c: symbol_pos_int(c) for c in s if c in string.ascii_letters}
        # TODO find nicer solution for N and other pre-mapped letters
        preliminary_expr = parse_expr(s, local_dict=local_dict)
        # Replace all free symbols with positive integer versions:
        local_dict.update(
            {s.name: symbol_pos_int(s.name) for s in preliminary_expr.free_symbols})
        return parse_expr(s, local_dict=local_dict)


def prefix_indent(prefix, textblock, later_prefix=' '):
    """
    Prefix and indent all lines in *textblock*.

    *prefix* is a prefix string
    *later_prefix* is used on all but the first line, if it is a single character
                   it will be repeated to match length of *prefix*
    """
    textblock = textblock.split('\n')
    line = prefix + textblock[0] + '\n'
    if len(later_prefix) == 1:
        later_prefix = ' '*len(prefix)
    line = line + '\n'.join([later_prefix + x for x in textblock[1:]])
    if line[-1] != '\n':
        return line + '\n'
    else:
        return line


def transform_multidim_to_1d_decl(decl):
    """
    Transform ast of multidimensional declaration to a single dimension declaration.

    In-place operation!

    Returns name and dimensions of array (to be used with transform_multidim_to_1d_ref())
    """
    dims = []
    type_ = decl.type
    while type(type_) is c_ast.ArrayDecl:
        dims.append(type_.dim)
        type_ = type_.type

    if dims:
        # Multidimensional array
        decl.type.dim = reduce(lambda l, r: c_ast.BinaryOp('*', l, r), dims)
        decl.type.type = type_

    return decl.name, dims


def transform_multidim_to_1d_ref(aref, dimension_dict):
    """
    Transform ast of multidimensional reference to a single dimension reference.

    In-place operation!
    """
    dims = []
    name = aref
    while type(name) is c_ast.ArrayRef:
        dims.append(name.subscript)
        name = name.name

    subscript_list = []
    for i, d in enumerate(dims):
        if i == 0:
            subscript_list.append(d)
        else:
            subscript_list.append(c_ast.BinaryOp('*', d, reduce(
                lambda l, r: c_ast.BinaryOp('*', l, r),
                dimension_dict[name.name][-1:-i-1:-1])))

    aref.subscript = reduce(
        lambda l, r: c_ast.BinaryOp('+', l, r), subscript_list)
    aref.name = name


def transform_array_decl_to_malloc(decl, with_init=True):
    """
    Transform ast of "type var_name[N]" to "type* var_name = aligned_malloc(sizeof(type)*N, 32)"

    In-place operation.

    :param with_init: if False, ommit malloc
    """
    if type(decl.type) is not c_ast.ArrayDecl:
        # Not an array declaration, can be ignored
        return

    type_ = c_ast.PtrDecl([], decl.type.type)
    if with_init:
        decl.init = c_ast.FuncCall(
            c_ast.ID('aligned_malloc'),
            c_ast.ExprList([
                c_ast.BinaryOp(
                    '*',
                    c_ast.UnaryOp(
                        'sizeof',
                        c_ast.Typename(None, [], c_ast.TypeDecl(
                            None, [], decl.type.type.type))),
                    decl.type.dim),
                c_ast.Constant('int', '32')]))
    decl.type = type_


def find_node_type(ast, node_type):
    """Return list of nodes with *node_type* in *ast*."""
    if type(ast) is node_type:
        return [ast]
    elif type(ast) is list:
        return reduce(operator.add, list(map(lambda a: find_node_type(a, node_type), ast)), [])
    elif ast is None:
        return []
    else:
        return reduce(operator.add,
                      [find_node_type(o[1], node_type) for o in ast.children()], [])


def find_pragmas(ast):
    """Return list of pragmas in AST."""
    if type(ast) is c_ast.Pragma:
        return [ast]


def force_iterable(f):
    """Will make any functions return an iterable objects by wrapping its result in a list."""
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper


def reduce_path(path):
    """Reduce absolute path to relative (if shorter) for easier readability."""
    relative_path = os.path.relpath(path)
    if len(relative_path) < len(path):
        return relative_path
    else:
        return path


class Kernel(object):
    """Kernel information with functions to analyze and report access patterns."""

    # Datatype sizes in bytes
    datatypes_size = {('double', '_Complex'): 16,
                      ('float', '_Complex'): 8,
                      ('double',): 8,
                      ('float',): 4}

    def __init__(self, machine=None):
        """Create kernel representation."""
        self._machine = machine
        self._loop_stack = []
        self.variables = {}
        self.sources = {}
        self.destinations = {}
        self._flops = {}
        self.datatype = None
        self.constants = None

        self.clear_state()

    def check(self):
        """Check that information about kernel makes sens and is valid."""
        datatypes = [v[0] for v in self.variables.values()]
        assert len(set(datatypes)) <= 1, 'mixing of datatypes within a kernel is not supported.'

        # TODO add combine all tests here

    def set_constant(self, name, value):
        """
        Set constant of name to value.

        :param name: may be a str or a sympy.Symbol
        :param value: must be an int
        """
        assert isinstance(name, str) or isinstance(name, sympy.Symbol), \
            "constant name needs to be of type str, unicode or a sympy.Symbol"
        assert type(value) is int, "constant value needs to be of type int"
        if isinstance(name, sympy.Symbol):
            self.constants[name] = value
        else:
            self.constants[symbol_pos_int(name)] = value
        # Ensure constants will always be sorted by keys
        self.constants = collections.OrderedDict(
            sorted(self.constants.items(), key=lambda i: str(i[0])))

    def set_variable(self, name, type_, size):
        """
        Register variable of name and type_, with a (multidimensional) size.

        :param name: variable name as it appears in code
        :param type_: may be any key from Kernel.datatypes_size (typically float or double)
        :param size: either None for scalars or an n-tuple of ints for an n-dimensional array
        """
        assert type_ in self.datatypes_size, 'only float, double, float _Complex and ' \
                                             'double _Complex variables are supported'
        if self.datatype is None:
            self.datatype = type_
        else:
            assert type_ == self.datatype, 'mixing of datatypes within a kernel is not supported.'
        assert type(size) in [tuple, type(None)], 'size has to be defined as tuple or None'
        self.variables[name] = (type_, size)

    def clear_state(self):
        """Clear mutable internal states (constants, asm_blocks and asm_block_idx)."""
        self.constants = collections.OrderedDict()
        self.subs_consts.cache_clear()  # clear LRU cache of function

    @lru_cache(40)
    def subs_consts(self, expr):
        """Substitute constants in expression unless it is already a number."""
        if isinstance(expr, numbers.Number):
            return expr
        else:
            return expr.subs(self.constants)

    def array_sizes(self, in_bytes=False, subs_consts=False):
        """
        Return a dictionary with all arrays sizes.

        :param in_bytes: If True, output will be in bytes, not element counts.
        :param subs_consts: If True, output will be numbers and not symbolic.

        Scalar variables are ignored.
        """
        var_sizes = {}

        for var_name, var_info in self.variables.items():
            var_type, var_size = var_info

            # Skiping scalars
            if var_size is None:
                continue

            var_sizes[var_name] = reduce(operator.mul, var_size, 1)

            # Multiply by bytes per element if requested
            if in_bytes:
                element_size = self.datatypes_size[var_type]
                var_sizes[var_name] *= element_size

        if subs_consts:
            return {k: self.subs_consts(v) for k, v in var_sizes.items()}
        else:
            return var_sizes

    def _calculate_relative_offset(self, name, access_dimensions):
        """
        Return the offset from the iteration center in number of elements.

        The order of indices used in access is preserved.
        """
        # TODO to be replaced with compile_global_offsets
        offset = 0
        base_dims = self.variables[name][1]

        for dim, offset_info in enumerate(access_dimensions):
            offset_type, idx_name, dim_offset = offset_info
            assert offset_type == 'rel', 'Only relative access to arrays is supported at the moment'

            if offset_type == 'rel':
                offset += self.subs_consts(
                    dim_offset*reduce(operator.mul, base_dims[dim+1:], sympy.Integer(1)))
            else:
                # should not happen
                pass

        return offset

    def _remove_duplicate_accesses(self):
        """
        Remove duplicate source and destination accesses
        """
        self.destinations = {var_name: set(acs) for var_name, acs in self.destinations.items()}
        self.sources = {var_name: set(acs) for var_name, acs in self.sources.items()}

    def access_to_sympy(self, var_name, access):
        """
        Transform a (multidimensional) variable access to a flattend sympy expression.

        Also works with flat array accesses.
        """
        if var_name not in self.variables:
            raise ValueError("No declaration of variable {!r} found.".format(var_name))
        base_sizes = self.variables[var_name][1]

        expr = sympy.Number(0)

        for dimension, a in enumerate(access):
            base_size = reduce(operator.mul, base_sizes[dimension+1:], sympy.Integer(1))

            expr += base_size*a

        return expr

    def iteration_length(self, dimension=None):
        """
        Return the number of global loop iterations that are performed.

        If dimension is not None, it is the loop dimension that is returned
        (-1 is the inner most loop and 0 the outermost)
        """
        total_length = 1

        if dimension is not None:
            loops = [self._loop_stack[dimension]]
        else:
            loops = reversed(self._loop_stack)

        for var_name, start, end, incr in loops:
            # This unspools the iterations:
            length = end-start
            total_length = total_length*length
        return self.subs_consts(total_length)

    def get_loop_stack(self, subs_consts=False):
        """Yield loop stack dictionaries in order from outer to inner."""
        for l in self._loop_stack:
            if subs_consts:
                yield {'index': l[0],
                       'start': self.subs_consts(l[1]),
                       'stop': self.subs_consts(l[2]),
                       'increment': self.subs_consts(l[3])}
            else:
                yield {'index': l[0], 'start': l[1], 'stop': l[2], 'increment': l[3]}

    def index_order(self, sources=True, destinations=True):
        """
        Return the order of indices as they appear in array references.

        Use *source* and *destination* to filter output
        """
        if sources:
            arefs = chain(*self.sources.values())
        else:
            arefs = []
        if destinations:
            arefs = chain(arefs, *self.destinations.values())

        ret = []
        for a in [aref for aref in arefs if aref is not None]:
            ref = []
            for expr in a:
                ref.append(expr.free_symbols)
            ret.append(ref)

        return ret

    def compile_sympy_accesses(self, sources=True, destinations=True):
        """
        Return a dictionary of lists of sympy accesses, for each variable.

        Use *source* and *destination* to filter output
        """
        sympy_accesses = defaultdict(list)
        # Compile sympy accesses
        for var_name in self.variables:
            if sources:
                for r in self.sources.get(var_name, []):
                    if r is None:
                        continue
                    sympy_accesses[var_name].append(self.access_to_sympy(var_name, r))
            if destinations:
                for w in self.destinations.get(var_name, []):
                    if w is None:
                        continue
                    sympy_accesses[var_name].append(self.access_to_sympy(var_name, w))

        return sympy_accesses

    def compile_relative_distances(self, sympy_accesses=None):
        """
        Return load and store distances between accesses.

        :param sympy_accesses: optionally restrict accesses, default from compile_sympy_accesses()

        e.g. if accesses are to [+N, +1, -1, -N], relative distances are [N-1, 2, N-1]

        returned is a dict of list of sympy expressions, for each variable
        """
        if sympy_accesses is None:
            sympy_accesses = self.compile_sympy_accesses()

        sympy_distances = defaultdict(list)
        for var_name, accesses in sympy_accesses.items():
            for i in range(1, len(accesses)):
                sympy_distances[var_name].append((accesses[i-1]-accesses[i]).simplify())

        return sympy_distances

    def global_iterator_to_indices(self, git=None):
        """
        Return sympy expressions translating global_iterator to loop indices.

        If global_iterator is given, an integer is returned
        """
        # unwind global iteration count into loop counters:
        base_loop_counters = {}
        global_iterator = symbol_pos_int('global_iterator')
        idiv = implemented_function(sympy.Function(str('idiv')), lambda x, y: x//y)
        total_length = 1
        last_incr = 1
        for var_name, start, end, incr in reversed(self._loop_stack):
            loop_var = symbol_pos_int(var_name)

            # This unspools the iterations:
            length = end-start  # FIXME is incr handled correct here?
            counter = start+(idiv(global_iterator*last_incr, total_length)*incr) % length
            total_length = total_length*length
            last_incr = incr

            base_loop_counters[loop_var] = sympy.lambdify(
                global_iterator,
                self.subs_consts(counter), modules=[numpy, {'Mod': numpy.mod}])

            if git is not None:
                try:  # Try to resolve to integer if global_iterator was given
                    base_loop_counters[loop_var] = sympy.Integer(self.subs_consts(counter))
                    continue
                except (ValueError, TypeError):
                    base_loop_counters[loop_var] = base_loop_counters[loop_var](git)

        return base_loop_counters

    @lru_cache(1)
    def global_iterator(self):
        """
        Return global iterator sympy expression
        """
        global_iterator = sympy.Integer(0)
        total_length = sympy.Integer(1)
        for var_name, start, end, incr in reversed(self._loop_stack):
            loop_var = symbol_pos_int(var_name)
            length = end - start  # FIXME is incr handled correct here?
            global_iterator += (loop_var - start) * total_length
            total_length *= length
        return global_iterator

    def indices_to_global_iterator(self, indices):
        """
        Transform a dictionary of indices to a global iterator integer.

        Inverse of global_iterator_to_indices().
        """
        global_iterator = self.subs_consts(self.global_iterator().subs(indices))
        return global_iterator

    def max_global_iteration(self):
        """Return global iterator with last iteration number"""
        return self.indices_to_global_iterator({
            symbol_pos_int(var_name): end-1 for var_name, start, end, incr in self._loop_stack
        })

    def compile_global_offsets(self, iteration=0, spacing=0):
        """
        Return load and store offsets on a virtual address space.

        :param iteration: controls the inner index counter
        :param spacing: sets a spacing between the arrays, default is 0

        All array variables (non scalars) are laid out linearly starting from 0. An optional
        spacing can be set. The accesses are based on this layout.

        The iteration 0 is the first iteration. All loops are mapped to this linear iteration
        space.

        Accesses to scalars are ignored.

        Returned are load and store byte-offset pairs for each iteration.
        """
        global_load_offsets = []
        global_store_offsets = []

        if isinstance(iteration, range):
            iteration = numpy.arange(iteration.start, iteration.stop, iteration.step, dtype='O')
        else:
            if not isinstance(iteration, collections.abc.Sequence):
                iteration = [iteration]
            iteration = numpy.array(iteration, dtype='O')

        # loop indices based on iteration
        # unwind global iteration count into loop counters:
        base_loop_counters = self.global_iterator_to_indices()
        total_length = self.iteration_length()

        assert iteration.max() < self.subs_consts(total_length), \
            "Iterations go beyond what is possible in the original code ({} vs {}). " \
            "One common reason, is that the iteration length are unrealistically small.".format(
                iteration.max(), self.subs_consts(total_length))

        # Get sizes of arrays and base offsets for each array
        var_sizes = self.array_sizes(in_bytes=True, subs_consts=True)
        base_offsets = {}
        base = 0
        # Always arrange arrays in alphabetical order in memory, for reproducibility
        for var_name, var_size in sorted(var_sizes.items(), key=lambda v: v[0]):
            base_offsets[var_name] = base
            array_total_size = self.subs_consts(var_size + spacing)
            # Add bytes to align by 64 byte (typical cacheline size):
            array_total_size = ((int(array_total_size) + 63) & ~63)
            base += array_total_size

        # Gather all read and write accesses to the array:
        for var_name, var_size in var_sizes.items():
            element_size = self.datatypes_size[self.variables[var_name][0]]
            for r in self.sources.get(var_name, []):
                offset_expr = self.access_to_sympy(var_name, r)
                # Ignore accesses that always go to the same location (constant offsets)
                if not any([s in base_loop_counters.keys() for s in offset_expr.free_symbols]):
                    continue
                offset = force_iterable(sympy.lambdify(
                    base_loop_counters.keys(),
                    self.subs_consts(
                        offset_expr*element_size
                        + base_offsets[var_name]), numpy))
                # TODO possibly differentiate between index order
                global_load_offsets.append(offset)
            for w in self.destinations.get(var_name, []):
                offset_expr = self.access_to_sympy(var_name, w)
                # Ignore accesses that always go to the same location (constant offsets)
                if not any([s in base_loop_counters.keys() for s in offset_expr.free_symbols]):
                    continue
                offset = force_iterable(sympy.lambdify(
                    base_loop_counters.keys(),
                    self.subs_consts(
                        offset_expr*element_size
                        + base_offsets[var_name]), numpy))
                # TODO possibly differentiate between index order
                global_store_offsets.append(offset)
                # TODO take element sizes into account, return in bytes

        # Generate numpy.array for each counter
        counter_per_it = [v(iteration) for v in base_loop_counters.values()]

        # Old and slow - left for reference
        ## Data access as they appear with iteration order
        #return zip_longest(zip(*[o(*counter_per_it) for o in global_load_offsets]),
        #                   zip(*[o(*counter_per_it) for o in global_store_offsets]),
        #                   fillvalue=None)

        # Data access as they appear with iteration order
        load_offsets = []
        for o in global_load_offsets:
            load_offsets.append(o(*counter_per_it))
        # Convert to numpy ndarray and transpose to get offsets per iterations
        load_offsets = numpy.asarray(load_offsets).T

        store_offsets = []
        for o in global_store_offsets:
            store_offsets.append(o(*counter_per_it))
        store_offsets = numpy.asarray(store_offsets).T

        # Combine loads and stores
        store_width = store_offsets.shape[1] if len(store_offsets.shape) > 1 else 0
        dtype = [('load', load_offsets.dtype, (load_offsets.shape[1],)),
                 ('store', store_offsets.dtype, (store_width,))]
        offsets = numpy.empty(max(load_offsets.shape[0], store_offsets.shape[0]), dtype=dtype)
        offsets['load'] = load_offsets
        offsets['store'] = store_offsets

        return offsets

    @property
    def bytes_per_iteration(self):
        """
        Consecutive bytes written out per high-level iterations (as counted by loop stack).

        Is used to compute number of iterations per cacheline.
        """
        # TODO Find longst consecutive writes to any variable and use as basis
        var_name = list(self.destinations)[0]
        var_type = self.variables[var_name][0]
        # FIXME this is correct most of the time, but not guaranteed:
        # Multiplying datatype size with step increment of inner-most loop
        return self.datatypes_size[var_type] * self._loop_stack[-1][3]

    def print_kernel_info(self, output_file=sys.stdout):
        """Print kernel information in human readble format."""
        table = ('     idx |        min        max       step\n' +
                 '---------+---------------------------------\n')
        for l in self._loop_stack:
            table += '{:>8} | {!r:>10} {!r:>10} {!r:>10}\n'.format(*l)
        print(prefix_indent('loop stack:        ', table), file=output_file)

        table = ('    name |  offsets   ...\n' +
                 '---------+------------...\n')
        for name, offsets in list(self.sources.items()):
            prefix = '{:>8} | '.format(name)
            right_side = '\n'.join(['{!r:}'.format(o) for o in offsets])
            table += prefix_indent(prefix, right_side, later_prefix='         | ')
        print(prefix_indent('data sources:      ', table), file=output_file)

        table = ('    name |  offsets   ...\n' +
                 '---------+------------...\n')
        for name, offsets in list(self.destinations.items()):
            prefix = '{:>8} | '.format(name)
            right_side = '\n'.join(['{!r:}'.format(o) for o in offsets])
            table += prefix_indent(prefix, right_side, later_prefix='         | ')
        print(prefix_indent('data destinations: ', table), file=output_file)

        table = (' op | count \n' +
                 '----+-------\n')
        for op, count in list(self._flops.items()):
            table += '{:>3} | {:>4}\n'.format(op, count)
        table += '     =======\n'
        table += '      {:>4}'.format(sum(self._flops.values()))
        print(prefix_indent('FLOPs:     ', table), file=output_file)

    def print_variables_info(self, output_file=sys.stdout):
        """Print variables information in human readble format."""
        table = ('    name |   type          size             \n' +
                 '---------+----------------------------------\n')
        for name, var_info in list(self.variables.items()):
            table += '{:>8} | {:>15} {!s:<10}\n'.format(name, ' '.join(var_info[0]), var_info[1])
        print(prefix_indent('variables: ', table), file=output_file)

    def print_constants_info(self, output_file=sys.stdout):
        """Print constants information in human readble format."""
        table = ('    name | value     \n' +
                 '---------+-----------\n')
        for name, value in list(self.constants.items()):
            table += '{!s:>8} | {:<10}\n'.format(name, value)
        print(prefix_indent('constants: ', table), file=output_file)

    def incore_analysis(self, *args, **kwargs):
        """Run in-core analysis (e.g., IACA)."""
        raise NotImplementedError("Kernel does not support compilation and in-core analysis. "
                                  "Try a different model or kernel input format.")

    def build_executable(self, *args, **kwargs):
        """Compile and build binary."""
        raise NotImplementedError("Kernel does not support compilation. Try a different model or "
                                  "kernel input format.")


class KernelCode(Kernel):
    """
    Kernel information gathered from code using pycparser.

    This version allows compilation and generation of code for iaca and likwid benchmarking
    """

    def __init__(self, kernel_code, machine, filename=None, keep_intermediates=True):
        """
        Create kernel representation from source code str and machine object.

        :param kernel_code: string with kernel code file content
        :param machine: MachineModel object
        :param filename: used for prettier error messages and as storage location prefix
        :param keep_intermediates: if set to True, intermediate files (for and by compilation) will
                                   be preserved. If set to False, they will be deleted after use.
        """
        super(KernelCode, self).__init__(machine=machine)

        # Initialize state
        self.clear_state()

        self.kernel_code = kernel_code
        self._filename = filename
        self._keep_intermediates = keep_intermediates
        parser = CParser()
        try:
            self.kernel_ast = parser.parse(self._strip_comments(self._as_function()),
                                           filename=filename).ext[0].body
        except plyparser.ParseError as e:
            print('Error parsing kernel code:', e)
            sys.exit(1)

        self._process_code()

        self.check()
    
    def get_intermediate_location(
            self, name, machine_and_compiler_dependent=True, other_dependencies=[]):
        """
        Get a suitable and reproduceble file path string for intermediate files.

        :param name: filename to use for caching
        :param machine_and_compiler_dependent: set to False if file content does not depend on
                                               machine file or compiler settings
        :param other_dependencies: list of strings to use in path. slashes are stripped
        """
        if self._filename:
            base_name = os.path.join(os.path.dirname(self._filename),
                                     '.' + os.path.basename(self._filename) + '_kerncraft')
        else:
            base_name = tempfile.mkdtemp()

        if not self._keep_intermediates:
            # Remove directory and all content up on program exit
            atexit.register(shutil.rmtree, base_name, ignore_errors=True)

        if machine_and_compiler_dependent:
            compiler, compiler_args = self._machine.get_compiler()
            compiler_args = '_'.join(compiler_args).replace('/', '')
            base_name += '/{}/{}/{}/'.format(
                self._machine.get_identifier(), compiler, compiler_args)
        
        for i in other_dependencies:
            base_name = os.path.join(base_name, str(i).replace('/', ''))

        # Create dirs recursively
        os.makedirs(base_name, exist_ok=True)

        # Build actual file path
        file_path = os.path.join(base_name, name)

        return reduce_path(file_path)

    def _check_freshness(self, file_path):
        """Return True, if file_path exists and file is up-to-date."""
        if os.path.exists(file_path):
            file_modified = datetime.utcfromtimestamp(os.stat(file_path).st_mtime)
            # Check if file is newer than machine file, kerncraft code and input kernel code
            if (file_modified > self._machine.get_last_modified_datetime() and  # machine file
                file_modified > kerncraft.get_last_modified_datetime() and  # kerncraft
                file_modified > datetime.utcfromtimestamp(os.stat(self._filename).st_mtime)):
                    return True
        return False

    def lock_intermediate(self, file_path):
        """
        Lock intermediate. Depending on state, readable or writable.

        A sepeate file_path+'.lock' file is used. It is the callees responsibility to close the lock

        :param file_path: path to baser lock file on.

        :return: tuple: (acquired lock mode, lock file pointer)

        lock modes are: fcntl.LOCK_SH, which means file_path is good for read-only access
                        fcntl.LOCK_EX, which means file_path is good for write access and MUST be 
                        create/updated.
        """
        lock_filename = file_path + '.lock'
        # 1. Open lockfile (create and write)
        lock_fp = open(lock_filename, 'w+')
        # 2. Acquire SH lock (blocking)
        try:
            fcntl.flock(lock_fp, fcntl.LOCK_SH)
        except OSError:
            print("WARNING: locking does not work on this filesystem! Assuming exclusive access! "
                  "THIS MAY BREAK IF MULTIPLE KERNCRAFT INSTANCES RUN IN PARALLEL!",
                  file=sys.stderr)
            return (fcntl.LOCK_EX, lock_fp)
        # 3. Check existence and freshness
        if self._check_freshness(file_path):
            # -> READ MODE
            return (fcntl.LOCK_SH, lock_fp)
        # 4. Release SH lock (to allow other processes already awaiting an exclusive lock to enter)
        fcntl.flock(lock_fp, fcntl.LOCK_UN)
        # 5. Acquire EX lock (blocking)
        fcntl.flock(lock_fp, fcntl.LOCK_EX)
        # 6. Check if file is now fresh (things may have changed!)
        if self._check_freshness(file_path):
            # Acquire SH lock (this will replace EX lock in-place)
            fcntl.flock(lock_fp, fcntl.LOCK_SH)
            # -> READ MODE
            return (fcntl.LOCK_SH, lock_fp)
        # else: -> WRITE MODE
        return (fcntl.LOCK_EX, lock_fp)

    def release_exclusive_lock(self, lock_fd):
        """
        Release exclusive lock and degrade to shared lock.

        Shared lock is released by closing the file descriptor.
        """
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
        except OSError:
            print("WARNING: locking does not work on this filesystem! Assuming exclusive access! "
                  "THIS MAY BREAK IN SHARED USE!",
                  file=sys.stderr)

    def _strip_comments(self, code):
        clean_code = []
        for l in code.split('\n'):
            i = l.find('//')
            if i > -1:
                clean_code.append(l[:i])
            else:
                clean_code.append(l)
        return '\n'.join(clean_code)

    def print_kernel_code(self, output_file=sys.stdout):
        """Print source code of kernel."""
        print(self.kernel_code, file=output_file)

    def _as_function(self, func_name='test', filename=None):
        if filename is None:
            filename = ''
        else:
            filename = '"{}"'.format(filename)
        return '#line 0 \nvoid {}() {{\n#line 1 {}\n{}\n#line 999 \n}}'.format(
            func_name, filename, self.kernel_code)

    def clear_state(self):
        """Clear mutable internal states."""
        super(KernelCode, self).clear_state()
        self.asm_block = None
        self.pointer_increment = None

    def _process_code(self):
        assert type(self.kernel_ast) is c_ast.Compound, "Kernel has to be a compound statement"

        declarations = []
        loop_nest = []
        swaps = []

        # Check that code follows sections:
        # Section in code are (in this specific order):
        # 'declarations' (any number of array and scalar variable declarations)
        # 'loopnest' (a single loop nest)
        # 'swaps' (any number of swaps, may be none)
        section = 'declarations'
        for s in self.kernel_ast.block_items:
            if section == 'declarations':
                if type(s) in [c_ast.Decl]:
                    declarations.append(s)
                    continue
                # anything not a Declaration terminates the declaration section
                else:
                    section = 'loopnest'
            if section == 'loopnest':
                # a single loop is expected, which may be preceded with Pragmas
                if type(s) is c_ast.Pragma:
                    loop_nest.append(s)
                    continue
                elif type(s) is c_ast.For:
                    loop_nest.append(s)
                    section = 'swaps'
                    continue
                else:
                    raise ValueError("Expected for loop or pragma(s), found {} instead.".format(s))
            if section == 'swaps':
                if type(s) is c_ast.FuncCall and s.name.name == 'swap':
                    swaps.append(s)
                    continue
                else:
                    raise ValueError("Beyond the for loop, only function calls of 'swap' may be "
                                     "placed, found {} instead.".format(s))
            else:
                raise ValueError("Malformed code, does not follow declaration-loopnest-swaps "
                                 "structure.")

        for item in declarations:
            array = type(item.type) is c_ast.ArrayDecl

            if array:
                dims = []
                t = item.type
                while type(t) is c_ast.ArrayDecl:
                    dims.append(self.conv_ast_to_sym(t.dim))
                    t = t.type

                self.set_variable(item.name, tuple(t.type.names), tuple(dims))

            else:
                self.set_variable(item.name, tuple(item.type.type.names), None)

        self._p_for(loop_nest[-1])
        self.swaps = swaps

    def conv_ast_to_sym(self, math_ast):
        """
        Convert mathematical expressions to a sympy representation.

        May only contain paranthesis, addition, subtraction and multiplication from AST.
        """
        if type(math_ast) is c_ast.ID:
            return symbol_pos_int(math_ast.name)
        elif type(math_ast) is c_ast.Constant:
            return sympy.Integer(math_ast.value)
        else:  # elif type(dim) is c_ast.BinaryOp:
            op = {
                '*': operator.mul,
                '+': operator.add,
                '-': operator.sub
            }

            return op[math_ast.op](
                self.conv_ast_to_sym(math_ast.left),
                self.conv_ast_to_sym(math_ast.right))

    def _get_offsets(self, aref, dim=0):
        """
        Return a tuple of offsets of an ArrayRef object in all dimensions.

        The index order is right to left (c-code order).
        e.g. c[i+1][j-2] -> (j-2, i+1)

        If aref is actually a c_ast.ID, None will be returned.
        """
        if isinstance(aref, c_ast.ID):
            return None

        # Check for restrictions
        assert type(aref.name) in [c_ast.ArrayRef, c_ast.ID], \
            "array references must only be used with variables or other array references"
        assert type(aref.subscript) in [c_ast.ID, c_ast.Constant, c_ast.BinaryOp], \
            'array subscript must only contain variables or binary operations'

        # Convert subscript to sympy and append
        idxs = [self.conv_ast_to_sym(aref.subscript)]

        # Check for more indices (multi-dimensional access)
        if type(aref.name) is c_ast.ArrayRef:
            idxs += self._get_offsets(aref.name, dim=dim+1)

        # Reverse to preserver order (the subscripts in the AST are traversed backwards)
        if dim == 0:
            idxs.reverse()

        return tuple(idxs)

    @classmethod
    def _get_basename(cls, aref):
        """
        Return base name of ArrayRef object.

        e.g. c[i+1][j-2] -> 'c'
        """
        if isinstance(aref.name, c_ast.ArrayRef):
            return cls._get_basename(aref.name)
        elif isinstance(aref.name, str):
            return aref.name
        else:
            return aref.name.name

    def _p_for(self, floop):
        # Check for restrictions
        assert type(floop) is c_ast.For, "May only be a for loop"
        assert hasattr(floop, 'init') and hasattr(floop, 'cond') and hasattr(floop, 'next'), \
            "Loop must have initial, condition and next statements."
        assert type(floop.init) is c_ast.DeclList, \
            "Initialization of loops need to be declarations."
        assert len(floop.init.decls) == 1, "Only single declaration is allowed in init. of loop."
        assert floop.cond.op in '<', "only lt (<) is allowed as loop condition"
        assert type(floop.cond.left) is c_ast.ID, 'left of cond. operand has to be a variable'
        assert type(floop.cond.right) in [c_ast.Constant, c_ast.ID, c_ast.BinaryOp], \
            'right of cond. operand has to be a constant, a variable or a binary operation'
        assert type(floop.next) in [c_ast.UnaryOp, c_ast.Assignment], \
            'next statement has to be a unary or assignment operation'
        assert floop.next.op in ['++', 'p++', '+='], 'only ++ and += next operations are allowed'
        assert type(floop.stmt) in [c_ast.Compound, c_ast.Assignment, c_ast.For], \
            'the inner loop may contain only assignments or compounds of assignments'

        if type(floop.cond.right) is c_ast.ID:
            const_name = floop.cond.right.name
            iter_max = symbol_pos_int(const_name)
        elif type(floop.cond.right) is c_ast.Constant:
            iter_max = sympy.Integer(floop.cond.right.value)
        else:  # type(floop.cond.right) is c_ast.BinaryOp
            bop = floop.cond.right
            assert bop.op in '+-*', ('only addition (+), substraction (-) and multiplications (*) '
                                     'are accepted operators')
            iter_max = self.conv_ast_to_sym(bop)

        iter_min = self.conv_ast_to_sym(floop.init.decls[0].init)

        if type(floop.next) is c_ast.Assignment:
            assert type(floop.next.lvalue) is c_ast.ID, \
                'next operation may only act on loop counter'
            assert type(floop.next.rvalue) is c_ast.Constant, 'only constant increments are allowed'
            assert floop.next.lvalue.name == floop.cond.left.name == floop.init.decls[0].name, \
                'initial, condition and next statement of for loop must act on same loop ' \
                'counter variable'
            step_size = int(floop.next.rvalue.value)
        else:
            assert type(floop.next.expr) is c_ast.ID, 'next operation may only act on loop counter'
            assert floop.next.expr.name == floop.cond.left.name == floop.init.decls[0].name, \
                'initial, condition and next statement of for loop must act on same loop ' \
                'counter variable'
            assert isinstance(floop.next, c_ast.UnaryOp), 'only assignment or unary operations ' \
                'are allowed for next statement of loop.'
            assert floop.next.op in ['++', 'p++', '--', 'p--'], 'Unary operation can only be ++ ' \
                'or -- in next statement'
            if floop.next.op in ['++', 'p++']:
                step_size = sympy.Integer('1')
            else:  # floop.next.op in ['--', 'p--']:
                step_size = sympy.Integer('-1')

        # Document for loop stack
        self._loop_stack.append(
            # (index name, min, max, step size)
            (floop.init.decls[0].name, iter_min, iter_max, step_size)
        )

        # Traverse tree
        if type(floop.stmt) is c_ast.For:
            self._p_for(floop.stmt)
        elif type(floop.stmt) is c_ast.Assignment:
            self._p_assignment(floop.stmt)
        # Handle For if it is the last statement, only preceeded by Pragmas
        elif type(floop.stmt.block_items[-1]) is c_ast.For and \
                all([type(s) == c_ast.Pragma for s in floop.stmt.block_items[:-1]]):
            self._p_for(floop.stmt.block_items[-1])
        else:  # type(floop.stmt) is c_ast.Compound
            # Handle Assignments
            for assgn in floop.stmt.block_items:
                # Ignore pragmas
                if type(assgn) is c_ast.Pragma:
                    continue
                elif type(assgn) is c_ast.Assignment:
                    self._p_assignment(assgn)
                else:
                    raise ValueError("Assignments are only allowed in inner most loop.")

    def _p_assignment(self, stmt):
        # Check for restrictions
        assert type(stmt) is c_ast.Assignment, \
            "Only assignment and pragma statements are allowed in loops."
        assert type(stmt.lvalue) in [c_ast.ArrayRef, c_ast.ID], \
            "Only assignment to array element or varialbe is allowed."

        write_and_read = False
        if stmt.op != '=':
            write_and_read = True
            op = stmt.op.strip('=')
            self._flops[op] = self._flops.get(op, 0)+1

        # Document data destination
        # self.destinations[dest name] = [dest offset, ...])
        self.destinations.setdefault(self._get_basename(stmt.lvalue), set())
        self.destinations[self._get_basename(stmt.lvalue)].add(
            self._get_offsets(stmt.lvalue))

        if write_and_read:
            # this means that +=, -= or something of that sort was used
            self.sources.setdefault(self._get_basename(stmt.lvalue), set())
            self.sources[self._get_basename(stmt.lvalue)].add(
                self._get_offsets(stmt.lvalue))

        # Traverse tree
        self._p_sources(stmt.rvalue)

    def _p_sources(self, stmt):
        sources = []
        assert type(stmt) in \
            [c_ast.ArrayRef, c_ast.Constant, c_ast.ID, c_ast.BinaryOp, c_ast.UnaryOp], \
            'only references to arrays, constants and variables as well as binary operations ' + \
            'are supported'
        assert type(stmt) is not c_ast.UnaryOp or stmt.op in ['-', '--', '++', 'p++', 'p--'], \
            'unary operations are only allowed with -, -- and ++'

        if type(stmt) in [c_ast.ArrayRef, c_ast.ID]:
            # Document data source
            bname = self._get_basename(stmt)
            self.sources.setdefault(bname, set())
            self.sources[bname].add(self._get_offsets(stmt))
        elif type(stmt) is c_ast.BinaryOp:
            # Traverse tree
            self._p_sources(stmt.left)
            self._p_sources(stmt.right)

            self._flops[stmt.op] = self._flops.get(stmt.op, 0)+1
        elif type(stmt) is c_ast.UnaryOp:
            self._p_sources(stmt.expr)
            self._flops[stmt.op] = self._flops.get(stmt.op[-1], 0)+1

        return sources

    def get_index_type(self, loop_nest=None):
        """
        Return index type used in loop nest.

        If index type between loops differ, an exception is raised.
        """
        if loop_nest is None:
            loop_nest = self.get_kernel_loop_nest()
        if type(loop_nest) is c_ast.For:
            loop_nest = [loop_nest]
        index_types = (None, None)
        for s in loop_nest:
            if type(s) is c_ast.For:
                if type(s.stmt) in [c_ast.For, c_ast.Compound]:
                    other = self.get_index_type(loop_nest=s.stmt)
                else:
                    other = None
                index_types = (s.init.decls[0].type.type.names, other)
                break
        if index_types[0] == index_types[1] or index_types[1] is None:
            return index_types[0]
        else:
            raise ValueError("Loop indices must have same type, found {}.".format(index_types))

    def _build_const_declartions(self, with_init=True):
        """
        Generate constants declarations

        :return: list of declarations
        """
        decls = []

        # Use type as provided by user in loop indices
        index_type = self.get_index_type()

        i = 2  # subscript for cli input, 1 is reserved for repeat
        for k in self.constants:
            # const long long N = strtoul(argv[2])
            # with increasing N and 1
            # TODO change subscript of argv depending on constant count
            type_decl = c_ast.TypeDecl(k.name, ['const'], c_ast.IdentifierType(index_type))
            init = None
            if with_init:
                init = c_ast.FuncCall(
                    c_ast.ID('atoi'),
                    c_ast.ExprList([c_ast.ArrayRef(c_ast.ID('argv'),
                                                   c_ast.Constant('int', str(i)))]))
            i += 1
            decls.append(c_ast.Decl(
                k.name, ['const'], [], [],
                type_decl, init, None))

        return decls

    def get_array_declarations(self):
        """Return array declarations."""
        return [d for d in self.kernel_ast.block_items
                if type(d) is c_ast.Decl and type(d.type) is c_ast.ArrayDecl]

    def get_kernel_loop_nest(self):
        """
        Return kernel loop nest, with openmp pragmas insert unless already present.
        """
        loop_nest = [s for s in self.kernel_ast.block_items
                     if type(s) in [c_ast.For, c_ast.Pragma, c_ast.FuncCall]]
        assert len(loop_nest) >= 1, "Found to few for statements in kernel"

        omp_pragmas = [p for p in find_node_type(loop_nest, c_ast.Pragma)
                        if 'omp' in p.string]
        # if omp pragmas were found: replace "parallel for" -> "for"
        for op in omp_pragmas:
            op.string = op.string.replace(' parallel', '')
        # if no omp for pragmas are present, insert suitable one at start of outer loop
        if not omp_pragmas:
            loop_nest.insert(0, c_ast.Pragma("omp for"))

        return loop_nest

    def _build_array_declarations(self, with_init=True):
        """
        Generate declaration statements for arrays.

        Also transforming multi-dim to 1d arrays and initializing with malloc.

        :param with_init: ommit malloc initialization

        :return: list of declarations nodes, dictionary of array names and original dimensions
        """
        # copy array declarations from from kernel ast
        array_declarations = deepcopy(self.get_array_declarations())
        array_dict = []
        for d in array_declarations:
            # We need to transform
            array_dict.append(transform_multidim_to_1d_decl(d))
            transform_array_decl_to_malloc(d, with_init=with_init)
        return array_declarations, dict(array_dict)

    def _find_inner_most_loop(self, loop_nest):
        """Return inner most for loop in loop nest"""
        r = None
        for s in loop_nest:
            if type(s) is c_ast.For:
                return self._find_inner_most_loop(s) or s
            else:
                r = r or self._find_inner_most_loop(s)
        return r

    def _build_array_initializations(self, array_dimensions):
        """
        Generate initialization statements for arrays.

        :param array_dimensions: dictionary of array dimensions

        :return: list of nodes
        """
        with set_recursionlimit(100000):
            kernel = deepcopy(self.get_kernel_loop_nest())
        # traverse to the inner most for loop:
        inner_most = self._find_inner_most_loop(kernel)
        orig_inner_stmt = inner_most.stmt
        inner_most.stmt = c_ast.Compound([])

        rand_float_str = str(random.uniform(1.0, 0.1))

        # find all array references in original orig_inner_stmt
        for aref in find_node_type(orig_inner_stmt, c_ast.ArrayRef):
            # transform to 1d references
            transform_multidim_to_1d_ref(aref, array_dimensions)
            # build static assignments and inject into inner_most.stmt
            inner_most.stmt.block_items.append(c_ast.Assignment(
                '=', aref, c_ast.Constant('float', rand_float_str)))

        return kernel

    def _build_dummy_calls(self):
        """
        Generate false if branch with dummy calls

        Requires kerncraft.h to be included, which defines dummy(...) and var_false.

        :return: dummy statement
        """
        # Make sure nothing gets removed by inserting dummy calls
        dummy_calls = []
        for d in self.kernel_ast.block_items:
            # Only consider toplevel declarations from kernel ast
            if type(d) is not c_ast.Decl: continue
            if type(d.type) is c_ast.ArrayDecl:
                dummy_calls.append(c_ast.FuncCall(
                    c_ast.ID('dummy'),
                    c_ast.ExprList([c_ast.ID(d.name)])))
            else:
                dummy_calls.append(c_ast.FuncCall(
                    c_ast.ID('dummy'),
                    c_ast.ExprList([c_ast.UnaryOp('&', c_ast.ID(d.name))])))
        dummy_stmt = c_ast.If(
            cond=c_ast.ID('var_false'),
            iftrue=c_ast.Compound(dummy_calls),
            iffalse=None)
        return dummy_stmt

    def _build_kernel_function_declaration(self, name='kernel'):
        """Build and return kernel function declaration"""
        array_declarations, array_dimensions = self._build_array_declarations(with_init=False)
        scalar_declarations = self.get_scalar_declarations(pointer=True, suffix='_')
        const_declarations = self._build_const_declartions(with_init=False)
        return c_ast.FuncDecl(args=c_ast.ParamList(params=array_declarations +
                                                          scalar_declarations +
                                                          const_declarations),
                              type=c_ast.TypeDecl(declname=name,
                                                  quals=[],
                                                  type=c_ast.IdentifierType(names=['void'])))

    def get_scalar_declarations(self, pointer=False, suffix=''):
        """
        Give list of scalars used in code.
        
        optionatlly with added suffix and converted to pointers
        """
        decls = [d for d in self.kernel_ast.block_items
                 if type(d) is c_ast.Decl and type(d.type) is c_ast.TypeDecl]
        for i, d in enumerate(decls):
            if suffix or pointer:
                d = deepcopy(d)
                decls[i] = d
            if suffix:
                d.name += suffix
                d.type.declname += suffix
            if pointer:
                d.type = c_ast.PtrDecl(quals=[], type=d.type)
        return decls

    def _build_scalar_extern_declarations(self):
        """Build and return scalar variable declarations, with extern attribute."""
        scalar_declarations = deepcopy(self.get_scalar_declarations())
        for d in scalar_declarations:
            d.storage = ['extern']
        return scalar_declarations

    def _build_scalar_initializations(self):
        """Build and return scalar variable initialization."""
        random.seed(2342)  # we want reproducible random numbers
        scalar_inits = []
        for d in deepcopy(self.get_scalar_declarations()):
            if d.type.type.names[0] in ['double', 'float']:
                d.init = c_ast.Constant('float', str(random.uniform(1.0, 0.1)))
            elif d.type.type.names[0] in ['int', 'long', 'long long',
                                          'unsigned int', 'unsigned long', 'unsigned long long']:
                d.init = c_ast.Constant('int', 2)
            scalar_inits.append(d)

        return scalar_inits

    def get_kernel_header(self, name='kernel'):
        """
        Generate and store kernel.h

        :return: tuple of filename of header and file pointer of lockfile
        """
        file_name = 'kernel.h'
        file_path = self.get_intermediate_location(
            file_name, machine_and_compiler_dependent=False)
        lock_mode, lock_fp = self.lock_intermediate(file_path)
        if lock_mode == fcntl.LOCK_SH:
            # use cache
            with open(file_path) as f:
                code = f.read()
        else:  # lock_mode == fcntl.LOCK_EX
            # needs update
            func_decl = self._build_kernel_function_declaration(name=name)
            code = LessParanthesizingCGenerator().visit(
                c_ast.FileAST(ext=[func_decl]))
            with open(file_path, 'w') as f:
                f.write(code)
            self.release_exclusive_lock(lock_fp)  # degrade to shared lock

        return file_name, lock_fp

    def get_kernel_code(self, openmp=False, name='kernel'):
        """
        Generate and return compilable source code with kernel function from AST.

        :param openmp: include openmp paragmas (or strip them)
        :param name: name of kernel function

        :return: tuple of filename of header and file pointer of lockfile
        """
        assert self.kernel_ast is not None, "AST does not exist, this could be due to running " \
                                            "based on a kernel description rather than code."
        filename = 'kernel'
        if openmp:
            filename += '-omp'
        filename += '.c'
        file_path = self.get_intermediate_location(
            filename, machine_and_compiler_dependent=False)
        lock_mode, lock_fp = self.lock_intermediate(file_path)

        if lock_mode == fcntl.LOCK_SH:
            # use cache
            with open(file_path) as f:
                code = f.read()
        else:  # lock_mode == fcntl.LOCK_EX
            # needs update
            array_declarations, array_dimensions = self._build_array_declarations()

            # Prepare actual kernel loop nest
            with set_recursionlimit(100000):
                kernel = deepcopy(self.get_kernel_loop_nest())
            # find all array references in kernel
            for aref in find_node_type(kernel, c_ast.ArrayRef):
                # transform to 1d references
                transform_multidim_to_1d_ref(aref, array_dimensions)
            # Declar scalars, initialize and copy from (and to) pointer
            scalar_decl = []
            scalar_init = []
            scalar_copy_back = []
            for sd in deepcopy(self.get_scalar_declarations()):
                sd.storage = ['static']
                scalar_decl.append(sd)
                s_id = c_ast.ID(name=sd.name)
                ptr = c_ast.UnaryOp(op='*', expr=c_ast.ID(name=sd.name + '_'))
                scalar_init.append(c_ast.Assignment(op='=', lvalue=s_id, rvalue=ptr))
                scalar_copy_back.append(c_ast.Assignment(op='=', lvalue=ptr, rvalue=s_id))
            kernel = scalar_decl + scalar_init + kernel + scalar_copy_back

            function_ast = c_ast.FuncDef(decl=c_ast.Decl(
                name=name, type=self._build_kernel_function_declaration(name=name), quals=[],
                storage=[], funcspec=[], init=None, bitsize=None),
                body=c_ast.Compound(block_items=kernel),
                param_decls=None)

            # Generate code
            with set_recursionlimit(100000):
                code = LessParanthesizingCGenerator().visit(function_ast)
            
            if not openmp:
                # remove all omp pragmas
                code = re.sub('#pragma omp[^\n]*\n', '', code)

            # Store to file
            with open(file_path, 'w') as f:
                f.write(code)
            self.release_exclusive_lock(lock_fp)  # degrade to shared lock

        return file_path, lock_fp

    def _build_kernel_call(self, name='kernel'):
        """Generate and return kernel call ast."""
        return c_ast.FuncCall(name=c_ast.ID(name=name), args=c_ast.ExprList(exprs=(
            [c_ast.ID(name=d.name) for d in self._build_array_declarations()[0]] +
            [c_ast.UnaryOp(op='&', expr=c_ast.ID(name=d.name))
             for d in self.get_scalar_declarations()] +
            [c_ast.ID(name=d.name) for d in self._build_const_declartions()])))

    CODE_TEMPLATE = textwrap.dedent("""
        #include <likwid.h>
        #include <stdlib.h>
        #include "kerncraft.h"
        #include "kernel.h"

        void dummy(void *);
        extern int var_false;

        int main(int argc, char **argv) {
          // Declaring constants
          DECLARE_CONSTS;
          // Declaring arrays
          DECLARE_ARRAYS;
          // Declaring and initializing scalars
          DECLARE_INIT_SCALARS;

          likwid_markerInit();
          #pragma omp parallel
          {
            likwid_markerRegisterRegion("loop");
            #pragma omp barrier

            // Initializing arrays in same order as touched in kernel loop nest
            INIT_ARRAYS;

            // Dummy call
            DUMMY_CALLS;

            for(int warmup = 1; warmup >= 0; --warmup) {
              int repeat = 2;
              if(warmup == 0) {
                repeat = atoi(argv[1]);
                likwid_markerStartRegion("loop");
              }

              for(; repeat > 0; --repeat) {
                KERNEL_CALL;
                DUMMY_CALLS;
              }

            }
            likwid_markerStopRegion("loop");
          }
          likwid_markerClose();
          return 0;
        }
        """)

    def get_main_code(self, kernel_function_name='kernel'):
        """
        Generate and return compilable source code from AST.

        :return: tuple of filename and shared lock file pointer
        """
        # TODO produce nicer code, including help text and other "comfort features".
        assert self.kernel_ast is not None, "AST does not exist, this could be due to running " \
                                            "based on a kernel description rather than code."

        file_path = self.get_intermediate_location('main.c', machine_and_compiler_dependent=False)
        lock_mode, lock_fp = self.lock_intermediate(file_path)

        if lock_mode == fcntl.LOCK_SH:
            # use cache
            with open(file_path) as f:
                code = f.read()
        else:  # lock_mode == fcntl.LOCK_EX
            # needs update
            parser = CParser()
            template_code = self.CODE_TEMPLATE
            template_ast = parser.parse(clean_code(template_code,
                                                   macros=True, comments=True, pragmas=False))
            ast = deepcopy(template_ast)

            # Define and replace DECLARE_CONSTS
            replace_id(ast, "DECLARE_CONSTS", self._build_const_declartions(with_init=True))

            # Define and replace DECLARE_ARRAYS
            array_declarations, array_dimensions = self._build_array_declarations()
            replace_id(ast, "DECLARE_ARRAYS", array_declarations)

            # Define and replace DECLARE_INIT_SCALARS
            replace_id(ast, "DECLARE_INIT_SCALARS", self._build_scalar_initializations())

            # Define and replace DUMMY_CALLS
            replace_id(ast, "DUMMY_CALLS", self._build_dummy_calls())

            # Define and insert kernel declaration at top
            #ast.ext.insert(0, self._build_kernel_function_declaration(
            #    name=kernel_function_name))
            #ast.ext[:0] = self._build_scalar_extern_declarations()

            # Define and replace KERNEL_CALL
            replace_id(ast, "KERNEL_CALL", self._build_kernel_call())

            # Define and replace INIT_ARRAYS based on previously generated kernel
            replace_id(ast, "INIT_ARRAYS", self._build_array_initializations(array_dimensions))

            # Generate code
            code = LessParanthesizingCGenerator().visit(ast)

            # Insert missing #includes from template to top of code
            code = '\n'.join([l for l in template_code.split('\n') if l.startswith("#include")]) + \
                   '\n\n' + code

            # Store to file
            with open(file_path, 'w') as f:
                f.write(code)
            self.release_exclusive_lock(lock_fp)  # degrade to shared lock

        return file_path, lock_fp


    def assemble_to_object(self, in_filename, verbose=False):
        """
        Assemble *in_filename* assembly into *out_filename* object.

        Returns tuple of filename to binary file and shared lock file pointer
        """
        # Build file name
        file_base_name = os.path.splitext(os.path.basename(in_filename))[0]
        out_filename = self.get_intermediate_location(file_base_name + '.o')
        lock_mode, lock_fp = self.lock_intermediate(out_filename)

        if lock_mode == fcntl.LOCK_SH:
            # use cached version
            pass
        else:  # lock_mode == fcntl.LOCK_EX
            # needs update
            compiler, compiler_args = self._machine.get_compiler()

            # Compile to object file
            compiler_args.append('-c')

            cmd = [compiler] + [in_filename] + compiler_args + ['-o', out_filename]

            if verbose:
                print('Executing (assemble_to_object): ', ' '.join(cmd))

            try:
                # Assemble all to a binary
                subprocess.check_output(cmd)
                self.release_exclusive_lock(lock_fp)  # degrade to shared lock
            except subprocess.CalledProcessError as e:
                print("Assembly failed:", e, file=sys.stderr)
                sys.exit(1)

        return out_filename, lock_fp

    def compile_kernel(self, assembly=False, openmp=False, verbose=False):
        """
        Compile source (from as_code(type_)) to assembly or object and return (fileptr, filename).

        :return: tuple of filename of compiled kernel and shared lock file pointer
        """
        compiler, compiler_args = self._machine.get_compiler()

        filename = 'kernel'
        if openmp:
            filename += '-omp'
        if assembly:
            compiler_args += ['-S']
            filename += '.s'
        else:
            filename += '.o'
        out_filename = self.get_intermediate_location(filename)
        lock_mode, out_lock_fp = self.lock_intermediate(out_filename)

        if lock_mode == fcntl.LOCK_SH:
            # use cached version
            if verbose:
                print('Executing (compile_kernel): ', 'using cached', out_filename)
        else:  # lock_mode == fcntl.LOCK_EX
            # needs update
            compiler_args += ['-std=c99']

            header_filename, header_lock_fp = self.get_kernel_header()
            in_filename, in_lock_fp = self.get_kernel_code(openmp=openmp)
            cmd = ([compiler] +
                   [in_filename,
                    '-c',
                    '-I'+reduce_path(os.path.abspath(os.path.dirname(
                        os.path.realpath(__file__)))+'/headers/'),
                    '-o', out_filename] +
                   compiler_args)

            if verbose:
                print('Executing (compile_kernel): ', ' '.join(cmd))

            try:
                subprocess.check_output(cmd)
                in_lock_fp.close()
                header_lock_fp.close()

            except subprocess.CalledProcessError as e:
                print("Compilation failed:", e, file=sys.stderr)
                sys.exit(1)

            # FIXME TODO FIXME TODO FIXME TODO
            # Hacky workaround for icc issue (icc may issue vkmovb instructions with AVX512, which 
            # are invalid and should be kmovb):
            if compiler == 'icc' and assembly:
                with open(out_filename, 'r+') as f:
                    assembly = f.read()
                    f.seek(0)
                    f.write(assembly.replace('vkmovb', 'kmovb'))
                    f.truncate()
            # FIXME TODO FIXME TODO FIXME TODO

            self.release_exclusive_lock(out_lock_fp)  # degrade to shared lock

        return out_filename, out_lock_fp


    def incore_analysis(self, asm_block='auto', pointer_increment='auto_with_manual_fallback',
                        model=None, verbose=False):
        """
        Run an in-core analysis and return its outcome.

        :param asm_block: controls how the to-be-marked block is chosen. "auto" (default) results in
                          the largest block, "manual" results in interactive and a number in the
                          according block.
        :param pointer_increment: number of bytes the pointer is incremented after the loop or
                                   - 'auto': automatic detection, RuntimeError raised if failed
                                   - 'auto_with_manual_fallback': like auto, fallsback to manual
                                     input
                                   - 'manual': prompt user
        :param model: which model to use, "IACA", "OSACA" or "LLVM-MCA"
        """
        # Get model and parameter
        model = self._machine.get_incore_model(model)
        model_parameter = self._machine['in-core model'][model]

        analysis_filename = self.get_intermediate_location('incore_analysis.pickle.lzma',
            other_dependencies=[model, str(model_parameter)])
        analysis_lock_mode, analysis_lock_fp = self.lock_intermediate(analysis_filename)
        if analysis_lock_mode == fcntl.LOCK_SH:
            # use cached analysis
            analysis, self.pointer_increment = compress_pickle.load(analysis_filename)
            analysis_lock_fp.close()  # release lock
            return analysis, self.pointer_increment
        
        marked_filename = self.get_intermediate_location('kernel-marked.s',
            other_dependencies=[asm_block, pointer_increment])
        lock_mode, marked_lock_fp = self.lock_intermediate(marked_filename)
        if lock_mode == fcntl.LOCK_SH:
            # use cached maked assembly and extract asm_block and pointer_increment
            with open(marked_filename) as f:
                marked_asm = f.read()
            m = re.search(r'pointer_increment=([0-9]+)', marked_asm)
            if m:
                self.pointer_increment = int(m.group(1))
            else:
                os.unlink(marked_filename)
                print("Could not find `pointer_increment=<byte increment>`. Plase place into file.",
                      file=sys.stderr)
                sys.exit(1)
        else:  # analysis_lock_mode == fcntl.LOCK_EX
            # marked assembly needs update
            asm_filename, asm_lock_fp = self.compile_kernel(assembly=True, verbose=verbose)
            with open(asm_filename, 'r') as in_file, open(marked_filename, 'w') as out_file:
                asm_block, self.pointer_increment = incore_model.asm_instrumentation(
                    in_file, out_file,
                    block_selection=asm_block,
                    pointer_increment=pointer_increment,
                    isa=self._machine['isa'])
            asm_lock_fp.close()
            self.release_exclusive_lock(marked_lock_fp)  # degrade to shared lock

        if model == 'OSACA':
            analysis = incore_model.osaca_analyse_instrumented_assembly(
                marked_filename, model_parameter)
        elif model == 'LLVM-MCA':
            analysis = incore_model.llvm_mca_analyse_instrumented_assembly(
                marked_filename, model_parameter)
        elif model == 'IACA':
            obj_name, obj_lock_fp = self.assemble_to_object(marked_filename, verbose=verbose)
            analysis = incore_model.iaca_analyse_instrumented_binary(obj_name, model_parameter)
            obj_lock_fp.close()
        else:
            raise ValueError("Unknown micro-architecture model: {!r}".format(model))
        compress_pickle.dump((analysis, self.pointer_increment), analysis_filename)
        analysis_lock_fp.close()
        marked_lock_fp.close()
        return analysis, self.pointer_increment

    def build_executable(self, lflags=None, verbose=False, openmp=False):
        """Compile source to executable with likwid capabilities and return the executable name."""
        compiler, compiler_args = self._machine.get_compiler()

        filename = 'kernel'
        if openmp:
            filename += '-omp'
        out_filename = self.get_intermediate_location(filename)
        lock_mode, out_lock_fp = self.lock_intermediate(out_filename)

        if lock_mode == fcntl.LOCK_SH:
            # use cached version
            if verbose:
                print('Executing (build_executable): ', 'using cached', out_filename)
        else:  # lock_mode == fcntl.LOCK_EX
            # needs update
            if not (('LIKWID_INCLUDE' in os.environ or 'LIKWID_INC' in os.environ) and
                    'LIKWID_LIB' in os.environ):
                print('Could not find LIKWID_INCLUDE (e.g., "-I/app/likwid/4.1.2/include") and '
                      'LIKWID_LIB (e.g., "-L/apps/likwid/4.1.2/lib") environment variables',
                      file=sys.stderr)
                sys.exit(1)

            compiler_args += [
                '-std=c99',
                '-I'+reduce_path(os.path.abspath(os.path.dirname(
                    os.path.realpath(__file__)))+'/headers/'),
                os.environ.get('LIKWID_INCLUDE', ''),
                os.environ.get('LIKWID_INC', ''),
                '-llikwid']

            # This is a special case for unittesting
            if os.environ.get('LIKWID_LIB') == '':
                compiler_args = compiler_args[:-1]

            if lflags is None:
                lflags = []
            lflags += os.environ['LIKWID_LIB'].split(' ') + ['-pthread']
            compiler_args += os.environ['LIKWID_LIB'].split(' ') + ['-pthread']

            main_filename, main_lock_fp = self.get_main_code()
            kernel_obj_filename, kernel_obj_lock_fp = self.compile_kernel(
                openmp=openmp, verbose=verbose)

            infiles = [reduce_path(os.path.abspath(os.path.dirname(
                os.path.realpath(__file__)))+'/headers/dummy.c'),
                       kernel_obj_filename, main_filename]

            cmd = [compiler] + infiles + compiler_args + ['-o', out_filename]
            # remove empty arguments
            cmd = list(filter(bool, cmd))
            if verbose:
                print('Executing (build_executable): ', ' '.join(cmd))
            try:
                subprocess.check_output(cmd)
                main_lock_fp.close()
                kernel_obj_lock_fp.close()
                self.release_exclusive_lock(out_lock_fp)  # degrade to shared lock
            except subprocess.CalledProcessError as e:
                print("Build failed:", e, file=sys.stderr)
                sys.exit(1)

        return out_filename, out_lock_fp


class KernelDescription(Kernel):
    """
    Kernel information gathered from YAML kernel description file.

    This class does NOT allow compilation, required for IACA analysis (ECMCPU and RooflineIACA)
    and LIKWID benchmarking (benchmark).
    """

    def incore_analysis(self, *args, **kwargs):
        raise NotImplementedError("IACA analysis is not possible based on a Kernel Description")

    def build_executable(self, *args, **kwargs):
        raise NotImplementedError("Building and compilation is not possible based on a Kernel "
                                  "Description")

    def __init__(self, description, machine=None):
        """
        Create kernel representation from a description dictionary.

        :param description: must have a dictionary like interface (e.g., a YAML object).
        """
        super(KernelDescription, self).__init__(machine=machine)

        # Loops
        self._loop_stack = list([
            (l['index'], string_to_sympy(l['start']),
             string_to_sympy(l['stop']), string_to_sympy(l['step']))
            for l in description['loops']
        ])

        # Variables
        for var_name, v in description['arrays'].items():
            self.set_variable(var_name, v['type'], string_to_sympy(v['dimension']))

        # Datatype
        self.datatype = list(self.variables.values())[0][0]

        # Data sources
        self.sources = {
            var_name: set([string_to_sympy(idx) for idx in v])
            for var_name, v in description['data sources'].items()
        }

        # Data destinations
        self.destinations = {
            var_name: set([string_to_sympy(idx) for idx in v])
            for var_name, v in description['data destinations'].items()
        }

        # Flops
        self._flops = description['flops']

        self.check()


class BinaryDescription(Kernel):
    """
    Binary information gathered from command line.

    This class does NOT allow compilation, required for IACA analysis (ECMCPU and RooflineIACA)
    and LIKWID benchmarking (benchmark).

    To be used only for stand-alone benchmarking.
    """

    def incore_analysis(self, *args, **kwargs):
        raise NotImplementedError("IACA analysis is not possible based on a Kernel Description")

    def build_executable(self, *args, **kwargs):
        raise NotImplementedError("Building and compilation is not possible based on a Kernel "
                                  "Description")

    def __init__(self, args=None, machine=None):
        """
        Create kernel representation from a description dictionary.

        :param args: read in arguments from command line.
        """

        if args is None:
            raise ValueError("Command Line Arguments must be provided in BinaryDescription.")

        super(BinaryDescription, self).__init__(machine=machine)

        self.args = args

        self.binary = args.binary

        self.define = args.define

        self.loops = args.loop

        self.regions = set(args.marker['region'])

        for region in args.loop.keys():
            for i in range(len(self.loops[region])):
                name = self.loops[region][i]['variable']
                if name:
                    self.loops[region][i]['offset'] = \
                        self.define[name]['value'] - self.loops[region][i]['end']


        # specified regions for repetitions
        repetition_regions = args.repetitions.keys()
        self.repetitions = {r: {'marker': None, 'variable': None, 'value': 1}
                               for r in repetition_regions}

        for region in repetition_regions:
            if args.repetitions[region] == 'marker':
                # obtain repetitions from likwid-marker
                if not args.marker['use_marker']:
                    print('Please enable --marker in order to obtain the number of repetitions '
                          'from likwid markers.')
                    sys.exit(-1)
                self.repetitions[region]['marker'] = True
                self.repetitions[region]['value'] = None
            else:
                try:
                    # number of repetitions is specified as number
                    self.repetitions[region]['value'] = int(args.repetitions[region])
                except:
                    # number of repetitions is given in terms of a variable
                    self.repetitions[region]['variable'] = args.repetitions[region]
                    self.repetitions[region]['value'] = None

        # Datatype
        self.datatype = (args.datatype,)

        # Number of Flops
        for region in args.flops.keys():
            self._flops[region] = args.flops[region]

        if len(self.regions) > 1:
            self.regions.discard('')

        self.check()

    def check(self):
        """Check that information about kernel makes sens and is valid."""
        datatypes = [v[0] for v in self.variables.values()]
        assert len(set(datatypes)) <= 1, 'mixing of datatypes within a kernel is not supported.'

        # check if all values are specified for all regions
        for region in self.regions:
            # use ''-values as default for not specified regions
            try:
                if not region in self.loops:
                    self.loops[region] = self.loops['']
                    if self.args.verbose > 1:
                        print('WARNING: used default value for loop information in region '
                              '{}.'.format(region))
                if not region in self.repetitions:
                    self.repetitions[region] = self.repetitions['']
                    if self.args.verbose > 1:
                        print('WARNING: used default value for repetition information in region '
                              '{}.'.format(region))
                if not region in self._flops:
                    self._flops[region] = self._flops['']
                    if self.args.verbose > 1:
                        print('WARNING: used default value for flop information in region '
                              '{}.'.format(region))

            except KeyError:
                print('WARNING: could not find default values for missing region information. '
                      'Please check your command line arguments.')
                sys.exit(-1)

        for name, variable in self.define.items():
            # if variable is marked adjustable but does not specify a region, assume it holds for all regions
            if variable['adjustable'] and variable['region'] == ['']:
                variable['region'] +=  list(self.regions)


    def print_kernel_info(self, output_file=sys.stdout):
        """Print kernel information in human readable format."""
        table = ('  region |        min        max       step\n' +
                 '---------+---------------------------------\n')
        for region in self.regions:
            table += '{:>8} |                                 \n'.format(region)

            for l in self.loops[region]:
                table += '         | {!r:>10} {!r:>10} {!r:>10}\n'.format(
                    l['start'], l['end'], l['step'])

        print(prefix_indent('loop stack:        ', table), file=output_file)


    def region__bytes_per_iteration(self, region):
        """
        Consecutive bytes written out per high-level iterations (as counted by loop stack).

        Is used to compute number of iterations per cacheline.
        """

        # Multiplying datatype size with step increment of inner-most loop
        return self.datatypes_size[self.datatype] * self.loops[region][-1]['step']


    def region__iterations_per_repetition(self, region):
        iterations_per_repetition = 1

        for loop in self.loops[region]:
            iterations_per_repetition = \
                iterations_per_repetition * floor((loop['end'] - loop['start']) / loop['step'])

        return iterations_per_repetition

    def set_constant(self, name, value):
        """
        Set constant of name to value.

        :param name: may be a str or a sympy.Symbol
        :param value: must be an int or float
        """
        assert isinstance(name, str) or isinstance(name, sympy.Symbol), \
            "constant name needs to be of type str, unicode or a sympy.Symbol"
        assert type(value) is int or type(value) is float, \
            "constant value needs to be of type int or float"
        if isinstance(name, sympy.Symbol):
            self.constants[name] = value
        else:
            self.constants[symbol_pos_int(name)] = value