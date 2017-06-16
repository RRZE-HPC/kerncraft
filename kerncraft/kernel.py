#!/usr/bin/env python

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from copy import deepcopy
import operator
import tempfile
import subprocess
import os
import os.path
import sys
import numbers
import collections
from functools import reduce
from string import ascii_letters
from distutils.spawn import find_executable
from itertools import chain
from collections import defaultdict
from pprint import pprint

import sympy
from sympy.utilities.lambdify import implemented_function
from sympy.parsing.sympy_parser import parse_expr
import numpy
from six.moves import filter
from six.moves import map
from six.moves import zip_longest
import six
from pylru import lrudecorator

from .pycparser import CParser, c_ast, plyparser
from .pycparser.c_generator import CGenerator

from . import iaca_marker as iaca


def prefix_indent(prefix, textblock, later_prefix=' '):
    textblock = textblock.split('\n')
    s = prefix + textblock[0] + '\n'
    if len(later_prefix) == 1:
        later_prefix = ' '*len(prefix)
    s = s+'\n'.join([later_prefix+x for x in textblock[1:]])
    if s[-1] != '\n':
        return s + '\n'
    else:
        return s


def trasform_multidim_to_1d_decl(decl):
    '''
    Transforms ast of multidimensional declaration to a single dimension declaration.
    In-place operation!

    Returns name and dimensions of array (to be used with transform_multidim_to_1d_ref())
    '''
    dims = []
    t = decl.type
    while type(t) is c_ast.ArrayDecl:
        dims.append(t.dim)
        t = t.type

    if dims:
        # Multidimensional array
        decl.type.dim = reduce(lambda l, r: c_ast.BinaryOp('*', l, r), dims)
        decl.type.type = t

    return decl.name, dims


def transform_multidim_to_1d_ref(aref, dimension_dict):
    '''
    Transforms ast of multidimensional reference to a single dimension reference.
    In-place operation!
    '''
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


def transform_array_decl_to_malloc(decl):
    '''Transforms ast of "type var_name[N]" to "type* var_name = __mm_malloc(N, 32)"
    (in-place)'''
    if type(decl.type) is not c_ast.ArrayDecl:
        # Not an array declaration, can be ignored
        return

    type_ = c_ast.PtrDecl([], decl.type.type)
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


def find_array_references(ast):
    '''returns list of array references in AST'''
    if type(ast) is c_ast.ArrayRef:
        return [ast]
    elif type(ast) is list:
        return list(map(find_array_references, ast))
    elif ast is None:
        return []
    else:
        return reduce(operator.add, [find_array_references(o[1]) for o in ast.children()], [])


# Make sure that functions will return iterable objects:
def force_iterable(f):
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper


class Kernel(object):
    '''This class captures the kernel information, analyzes it and reports access pattern'''
    # Datatype sizes in bytes
    datatypes_size = {'double': 8, 'float': 4}

    def __init__(self):
        self._loop_stack = []
        self.variables = {}
        self._sources = {}
        self._destinations = {}
        self._flops = {}
        self.datatype = None

        self.clear_state()

    def check(self):
        '''Checks that information about kernel makes sens and is valid.'''
        datatypes = [v[0] for v in self.variables.values()]
        assert len(set(datatypes)) <= 1, 'mixing of datatypes within a kernel is not supported.'

        # TODO add combine all tests here

    def set_constant(self, name, value):
        assert isinstance(name, six.string_types) or isinstance(name, sympy.Symbol), \
            "constant name needs to be of type str, unicode or a sympy.Symbol"
        assert type(value) is int, "constant value needs to be of type int"
        if isinstance(name, sympy.Symbol):
            self.constants[name] = value
        else:
            self.constants[sympy.Symbol(name, positive=True)] = value

    def set_variable(self, name, type_, size):
        assert type_ in self.datatypes_size, 'only float and double variables are supported'
        if self.datatype is None:
            self.datatype = type_
        else:
            assert type_ == self.datatype, 'mixing of datatypes within a kernel is not supported.'
        assert type(size) in [list, type(None)], 'size has to be defined as tuple'
        self.variables[name] = (type_, size)

    def clear_state(self):
        '''Clears changable internal states
        (constants, asm_blocks and asm_block_idx)'''
        self.constants = {}
        self.subs_consts.clear()  # clear LRU cache of function

    @lrudecorator(40)
    def subs_consts(self, expr):
        '''
        Substitutes constants in expression unless it is already a number
        '''
        if isinstance(expr, numbers.Number):
            return expr
        else:
            return expr.subs(self.constants)

    def array_sizes(self, in_bytes=False, subs_consts=False):
        '''Returns a dictionary with all arrays sizes (optunally in bytes, otherwise in elements).

        :param in_bytes: If True, output will be in bytes, not in element counts.
        :param subs_consts: If True, output will be numbers and not symbolic.

        Scalar variables are ignored.
        '''
        var_sizes = {}

        for var_name, var_info in self.variables.items():
            var_type, var_size = var_info

            # Skiping sclars
            if var_size is None:
                continue

            var_sizes[var_name] = reduce(operator.mul, var_size, 1)

            # Multiply by bytes per element if requested
            if in_bytes:
                element_size = self.datatypes_size[var_type]
                var_sizes[var_name] *= element_size

        if subs_consts:
            return {k: self.subs_consts(v) for k,v in var_sizes.items()}
        else:
            return var_sizes

    def _calculate_relative_offset(self, name, access_dimensions):
        '''
        returns the offset from the iteration center in number of elements and the order of indices
        used in access.
        '''
        # TODO to be replaced with compile_global_offsets
        offset = 0
        base_dims = self.variables[name][1]

        for dim, offset_info in enumerate(access_dimensions):
            offset_type, idx_name, dim_offset = offset_info
            assert offset_type == 'rel', 'Only relative access to arrays is supported at the moment'

            if offset_type == 'rel':
                offset += self.kernel.subs_consts(
                   dim_offset*reduce(operator.mul, base_dims[dim+1:], sympy.Integer(1)))
            else:
                # should not happen
                pass

        return offset

    def access_to_sympy(self, var_name, access):
        '''Transforms a variable access to a sympy expression'''
        base_sizes = self.variables[var_name][1]

        expr = sympy.Number(0)

        for dimension, a in enumerate(access):
            base_size = reduce(operator.mul, base_sizes[dimension+1:], sympy.Integer(1))

            expr += base_size*a

        return expr

    def iteration_length(self, dimension=None):
        '''Returns the number of global loop iterations that are performed

        If dimension is not None, it is the loop dimension that is returned
        (-1 is the inner most loop and 0 the outermost)'''

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
        for l in self._loop_stack:
            if subs_consts:
                yield {'index': l[0],
                       'start': self.subs_consts(l[1]),
                       'stop': self.subs_consts(l[2]),
                       'increment': self.subs_consts(l[3])}
            else:
                yield {'index': l[0], 'start': l[1], 'stop': l[2], 'increment': l[3]}

    def index_order(self, sources=True, destinations=True):
        '''
        Returns the order of indices as they appear in array references

        Use *source* and *destination* to reduce output
        '''
        if sources:
            arefs = chain(*self._sources.values())
        else:
            arefs = []
        if destinations:
            arefs = chain(arefs, *self._destinations.values())

        ret = []
        for a in [aref for aref in arefs if aref is not None]:
            ref = []
            for expr in a:
                ref.append(expr.free_symbols)
            ret.append(ref)

        return ret

    def compile_sympy_accesses(self, sources=True, destinations=True):
        '''returns a dictionary of lists of sympy accesses, for each variable'''
        sympy_accesses = defaultdict(list)
        # Compile sympy accesses
        for var_name in self.variables:
            if sources:
                for r in self._sources.get(var_name, []):
                    if r is None: continue
                    sympy_accesses[var_name].append(self.access_to_sympy(var_name, r))
            if destinations:
                for w in self._destinations.get(var_name, []):
                    if w is None: continue
                    sympy_accesses[var_name].append(self.access_to_sympy(var_name, w))

        return sympy_accesses

    def compile_relative_distances(self, sympy_accesses=None):
        '''Returns load and store distances between accesses.

        :param sympy_accesses: optionally restrict accesses, default from compile_sympy_accesses()

        e.g. if accesses are to [+N, +1, -1, -N], relative distances are [N-1, 2, N-1]

        returned is a dict of list of sympy expressions, for each variable'''
        if sympy_accesses is None:
            sympy_accesses = self.compile_sympy_accesses()

        sympy_distances = defaultdict(list)
        for var_name, accesses in sympy_accesses.items():
            for i in range(1, len(accesses)):
                sympy_distances[var_name].append((accesses[i-1]-accesses[i]).simplify())

        return sympy_distances

    def global_iterator_to_indices(self, git=None):
        '''Returns sympy expressions translating global_iterator to loop indices,
        or if global_iterator is given, an integer is returned'''
        # unwind global iteration count into loop counters:
        base_loop_counters = {}
        if git is None:
            global_iterator = sympy.Symbol('global_iterator', positive=True)
        else:
            global_iterator = git
        idiv = implemented_function(sympy.Function(str('idiv')), lambda x, y: x//y)
        total_length = 1
        last_incr = 1
        for var_name, start, end, incr in reversed(self._loop_stack):
            loop_var = sympy.Symbol(var_name, positive=True)

            # This unspools the iterations:
            length = end-start  # FIXME is incr handled correct here?
            counter = start+(idiv(global_iterator*last_incr, total_length)*incr) % length
            total_length = total_length*length
            last_incr = incr

            if git is not None:
                try:  # Try to resolve to integer if global_iterator was given
                    base_loop_counters[loop_var] = sympy.Integer(self.subs_consts(counter))
                    continue
                except:
                    pass
            base_loop_counters[loop_var] = sympy.lambdify(
                global_iterator,
                self.subs_consts(counter), modules=[numpy, {'Mod': numpy.mod}])

        return base_loop_counters

    def indices_to_global_iterator(self, indices):
        '''Transforms a dictionary of indices to a global iterator integer.

        Inverse of global_iterator_to_indices().'''
        global_iterator = sympy.Integer(0)
        total_length = 1
        for var_name, start, end, incr in reversed(self._loop_stack):
            loop_var = sympy.Symbol(var_name, positive=True)
            length = end-start  # FIXME is incr handled correct here?
            global_iterator += (indices[loop_var] - start)*total_length
            total_length = total_length*length

        return self.subs_consts(global_iterator)


    def compile_global_offsets(self, iteration=0, spacing=0):
        '''Returns load and store offsets on a virtual address space.

        :param iteration: controlls the inner index counter
        :param spacing: sets a spacing between the arrays, default is 0

        All array variables (non scalars) are layed out linearly starting from 0. An optional
        spacing can be set. The accesses are based on this layout.

        The iteration 0 is the first itreation. All loops are mapped to this linear iteration space.

        Accesses to scalars are ignored.

        Returned are load and store byte-offset pairs for each iteration.
        '''
        global_load_offsets = []
        global_store_offsets = []

        if not isinstance(iteration, collections.Sequence):
            iteration = [iteration]
        iteration = numpy.array(iteration)

        # loop indices based on iteration
        # unwind global iteration count into loop counters:
        base_loop_counters = self.global_iterator_to_indices()
        total_length = self.iteration_length()

        assert max(iteration) < self.subs_consts(total_length), \
            "Iterations go beyond what is possible in the original code. One common reason is, " + \
            "that the iteration length are unrealistically small."

        # Get sizes of arrays and base offsets for each array
        var_sizes = self.array_sizes(in_bytes=True, subs_consts=True)
        base_offsets = {}
        base = 0
        # Always arange arrays in alphabetical order in memory, for reproducability
        for var_name, var_size in sorted(var_sizes.items(), key=lambda v: v[0]):
            base_offsets[var_name] = base
            array_total_size = self.subs_consts(var_size + spacing)
            # Add bytes to align by 64 byte (typical cacheline size):
            array_total_size = ((int(array_total_size)+63)& ~63)
            base += array_total_size

        # Gather all read and write accesses to the array:
        for var_name, var_size in var_sizes.items():
            element_size = self.datatypes_size[self.variables[var_name][0]]
            for r in self._sources.get(var_name, []):
                offset_expr = self.access_to_sympy(var_name, r)
                offset = force_iterable(sympy.lambdify(
                    base_loop_counters.keys(),
                    self.subs_consts(
                        offset_expr*element_size
                        + base_offsets[var_name]), numpy))
                # TODO possibly differentiate between index order
                global_load_offsets.append(offset)
            for w in self._destinations.get(var_name, []):
                offset_expr = self.access_to_sympy(var_name, w)
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

        # Data access as they appear with iteration order
        return zip_longest(zip(*[o(*counter_per_it) for o in global_load_offsets]),
                           zip(*[o(*counter_per_it) for o in global_store_offsets]),
                           fillvalue=None)

    def print_kernel_info(self, output_file=sys.stdout):
        table = ('     idx |        min        max       step\n' +
                 '---------+---------------------------------\n')
        for l in self._loop_stack:
            table += '{:>8} | {!r:>10} {!r:>10} {!r:>10}\n'.format(*l)
        print(prefix_indent('loop stack:        ', table), file=output_file)

        table = ('    name |  offsets   ...\n' +
                 '---------+------------...\n')
        for name, offsets in list(self._sources.items()):
            prefix = '{:>8} | '.format(name)
            right_side = '\n'.join(['{!r:}'.format(o) for o in offsets])
            table += prefix_indent(prefix, right_side, later_prefix='         | ')
        print(prefix_indent('data sources:      ', table), file=output_file)

        table = ('    name |  offsets   ...\n' +
                 '---------+------------...\n')
        for name, offsets in list(self._destinations.items()):
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
        table = ('    name |   type size             \n' +
                 '---------+-------------------------\n')
        for name, var_info in list(self.variables.items()):
            table += '{:>8} | {:>6} {!s:<10}\n'.format(name, var_info[0], var_info[1])
        print(prefix_indent('variables: ', table), file=output_file)

    def print_constants_info(self, output_file=sys.stdout):
        table = ('    name | value     \n' +
                 '---------+-----------\n')
        for name, value in list(self.constants.items()):
            table += '{!s:>8} | {:<10}\n'.format(name, value)
        print(prefix_indent('constants: ', table), file=output_file)


class KernelCode(Kernel):
    '''
    Kernel information gathered from code using pycparser

    This version allows compilation and generation of code for iaca and likwid benchmarking
    '''
    def __init__(self, kernel_code, filename=None):
        super(KernelCode, self).__init__()

        # Initialize state
        self.asm_blocks = {}
        self.asm_block_idx = None

        self.kernel_code = kernel_code
        self._filename = filename
        # need to refer to local lextab, otherwise the systemwide lextab would be imported
        parser = CParser(lextab='kerncraft.pycparser.lextab',
                         yacctab='kerncraft.pycparser.yacctab')
        try:
            self.kernel_ast = parser.parse(self._as_function(), filename=filename).ext[0].body
        except plyparser.ParseError as e:
            print('Error parsing kernel code:', e)
            sys.exit(1)

        self._process_code()

        self.check()

    def print_kernel_code(self, output_file=sys.stdout):
        print(self.kernel_code, file=output_file)

    def _as_function(self, func_name='test', filename=None):
        if filename is None:
            filename = ''
        else:
            filename ='"{}"'.format(filename)
        return '#line 0 \nvoid {}() {{\n#line 1 {}\n{}\n#line 999 \n}}'.format(func_name, filename, self.kernel_code)

    def clear_state(self):
        '''Clears changable internal states'''
        super(KernelCode, self).clear_state()
        self.asm_blocks = {}
        self.asm_block_idx = None
        self.subs_consts.clear()  # clear LRU cache of function

    def _process_code(self):
        assert type(self.kernel_ast) is c_ast.Compound, "Kernel has to be a compound statement"
        assert all([type(s) in [c_ast.Decl, c_ast.Pragma]
                    for s in self.kernel_ast.block_items[:-1]]), \
            'all statments before the for loop need to be declarations or pragmas'
        assert type(self.kernel_ast.block_items[-1]) is c_ast.For, \
            'last statment in kernel code must be a loop'

        for item in self.kernel_ast.block_items[:-1]:
            if type(item) is c_ast.Pragma: continue
            array = type(item.type) is c_ast.ArrayDecl

            if array:
                dims = []
                t = item.type
                while type(t) is c_ast.ArrayDecl:
                    dims.append(self.conv_ast_to_sym(t.dim))
                    t = t.type

                assert len(t.type.names) == 1, "only single types are supported"
                self.set_variable(item.name, t.type.names[0], list(dims))

            else:
                assert len(item.type.type.names) == 1, "only single types are supported"
                self.set_variable(item.name, item.type.type.names[0], None)

        floop = self.kernel_ast.block_items[-1]
        self._p_for(floop)

    def conv_ast_to_sym(self, math_ast):
        '''
        converts mathematical expressions containing paranthesis, addition, subtraction and
        multiplication from AST to a sympy representation.
        '''
        if type(math_ast) is c_ast.ID:
            return sympy.Symbol(math_ast.name, positive=True)
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
        '''
        returns a list of offsets of an ArrayRef object in all dimensions

        the index order is right to left (c-code order).
        e.g. c[i+1][j-2] -> [-2, +1]

        if aref is actually an ID, None will be returned
        '''
        if isinstance(aref, c_ast.ID):
            return None

        # Check for restrictions
        assert type(aref.name) in [c_ast.ArrayRef, c_ast.ID], \
            "array references must only be used with variables or other array references"
        assert type(aref.subscript) in [c_ast.ID, c_ast.Constant, c_ast.BinaryOp], \
            'array subscript must only contain variables or binary operations'

        idxs = []

        # Convert subscript to sympy and append
        idxs.append(self.conv_ast_to_sym(aref.subscript))

        # Check for more indices (multi-dimensional access)
        if type(aref.name) is c_ast.ArrayRef:
            idxs += self._get_offsets(aref.name, dim=dim+1)

        # Reverse to preserver order (the subscripts in the AST are traversed backwards)
        if dim == 0:
            idxs.reverse()

        return idxs

    @classmethod
    def _get_basename(cls, aref):
        '''
        returns base name of ArrayRef object

        e.g. c[i+1][j-2] -> 'c'
        '''

        if isinstance(aref.name, c_ast.ArrayRef):
            return cls._get_basename(aref.name)
        elif isinstance(aref.name, six.string_types):
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
            iter_max = sympy.Symbol(const_name, positive=True)
        elif type(floop.cond.right) is c_ast.Constant:
            iter_max = sympy.Integer(floop.cond.right.value)
        else:  # type(floop.cond.right) is c_ast.BinaryOp
            bop = floop.cond.right
            #assert type(bop.left) is c_ast.ID, 'left of operator has to be a variable'
            #assert type(bop.right) is c_ast.Constant, 'right of operator has to be a constant'
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
        elif type(floop.stmt) is c_ast.Compound and \
                len(floop.stmt.block_items) == 1 and \
                type(floop.stmt.block_items[0]) is c_ast.For:
            self._p_for(floop.stmt.block_items[0])
        elif type(floop.stmt) is c_ast.Compound and \
                len(floop.stmt.block_items) == 2 and \
                type(floop.stmt.block_items[0]) is c_ast.Pragma and \
                type(floop.stmt.block_items[1]) is c_ast.For:
            self._p_for(floop.stmt.block_items[1])
        elif type(floop.stmt) is c_ast.Assignment:
            self._p_assignment(floop.stmt)
        else:  # type(floop.stmt) is c_ast.Compound
            for assgn in floop.stmt.block_items:
                self._p_assignment(assgn)

    def _p_assignment(self, stmt):
        # Check for restrictions
        assert type(stmt) is c_ast.Assignment, \
            "Only assignment statements are allowed in loops."
        assert type(stmt.lvalue) in [c_ast.ArrayRef, c_ast.ID], \
            "Only assignment to array element or varialbe is allowed."

        write_and_read = False
        if stmt.op != '=':
            write_and_read = True
            op = stmt.op.strip('=')
            self._flops[op] = self._flops.get(op, 0)+1

        # Document data destination
        # self._destinations[dest name] = [dest offset, ...])
        self._destinations.setdefault(self._get_basename(stmt.lvalue), [])
        self._destinations[self._get_basename(stmt.lvalue)].append(
            self._get_offsets(stmt.lvalue))

        if write_and_read:
            # this means that +=, -= or something of that sort was used
            self._sources.setdefault(self._get_basename(stmt.lvalue), [])
            self._sources[self._get_basename(stmt.lvalue)].append(
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
            self._sources.setdefault(bname, [])
            self._sources[bname].append(self._get_offsets(stmt))
        elif type(stmt) is c_ast.BinaryOp:
            # Traverse tree
            self._p_sources(stmt.left)
            self._p_sources(stmt.right)

            self._flops[stmt.op] = self._flops.get(stmt.op, 0)+1
        elif type(stmt) is c_ast.UnaryOp:
            self._p_sources(stmt.expr)
            self._flops[stmt.op] = self._flops.get(stmt.op[-1], 0)+1

        return sources

    def as_code(self, type_='iaca'):
        '''
        generates compilable source code from AST

        *type* can be iaca or likwid.
        '''
        assert self.kernel_ast is not None, "AST does not exist, this could be due to running of " \
             "kernel description rather then code."

        ast = deepcopy(self.kernel_ast)
        declarations = [d for d in ast.block_items if type(d) is c_ast.Decl]

        # transform multi-dimensional declarations to one dimensional references
        array_dimensions = dict(list(map(trasform_multidim_to_1d_decl, declarations)))
        # transform to pointer and malloc notation (stack can be too small)
        list(map(transform_array_decl_to_malloc, declarations))

        # add declarations for constants
        i = 1  # subscript for cli input
        for k in self.constants:
            # cont int N = atoi(argv[1])
            # TODO change subscript of argv depending on constant count
            type_decl = c_ast.TypeDecl(k.name, ['const'], c_ast.IdentifierType(['int']))
            init = c_ast.FuncCall(
                c_ast.ID('atoi'),
                c_ast.ExprList([c_ast.ArrayRef(c_ast.ID('argv'), c_ast.Constant('int', str(i)))]))
            i += 1
            ast.block_items.insert(0, c_ast.Decl(
                k.name, ['const'], [], [],
                type_decl, init, None))

        if type_ == 'likwid':
            # Call likwid_markerInit()
            ast.block_items.insert(0, c_ast.FuncCall(c_ast.ID('likwid_markerInit'), None))
            # Call likwid_markerThreadInit()
            ast.block_items.insert(1, c_ast.FuncCall(c_ast.ID('likwid_markerThreadInit'), None))
            # Call likwid_markerClose()
            ast.block_items.append(c_ast.FuncCall(c_ast.ID('likwid_markerClose'), None))

        # inject array initialization
        for d in declarations:
            i = ast.block_items.index(d)

            # Build ast to inject
            if array_dimensions[d.name]:
                # this is an array, we need a for loop to initialize it
                # for(init; cond; next) stmt

                # Init: int i = 0;
                counter_name = 'i'
                while counter_name in array_dimensions:
                    counter_name = chr(ord(counter_name)+1)

                init = c_ast.DeclList([
                    c_ast.Decl(
                        counter_name, [], [], [], c_ast.TypeDecl(
                            counter_name, [], c_ast.IdentifierType(['int'])),
                        c_ast.Constant('int', '0'),
                        None)],
                    None)

                # Cond: i < ... (... is length of array)
                cond = c_ast.BinaryOp(
                    '<',
                    c_ast.ID(counter_name),
                    reduce(lambda l, r: c_ast.BinaryOp('*', l, r), array_dimensions[d.name]))

                # Next: i++
                next_ = c_ast.UnaryOp('++', c_ast.ID(counter_name))

                # Statement
                stmt = c_ast.Assignment(
                    '=',
                    c_ast.ArrayRef(c_ast.ID(d.name), c_ast.ID(counter_name)),
                    c_ast.Constant('float', '0.23'))

                ast.block_items.insert(i+1, c_ast.For(init, cond, next_, stmt))

                # inject dummy access to arrays, so compiler does not over-optimize code
                # with if around it, so code will actually run
                ast.block_items.insert(
                    i+2, c_ast.If(
                        cond=c_ast.ID('var_false'),
                        iftrue=c_ast.Compound([
                            c_ast.FuncCall(
                                c_ast.ID('dummy'),
                                c_ast.ExprList([c_ast.ID(d.name)]))]),
                        iffalse=None))
            else:
                # this is a scalar, so a simple Assignment is enough
                ast.block_items.insert(
                    i+1, c_ast.Assignment('=', c_ast.ID(d.name), c_ast.Constant('float', '0.23')))

                # inject dummy access to scalar, so compiler does not over-optimize code
                # TODO put if around it, so code will actually run
                ast.block_items.insert(
                    i+2, c_ast.If(
                        cond=c_ast.ID('var_false'),
                        iftrue=c_ast.Compound([
                            c_ast.FuncCall(
                                c_ast.ID('dummy'),
                                c_ast.ExprList([c_ast.UnaryOp('&', c_ast.ID(d.name))]))]),
                        iffalse=None))

        # transform multi-dimensional array references to one dimensional references
        list(map(lambda aref: transform_multidim_to_1d_ref(aref, array_dimensions),
                 find_array_references(ast)))

        if type_ == 'likwid':
            # Instrument the outer for-loop with likwid
            ast.block_items.insert(-2, c_ast.FuncCall(
                c_ast.ID('likwid_markerStartRegion'),
                c_ast.ExprList([c_ast.Constant('string', '"loop"')])))

            dummies = []
            # Make sure nothing gets removed by inserting dummy calls
            for d in declarations:
                if array_dimensions[d.name]:
                    dummies.append(c_ast.If(
                        cond=c_ast.ID('var_false'),
                        iftrue=c_ast.Compound([
                            c_ast.FuncCall(
                                c_ast.ID('dummy'),
                                c_ast.ExprList([c_ast.ID(d.name)]))]),
                        iffalse=None))
                else:
                    dummies.append(c_ast.If(
                        cond=c_ast.ID('var_false'),
                        iftrue=c_ast.Compound([
                            c_ast.FuncCall(
                                c_ast.ID('dummy'),
                                c_ast.ExprList([c_ast.UnaryOp('&', c_ast.ID(d.name))]))]),
                        iffalse=None))

            # Wrap everything in a reapeat loop
            # int repeat = atoi(argv[2])
            type_decl = c_ast.TypeDecl('repeat', [], c_ast.IdentifierType(['int']))
            init = c_ast.FuncCall(
                c_ast.ID('atoi'),
                c_ast.ExprList([c_ast.ArrayRef(
                    c_ast.ID('argv'), c_ast.Constant('int', str(len(self.constants)+1)))]))
            ast.block_items.insert(-3, c_ast.Decl(
                'repeat', ['const'], [], [],
                type_decl, init, None))
            # for(; repeat > 0; repeat--) {...}
            cond = c_ast.BinaryOp( '>', c_ast.ID('repeat'), c_ast.Constant('int', '0'))
            next_ = c_ast.UnaryOp('--', c_ast.ID('repeat'))
            stmt = c_ast.Compound([ast.block_items.pop(-2)]+dummies)

            ast.block_items.insert(-1, c_ast.For(None, cond, next_, stmt))

            ast.block_items.insert(-1, c_ast.FuncCall(
                c_ast.ID('likwid_markerStopRegion'),
                c_ast.ExprList([c_ast.Constant('string', '"loop"')])))

        # embedd Compound into main FuncDecl
        decl = c_ast.Decl('main', [], [], [], c_ast.FuncDecl(c_ast.ParamList([
            c_ast.Typename(None, [], c_ast.TypeDecl('argc', [], c_ast.IdentifierType(['int']))),
            c_ast.Typename(None, [], c_ast.PtrDecl([], c_ast.PtrDecl(
                [], c_ast.TypeDecl('argv', [], c_ast.IdentifierType(['char'])))))]),
            c_ast.TypeDecl('main', [], c_ast.IdentifierType(['int']))),
            None, None)

        ast = c_ast.FuncDef(decl, None, ast)

        # embedd Compound AST into FileAST
        ast = c_ast.FileAST([ast])

        # add dummy function declaration
        decl = c_ast.Decl('dummy', [], [], [], c_ast.FuncDecl(
            c_ast.ParamList([c_ast.Typename(None, [], c_ast.PtrDecl(
                [], c_ast.TypeDecl(None, [], c_ast.IdentifierType(['double']))))]),
            c_ast.TypeDecl('dummy', [], c_ast.IdentifierType(['void']))),
            None, None)
        ast.ext.insert(0, decl)

        # add external var_false declaration
        decl = c_ast.Decl('var_false', [], ['extern'], [], c_ast.TypeDecl(
                'var_false', [], c_ast.IdentifierType(['int'])
            ), None, None)
        ast.ext.insert(1, decl)


        # convert to code string
        code = CGenerator().visit(ast)

        # add "#include"s for dummy, var_false and stdlib (for malloc)
        code = '#include <stdlib.h>\n\n' + code
        code = '#include "kerncraft.h"\n' + code
        if type_ == 'likwid':
            code = '#include <likwid.h>\n' + code

        return code

    def assemble(self, compiler, in_filename,
                 out_filename=None, iaca_markers=True, asm_block='auto', asm_increment=0):
        '''
        Assembles *in_filename* to *out_filename*.

        If *out_filename* is not given a new file will created either temporarily or according
        to kernel file location.

        if *iaca_marked* is set to true, markers are inserted around the block with most packed
        instructions or (if no packed instr. were found) the largest block and modified file is
        saved to *in_file*.

        *asm_block* controlls how the to-be-marked block is chosen. "auto" (default) results in
        the largest block, "manual" results in interactive and a number in the according block.

        *asm_increment* is the increment of the store pointer during each iteration of the ASM block
        if it is 0 (default), automatic detection will be use and might lead to an interactive user
        interface.

        Returns two-tuple (filepointer, filename) to temp binary file.
        '''
        if not out_filename:
            suffix = ''
            if iaca_markers:
                suffix += '.iaca_marked'
            if self._filename:
                out_filename = os.path.abspath(os.path.splitext(self._filename)[0]+suffix)
            else:
                out_filename = tempfile.mkstemp(suffix=suffix)

        # insert iaca markers
        if iaca_markers:
            with open(in_filename, 'r') as in_file:
                lines = in_file.readlines()
            blocks = iaca.find_asm_blocks(lines)

            # TODO check for already present markers

            # Choose best default block:
            block_idx = iaca.select_best_block(blocks)
            if asm_block == 'manual':
                block_idx = iaca.userselect_block(blocks, default=block_idx)
            elif asm_block != 'auto':
                block_idx = asm_block

            self.asm_block = blocks[block_idx][1]

            # Use userinput for pointer_increment, if given
            if asm_increment != 0:
                self.asm_block['pointer_increment'] = asm_increment

            # If block's pointer_increment is None, let user choose
            if self.asm_block['pointer_increment'] is None:
                iaca.userselect_increment(self.asm_block)

            # Insert markers:
            lines = iaca.insert_markers(
                lines, self.asm_block['first_line'], self.asm_block['last_line'])

            # write back to file
            with open(in_filename, 'w') as in_file:
                in_file.writelines(lines)

        try:
            # Assamble all to a binary
            subprocess.check_output(
                [compiler, os.path.basename(in_file.name), 'dummy.s', '-o', out_filename],
                cwd=os.path.dirname(os.path.realpath(in_file.name)))
        except subprocess.CalledProcessError as e:
            print(u"Assemblation failed:", e, file=sys.stderr)
            sys.exit(1)
        finally:
            in_file.close()

        return out_filename

    def compile(self, compiler, compiler_args=None):
        '''
        Compiles source (from as_code(type_)) to assembly.

        Returns two-tuple (filepointer, filename) to assembly file.

        Output can be used with Kernel.assemble()
        '''
        # Making sure compiler is available:
        if find_executable(compiler) is None:
            print("Compiler ({}) was not found. Choose different one in machine file "
                  "or make sure it is found in PATH.".format(compiler), file=sys.stderr)
            sys.exit(1)

        if not self._filename:
            in_file = tempfile.NamedTemporaryFile(
                suffix='_compilable.c', mode='w', encoding='ascii'
            )
        else:
            in_file = open(self._filename+"_compilable.c", 'w')

        in_file.write(self.as_code())
        in_file.flush()

        if compiler_args is None:
            compiler_args = []
        compiler_args += ['-std=c99']

        try:
            subprocess.check_output(
                [compiler] +
                compiler_args +
                [os.path.basename(in_file.name),
                 '-S',
                 '-I'+os.path.abspath(os.path.dirname(os.path.realpath(__file__)))+'/headers/'],
                cwd=os.path.dirname(os.path.realpath(in_file.name)))

            subprocess.check_output(
                [compiler] + compiler_args + [
                    os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/headers/dummy.c'),
                    '-S'],
                cwd=os.path.dirname(os.path.realpath(in_file.name)))
        except subprocess.CalledProcessError as e:
            print(u"Compilation failed:", e, file=sys.stderr)
            sys.exit(1)
        finally:
            in_file.close()

        # Let's return the out_file name
        return os.path.splitext(in_file.name)[0]+'.s'

    def build(self, compiler, cflags=None, lflags=None, verbose=False):
        '''
        compiles source to executable with likwid capabilities

        returns the executable name
        '''
        if not (('LIKWID_INCLUDE' in os.environ or 'LIKWID_INC' in os.environ) and
                'LIKWID_LIB' in os.environ):
            print('Could not find LIKWID_INCLUDE and LIKWID_LIB environment variables',
                  file=sys.stderr)
            sys.exit(1)

        # Making sure compiler is available:
        if find_executable(compiler) is None:
            print("Compiler ({}) was not found. Choose different one in machine file "
                  "or make sure it is found in PATH.".format(compiler), file=sys.stderr)
            sys.exit(1)

        if cflags is None:
            cflags = []
        cflags += ['-std=c99',
                   '-I'+os.path.abspath(os.path.dirname(os.path.realpath(__file__)))+'/headers/',
                   os.environ.get('LIKWID_INCLUDE', ''),
                   os.environ.get('LIKWID_INC', ''),
                   '-llikwid']

        if lflags is None:
            lflags = []
        lflags += os.environ['LIKWID_LIB'].split(' ') + ['-pthread']

        if not self._filename:
            source_file = tempfile.NamedTemporaryFile(
                suffix='_compilable.c', mode='w', encoding='ascii'
            )
        else:
            source_file = open(self._filename+"_compilable.c", 'w')

        source_file.write(self.as_code(type_='likwid'))
        source_file.flush()

        infiles = [os.path.abspath(os.path.dirname(os.path.realpath(__file__)))+'/headers/dummy.c',
                   source_file.name]
        if self._filename:
            outfile = os.path.abspath(os.path.splitext(self._filename)[0]+'.likwid_marked')
        else:
            outfile = tempfile.mkstemp(suffix='.likwid_marked')
        cmd = [compiler] + infiles + cflags + lflags + ['-o', outfile]
        # remove empty arguments
        cmd = list(filter(bool, cmd))
        if verbose:
            print(' '.join(cmd))
        try:
            subprocess.check_output(cmd)
        except subprocess.CalledProcessError as e:
            print("Build failed:", e, file=sys.stderr)
            sys.exit(1)
        finally:
            source_file.close()

        return outfile


class KernelDescription(Kernel):
    '''
    Kernel information gathered from YAML kernel description file

    This class does NOT allow compilation (required by iaca analysis and likwid benchmarking).
    '''
    def __init__(self, description):
        super(KernelDescription, self).__init__()

        # Loops
        self._loop_stack = list([
            (l['index'], self.string_to_sympy(l['start']),
             self.string_to_sympy(l['stop']), self.string_to_sympy(l['step']))
            for l in description['loops']
        ])

        # Variables
        for var_name, v in description['arrays'].items():
            self.set_variable(var_name, v['type'], self.string_to_sympy(v['dimension']))

        # Datatype
        self.datatype = list(self.variables.values())[0][0]

        # Data sources
        self._sources = {
            var_name: list([self.string_to_sympy(idx) for idx in v])
            for var_name,v in description['data sources'].items()
        }

        # Data destinations
        self._destinations = {
            var_name: list([self.string_to_sympy(idx) for idx in v])
            for var_name,v in description['data destinations'].items()
        }

        # Flops
        self._flops = description['flops']

        self.check()

    @classmethod
    def string_to_sympy(cls, s):
        if isinstance(s, int):
            return sympy.Integer(s)
        elif isinstance(s, list):
            return list([cls.string_to_sympy(e) for e in s])
        elif s is None:
            return None
        else:
            # TODO find nicer solution for N and other pre-mapped letters
            return parse_expr(s, local_dict={c: sympy.Symbol(c, positive=True)
                                             for c in s if c in ascii_letters})
