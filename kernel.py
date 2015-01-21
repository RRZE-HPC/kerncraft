#!/usr/bin/env python

from __future__ import print_function

from pycparser import CParser, c_ast
import sympy  # TODO remove dependency to sympy


def prefix_indent(prefix, textblock, later_prefix=' '):
    textblock = textblock.split('\n')
    s = prefix + textblock[0] + '\n'
    if len(later_prefix) == 1:
        later_prefix = ' '*len(prefix)
    s = s+'\n'.join(map(lambda x: later_prefix+x, textblock[1:]))
    if s[-1] != '\n':
        return s + '\n'
    else:
        return s


def conv_ast_to_sympy(math_ast):
    '''
    converts mathematical expressions containing paranthesis, addition, subtraction and
    multiplication from AST to SymPy expresions.
    '''
    if type(math_ast) is c_ast.ID:
        return sympy.Symbol(math_ast.name)
    elif type(math_ast) is c_ast.Constant:
        return int(math_ast.value)
    else:  # elif type(dim) is c_ast.BinaryOp:
        sympy_op = {'*': lambda l,r: sympy.Mul(l, r, evaluate=False),
                    '+': lambda l,r: sympy.Add(l, r, evaluate=False),
                    '-': lambda l,r: sympy.Add(l, sympy.Mul(-1, r), evaluate=False)}
        
        op = sympy_op[math_ast.op]
        return op(conv_ast_to_sympy(math_ast.left), conv_ast_to_sympy(math_ast.right))


class Kernel:
    def __init__(self, kernel_code, constants=None, variables=None):
        '''This class captures the DSL kernel code, analyzes it and reports access pattern'''
        self.kernel_code = kernel_code
    
        parser = CParser()
        self.kernel_ast = parser.parse('void test() {'+kernel_code+'}').ext[0].body
        
        self._loop_stack = []
        self._sources = {}
        self._destinations = {}
        self._constants = {} if constants is None else constants
        self._variables = {} if variables is None else variables
    
    def set_constant(self, name, value):
        assert type(name) is str, "constant name needs to be of type str"
        assert type(value) is int, "constant value needs to be of type int"
        self._constants[name] = value
    
    def set_variable(self, name, type_, size):
        assert type_ in ['double', 'float'], 'only float and double variables are supported'
        assert type(size) in [tuple, type(None)], 'size has to be defined as tuple'
        self._variables[name] = (type_, size)
    
    def process(self):
        assert type(self.kernel_ast) is c_ast.Compound, "Kernel has to be a compound statement"
        assert all(map(lambda s: type(s) is c_ast.Decl, self.kernel_ast.block_items[:-1])), \
            'all statments befor the for loop need to be declarations'
        assert type(self.kernel_ast.block_items[-1]) is c_ast.For, \
            'last statment in kernel code must be a loop'
        
        for item in self.kernel_ast.block_items[:-1]:
            array = type(item.type) is c_ast.ArrayDecl
            
            if array:
                dims = []
                t = item.type
                while type(t) is c_ast.ArrayDecl:
                    dims.append(int(conv_ast_to_sympy(t.dim).subs(self._constants)))
                    t = t.type
                
                assert len(t.type.names) == 1, "only single types are supported"
                self.set_variable(item.name, t.type.names[0], tuple(dims))
                
            else:
                assert len(item.type.type.names) == 1, "only single types are supported"
                self.set_variable(item.name, item.type.type.names[0], None)
        
        floop = self.kernel_ast.block_items[-1]
        self._p_for(floop)
    
    def _get_offsets(self, aref, dim=0):
        '''
        returns a list of offsets of an ArrayRef object in all dimensions
        
        the index order is right to left (c-code order).
        e.g. c[i+1][j-2] -> [-2, +1]
        '''
        
        # Check for restrictions
        assert type(aref.name) in [c_ast.ArrayRef, c_ast.ID], \
            "array references must only be used with variables or other array references"
        assert type(aref.subscript) in [c_ast.ID, c_ast.Constant, c_ast.BinaryOp], \
            'array subscript must only contain variables or binary operations'
        
        idxs = []
        
        # TODO work-in-progress generisches auswerten von allem in [...]
        #idxs.append(('rel', conv_ast_to_sympy(aref.subscript)))
        if type(aref.subscript) is c_ast.BinaryOp:
            assert aref.subscript.op in '+-', \
                'binary operations in array subscript must by + or -'
            assert (type(aref.subscript.left) is c_ast.ID and \
                    type(aref.subscript.right) is c_ast.Constant), \
                'binary operation in array subscript may only have form "variable +- constant"'
            assert aref.subscript.left.name in map(lambda l: l[0], self._loop_stack), \
                'varialbes used in array indices has to be a loop counter'
            
            sign = 1 if aref.subscript.op == '+' else -1
            offset = sign*int(aref.subscript.right.value)
            
            idxs.append(('rel', aref.subscript.left.name, offset))
        elif type(aref.subscript) is c_ast.ID:
            assert aref.subscript.name in map(lambda l: l[0], self._loop_stack), \
                'varialbes used in array indices has to be a loop counter'
            idxs.append(('rel', aref.subscript.name, 0))
        else:  # type(aref.subscript) is c_ast.Constant
            idxs.append(('abs', int(aref.subscript.value)))
        
        if type(aref.name) is c_ast.ArrayRef:
            idxs += self._get_offsets(aref.name, dim=dim+1)
        
        if dim == 0:
            idxs.reverse()
        
        return idxs
    
    @classmethod
    def _get_basename(cls, aref):
        '''
        returns base name of ArrayRef object
        
        e.g. c[i+1][j-2] -> 'c'
        '''
        
        if type(aref.name) is c_ast.ArrayRef:
            return cls._get_basename(aref.name)
        else:
            return aref.name.name
    
    def _p_for(self, floop):
        # Check for restrictions
        assert type(floop) is c_ast.For, "May only be a for loop"
        assert hasattr(floop, 'init') and hasattr(floop, 'cond') and hasattr(floop, 'next'), \
            "Loop must have initial, condition and next statements."
        assert type(floop.init) is c_ast.Assignment, "Initialization of loops need to be " + \
            "assignments (declarations are not allowed or needed)"
        assert floop.cond.op in '<', "only lt (<) is allowed as loop condition"
        assert type(floop.cond.left) is c_ast.ID, 'left of cond. operand has to be a variable'
        assert type(floop.cond.right) in [c_ast.Constant, c_ast.ID, c_ast.BinaryOp], \
            'right of cond. operand has to be a constant, a variable or a binary operation'
        assert type(floop.next) in [c_ast.UnaryOp, c_ast.Assignment], 'next statement has to ' + \
            'be a unary or assignment operation'
        assert floop.next.op in ['++', 'p++', '+='], 'only ++ and += next operations are allowed'
        assert type(floop.stmt) in [c_ast.Compound, c_ast.Assignment, c_ast.For], 'the inner ' + \
            'loop may contain only assignments or compounds of assignments'

        if type(floop.cond.right) is c_ast.ID:
            const_name = floop.cond.right.name
            assert const_name in self._constants, 'loop right operand has to be defined as a ' +\
                 'constant in ECM object'
            iter_max = self._constants[const_name]
        elif type(floop.cond.right) is c_ast.Constant:
            iter_max = int(floop.cond.right.value)
        else:  # type(floop.cond.right) is c_ast.BinaryOp
            bop = floop.cond.right
            assert type(bop.left) is c_ast.ID, 'left of operator has to be a variable'
            assert type(bop.right) is c_ast.Constant, 'right of operator has to be a constant'
            assert bop.op in '+-', 'only plus (+) and minus (-) are accepted operators'
            
            sign = 1 if bop.op == '+' else -1
            iter_max = self._constants[bop.left.name]+sign*int(bop.right.value)
        
        if type(floop.next) is c_ast.Assignment:
            assert type(floop.next.lvalue) is c_ast.ID, \
                'next operation may only act on loop counter'
            assert type(floop.next.rvalue) is c_ast.Constant, 'only constant increments are allowed'
            assert floop.next.lvalue.name ==  floop.cond.left.name ==  floop.init.lvalue.name, \
                'initial, condition and next statement of for loop must act on same loop ' + \
                'counter variable'
            step_size = int(floop.next.rvalue.value)
        else:
            assert type(floop.next.expr) is c_ast.ID, 'next operation may only act on loop counter'
            assert floop.next.expr.name ==  floop.cond.left.name ==  floop.init.lvalue.name, \
                'initial, condition and next statement of for loop must act on same loop ' + \
                'counter variable'
            step_size = 1
        
        # Document for loop stack
        self._loop_stack.append(
            # (index name, min, max, step size)
            (floop.init.lvalue.name, floop.init.rvalue.value, iter_max, step_size)
        )
        # TODO add support for other stepsizes (even negative/reverse steps?)

        # Traverse tree
        if type(floop.stmt) is c_ast.For:
            self._p_for(floop.stmt)
        elif type(floop.stmt) is c_ast.Compound and \
                len(floop.stmt.block_items) == 1 and \
                type(floop.stmt.block_items[0]) is c_ast.For:
            self._p_for(floop.stmt.block_items[0])
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
        
        # Document data destination
        if type(stmt.lvalue) is c_ast.ArrayRef:
            # self._destinations[dest name] = [dest offset, ...])
            self._destinations.setdefault(self._get_basename(stmt.lvalue), [])
            self._destinations[self._get_basename(stmt.lvalue)].append(
                 self._get_offsets(stmt.lvalue))
            # TODO deactivated for now, since that notation might be useless
            # ArrayAccess(stmt.lvalue, array_info=self._variables[self._get_basename(stmt.lvalue)])
        else:  # type(stmt.lvalue) is c_ast.ID
            self._destinations.setdefault(stmt.lvalue.name, [])
            self._destinations[stmt.lvalue.name].append([('dir',)])
        
        # Traverse tree
        self._p_sources(stmt.rvalue)

    def _p_sources(self, stmt):
        sources = []
        
        assert type(stmt) in [c_ast.ArrayRef, c_ast.Constant, c_ast.ID, c_ast.BinaryOp], \
            'only references to arrays, constants and variables as well as binary operations ' + \
            'are supported'

        if type(stmt) is c_ast.ArrayRef:            
            # Document data source
            bname = self._get_basename(stmt)
            self._sources.setdefault(bname, [])
            self._sources[bname].append(self._get_offsets(stmt))
            # TODO deactivated for now, since that notation might be useless
            # ArrayAccess(stmt, array_info=self._variables[bname])
        elif type(stmt) is c_ast.ID:
            # Document data source
            self._sources.setdefault(stmt.name, [])
            self._sources[stmt.name].append([('dir',)])
        elif type(stmt) is c_ast.BinaryOp:
            # Traverse tree
            self._p_sources(stmt.left)
            self._p_sources(stmt.right)
        
        return sources
    
    def print_kernel_info(self):
        table = ('     idx |        min        max       step\n' +
                 '---------+---------------------------------\n')
        for l in self._loop_stack:
            table += '{:>8} | {:>10} {:>10} {:>+10}\n'.format(*l)
        print(prefix_indent('loop stack:        ', table))
        
        table = ('    name |  offsets   ...\n' +
                 '---------+------------...\n')
        for name, offsets in self._sources.items():
            prefix = '{:>8} | '.format(name)
            right_side = '\n'.join(map(lambda o: ', '.join(map(tuple.__repr__, o)), offsets))
            table += prefix_indent(prefix, right_side, later_prefix='         | ')
        print(prefix_indent('data sources:      ', table))
        
        table = ('    name |  offsets   ...\n' +
                 '---------+------------...\n')
        for name, offsets in self._destinations.items():
            prefix = '{:>8} | '.format(name)
            right_side = '\n'.join(map(lambda o: ', '.join(map(tuple.__repr__, o)), offsets))
            table += prefix_indent(prefix, right_side, later_prefix='         | ')
        print(prefix_indent('data destinations: ', table))

    def print_kernel_code(self):
        print(self.kernel_code)
    
    def print_variables_info(self):
        table = ('    name |   type size             \n' +
                 '---------+-------------------------\n')
        for name, var_info in self._variables.items():
            table += '{:>8} | {:>6} {:<10}\n'.format(name, var_info[0], var_info[1])
        print(prefix_indent('variables: ', table))
    
    def print_constants_info(self):
        table = ('    name | value     \n' +
                 '---------+-----------\n')
        for name, value in self._constants.items():
            table += '{:>8} | {:<10}\n'.format(name, value)
        print(prefix_indent('constants: ', table))