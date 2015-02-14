#!/usr/bin/env python

from __future__ import print_function

from copy import copy
import operator
import tempfile
import subprocess
import os.path
import operator

from pycparser import CParser, c_ast, c_generator
from pycparser.c_generator import CGenerator
import iaca_marker as iaca


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
                dimension_dict[name.name][-i-1:-1])))
    
    aref.subscript = reduce(
        lambda l, r: c_ast.BinaryOp('+', l, r), subscript_list)
    aref.name = name

def transform_array_decl_to_malloc(decl):
    '''Trans forms "type var_name[N]" to "type* var_name = malloc(sizeof(type)*N)'''
    if type(decl.type) is not c_ast.ArrayDecl:
        # Not an array declaration, can be ignored
        return
    
    type_ = c_ast.PtrDecl([], decl.type.type)
    init = c_ast.FuncCall(
        c_ast.ID('malloc'),
        c_ast.ExprList([
            c_ast.BinaryOp(
                '*',
                c_ast.UnaryOp(
                    'sizeof', 
                    c_ast.Typename([], c_ast.TypeDecl(None, [], decl.type.type.type))),
                decl.type.dim)]))
    
    decl.type = type_
    decl.init = init

def find_array_references(ast):
    '''returns list of array references in AST'''
    if type(ast) is c_ast.ArrayRef:
        return [ast]
    else:
        return reduce(operator.add, map(lambda o: find_array_references(o[1]), ast.children()), [])

class Kernel:
    def __init__(self, kernel_code, constants=None, variables=None, filename=None):
        '''This class captures the DSL kernel code, analyzes it and reports access pattern'''
        self.kernel_code = kernel_code
        self._filename = filename
    
        parser = CParser()
        self.kernel_ast = parser.parse(self.as_function()).ext[0].body
        
        self._loop_stack = []
        self._sources = {}
        self._destinations = {}
        self._constants = {} if constants is None else constants
        self._variables = {} if variables is None else variables
        self._flops = {}
        self.blocks = {}  # ASM block information, populated after call to assemble
        self.block_idx = None  # Block index used for marking (and containing the inner loop)
    
    def as_function(self, func_name='test'):
        return 'void {}() {{ {} }}'.format(func_name, self.kernel_code)
    
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
                    dims.append(self.conv_ast_to_int(t.dim))
                    t = t.type
                
                assert len(t.type.names) == 1, "only single types are supported"
                self.set_variable(item.name, t.type.names[0], tuple(dims))
                
            else:
                assert len(item.type.type.names) == 1, "only single types are supported"
                self.set_variable(item.name, item.type.type.names[0], None)
        
        floop = self.kernel_ast.block_items[-1]
        self._p_for(floop)
    
    def conv_ast_to_int(self, math_ast):
        '''
        converts mathematical expressions containing paranthesis, addition, subtraction and
        multiplication from AST to int.
        '''
        if type(math_ast) is c_ast.ID:
            return self._constants[math_ast.name]
        elif type(math_ast) is c_ast.Constant:
            return int(math_ast.value)
        else:  # elif type(dim) is c_ast.BinaryOp:
            op = {
                '*': operator.mul,
                '+': operator.add,
                '-': operator.sub
            }
        
            return int(op[math_ast.op](
                self.conv_ast_to_int(math_ast.left),
                self.conv_ast_to_int(math_ast.right)))
    
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
        assert type(floop.init) is c_ast.DeclList, "Initialization of loops need to be "+ \
            " declarations."
        assert len(floop.init.decls) == 1, "Only single declaration is allowed in init. of loop."
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
            assert floop.next.lvalue.name ==  floop.cond.left.name ==  floop.init.decls[0].name, \
                'initial, condition and next statement of for loop must act on same loop ' + \
                'counter variable'
            step_size = int(floop.next.rvalue.value)
        else:
            assert type(floop.next.expr) is c_ast.ID, 'next operation may only act on loop counter'
            assert floop.next.expr.name ==  floop.cond.left.name ==  floop.init.decls[0].name, \
                'initial, condition and next statement of for loop must act on same loop ' + \
                'counter variable'
            step_size = 1
        
        # Document for loop stack
        self._loop_stack.append(
            # (index name, min, max, step size)
            (floop.init.decls[0].name, floop.init.decls[0].init.value, iter_max, step_size)
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
        
        write_and_read = False
        if stmt.op != '=':
            write_and_read = True
            op = stmt.op.strip('=')
            self._flops[op] = self._flops.get(op, 0)+1
        
        # Document data destination
        if type(stmt.lvalue) is c_ast.ArrayRef:
            # self._destinations[dest name] = [dest offset, ...])
            self._destinations.setdefault(self._get_basename(stmt.lvalue), [])
            self._destinations[self._get_basename(stmt.lvalue)].append(
                 self._get_offsets(stmt.lvalue))
                 
            if write_and_read:
                # this means that +=, -= or something of that sort was used
                self._sources.setdefault(self._get_basename(stmt.lvalue), [])
                self._sources[self._get_basename(stmt.lvalue)].append(
                     self._get_offsets(stmt.lvalue))
                
        else:  # type(stmt.lvalue) is c_ast.ID
            self._destinations.setdefault(stmt.lvalue.name, [])
            self._destinations[stmt.lvalue.name].append([('dir',)])
            
            if write_and_read:
                # this means that +=, -= or something of that sort was used
                self._sources.setdefault(stmt.lvalue.name, [])
                self._sources[stmt.lvalue.name].append([('dir',)])
        
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
            
            self._flops[stmt.op] = self._flops.get(stmt.op, 0)+1
        
        return sources
        
    def as_code(self):
        ast = copy(self.kernel_ast)
        declarations = filter(lambda d: type(d) is c_ast.Decl, ast.block_items)
        
        # transform multi-dimensional declarations to one dimensional references
        array_dimensions = dict(map(trasform_multidim_to_1d_decl, declarations))
        # transform to pointer and malloc notation (stack can be too small)
        map(transform_array_decl_to_malloc, declarations)
        
        # add declarations for constants
        for k, v in self._constants.items():
            # cont int N = atoi(argv[1])
            # TODO change subscript of argv depending on constant count
            type_decl = c_ast.TypeDecl(k, ['const'], c_ast.IdentifierType(['int']))
            init = c_ast.FuncCall(
                c_ast.ID('atoi'),
                c_ast.ExprList([c_ast.ArrayRef(c_ast.ID('argv'), c_ast.Constant('int', '1'))]))
            ast.block_items.insert(0, c_ast.Decl(
                k, ['const'], [], [],
                type_decl, init, None))
        
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
                    c_ast.Constant('int', '0'))
                
                ast.block_items.insert(i+1, c_ast.For(init, cond, next_, stmt))
            
                # inject dummy access to arrays, so compiler does not over-optimize code
                # TODO put if around it, so code will actually run
                ast.block_items.insert(
                    i+2, c_ast.FuncCall(c_ast.ID('dummy'), c_ast.ExprList([c_ast.ID(d.name)])))
            else:
                # this is a scalar, so a simple Assignment is enough
                ast.block_items.insert(
                    i+1, c_ast.Assignment('=', c_ast.ID(d.name), c_ast.Constant('int', '0')))
                
                # inject dummy access to scalar, so compiler does not over-optimize code
                # TODO put if around it, so code will actually run
                ast.block_items.insert(
                    i+2,
                    c_ast.FuncCall(c_ast.ID('dummy'),
                    c_ast.ExprList([c_ast.UnaryOp('&', c_ast.ID(d.name))])))
        
        # transform multi-dimensional array references to one dimensional references
        map(lambda aref: transform_multidim_to_1d_ref(aref, array_dimensions),
            find_array_references(ast))
        
        # Mark the outer for-loop by injecting asm("nop");
        ast.block_items.insert(
            -1,
            c_ast.FuncCall(c_ast.ID('asm'), c_ast.ExprList([c_ast.Constant('string', '"nop"')])))
        ast.block_items.append(
            c_ast.FuncCall(c_ast.ID('asm'), c_ast.ExprList([c_ast.Constant('string', '"nop"')])))
        
        
        # embedd Compound into main FuncDecl
        decl = c_ast.Decl('main', [], [], [],
            c_ast.FuncDecl(
                c_ast.ParamList([
                    c_ast.Typename([], c_ast.TypeDecl('argc', [], c_ast.IdentifierType(['int']))),
                    c_ast.Typename([], 
                        c_ast.PtrDecl([],
                            c_ast.PtrDecl([],
                                c_ast.TypeDecl('argv', [], c_ast.IdentifierType(['char'])))))]),
                c_ast.TypeDecl('main', [], c_ast.IdentifierType(['int']))),
            None, None)
        
        ast = c_ast.FuncDef(decl, None, ast)
        
        # embedd Compound AST into FileAST
        ast = c_ast.FileAST([ast])
        
        # add dummy function declaration
        decl = c_ast.Decl('dummy', [], [], [],
            c_ast.FuncDecl(
                c_ast.ParamList([c_ast.Typename([], c_ast.PtrDecl([], 
                        c_ast.TypeDecl(None, [], c_ast.IdentifierType(['double']))))]),
                c_ast.TypeDecl('dummy', [], c_ast.IdentifierType(['void']))),
            None, None)
        
        ast.ext.insert(0, decl)
        
        # convert to code string
        code = CGenerator().visit(ast)
        
        # add "#include"s for dummy and stdlib (for malloc)
        code = '#include <stdlib.h>\n\n' + code
        
        return code
    
    def assemble(self, in_filename, out_filename=None, iaca_markers=True, asm_block='auto'):
        '''
        Assembles *in_filename* to *out_filename*.
        
        If *out_filename* is not given a new file will created either temporarily or according
        to kernel file location.
        
        if *iaca_marked* is set to true, markers are inserted around the block with most packed
        instructions or (if no packed instr. were found) the largest block and modified file is 
        saved to *in_file*.
        
        *asm_block* controlls how the to-be-marked block is chosen. "auto" (default) results in
        the largest block, "manual" results in interactive and a number in the according block.
        
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
            self.blocks = iaca.find_asm_blocks(lines)
            
            # TODO check for already present markers
    
            # Choose best default block:
            block_idx = iaca.select_best_block(self.blocks)
            if asm_block == 'manual':
                block_idx = iaca.userselect_block(self.blocks, default=block_idx)
            elif asm_block != 'auto':
                block_idx = asm_block
            self.block_idx = block_idx
            
            block = self.blocks[block_idx][1]
    
            # Insert markers:
            lines = iaca.insert_markers(lines, block['first_line'], block['last_line'])
    
            # write back to file
            with open(in_filename, 'w') as in_file:
                in_file.writelines(lines)
        
        try:
            # Assamble all to a binary
            subprocess.check_output(
                ["icc", os.path.basename(in_file.name), 'dummy.s', '-o', out_filename],
                cwd=os.path.dirname(in_file.name))
        finally:
            in_file.close()
        
        return out_filename
    
    def compile(self, compiler_args=None):
        '''
        Compiles source (from as_code()) to assembly.
        
        Returns two-tuple (filepointer, filename) to assembly file.
        
        Output can be used with Kernel.assemble()
        '''
        
        if not self._filename:
            in_file = tempfile.NamedTempfile(suffix='_compilable.c').file
        else:
            in_file = open(self._filename+"_compilable.c", 'w')

        in_file.write(self.as_code())
        in_file.flush()

	if compiler_args is None:
            compiler_args = []
        compiler_args += ['-O3', '-fno-alias', '-std=c99']
        
        try:
            subprocess.check_output(
                ["icc"]+compiler_args+[os.path.basename(in_file.name), '-S'],
                cwd=os.path.dirname(in_file.name))
            
            subprocess.check_output(
                ["icc"] + compiler_args + [
                    os.path.abspath(os.path.dirname(__file__)+'/headers/dummy.c'),
                    '-S'],
                cwd=os.path.dirname(in_file.name))
        finally:
            in_file.close()
        
        # Let's return the out_file name
        return os.path.splitext(in_file.name)[0]+'.s'
    
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
        
        table = (' op | count \n' +
                 '----+-------\n')
        for op, count in self._flops.items():
            table += '{:>3} | {:>4}\n'.format(op, count)
        table += '     =======\n'
        table += '      {:>4}'.format(sum(self._flops.values()))
        print(prefix_indent('FLOPs:     ', table))

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
