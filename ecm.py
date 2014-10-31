#!/usr/bin/python

from __future__ import print_function
from textwrap import dedent
from pycparser import CParser, c_ast
from pprint import pprint

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

class ECM:
    """
    class representation of the Execution-Cache-Memory Model

    more info to follow...
    """

    def __init__(self, kernel_code, core, machine, constants=None, variables=None):
        """
        *kernel* is a string of c-code describing the kernel loops and updates
        *core* is the  in-core throughput as tuple of (overlapping cycles, non-overlapping cycles)
        *machine* describes the machine (cpu, cache and memory) characteristics
        """
        self.kernel_code = kernel_code
        self.core = core
        self.machine = machine
        
        parser = CParser()
        self.kernel_ast = parser.parse('void test() {'+kernel_code+'}').ext[0].body
        
        self._arrays = {}
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
                    if type(t.dim) is c_ast.ID:
                        dims.append(self._constants[t.dim.name])
                    else:  # type(t.dim) is c_ast.Constant
                        dims.append(int(t.dim.value))
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
        
        if type(aref.subscript) is c_ast.BinaryOp:
            assert aref.subscript.op in '+-', \
                'binary operations in array subscript must by + or -'
            assert (type(aref.subscript.left) is c_ast.ID and \
                    type(aref.subscript.right) is c_ast.Constant), \
                'binary operation in array subscript may only have form "variable +- constant"'
            assert aref.subscript.left.name == self._loop_stack[-dim-1][0], \
                'order of varialbes used in array indices has to follow the order of for loop ' + \
                'counters'
            
            sign = 1 if aref.subscript.op == '+' else -1
            offset = sign*int(aref.subscript.right.value)
            
            idxs.append(('rel', offset))
        elif type(aref.subscript) is c_ast.ID:
            assert aref.subscript.name == self._loop_stack[-dim-1][0], \
                'order of varialbes used in array indices has to follow the order of for loop ' + \
                'counters'
            idxs.append(('rel', 0))
        else:  # type(aref.subscript) is c_ast.Constant
            idxs.append(('abs', int(aref.subscript.value)))
        
        if type(aref.name) is c_ast.ArrayRef:
            idxs += self._get_offsets(aref.name, dim=dim+1)
        
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
        for name, offsets in e._sources.items():
            prefix = '{:>8} | '.format(name)
            right_side = '\n'.join(map(lambda o: ', '.join(map(tuple.__repr__, o)), offsets))
            table += prefix_indent(prefix, right_side, later_prefix='         | ')
        print(prefix_indent('data sources:      ', table))
        
        table = ('    name |  offsets   ...\n' +
                 '---------+------------...\n')
        for name, offsets in e._destinations.items():
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

# Example kernels:
kernels = {
    'scale':
        {
            'kernel_code':
                """\
                double a[N], b[N];
                double s;
                
                for(i=0; i<N; ++i)
                    a[i] = s * b[i];
                """,
            'constants': [('N', 50)],
            
        },
    'copy':
        {
            'kernel_code':
                """\
                double a[N], b[N];
                
                for(i=0; i<N; ++i)
                    a[i] = b[i];
                """,
            'constants': [('N', 50)],
        },
    'add':
        {
            'kernel_code':
                """\
                double a[N], b[N], c[N];
                
                for(i=0; i<N; ++i)
                    a[i] = b[i] + c[i];
                """,
            'constants': [('N', 50)],
        },
    'triad':
        {
            'kernel_code':
                """\
                double a[N], b[N], c[N];
                double s;
                
                for(i=0; i<N; ++i)
                    a[i] = b[i] + s * c[i];
                """,
            'constants': [('N', 50)],
        },
    '1d-3pt':
        {
            'kernel_code':
                """\
                double a[N], b[N];
                
                for(i=1; i<N-1; ++i)
                    b[i] = c * (a[i-1] - 2.0*a[i] + a[i+1]);
                """,
            'constants': [('N', 50)],
        },
    '2d-5pt':
        {
            'kernel_code':
                """\
                double a[N][N];
                double b[N][N];
                double c;
                
                for(i=0; i<N; ++i)
                    for(j=0; j<N; ++j)
                        b[i][j] = c * (a[i-1][j] + a[i][j-1] + a[i][j] + a[i][j+1] + a[i+1][j]);
                """,
            'constants': [('N', 50)],
        },
    'uxx-stencil':
        {
            'kernel_code':
                """\
                double u1[N][N][N];
                double d1[N][N][N];
                double xx[N][N][N];
                double xy[N][N][N];
                double xz[N][N][N];
                double c1, c2, d;
                
                for(k=2; k<N-2; k++) {
                    for(j=2; j<N-2; j++) {
                        for(i=2; i<N-2; i++) {
                            d = 0.25*(d1[ k ][j][i] + d1[ k ][j-1][i]
                                    + d1[k-1][j][i] + d1[k-1][j-1][i]);
                            u1[k][j][i] = u1[k][j][i] + (dth/d)
                             * ( c1*(xx[ k ][ j ][ i ] - xx[ k ][ j ][i-1])
                               + c2*(xx[ k ][ j ][i+1] - xx[ k ][ j ][i-2])
                               + c1*(xy[ k ][ j ][ i ] - xy[ k ][j-1][ i ])
                               + c2*(xy[ k ][j+1][ i ] - xy[ k ][j-2][ i ])
                               + c1*(xz[ k ][ j ][ i ] - xz[k-1][ j ][ i ])
                               + c2*(xz[k+1][ j ][ i ] - xz[k-2][ j ][ i ]));
                }}}
                """,
            'constants': [('N', 50)],
        },
    '3d-long-range-stencil':
        {
            'kernel_code':
                """\
                double U[N][N][N];
                double V[N][N][N];
                double ROC[N][N][N];
                double c1, c2, c3, c4, lap;
                
                for(k=4; k < N-4; k++) {
                    for(j=4; j < N-4; j++) {
                        for(i=4; i < N-4; i++) {
                            lap = c0 * V[k][j][i]
                                + c1 * ( V[ k ][ j ][i+1] + V[ k ][ j ][i-1])
                                + c1 * ( V[ k ][j+1][ i ] + V[ k ][j-1][ i ])
                                + c1 * ( V[k+1][ j ][ i ] + V[k-1][ j ][ i ])
                                + c2 * ( V[ k ][ j ][i+2] + V[ k ][ j ][i-2])
                                + c2 * ( V[ k ][j+2][ i ] + V[ k ][j-2][ i ])
                                + c2 * ( V[k+2][ j ][ i ] + V[k-2][ j ][ i ])
                                + c3 * ( V[ k ][ j ][i+3] + V[ k ][ j ][i-3])
                                + c3 * ( V[ k ][j+3][ i ] + V[ k ][j-3][ i ])
                                + c3 * ( V[k+3][ j ][ i ] + V[k-3][ j ][ i ])
                                + c4 * ( V[ k ][ j ][i+4] + V[ k ][ j ][i-4])
                                + c4 * ( V[ k ][j+4][ i ] + V[ k ][j-4][ i ])
                                + c4 * ( V[k+4][ j ][ i ] + V[k-4][ j ][ i ]);
                            U[k][j][i] = 2.f * V[k][j][i] - U[k][j][i] 
                                       + ROC[k][j][i] * lap;
                }}}
                """,
            'constants': [('N', 50)],
        },
    }

if __name__ == '__main__':
    for name, info in kernels.items():
        print('='*80 + '\n{:^80}\n'.format(name) + '='*80)
        # Read machine description
        # TODO
        
        # Read and interpret IACA output
        # TODO
        
        # Create ECM object and give additional information about runtime
        e = ECM(dedent(info['kernel_code']), None, None)
        for const_name, const_value in info['constants']:
            e.set_constant(const_name, const_value)
        
        # Verify code and document data access and loop traversal
        e.process()
        e.print_kernel_code()
        print()
        e.print_variables_info()
        e.print_constants_info()
        e.print_kernel_info()
        
        # Analyze access patterns
        # TODO <-- this is my thesis
        
        # Report
        # TODO
        
        print
    