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
    return s+'\n'.join(map(lambda x: later_prefix+x, textblock[1:]))

class ECM:
    """
    class representation of the Execution-Cache-Memory Model

    more info to follow...
    """

    def __init__(self, kernel, core, machine, constants=None):
        """
        *kernel* is a string of c-code describing the kernel loops and updates
        *core* is the  in-core throughput as tuple of (overlapping cycles, non-overlapping cycles)
        *machine* describes the machine (cpu, cache and memory) characteristics
        """
        self.kernel = kernel
        self.core = core
        self.machine = machine
        
        self._arrays = {}
        self._loop_stack = []
        self._sources = {}
        self._destinations = {}
        self._constants = {} if constants is None else constants
    
    def set_constant(self, name, value):
        assert type(name) is str, "constant name needs to be of type str"
        assert type(value) is int, "constant value needs to be of type int"
        self._constants[name] = value
    
    def process(self):
        assert type(self.kernel) is c_ast.Compound, "Kernel has to be a compound statement"
        floop = self.kernel.block_items[0]
        self._p_for(floop)
    
    @classmethod
    def _get_offsets(cls, aref):
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
            
            sign = 1 if aref.subscript.op == '+' else -1
            if type(aref.subscript.right) is c_ast.Constant:
                offset = sign*int(aref.subscript.right.value)
            else:
                offset = sign*int(aref.subscript.left.value)
            
            idxs.append(('rel', offset))
        elif type(aref.subscript) is c_ast.ID:
            idxs.append(('rel', 0))
        else:  # type(aref.subscript) is c_ast.Constant
            idxs.append(('abs', int(aref.subscript.value)))
        
        if type(aref.name) is c_ast.ArrayRef:
            idxs += cls._get_offsets(aref.name)
        
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
        assert type(floop.cond.right) in [c_ast.Constant, c_ast.ID], \
            'right of cond. operand has to be a constant or a variable'
        assert type(floop.next) is c_ast.UnaryOp, 'next statement has to be a unary operation'
        assert floop.next.op in ['++', 'p++'], 'only incremental next operations are allowed'
        assert type(floop.next.expr) is c_ast.ID, 'next operation may only act on loop counter'
        assert floop.next.expr.name ==  floop.cond.left.name ==  floop.init.lvalue.name, \
            'initial, condition and next statement of for loop must act on same loop counter' + \
            'variable'

        if type(floop.cond.right) is c_ast.ID:
            const_name = floop.cond.right.name
            assert const_name in self._constants, 'loop right operand has to be defined as a ' +\
                 'constant in ECM object'
            iter_max = self._constants[const_name]
        else:  # type(floop.init.right) is c_ast.Constant
            iter_max = int(floop.cond.right.value)
        # TODO add support for c_ast.BinaryOp with +1 and -1 (like array indices)

        # Document for loop stack
        self._loop_stack.append(
            # (index name, min, max, step size)
            (floop.init.lvalue.name, floop.init.rvalue.value, iter_max, int('1'))
        )
        # TODO add support for other stepsizes (even negative/reverse steps?)

        # Traverse tree
        if type(floop.stmt) is c_ast.For:
            self._p_for(floop.stmt)
        else:
            # TODO support c_ast.Compound
            self._p_assignment(floop.stmt)

    def _p_assignment(self, stmt):
        # Check for restrictions
        assert type(stmt) is c_ast.Assignment, \
            "Only single assignment statements are allowed in loops."
        assert type(stmt.lvalue) in [c_ast.ArrayRef, c_ast.ID], \
            "Only assignment to array element or varialbe is allowed."
        
        # Document data destination
        if type(stmt.lvalue) is c_ast.ArrayRef:
            # self._destinations[dest name] = [dest offset, ...])
            self._destinations.setdefault(self._get_basename(stmt.lvalue), [])
            self._destinations[self._get_basename(stmt.lvalue)].append(
                 self._get_offsets(stmt.lvalue))
        else:  # type(stmt.lvalue) is c_ast.ID
            self._destinations.setdefault(stmt.lvalue.name.name, [])
            self._destinations[stmt.lvalue.name.name].append(('dir',))
        
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
    
    def print_info(self):
        table = ('     idx |        min        max       step\n' +
                 '---------+---------------------------------\n')
        for l in self._loop_stack:
            table += '{:>8} | {:>10} {:>10} {:>+10}\n'.format(*l)
        print(prefix_indent('loop stack:        ', table))
        
        table = ('    name |  offsets   ...\n' +
                 '---------+------------...\n')
        for name, offsets in e._sources.items():
            prefix = '{:>8} | '.format(name)
            right_side = ''
            for off in offsets:
                for o in off:
                    right_side += '{:>3}'.format(o[0])
                    right_side += ' {:>+3}'.format(o[1]) if len(o) == 2 else '    '
                    right_side += ' | '
                right_side = right_side[:-3] + '\n'
            table += prefix_indent(prefix, right_side, later_prefix='         | ') + '\n'
        print(prefix_indent('data sources:      ', table))
        
        table = ('    name |  offsets   ...\n' +
                 '---------+------------...\n')
        for name, offsets in e._destinations.items():
            prefix = '{:>8} | '.format(name)
            right_side = ''
            for off in offsets:
                for o in off:
                    right_side += '{:>3}'.format(o[0])
                    right_side += ' {:>+3}'.format(o[1]) if len(o) == 2 else '    '
                    right_side += ' | '
                right_side = right_side[:-3] + '\n'
            table += prefix_indent(prefix, right_side, later_prefix='         | ') + '\n'
        print(prefix_indent('data destinations: ', table))

# Example kernels:
kernels = {
    'scale':
        """\
        for(i=0; i<N; ++i)
            a[i] = s * b[i];""",
    'copy':
        """\
        for(i=0; i<N; ++i)
            a[i] = b[i];""",
    'add':
        """\
        for(i=0; i<N; ++i)
            a[i] = b[i] + c[i];""",
    'triad':
        """\
        for(i=0; i<N; ++i)
            a[i] = b[i] + s * c[i];""",
    '1d-3pt':
        """\
        for(i=1; i<N; ++i)
            b[i] = c * (a[i-1] - 2.0*a[i] + a[i+1]);""",
    '2d-5pt':
        """\
        for(i=0; i<N; ++i)
            for(j=0; j<N; ++j)
                b[i][j] = c * (a[i-1][j] + a[i][j-1] + a[i][j] + a[i][j+1] + a[i+1][j]);
        """,
    '2d-5pt-doublearray':
        """\
        for(i=0; i<N; ++i)
            for(j=0; j<N; ++j)
                b[i][j] = c * (a[i-1][j] + a[i][j-1] + a[i][j] + a[i][j+1] + a[i+1][j]);
        """,
    # TODO uxx stencil (from ipdps15-ECM paper)
    # TODO 3D long-range stencil (from ipdps15-ECM paper)
    }

if __name__ == '__main__':
    for k in kernels.keys():
        print('='*80 + '\n{:^80}\n'.format(k) + '='*80)
        # Read machine description
        # TODO
        
        # Read and interpret IACA output
        # TODO
        
        # Parse given kernel/snippet
        parser = CParser()
        kernel = parser.parse('void test() {'+kernels[k]+'}').ext[0].body
        print(dedent(kernels[k]))
        print()
        #kernel.show()
        
        e = ECM(kernel, None, None)
        e.set_constant('N', 50)
        
        # Verify code and document data access and loop traversal
        e.process()
        e.print_info()
        
        # Analyze access patterns
        # TODO <-- this is my thesis
        
        # Report
        # TODO
        
        print
    