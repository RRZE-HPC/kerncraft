#!/usr/bin/python

from textwrap import dedent
from pycparser import CParser, c_ast

class ECM:
    """
    class representation of the Execution-Cache-Memory Model

    more info to follow...
    """

    def __init__(self, kernel, core, machine):
        """
        *kernel* is a string of c-code describing the kernel loops and updates
        *core* is the  in-core throughput as tuple of (overlapping cycles, non-overlapping cycles)
        *machine* describes the machine (cpu, cache and memory) characteristics
        """
        self.kernel = kernel
        self.core = core
        self.machine = machine
    
    def process(self):
        assert type(self.kernel) is c_ast.Compound, "Kernel has to be a compound statement"
        floop = self.kernel.block_items[0]
        self._p_for(floop)

    def _p_for(self, floop):
        # Check for restrictions
        assert type(floop) is c_ast.For, "May only be a for loop"
        assert hasattr(floop, 'init') and hasattr(floop, 'cond') and hasattr(floop, 'next'), \
            "Loop must have initial, condition and next statements."
        assert type(floop.init) is c_ast.Assignment, "Initialization of loops need to be " + \
            "assignments (declarations are not allowed or needed)"
        assert floop.cond.op in '<>', "only lt (<) or (>) gt is allowed as loop condition"
        assert type(floop.cond.left) is c_ast.ID, 'left of cond. operand has to be a variable'
        assert type(floop.cond.right) is c_ast.Constant, \
            'right of cond. operand has to be a constant'
        assert type(floop.next) is c_ast.UnaryOp, 'next statement has to be a unary operation'
        assert floop.next.op in ['++', 'p++', '--', 'p--'], \
            'only ++ or -- next operations are allowed'
        assert type(floop.next.expr) is c_ast.ID, 'next operation may only act on loop counter'
        assert floop.next.expr.name ==  floop.cond.left.name ==  floop.init.lvalue.name, \
            'initial, condition and next statement of for loop must act on same loop counter' + \
            'variable'

        # Document for loop stack
        # TODO

        # Traverse tree
        if type(floop.stmt) is not c_ast.For:
            self._p_assignment(floop.stmt)
        else:
            self._p_for(floop.stmt)

    def _p_assignment(self, stmt):
        # Check for restrictions
        assert type(stmt) is c_ast.Assignment, \
            "Only single assignment statements are allowed in loops."
        
        # Document data destination
        # TODO
        
        # Traverse tree
        self._p_operations(stmt.rvalue)

    def _p_operations(self, stmt):
        assert type(stmt) in [c_ast.ArrayRef, c_ast.Constant, c_ast.ID, c_ast.BinaryOp], \
            'only references to arrays, constants and variables as well as binary operations ' + \
            'are supported'

        if type(stmt) is c_ast.ArrayRef:
            # Check for restrictions
            assert type(stmt.name) is c_ast.ID, "array references must only be used with variables"
            assert type(stmt.subscript) in [c_ast.ID, c_ast.Constant, c_ast.BinaryOp], \
                'array subscript must only contain constants, variables or binary operations'

            if type(stmt.subscript) is c_ast.BinaryOp:
                assert stmt.subscript.op in '+-', \
                    'binary operations in array subscript must by + or -'
                assert type(stmt.subscript.left) in [c_ast.ID, c_ast.Constant] and \
                    type(stmt.subscript.right) in [c_ast.ID, c_ast.Constant], \
                    'binary operation in array subscript may only act on variables or constants'
            
            # Document data source
            # TODO
        
        elif type(stmt) is c_ast.ID:
            # Document data source
            # TODO
        
        elif type(stmt) is c_ast.BinaryOp:
            # Traverse tree
            self._p_operations(stmt.left)
            self._p_operations(stmt.right)

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
        for(i=1; i<N-1; ++i)
            for(j=1; j<N-1; ++j)
                b[i*N+j] = c * (a[(i-1)*N+j] + a[i*N+j-1] + a[i*N+j] + a[i*N+j+1] + a[(i+1)*N+j]);
        """,
    '2d-5pt-doublearray':
        """\
        for(i=1; i<N-1; ++i)
            for(j=1; j<N-1; ++j)
                b[i][j] = c * (a[i-1][j] + a[i][j-1] + a[i][j] + a[i][j+1] + a[i+1][j]);
        """,
    # TODO uxx stencil (from ipdps15-ECM paper)
    # TODO 3D long-range stencil (from ipdps15-ECM paper)
    }

if __name__ == '__main__':
    # Read machine description
    # TODO

    # Read IACA output
    # TODO

    # Parse given kernel/snippet
    kernels['test'] = """\
        for(i=0; i<50; ++i)
            a[i] = c * (a[i-1] - 2.0*a[i] + a[i+1]);"""
    parser = CParser()
    kernel = parser.parse('void test() {'+kernels['test']+'}').ext[0].body
    kernel.show()
    

    e = ECM(kernel, None, None)

    # Verify code and document data access and loop traversal
    e.process()
    
    # Analyze access patterns
    # TODO <-- this is my thesis

    # Report
    # TODO

    pass