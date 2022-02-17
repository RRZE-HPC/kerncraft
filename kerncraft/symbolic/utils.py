import sympy

class int_floor(sympy.Function):
    @classmethod
    def eval(cls, x, y):
        if x.is_Number and y.is_Number:
            return x // y

    def _eval_is_integer(self):
        return True

class int_ceil(sympy.Function):
    @classmethod
    def eval(cls, x, y):
        if x.is_Number and y.is_Number:
            return sympy.ceiling(x / y)

    def _eval_is_integer(self):
        return True
