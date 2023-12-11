import numpy as np
from scipy.optimize import line_search


class Step:
    def __init__(self, value, method):
        self.value = value
        self.method = method

    def __call__(self, *args):
        return self.value


class Constant(Step):
    def __init__(self, value):
        super().__init__(value, "Constant")


class LineSearch(Step):
    def __init__(self):
        super().__init__(None, "Exact Line Search")

    def __call__(self, f, grad, x):
        self.value = line_search(f, grad, x, -grad(x))[0]
        return super().__call__()


class Backtracking(Step):
    def __init__(self, s, alpha, beta):
        super().__init__(None, "Backtracing Line Search")
        self.s = s
        self.alpha = alpha
        self.beta = beta

    def backtracing_line_search(self, f, grad, x, d):
        t = self.s
        while f(x) - f(x + t * d) < -self.alpha * t * np.dot(grad(x).T, d):
            t *= self.beta
        return t

    def __call__(self, f, grad, x):
        self.value = self.backtracing_line_search(f, grad, x, -grad(x))
        return super().__call__()
