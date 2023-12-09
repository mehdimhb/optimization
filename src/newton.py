import numpy as np
from src.math_utils import function, gradient, hessian_inverse


def newton_method(expression, x0, tolerance=1e-5, maximum_iteration=10000):
    k = 0
    x = np.array(x0)
    f = function(expression)
    grad = gradient(expression)
    hess_inv = hessian_inverse(expression)
    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        dk = - np.dot(hess_inv(x), grad(x))
        x = x + dk
        k += 1
    return x, f(x), np.linalg.norm(grad(x)), k
