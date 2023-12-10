import numpy as np

from src.math_utils import function, gradient, hessian_inverse


def newton_method(expression, x0, tolerance=1e-5, maximum_iteration=10000):
    """
    Performs Newton's method for optimization.

    Parameters:
    - expression (function): The objective function to be minimized.
    - x0 (array-like): Initial point.
    - tolerance (float, optional): The convergence threshold. Default is 1e-5.
    - maximum_iteration (int, optional): The maximum number of iterations. Default is 10000.

    Returns:
    - x (array-like): The optimized point.
    - f(x) (float): The value of the objective function at the optimized point.
    - ||grad(x)|| (float): The norm of the gradient at the optimized point.
    - k (int): The number of iterations performed.
    """
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
