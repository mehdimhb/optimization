import numpy as np

from src.math_utils import function, gradient


def gradient_decent_method(expression, x0, step_method, tolerance, maximum_iteration):
    """
    Performs gradient descent optimization with a given step size method.

    Parameters:
    - expression (function): The objective function to be minimized.
    - x0 (array-like): Initial point.
    - step_method (StepSize): The step size used in each iteration.
    - tolerance (float): The convergence threshold.
    - maximum_iteration (int): The maximum number of iterations.

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

    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        step_size = step_method(f, grad, x)
        x = x - step_size * grad(x)
        k += 1

    return x, f(x), np.linalg.norm(grad(x)), k
