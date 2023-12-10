import numpy as np
from scipy.optimize import line_search

from src.math_utils import function, gradient


def gdm_constant(expression, x0, step_size, tolerance=1e-10, maximum_iteration=10000):
    """
    Performs gradient descent optimization with a constant step size.

    Parameters:
    - expression (function): The objective function to be minimized.
    - x0 (array-like): Initial point.
    - step_size (float): The step size used in each iteration.
    - tolerance (float, optional): The convergence threshold. Default is 1e-10.
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

    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        x = x - step_size * grad(x)
        k += 1

    return x, f(x), np.linalg.norm(grad(x)), k


def gdm_exact_line_search(expression, x0, tolerance=1e-5, maximum_iteration=10000):
    """
    Performs gradient descent optimization with exact line search.

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

    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        step_size = line_search(f, grad, x, -grad(x))[0]
        x = x - step_size * grad(x)
        k += 1

    return x, f(x), np.linalg.norm(grad(x)), k


def backtracking_line_search(f, grad, x, d, s, alpha, beta):
    """
    Performs backtracking line search to find an appropriate step size.

    Parameters:
    - f (function): The objective function.
    - grad (function): The gradient function.
    - x (array-like): The current point.
    - d (array-like): The search direction.
    - s (float): The initial step size.
    - alpha (float): The constant used to control the sufficient decrease condition.
    - beta (float): The constant used to control the step size reduction.

    Returns:
    - t (float): The chosen step size.
    """
    t = s
    while f(x) - f(x + t * d) < -alpha * t * np.dot(grad(x).T, d):
        t *= beta
    return t


def gdm_backtracking(
    expression, x0, s, alpha, beta, tolerance=1e-5, maximum_iteration=10000
):
    """
    Performs gradient descent optimization with backtracking line search.

    Parameters:
    - expression (function): The objective function to be minimized.
    - x0 (array-like): Initial point.
    - s (float): The initial step size for backtracking line search.
    - alpha (float): The constant used to control the sufficient decrease condition in backtracking line search.
    - beta (float): The constant used to control the step size reduction in backtracking line search.
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

    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        step_size = backtracking_line_search(f, grad, x, -grad(x), s, alpha, beta)
        x = x - step_size * grad(x)
        k += 1

    return x, f(x), np.linalg.norm(grad(x)), k
