import numpy as np

from src.math_utils import (function, gradient, hessian_inverse, is_positive_definite)


def gradient_decent_method(expression, x0, step_method, scale, tolerance, maximum_iteration):
    """
    Performs gradient descent optimization with a given step size method.

    Parameters:
    - expression (function): The objective function to be minimized.
    - x0 (array-like): Initial point.
    - step_method (Step): The method for step size used in each iteration.
    - scaling (boolean): enable Scaling Matrix.
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
    points = [x]

    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        d = -np.dot(scale(x), grad(x))
        step_size = step_method(f, grad, x, d)
        x = x + step_size * d
        points.append(x)
        k += 1

    return x, f(x), np.linalg.norm(grad(x)), k, points


def newton_method(expression, x0, step_method, tolerance, maximum_iteration):
    """
    Performs Newton's method for optimization.

    Parameters:
    - expression (function): The objective function to be minimized.
    - x0 (array-like): Initial point.
    - step_method (Step): The method for step size used in each iteration if damped newton's method is considered.
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
    points = [x]

    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        d = -np.dot(hess_inv(x), grad(x))
        step_size = step_method(f, grad, x, d) if step_method else 1
        x = x + step_size*d
        points.append(x)
        k += 1

    return x, f(x), np.linalg.norm(grad(x)), k, points


def hybrid_gradient_newton_method(expression, x0, step_method, tolerance, maximum_iteration):
    """
    Performs gradient descent optimization with a given step size method.

    Parameters:
    - expression (function): The objective function to be minimized.
    - x0 (array-like): Initial point.
    - step_method (Step): The method for step size used in each iteration.
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
    hess_inv = hessian_inverse(expression)
    points = [x]

    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        if is_positive_definite(hess_inv(x)):
            d = -np.dot(hess_inv(x), grad(x))
        else:
            d = -grad(x)
        step_size = step_method(f, grad, x, d)
        x = x + step_size * d
        points.append(x)
        k += 1

    return x, f(x), np.linalg.norm(grad(x)), k, points
