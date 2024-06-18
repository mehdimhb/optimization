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
    # Initialize variables
    k = 0  # iteration counter
    x = np.array(x0)  # current point
    f = function(expression)  # the objective function
    grad = gradient(expression)  # gradient of the objective function
    points = [x]  # record of the points visited in the optimization process

    # Iterate until convergence or maximum iterations reached
    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        # Compute the next step direction using gradient descent
        # Multiply by scaling matrix if enabled
        d = -np.dot(scale(x), grad(x))
        # Compute the step size using the provided step method
        step_size = step_method(f, grad, x, d)
        # Update the current point by moving in the direction of the step
        x = x + step_size * d
        # Record the current point in the sequence of iterations
        points.append(x)
        # Increment the iteration counter
        k += 1

    # Return the optimized point, its value, the norm of the gradient, the number of iterations and the sequence of points visited
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
    # Initialize variables
    k = 0  # iteration counter
    x = np.array(x0)  # current point
    f = function(expression)  # the objective function
    grad = gradient(expression)  # gradient of the objective function
    hess_inv = hessian_inverse(expression)  # inverse of the Hessian matrix of the objective function
    points = [x]  # record of the points visited in the optimization process

    # Iterate until convergence or maximum iterations reached
    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        # Compute the next step direction using the Newton's method
        d = -np.dot(hess_inv(x), grad(x))
        # Adjust the step size based on the step method if given
        step_size = step_method(f, grad, x, d) if step_method else 1
        # Update the current point by moving in the direction of the step
        x = x + step_size*d
        # Record the current point in the sequence of iterations
        points.append(x)
        # Increment the iteration counter
        k += 1

    # Return the optimized point, its value, the norm of the gradient, the number of iterations and the sequence of points visited
    return x, f(x), np.linalg.norm(grad(x)), k, points


def hybrid_gradient_newton_method(expression, x0, step_method, tolerance, maximum_iteration):
    """
    Performs Hybrid Gradient-Newton Method optimization with a given step size method.

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
    # Initialize variables
    k = 0  # iteration counter
    x = np.array(x0)  # current point
    f = function(expression)  # the objective function
    grad = gradient(expression)  # gradient of the objective function
    hess_inv = hessian_inverse(expression)  # inverse of the Hessian matrix of the objective function
    points = [x]  # record of the points visited in the optimization process

    # Iterate until convergence or maximum iterations reached
    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        # Compute the next step direction using the Hybrid Gradient-Newton method
        if is_positive_definite(hess_inv(x)):
            # Use Newton's method if the Hessian matrix is positive definite
            d = -np.dot(hess_inv(x), grad(x))
        else:
            # Use gradient descent if the Hessian matrix is not positive definite
            d = -grad(x)
        # Adjust the step size based on the step method
        step_size = step_method(f, grad, x, d)
        # Update the current point by moving in the direction of the step
        x = x + step_size * d
        # Record the current point in the sequence of iterations
        points.append(x)
        # Increment the iteration counter
        k += 1

    # Return the optimized point, its value, the norm of the gradient, the number of iterations and the sequence of points visited
    return x, f(x), np.linalg.norm(grad(x)), k, points
