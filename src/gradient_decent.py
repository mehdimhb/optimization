import numpy as np
from scipy.optimize import line_search
from src.math_utils import function, gradient


def gdm_constant(expression, x0, step_size, tolerance=1e-10, maximum_iteration=10000):
    k = 0
    x = np.array(x0)
    f = function(expression)
    grad = gradient(expression)
    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        x = x - step_size * grad(x)
        k += 1
    return x, f(x), np.linalg.norm(grad(x)), k


def gdm_exact_line_search(expression, x0, tolerance=1e-5, maximum_iteration=10000):
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
    t = s
    while f(x) - f(x+t*d) < -alpha * t * np.dot(grad(x).T, d):
        t *= beta
    return t


def gdm_backtracking(expression, x0, s, alpha, beta, tolerance=1e-5, maximum_iteration=10000):
    k = 0
    x = np.array(x0)
    f = function(expression)
    grad = gradient(expression)
    while np.linalg.norm(grad(x)) > tolerance and k < maximum_iteration:
        step_size = backtracking_line_search(f, grad, x, -grad(x), s, alpha, beta)
        x = x - step_size * grad(x)
        k += 1
    return x, f(x), np.linalg.norm(grad(x)), k
