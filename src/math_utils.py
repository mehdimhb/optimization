import numpy as np
import sympy as sp


def variables(expression):
    return list(sp.ordered(expression.free_symbols))


def allow_to_take_input(expression, variables):
    return sp.lambdify(variables, expression, "numpy")


def change_input_to_vector(function):
    return lambda x: np.array(function(*x)).squeeze()


def make_function(expression, variables):
    return change_input_to_vector(allow_to_take_input(expression, variables))


def function(expression):
    return make_function(expression, variables(expression))


def large_input_function(f, X, Y):
    Z = []
    for xy in zip(X, Y):
        Z.append(f(xy))
    return Z


def gradient(expression):
    gradient_expression = sp.Matrix([expression]).jacobian(variables(expression))
    return make_function(gradient_expression, variables(expression))


def hessian_inverse(expression):
    hessian_inverse_expression = sp.hessian(expression, variables(expression))**-1
    return make_function(hessian_inverse_expression, variables(expression))


def hessian_inverse_diagonal_expression(expression):
    hessian_inverse_expression = sp.hessian(expression, variables(expression))**-1
    return sp.Matrix(
        *hessian_inverse_expression.shape,
        lambda i, j: hessian_inverse_expression[i, j] if i == j else 0
    )


def hessian_inverse_diagonal(expression):
    return make_function(hessian_inverse_diagonal_expression(expression), variables(expression))


def convert_array_to_sympy_function(array, expression):
    matrix = sp.sympify(array)
    return make_function(matrix, variables(expression))


def is_positive_definite(matrix):
    return sp.Matrix(matrix).is_positive_definite
