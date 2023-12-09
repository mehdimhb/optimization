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


def gradient(expression):
    gradient_expression = sp.Matrix([expression]).jacobian(variables(expression))
    return make_function(gradient_expression, variables(expression))


def hessian_inverse(expression):
    hessian_inverse_expression = sp.hessian(expression, variables(expression))**-1
    return make_function(hessian_inverse_expression, variables(expression))
