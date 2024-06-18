import numpy as np
import sympy as sp


def variables(expression):
    """
    Get the variables in the given expression.

    Args:
        expression (sympy.Expr): The expression to extract variables from.

    Returns:
        List[sympy.Symbol]: A list of the variables in the expression, sorted in alphabetical order.
    """
    return list(sp.ordered(expression.free_symbols))


def allow_to_take_input(expression, variables):
    """
    Creates a lambda function that allows input for the given expression using the provided variables.

    Args:
        expression (sympy.Expr): The expression to create a lambda function for.
        variables (List[sympy.Symbol]): The variables to use as input for the lambda function.

    Returns:
        function: A lambda function that takes input as a numpy array and
        evaluates the expression using the provided variables.
    """
    return sp.lambdify(variables, expression, "numpy")


def change_input_to_vector(function):
    """
    Creates a lambda function that takes input as a vector and applies it to the given function.

    Args:
        function (Callable): The function to be applied to the input.

    Returns:
        Callable: A lambda function that takes input as a vector and applies it to the given function.
        The result is converted to a numpy array and squeezed.
    """
    return lambda x: np.array(function(*x)).squeeze()


def make_function(expression, variables):
    """
    Creates a lambda function that takes input as a vector and applies it to the given expression
    using the provided variables.

    Args:
        expression (sympy.Expr): The expression to create a lambda function for.
        variables (List[sympy.Symbol]): The variables to use as input for the lambda function.

    Returns:
        Callable: A lambda function that takes input as a vector and applies it to
                  the given expression using the provided variables.
                  The result is converted to a numpy array and squeezed.
    """
    return change_input_to_vector(allow_to_take_input(expression, variables))


def function(expression):
    """
    Creates a lambda function that takes input as a vector and applies it to the given expression
    using the provided variables.

    Args:
        expression (sympy.Expr): The expression to create a lambda function for.

    Returns:
        Callable: A lambda function that takes input as a vector and applies it to the given expression
                   using the provided variables. The result is converted to a numpy array and squeezed.
    """
    return make_function(expression, variables(expression))


def large_input_function(f, X, Y):
    """
    Generates a list of values by applying a function to pairs of elements from two lists.

    Args:
        f (Callable): The function to apply to each pair of elements from X and Y.
        X (Iterable): The first list of elements.
        Y (Iterable): The second list of elements.

    Returns:
        List: A list of values obtained by applying the function f to each pair of elements from X and Y.
    """
    Z = []
    for xy in zip(X, Y):
        Z.append(f(xy))
    return Z


def gradient(expression):
    """
    Calculates the gradient of the given expression.

    Args:
        expression: The expression for which the gradient is calculated.

    Returns:
        The gradient expression.
    """
    gradient_expression = sp.Matrix([expression]).jacobian(variables(expression))
    return make_function(gradient_expression, variables(expression))


def hessian_inverse(expression):
    """
    Calculates the inverse of the Hessian matrix of the given expression.

    Args:
        expression: The input expression for which the Hessian matrix is calculated.

    Returns:
        The inverse of the Hessian matrix as a function.
    """
    hessian_inverse_expression = sp.hessian(expression, variables(expression))**-1
    return make_function(hessian_inverse_expression, variables(expression))


def hessian_inverse_diagonal_expression(expression):
    """
    Calculates the diagonal elements of the inverse of the Hessian matrix of the given expression.

    Args:
        expression (sympy.Expr): The input expression for which the Hessian matrix is calculated.

    Returns:
        sympy.Matrix: The diagonal elements of the inverse of the Hessian matrix as a sympy.Matrix object.

    """
    hessian_inverse_expression = sp.hessian(expression, variables(expression))**-1
    return sp.Matrix(
        *hessian_inverse_expression.shape,
        lambda i, j: hessian_inverse_expression[i, j] if i == j else 0
    )


def hessian_inverse_diagonal(expression):
    """
    Calculates the diagonal elements of the inverse of the Hessian matrix of the given expression.

    Args:
        expression (sympy.Expr): The input expression for which the Hessian matrix is calculated.

    Returns:
        sympy.Function: The diagonal elements of the inverse of the Hessian matrix as a function.
    """
    return make_function(hessian_inverse_diagonal_expression(expression), variables(expression))


def convert_array_to_sympy_function(array, expression):
    """
    Converts an array to a sympy function based on the given expression.

    Args:
        array (array): The input array to be converted.
        expression (sympy.Expr): The expression used for conversion.

    Returns:
        sympy.Function: The sympy function created from the array and expression.
    """
    matrix = sp.sympify(array)
    return make_function(matrix, variables(expression))


def is_positive_definite(matrix):
    """
    Check if a given matrix is positive definite.

    Args:
        matrix (numpy.ndarray or sympy.Matrix): The matrix to check.

    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    return sp.Matrix(matrix).is_positive_definite
