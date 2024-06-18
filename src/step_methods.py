import numpy as np
from scipy.optimize import line_search


class Step:
    def __init__(self, value, method):
        """
        Initializes a new instance of the Step class.

        Args:
            value (Any): The value of the step.
            method (str): The method used for the step.

        Returns:
            None
        """
        self.value = value
        self.method = method

    def __call__(self, *args):
        return self.value


class Constant(Step):
    def __init__(self, value):
        super().__init__(value, "Constant")


class LineSearch(Step):
    def __init__(self):
        super().__init__(None, "Exact Line Search")

    def __call__(self, f, grad, x, d):
        """
        Calls the line search algorithm to find the optimal step size for gradient descent optimization.

        Args:
            f (function): The objective function to be minimized.
            grad (function): The gradient of the objective function.
            x (array-like): The current point.
            d (array-like): The direction of the step.

        Returns:
            float: The optimal step size.

        Description:
            This function calls the line search algorithm to find the optimal step size for gradient descent optimization.
            It takes the objective function `f`, its gradient `grad`, the current point `x`, and the direction of the step `d` as input.
            It uses the `line_search` function from the `scipy.optimize` module to find the optimal step size.
            The optimal step size is stored in the `value` attribute of the `LineSearch` object.
            The function then calls the `__call__` method of the parent class using `super().__call__()`.
            The return value is the result of the parent class's `__call__` method.
        """
        self.value = line_search(f, grad, x, d)[0]
        return super().__call__()


class Backtracking(Step):
    def __init__(self, s, alpha, beta):
        super().__init__(None, "Backtracing Line Search")
        self.s = s
        self.alpha = alpha
        self.beta = beta

    def backtracing_line_search(self, f, grad, x, d):
        """
        Performs backtracking line search to find the optimal step size for gradient descent optimization.

        Args:
            f (function): The objective function to be minimized.
            grad (function): The gradient of the objective function.
            x (array-like): The current point.
            d (array-like): The direction of the step.

        Returns:
            float: The optimal step size.

        Description:
            This function performs backtracking line search to find the optimal step size for gradient descent optimization.
            It starts with an initial step size `t` and iteratively decreases it until the Armijo condition is satisfied.
            The Armijo condition is given by:
                f(x + t * d) <= f(x) - alpha * t * np.dot(grad(x).T, d)
            where `f` is the objective function, `grad` is the gradient of the objective function, `x` is the current point,
            `d` is the direction of the step, `alpha` is a parameter that controls the rate of convergence, and `t` is the step size.
            The step size is updated using the formula:
                t *= beta
            where `beta` is a parameter that controls the step size reduction rate.
        """
        t = self.s
        while f(x) - f(x + t * d) < -self.alpha * t * np.dot(grad(x).T, d):
            t *= self.beta
        return t

    def __call__(self, f, grad, x, d):
        self.value = self.backtracing_line_search(f, grad, x, d)
        return super().__call__()
