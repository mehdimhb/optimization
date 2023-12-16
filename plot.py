import numpy as np
import matplotlib.pyplot as plt
from src.math_utils import function, large_input_function


def contour(expression, center, radius, levels, filled, points):
    x_range = np.linspace(center[0]-radius, center[0]+radius, 1000)
    y_range = np.linspace(center[1]-radius, center[1]+radius, 1000)
    X, Y = np.meshgrid(x_range, y_range)
    Z = large_input_function(function(expression), X, Y)
    fig, ax = plt.subplots(1, 1)
    if filled:
        plot = ax.contourf(X, Y, Z, levels)
    else:
        plot = ax.contour(X, Y, Z, levels)
    fig.colorbar(plot)
    ax.scatter(points[0], points[1], color="#E73F33")
    ax.plot(points[0], points[1], color="#E73F33")
    return fig
