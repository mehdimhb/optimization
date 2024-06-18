import streamlit as st
import sympy as sp
import numpy as np

from src.optimization_methods import (gradient_decent_method,
                                      newton_method,
                                      hybrid_gradient_newton_method)
from src.step_methods import (Constant, LineSearch, Backtracking)
from src.math_utils import (hessian_inverse_diagonal,
                            hessian_inverse_diagonal_expression,
                            convert_array_to_sympy_function)
from plot import contour


# Set page title
st.set_page_config(
    page_title="Optimization"
)
st.title("Optimization")

# Set subheader for method selection
st.subheader("Method")

# Get function from user input
columns = st.columns(2)
function = columns[0].text_input("Function")

# If function is provided, process it
if function != "":
    function = sp.sympify(function)
    columns[1].latex(function)
    no_of_variables = len(function.free_symbols)
    columns = st.columns(no_of_variables)
    x0 = []
    for i in range(no_of_variables):
        exec(f"x0.append(columns[{i}].text_input(r'$x_{i+1}^{0}$'))")
    scale = convert_array_to_sympy_function(np.identity(no_of_variables), function)
else:
    x0 = [st.text_input(r'$x^{0}$')]

# Get method from user input
method = st.selectbox(
    "Method",
    ["Gradient Decent Method", "Newton's Method", "Damped Newton's Method", "Hybrid Gradient-Newton Method"])

# If method is not Newton's Method, get step size selection rule from user input
if method != "Newton's Method":
    columns = st.columns(2)
    step_size_selection_rule = columns[0].selectbox(
        "Step Size Selection Rule", ["Constant", "Exact Line Search", "Backtracing Line Search"]
    )
    match step_size_selection_rule:
        case "Constant":
            # Get constant step size from user input
            c = columns[1].number_input("Step Size", step=0.001, format="%0.3f")
            step_method = Constant(c)
        case "Exact Line Search":
            step_method = LineSearch()
        case "Backtracing Line Search":
            # Get backtracking line search parameters from user input
            s = columns[1].number_input("S", min_value=0.0, step=0.01)
            alpha = columns[1].slider(r'$\alpha$', 0.0, 1.0, step=0.01)
            beta = columns[1].slider(r'$\beta$', 0.0, 1.0, step=0.01)
            step_method = Backtracking(s, alpha, beta)
    if method == "Gradient Decent Method":
        col = st.columns([1.5, 2, 2])
        scale_matrix = col[0].toggle(r"Use Scaling Matrix")
        if scale_matrix:
            scaling_matrix_type = col[1].radio(
                "Choose Type of Scaling Matrix:", [r"$D_{k}=diag(\nabla^{2}f(x_{k})^{-1})$", "Custom"]
            )
            match scaling_matrix_type:
                case r"$D_{k}=diag(\nabla^{2}f(x_{k})^{-1})$":
                    if function != "":
                        col[2].latex(sp.latex(hessian_inverse_diagonal_expression(function)))
                        scale = hessian_inverse_diagonal(function)
                case "Custom":
                    if function != "":
                        col2 = col[2].columns(no_of_variables)
                        scale = np.empty((no_of_variables, no_of_variables), dtype="U50")
                        for i in range(no_of_variables):
                            for j in range(no_of_variables):
                                scale[i, j] = col2[j].text_input(f'x{i+1}{j+1}', label_visibility='hidden')
                        if "" not in scale:
                            col[2].latex(sp.latex(sp.sympify(scale)))
                            scale = convert_array_to_sympy_function(scale, function)

# Get tolerance and maximum iteration from user input
columns = st.columns(2)
tolerance = columns[0].number_input("Tolerance (1e-N)", 1, value=5)
tolerance = float(f"1e-{tolerance}")
maximum_iteration = columns[1].number_input("Maximum Iteration", min_value=1, value=1000, step=500)
is_round = st.checkbox("Round Result According to Tolerance")

# Set subheader for plot
st.subheader("Plot")

# If function is provided and it is 2D, get plotting parameters from user input
if function != "" and no_of_variables == 2:
    plotting = True
    columns = st.columns(3)
    type_of_contour = columns[0].radio("Type", ["Line Contour", "Filled Contour"], index=1)
    filled = True if type_of_contour == "Filled Contour" else False
    levels = columns[1].number_input("Levels", 1, value=10, step=1)
    radius = columns[2].number_input("Radius", 0.01, value=3.00, step=0.01)
else:
    plotting = False
    st.write("Plot is not supported for this function")

# If user clicks "Run" button, run optimization method and get result
if st.button("Run", type="primary"):
    x0 = list(map(eval, x0))
    with st.spinner('Calculating...'):
        match method:
            case "Gradient Decent Method":
                result = gradient_decent_method(function, x0, step_method, scale, tolerance, maximum_iteration)
            case "Newton's Method":
                result = newton_method(function, x0, None, tolerance, maximum_iteration)
            case "Damped Newton's Method":
                result = newton_method(function, x0, step_method, tolerance, maximum_iteration)
            case "Hybrid Gradient-Newton Method":
                result = hybrid_gradient_newton_method(function, x0, step_method, tolerance, maximum_iteration)
    unrounded_result = result
    if is_round:
        result = list(result)
        for i in range(len(result)):
            result[i] = np.round(result[i], int(-np.log10(tolerance)))
            if isinstance(result[i], np.ndarray):
                result[i][result[i] == 0] = 0
            else:
                if result[i] == 0:
                    result[i] = 0
    if len(result[0]) == 1:
        st.metric("Optimal Point", result[0][0])
    else:
        st.metric(r"Optimal Point: $\\x_{1}$", result[0][0])
        for i, x in enumerate(result[0][1:]):
            exec(f'st.metric(r"$x_{i+2}$", x)')
    st.metric("Optimal Value", result[1])
    st.metric("Norm of Gradient", result[2])
    st.metric("No of Iteration", result[3])

    if plotting:
        with st.spinner('Plotting...'):
            points = [
                [p[i] for p in unrounded_result[4] if np.abs(p-unrounded_result[0]).sum(-1) < 2*radius]
                for i in range(no_of_variables)
            ]
            st.pyplot(contour(function, unrounded_result[0], radius, levels, filled, points))
