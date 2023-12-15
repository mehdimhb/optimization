import streamlit as st
import sympy as sp
import numpy as np

from src.gradient_decent import (gradient_decent_method, newton_method)
from src.step_methods import (Constant, LineSearch, Backtracking)
from src.math_utils import (hessian_inverse_diagonal,
                            hessian_inverse_diagonal_expression,
                            convert_array_to_sympy_function)


st.set_page_config(
    page_title="Optimization"
)

st.title("Optimization")

columns = st.columns(2)
function = columns[0].text_input("Function")

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

method = st.selectbox("Method", ["Gradient Decent Method", "Newton's Method", "Damped Newton's Method"])

if method == "Gradient Decent Method" or method == "Damped Newton's Method":
    columns = st.columns(2)
    step_size_selection_rule = columns[0].selectbox(
        "Step Size Selection Rule", ["Constant", "Exact Line Search", "Backtracing Line Search"]
    )
    match step_size_selection_rule:
        case "Constant":
            c = columns[1].number_input("Step Size", step=0.001, format="%0.3f")
            step_method = Constant(c)
        case "Exact Line Search":
            step_method = LineSearch()
        case "Backtracing Line Search":
            s = columns[1].number_input("S", min_value=0.0, step=0.01)
            alpha = columns[1].slider(r'$\alpha$', 0.0, 1.0, step=0.01)
            beta = columns[1].slider(r'$\beta$', 0.0, 1.0, step=0.01)
            step_method = Backtracking(s, alpha, beta)
    if method == "Gradient Decent Method":
        col = st.columns([1.5, 2, 2])
        scale_matrix = col[0].toggle(r"Use Scaling Matrix")
        if scale_matrix:
            scaling_matrix_type = col[1].radio(
                "choose type of Scaling Matrix:", [r"$D_{k}=diag(\nabla^{2}f(x_{k})^{-1})$", "Custom"]
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

columns = st.columns(2)
tolerance = columns[0].number_input("Tolerance (1-eN)", 1, value=5)
tolerance = float(f"1e-{tolerance}")
maximum_iteration = columns[1].number_input("Maximum Iteration", min_value=1, value=1000, step=500)

if st.button("Run", type="primary"):
    x0 = list(map(float, x0))
    if scale_matrix and scaling_matrix_type == "Custom":
        scale = convert_array_to_sympy_function(scale, function)
    with st.spinner('Calculating...'):
        match method:
            case "Gradient Decent Method":
                result = gradient_decent_method(function, x0, step_method, scale, tolerance, maximum_iteration)
            case "Newton's Method":
                result = newton_method(function, x0, None, tolerance, maximum_iteration)
            case "Damped Newton's Method":
                result = newton_method(function, x0, step_method, tolerance, maximum_iteration)
    if len(result) == 1:
        st.metric("Optimal Point", result[0][0])
    else:
        st.metric(r"Optimal Point: $\\x_{1}$", result[0][0])
        for i, x in enumerate(result[0][1:]):
            exec(f'st.metric(r"$x_{i+2}$", x)')
    st.metric("Optimal Value", result[1])
    st.metric("Norm of Gradient", result[2])
    st.metric("No of Iteration", result[3])
