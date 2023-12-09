import streamlit as st
import sympy as sp
from src.gradient_decent import gdm_constant, gdm_exact_line_search, gdm_backtracking
from src.newton import newton_method


st.set_page_config(
    page_title="Optimization"
)

st.title("Optimization")

function = st.text_input("Function")

if function != "":
    function = sp.sympify(function)
    no_of_variables = len(function.free_symbols)
    columns = st.columns(no_of_variables)
    x0 = []
    for i in range(no_of_variables):
        exec(f"x0.append(columns[{i}].text_input(r'$x_{i+1}^{0}$'))")
else:
    x0 = [st.text_input(r'$x^{0}$')]

method = st.selectbox("Method", ["Gradient Decent Method", "Newton's Method"])

if method == "Gradient Decent Method":
    columns = st.columns(2)
    step_size_selection_rule = columns[0].selectbox(
        "Step Size Selection Rule", ["Constant", "Exact Line Search", "Backtracing"]
    )
    match step_size_selection_rule:
        case "Constant":
            c = columns[1].number_input("Step Size", step=0.001, format="%0.3f")
        case "Backtracing":
            s = columns[1].number_input("S", min_value=0.0, step=0.001, format="%0.3f")
            alpha = columns[1].slider(r'$\alpha$', 0.0, 1.0, step=0.01)
            beta = columns[1].slider(r'$\beta$', 0.0, 1.0, step=0.01)

columns = st.columns(2)
tolerance = columns[0].number_input("Tolerance (1-eN)", 1, value=5)
tolerance = float(f"1e-{tolerance}")
maximum_iteration = columns[1].number_input("Maximum Iteration", min_value=1, value=1000, step=500)

if st.button("Run", type="primary"):
    x0 = list(map(float, x0))
    with st.spinner('Calculating...'):
        match method:
            case "Gradient Decent Method":
                match step_size_selection_rule:
                    case "Constant":
                        result = gdm_constant(function, x0, c, tolerance, maximum_iteration)
                    case "Exact Line Search":
                        result = gdm_exact_line_search(function, x0, tolerance, maximum_iteration)
                    case "Backtracing":
                        result = gdm_backtracking(function, x0, s, alpha, beta, tolerance, maximum_iteration)
            case "Newton's Method":
                result = newton_method(function, x0, tolerance, maximum_iteration)
    if len(result) == 1:
        st.metric("Optimal Point", result[0][0])
    else:
        st.metric(r"Optimal Point: $\\x_{1}$", result[0][0])
        for i, x in enumerate(result[0][1:]):
            exec(f'st.metric(r"$x_{i+2}$", x)')
    st.metric("Optimal Value", result[1])
    st.metric("Norm of Gradient", result[2])
    st.metric("No of Iteration", result[3])