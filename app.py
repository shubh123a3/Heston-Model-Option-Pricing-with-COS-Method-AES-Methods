import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import scipy.stats as st
import enum
from scipy.stats import norm


# Enum for option type
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0


# COS method functions
def CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L):
    K = np.array(K).reshape([len(K), 1])
    i = 1j
    x0 = np.log(S0 / K)
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    k = np.linspace(0, N - 1, N).reshape([N, 1])
    u = k * np.pi / (b - a)
    H_k = CallPutCoefficients(CP, a, b, k)
    mat = np.exp(i * np.outer((x0 - a), u))
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))
    return value


def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)), 2.0))
    expr1 = np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(
        k * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    return {'chi': chi, 'psi': psi}


def CallPutCoefficients(CP, a, b, k):
    if CP == OptionType.CALL:
        c, d = 0.0, b
        coef = Chi_Psi(a, b, c, d, k)
        H_k = 2.0 / (b - a) * (coef['chi'] - coef['psi'])
    elif CP == OptionType.PUT:
        c, d = a, 0.0
        coef = Chi_Psi(a, b, c, d, k)
        H_k = 2.0 / (b - a) * (-coef['chi'] + coef['psi'])
    return H_k


# Heston model functions
def ChFHestonModel(r, tau, kappa, gamma, vbar, v0, rho):
    i = 1j
    D1 = lambda u: np.sqrt(np.power(kappa - gamma * rho * i * u, 2) + (u ** 2 + i * u) * gamma ** 2)
    g = lambda u: (kappa - gamma * rho * i * u - D1(u)) / (kappa - gamma * rho * i * u + D1(u))
    C = lambda u: (1 - np.exp(-D1(u) * tau)) / (gamma ** 2 * (1 - g(u) * np.exp(-D1(u) * tau))) * (
                kappa - gamma * rho * i * u - D1(u))
    A = lambda u: r * i * u * tau + kappa * vbar * tau / gamma ** 2 * (
                kappa - gamma * rho * i * u - D1(u)) - 2 * kappa * vbar / gamma ** 2 * np.log(
        (1 - g(u) * np.exp(-D1(u) * tau)) / (1 - g(u)))
    cf = lambda u: np.exp(A(u) + C(u) * v0)
    return cf


# Path generation functions
def GeneratePathsHestonEuler(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0, seed):
    np.random.seed(seed)
    Z1 = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    Z2 = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W1 = np.zeros([NoOfPaths, NoOfSteps + 1])
    W2 = np.zeros([NoOfPaths, NoOfSteps + 1])
    V = np.zeros([NoOfPaths, NoOfSteps + 1])
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    V[:, 0] = v0
    X[:, 0] = np.log(S_0)
    dt = T / NoOfSteps
    for i in range(NoOfSteps):
        if NoOfPaths > 1:
            Z1[:, i] = (Z1[:, i] - np.mean(Z1[:, i])) / np.std(Z1[:, i])
            Z2[:, i] = (Z2[:, i] - np.mean(Z2[:, i])) / np.std(Z2[:, i])
        Z2[:, i] = rho * Z1[:, i] + np.sqrt(1 - rho ** 2) * Z2[:, i]
        W1[:, i + 1] = W1[:, i] + np.sqrt(dt) * Z1[:, i]
        W2[:, i + 1] = W2[:, i] + np.sqrt(dt) * Z2[:, i]
        V[:, i + 1] = V[:, i] + kappa * (vbar - V[:, i]) * dt + gamma * np.sqrt(V[:, i]) * (W1[:, i + 1] - W1[:, i])
        V[:, i + 1] = np.maximum(V[:, i + 1], 0.0)
        X[:, i + 1] = X[:, i] + (r - 0.5 * V[:, i]) * dt + np.sqrt(V[:, i]) * (W2[:, i + 1] - W2[:, i])
    S = np.exp(X)
    return {"time": np.linspace(0, T, NoOfSteps + 1), "S": S}


def CIR_Sample(NoOfPaths, kappa, gamma, vbar, s, t, v_s):
    delta = 4 * kappa * vbar / gamma ** 2
    c = gamma ** 2 * (1 - np.exp(-kappa * (t - s))) / (4 * kappa)
    kappa_bar = 4 * kappa * v_s * np.exp(-kappa * (t - s)) / (gamma ** 2 * (1 - np.exp(-kappa * (t - s))))
    return c * np.random.noncentral_chisquare(delta, kappa_bar, NoOfPaths)


def GeneratePathsHestonAES(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0, seed):
    np.random.seed(seed)
    Z1 = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W1 = np.zeros([NoOfPaths, NoOfSteps + 1])
    V = np.zeros([NoOfPaths, NoOfSteps + 1])
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    V[:, 0] = v0
    X[:, 0] = np.log(S_0)
    dt = T / NoOfSteps
    for i in range(NoOfSteps):
        if NoOfPaths > 1:
            Z1[:, i] = (Z1[:, i] - np.mean(Z1[:, i])) / np.std(Z1[:, i])
        W1[:, i + 1] = W1[:, i] + np.sqrt(dt) * Z1[:, i]
        V[:, i + 1] = CIR_Sample(NoOfPaths, kappa, gamma, vbar, 0, dt, V[:, i])
        k0 = (r - rho / gamma * kappa * vbar) * dt
        k1 = (rho * kappa / gamma - 0.5) * dt - rho / gamma
        k2 = rho / gamma
        X[:, i + 1] = X[:, i] + k0 + k1 * V[:, i] + k2 * V[:, i + 1] + np.sqrt((1 - rho ** 2) * V[:, i]) * (
                    W1[:, i + 1] - W1[:, i])
    S = np.exp(X)
    return {"time": np.linspace(0, T, NoOfSteps + 1), "S": S}


def EUOptionPriceFromMCPathsGeneralized(CP, S, K, T, r):
    result = np.zeros([len(K), 1])
    if CP == OptionType.CALL:
        for idx, k in enumerate(K):
            result[idx] = np.exp(-r * T) * np.mean(np.maximum(S - k, 0.0))
    elif CP == OptionType.PUT:
        for idx, k in enumerate(K):
            result[idx] = np.exp(-r * T) * np.mean(np.maximum(k - S, 0.0))
    return result


# Streamlit app
def main():
    st.title("Heston Model Option Pricing")

    # Sidebar parameters
    st.sidebar.header("Model Parameters")
    S_0 = st.sidebar.number_input("Spot Price (S₀)", 100.0)
    r = st.sidebar.slider("Risk-Free Rate (r)", 0.01,0.9)
    T = st.sidebar.slider("Time to Maturity (T)", 1,10 )
    kappa = st.sidebar.number_input("Kappa (κ)", 0.5)
    gamma = st.sidebar.number_input("Gamma (γ)", 1.0)
    rho = st.sidebar.number_input("Rho (ρ)", -0.9)
    vbar = st.sidebar.number_input("Long-Term Variance (v̄)", 0.04)
    v0 = st.sidebar.number_input("Initial Variance (v₀)", 0.04)
    option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])
    CP = OptionType.CALL if option_type == "Call" else OptionType.PUT

    st.sidebar.header("Simulation Parameters")
    NoOfPaths = st.sidebar.number_input("Number of Paths", 2500)
    NoOfSteps = st.sidebar.number_input("Number of Steps", 500)
    seed = st.sidebar.number_input("Random Seed", 3)

    st.sidebar.header("Strike Parameters")
    strike_min = st.sidebar.number_input("Minimum Strike", 80.0)
    strike_max = st.sidebar.number_input("Maximum Strike", 150.0)
    num_strikes = st.sidebar.number_input("Number of Strikes", 30)
    K = np.linspace(strike_min, strike_max, num_strikes)

    # Calculate exact prices
    cf = ChFHestonModel(r, T, kappa, gamma, vbar, v0, rho)
    optValueExact = CallPutOptionPriceCOSMthd(cf, CP, S_0, r, T, K, 1000, 8)

    # Generate paths
    pathsEULER = GeneratePathsHestonEuler(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0, seed)
    pathsAES = GeneratePathsHestonAES(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0, seed)

    # Calculate option prices
    OptPrice_EULER = EUOptionPriceFromMCPathsGeneralized(CP, pathsEULER["S"][:, -1], K, T, r)
    OptPrice_AES = EUOptionPriceFromMCPathsGeneralized(CP, pathsAES["S"][:, -1], K, T, r)

    # Plot results
    st.header("Option Price Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=K, y=optValueExact.flatten(), name='Exact (COS)'))
    fig.add_trace(go.Scatter(x=K, y=OptPrice_EULER.flatten(), name='Euler'))
    fig.add_trace(go.Scatter(x=K, y=OptPrice_AES.flatten(), name='AES'))
    fig.update_layout(xaxis_title="Strike Price", yaxis_title="Option Price",
                      legend_title="Method")
    st.plotly_chart(fig)

    # Plot sample paths
    st.header("Sample Paths")
    num_sample_paths = min(10, NoOfPaths)
    time_grid = pathsEULER["time"]

    fig_paths = make_subplots(rows=1, cols=2, subplot_titles=("Euler Scheme", "AES Scheme"))
    for i in range(num_sample_paths):
        fig_paths.add_trace(go.Scatter(x=time_grid, y=pathsEULER["S"][i], mode='lines',
                                       name=f'Path {i + 1}', showlegend=False), row=1, col=1)
        fig_paths.add_trace(go.Scatter(x=time_grid, y=pathsAES["S"][i], mode='lines',
                                       name=f'Path {i + 1}', showlegend=False), row=1, col=2)
    fig_paths.update_layout(height=400, width=1000)
    st.plotly_chart(fig_paths)

    # Convergence analysis
    st.header("Convergence Analysis")
    K_conv = st.sidebar.number_input("Convergence Strike Price", 140.0)
    dt_values = np.array([1.0, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64])
    NoOfStepsV = [int(T / dt) for dt in dt_values]

    # Calculate exact price for convergence strike
    optValueExact_conv = CallPutOptionPriceCOSMthd(cf, CP, S_0, r, T, [K_conv], 1000, 8)

    errorEuler = []
    errorAES = []
    for steps in NoOfStepsV:
        # Euler paths
        paths_euler_conv = GeneratePathsHestonEuler(NoOfPaths, steps, T, r, S_0, kappa, gamma, rho, vbar, v0, seed)
        price_euler = EUOptionPriceFromMCPathsGeneralized(CP, paths_euler_conv["S"][:, -1], [K_conv], T, r)
        errorEuler.append(price_euler[0][0] - optValueExact_conv[0][0])

        # AES paths
        paths_aes_conv = GeneratePathsHestonAES(NoOfPaths, steps, T, r, S_0, kappa, gamma, rho, vbar, v0, seed)
        price_aes = EUOptionPriceFromMCPathsGeneralized(CP, paths_aes_conv["S"][:, -1], [K_conv], T, r)
        errorAES.append(price_aes[0][0] - optValueExact_conv[0][0])
    dtV = np.array([1.0, 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0])
    NoOfStepsV = [int(T / x) for x in dtV]
    K=st.number_input(" test Strike Price", 100.0)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Euler Scheme")
        for i in range(0, len(NoOfStepsV)):
            st.markdown("K ={0}, dt = {1} = {2}".format(K, dtV[i], errorEuler[i]))
    with col2:
        st.markdown("AES Scheme")
        for i in range(0, len(NoOfStepsV)):
            st.markdown("K ={0}, dt = {1} = {2}".format(K, dtV[i], errorAES[i]))


    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=dt_values, y=errorEuler, mode='lines+markers', name='Euler'))
    fig_conv.add_trace(go.Scatter(x=dt_values, y=errorAES, mode='lines+markers', name='AES'))
    fig_conv.update_layout(xaxis_title="Time Step (Δt)", yaxis_title="Error",
                           xaxis_type="log", yaxis_type="log",
                           legend_title="Method")
    st.plotly_chart(fig_conv)


if __name__ == "__main__":
    main()