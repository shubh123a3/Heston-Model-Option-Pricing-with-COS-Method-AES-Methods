# Heston Model Option Pricing with COS and AES Methods

## Project Overview

This project implements the Heston Model for option pricing, utilizing the Fourier-Cosine (COS) method and the Almost Exact Simulation (AES) method. It includes Monte Carlo simulations, characteristic function calculations, and variance processes to achieve efficient and accurate derivative pricing.

## Table of Contents

1. [Introduction](#introduction)
2. [The Heston Model](#the-heston-model)
3. [Cox-Ingersoll-Ross (CIR) Process](#cox-ingersoll-ross-cir-process)
4. [Fourier-Cosine (COS) Method](#fourier-cosine-cos-method)
5. [Almost Exact Simulation (AES) Method](#almost-exact-simulation-aes-method)
6. [Implementation Details](#implementation-details)
7. [Usage](#usage)
8. [References](#references)

## Introduction

Option pricing is a fundamental aspect of financial engineering, and various models have been developed to capture the complexities of financial markets. The Heston Model is renowned for its ability to incorporate stochastic volatility, providing a more accurate representation of market behaviors compared to models with constant volatility assumptions. This project focuses on implementing the Heston Model using the COS and AES methods to enhance computational efficiency and accuracy.

## The Heston Model

The Heston Model, introduced by Steven Heston in 1993, is a stochastic volatility model that describes the evolution of an asset price and its variance. The model is defined by the following set of stochastic differential equations (SDEs):

\[
\begin{aligned}
dS_t &= \mu S_t \, dt + \sqrt{v_t} S_t \, dW_t^S, \\
dv_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^v,
\end{aligned}
\]

where:

- \( S_t \): Asset price at time \( t \),
- \( v_t \): Variance of the asset price at time \( t \),
- \( \mu \): Drift term representing the expected return,
- \( \kappa \): Rate at which \( v_t \) reverts to the long-term mean \( \theta \),
- \( \theta \): Long-term mean of the variance,
- \( \sigma \): Volatility of the variance process (volatility of volatility),
- \( W_t^S \) and \( W_t^v \): Wiener processes with correlation \( \rho \).

The correlation between the two Wiener processes introduces a relationship between the asset price and its variance, allowing the model to capture the observed market phenomena such as volatility clustering and the leverage effect.

## Cox-Ingersoll-Ross (CIR) Process

The variance process \( v_t \) in the Heston Model follows a Cox-Ingersoll-Ross (CIR) process, which is a type of mean-reverting square-root diffusion process. The CIR process is defined as:

\[
dv_t = \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^v.
\]

Key properties of the CIR process include:

- **Mean Reversion**: The process tends to revert to the long-term mean \( \theta \) at a rate \( \kappa \).
- **Non-Negativity**: The square-root term ensures that the variance \( v_t \) remains non-negative, which is essential for modeling volatility.

The CIR process is widely used in financial modeling due to these properties, making it suitable for describing the evolution of interest rates and variances.

## Fourier-Cosine (COS) Method

The Fourier-Cosine (COS) method is a numerical technique used for option pricing, particularly effective when the characteristic function of the underlying asset's logarithm is known. The method involves the following steps:

1. **Characteristic Function**: Compute the characteristic function \( \varphi(u) \) of the log-asset price.
2. **Integration Interval**: Determine a suitable integration interval \([a, b]\) to capture the significant contributions of the integrand.
3. **Cosine Expansion**: Approximate the probability density function using a Fourier-Cosine series expansion.
4. **Option Valuation**: Calculate the option price by integrating the payoff function against the approximated density function.

The COS method is known for its rapid convergence and computational efficiency, making it a valuable tool for pricing European-style options under various stochastic processes.

## Almost Exact Simulation (AES) Method

The Almost Exact Simulation (AES) method is a technique used to simulate the paths of stochastic processes with high accuracy. In the context of the Heston Model, the AES method involves:

1. **Discretization**: Divide the time horizon into small intervals.
2. **Variance Process Simulation**: Simulate the variance process \( v_t \) using an appropriate scheme that preserves its statistical properties.
3. **Asset Price Simulation**: Simulate the asset price \( S_t \) conditional on the variance path.

The AES method aims to closely replicate the true distribution of the stochastic processes, reducing discretization errors and providing more accurate simulations compared to standard methods.

## Implementation Details

The implementation of the Heston Model with the COS and AES methods involves the following components:

- **Characteristic Function Calculation**: Derive and implement the characteristic function of the log-asset price under the Heston dynamics.
- **COS Method Application**: Apply the COS method to price European options by integrating the payoff function against the approximated density function.
- **AES Method Application**: Implement the AES method to simulate the paths of the asset price and variance processes for Monte Carlo simulations.
- **Parameter Calibration**: Calibrate the model parameters (\( \kappa \), \( \theta \), \( \sigma \), \( \rho \), \( v_0 \)) to market data to ensure accurate pricing.

The code is organized into modules for clarity and reusability, with detailed comments and documentation provided for each function.

## Usage

To use this implementation:

1. **Install Dependencies**: Ensure that the required Python packages are installed as specified in the `requirements.txt` file.
2. **Run Simulations**: Use the provided scripts to perform option pricing and simulations. The `heston_model.py` script contains functions for pricing options using the COS method, while the `aes_simulation.py` script implements the AES method for path simulations.
3. **Analyze Results**: The results of the simulations and pricing can be analyzed using the Jupyter Notebook `heston_model.ipynb`, which provides visualizations and further explanations.

## References

- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options. *The Review of Financial Studies*, 6(2), 327-343.
- Fang, F., & Oosterlee, C. W. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. *SIAM Journal on Scientific Computing*, 31(2), 826-848.
- Broadie, M., & Kaya, O. (2006). Exact Simulation of Stochastic Volatility and Other Affine Jump Diffusion Processes. *Operations Research*, 54(2), 217-231.

 
