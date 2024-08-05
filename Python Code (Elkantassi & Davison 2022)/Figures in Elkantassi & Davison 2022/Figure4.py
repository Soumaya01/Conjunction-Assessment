############################################################################################################
# This script reproduces Figure 4 from Elkantassi & Davison 2022. It illustrates the evidence functions based 
# on the likelihood root r(psi), Wald statistic w(psi), exact probability, and modified likelihood root rstar(psi)
# for the Rayleigh distribution.
############################################################################################################


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

def r(y, psi=1):
    """
    Computes the likelihood root r(psi).
    
    Args:
    - y (float): Observation value.
    - psi (float): Parameter value.
    
    Returns:
    - float: Likelihood root value.
    """
    psi_hat = y / np.sqrt(2)
    return np.sign(psi_hat - psi) * np.sqrt(2 * (2 * np.log(psi / psi_hat) + (psi_hat / psi)**2 - 1))

def w(y, psi=1):
    """
    Computes the Wald statistic w(psi).
    
    Args:
    - y (float): Observation value.
    - psi (float): Parameter value.
    
    Returns:
    - float: Wald statistic value.
    """
    psi_hat = y / np.sqrt(2)
    return 2 * (1 - psi / psi_hat)

def q(y, psi=1):
    """
    Computes the q(psi) function.
    
    Args:
    - y (float): Observation value.
    - psi (float): Parameter value.
    
    Returns:
    - float: q(psi) value.
    """
    psi_hat = y / np.sqrt(2)
    return -(1 - psi_hat**2 / psi**2)

def r_star(ro_psi, qo_psi):
    """
    Computes the modified likelihood root rstar(psi).
    
    Args:
    - ro_psi (float): Likelihood root value.
    - qo_psi (float): q(psi) value.
    
    Returns:
    - float: Modified likelihood root value.
    """
    return ro_psi + np.log(qo_psi / ro_psi) / ro_psi

# Setup parameters
yo = np.sqrt(2)
psio_hat = yo / np.sqrt(2)
psi = np.linspace(1e-5, 5, 200)

# Compute evidence functions
ro_psi = r(yo, psi)
wo_psi = w(yo, psi)
qo_psi = q(yo, psi)
rstaro_psi = r_star(ro_psi, qo_psi)
exact = np.exp(-(psio_hat / psi)**2)

# Define quantiles for horizontal lines
qd = np.array([10**(-5), 10**(-4), 10**(-3), 10**(-2), 0.025, 0.05, 0.5] + (1 - np.array([10**(-5), 10**(-4), 10**(-3), 10**(-2), 0.025, 0.05])).tolist())

# Plot evidence functions
plt.figure(figsize=(12, 6))

# Left subplot: Evidence function
plt.subplot(1, 2, 1)
plt.plot(psi, norm.cdf(ro_psi), 'k-', label='r(psi)')
plt.plot(psi, 1 - exact, 'cyan', label='Exact', linewidth=2)  # Corrected evidence function for exact
plt.plot(psi, norm.cdf(wo_psi), 'b--', linewidth=2, label='w(psi)')
plt.plot(psi, norm.cdf(rstaro_psi), 'r-.', label='rstar(psi)')
plt.hlines(qd, psi.min(), psi.max(), colors='grey', linestyles='dotted')
plt.ylim(0, 1)
plt.xlabel(r'$\psi$')
plt.ylabel('Evidence function')
plt.legend()

# Right subplot: Pivot
plt.subplot(1, 2, 2)
plt.plot(psi, ro_psi, 'k-', label='r(psi)')
plt.plot(psi, wo_psi, 'b--', linewidth=2, label='w(psi)')
plt.plot(psi, rstaro_psi, 'r-.', linewidth=2, label='rstar(psi)')
plt.plot(psi, norm.ppf(1 - exact), 'cyan', label='qnorm(exact)', linewidth=1)  # Correct pivot plot for exact
plt.hlines(norm.ppf(qd), psi.min(), psi.max(), colors='grey', linestyles='dotted')
plt.ylim(-4.5, 4.5)
plt.xlabel(r'$\psi$')
plt.ylabel('Pivot')
plt.legend()

plt.tight_layout()
plt.show()
