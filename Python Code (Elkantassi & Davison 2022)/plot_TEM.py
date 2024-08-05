########################################################################################################
# Functions to plot evidence functions and remove singularities from outputs of the tem_xD.py function #
########################################################################################################

# Importing standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from scipy.stats import chi2


def plot_prob(out, psi0, q, prob, xlab, rstarB=False):
    """
    Plots the evidence function or the pivots as a function of the interest parameter.
    
    Parameters:
    - out (dict): a dictionary containing the results.
    - psi0 (float): value of psi under the null hypothesis.
    - q (list of float): probabilities = 1 - nominal value, 0 <= q <= 1.
    - prob (bool): True if interested in the evidence function; False for pivots.
    - xlab (str): label of the x-axis.
    - rstarB (bool): If True, plot the rstar_B values.
    """
    psi = out['psi']
    ro_psi = out['r']
    wo_psi = (out['normal'][0] - psi) / out['normal'][1]
    rstaro_psi = out['rstar']
    if rstarB:
        rstaro_B_psi = out['rstar_B']

    plt.figure(figsize=(10, 6))
    
    if prob:
        plt.plot(psi, 1 - norm.cdf(ro_psi), label='r', color='black')
        plt.plot(psi, 1 - norm.cdf(wo_psi), label='wo', color='blue', linestyle='--')
        plt.plot(psi, 1 - norm.cdf(rstaro_psi), label='rstar', color='red', linestyle='--')
        if rstarB:
            plt.plot(psi, 1 - norm.cdf(rstaro_B_psi), label='rstar_B', color='green', linestyle='-.')
        plt.ylim(0, 1)
        plt.ylabel("Evidence function")
        for qi in q:
            plt.axhline(y=qi, color='grey', linestyle='--')
    else:
        plt.plot(psi, ro_psi, label='r', color='black')
        plt.plot(psi, wo_psi, label='wo', color='blue', linestyle='--')
        plt.plot(psi, rstaro_psi, label='rstar', color='red', linestyle='--')
        if rstarB:
            plt.plot(psi, rstaro_B_psi, label='rstar_B', color='green', linestyle='-.')
        plt.ylim(-3, 3)
        plt.ylabel("Pivots")
        for qi in q:
            plt.axhline(y=norm.ppf(qi), color='grey', linestyle='--')
    
    plt.axvline(x=psi0, color='grey', linestyle='--')
    plt.xlabel(xlab)
    plt.legend()
    plt.grid(True)
    plt.show()
# =============================================================================
# # Example usage
# # Assuming `results` is the dictionary returned by the `tem_2D` function or an equivalent function 
# psi0 = 20
# q = [0.05,0.1,0,5,0.9, 0.95]
# prob = True
# xlab = 'Psi'
# plot_prob(results, psi0, q, prob, xlab)

# =============================================================================
##########################################################################################

def smooth_curves(out, h):  
    """Remove singularities from rstar and r.
    
    Parameters:
    - out (dict): a dictionary containing the results.
    - h (float): threshold for smoothing, usually small (approx. 0.2).
    
    Returns:
    - dict: Updated dictionary after removing the singularity."""
    
    # Step 1: Remove failed fits
    fit_failed = np.isnan(out['r'])
    if np.any(fit_failed):
        out['r'] = out['r'][~fit_failed]
        out['rstar'] = out['rstar'][~fit_failed]
        out['q'] = out['q'][~fit_failed]
        out['psi'] = out['psi'][~fit_failed]
    
    w = chi2.cdf(out['r']**2, df=1)  # Use df=1 for proper chi-square distribution
    
    # If any correction for q failed and returned NA
    cor_failed = np.isnan(out['rstar'])
    # If equi-spaced values for psi between MLE and other, then we have r = 0
    cor_failed = np.logical_or(cor_failed, np.abs(out['r']) <= h)
    
    if np.any(cor_failed):
        resp = out['rstar'][~cor_failed] - out['r'][~cor_failed]
        regr = out['r'][~cor_failed]
        w = w[~cor_failed]
    else:
        resp = out['rstar'] - out['r']
        regr = out['r']
    
    # Ensure regr values are unique by filtering
    unique_regr, unique_indices = np.unique(regr, return_index=True)
    unique_resp = resp[unique_indices]
    unique_w = w[unique_indices]
    
    # Fit smoothing spline
    spline = UnivariateSpline(unique_regr, unique_resp, w=unique_w, s=1)
    fitted_spline = spline(unique_regr)
    
    # Compute difference between fitted values and rstar
    departure = fitted_spline - unique_resp
    
    # Outlier detection via chi-square test
    def scores(x, prob):
        return (x - np.mean(x))**2 / np.var(x) > chi2.ppf(prob, df=1)
    
    bad = scores(departure, 0.95)
    if np.any(bad):
        bad = np.logical_and(bad, np.logical_and(bad < 0.85 * len(departure), bad > 0.15 * len(departure)))
    
    if np.any(bad):
        unique_resp[bad] = np.nan
        unique_w = unique_w[~bad]
    
    # Fit smoothing spline again with less smoothness if there are bad points
    if np.any(bad):
        spline = UnivariateSpline(unique_regr, unique_resp, w=unique_w, s=-1)
    
    out['spline'] = spline
    out['rstar'] = spline(out['r']) + out['r']
    
    return out
