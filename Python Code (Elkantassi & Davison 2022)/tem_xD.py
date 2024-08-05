# Importing standard and third-party libraries
import numpy as np
from scipy.optimize import minimize
from numpy.linalg import det, inv, LinAlgError, eigvals
from math import cos, sin, sqrt
from scipy.stats import multivariate_normal
import numdifftools as nd
import warnings

#############################################################
############### Tangent Exponential model in 3D #############
#############################################################

def is_positive_definite(H):
    """Check if a matrix H is positive definite."""
    try:
        np.linalg.cholesky(H)
        return True
    except LinAlgError:
        return False


def tem_2D(psi, nlogL, gr_rest, j_theta, MLEs, th_init, phi, d_phi, data, delta, z, n_psi, low, up, method):
    """
    Computes the Tangent Exponential Models (TEM) in 2D in a scenario of a known velocity.

    Args:
    - psi (float or array): A specific value or vector for the parameter of interest psi. If psi is None, it will be set inside this function.       
    - nlogL (function): Minus the log-likelihood function.
    - gr_rest (function): Gradient of the minus log-likelihood function.
    - MLEs (function): Function to compute Maximum Likelihood Estimates.
    - phi (function): Canonical parameter of the exponential family.
    - d_phi (function): Derivative of the phi function.
    - data (dict): Contains (w, D), D being the known variance-covariance matrix projected into the conjunction plane.
                   w is the observed relative position in the conjunction plane.
    - delta (float): Lower bound for the grid of psi values.
    - n_psi (int): Number of points in the grid of psi.
    - z (float): Z-score or multiplier for the standard error.
    - low, up (floats): Bounds for the optimization (lower, upper).
    - method (str): Optimization method to be used.

    Returns:
    - dict: Contains pivots for each psi value.
    """
    
    def q_a(theta, data):
        """Compute the q statistic for given theta and data."""
        psi, lam = theta
        numerator = psi * (data['w'][0] * cos(lam) + data['w'][1] * sin(lam)) - psi**2
        denominator = sqrt(psi * (data['w'][0] * cos(lam) * data['D'][1, 1] + data['w'][1] * sin(lam) * data['D'][0, 0]) +
                           psi**2 * cos(2 * lam) * (data['D'][0, 0] - data['D'][1, 1]))
        return numerator / denominator

    def r2_stat(theta, data):
        """Compute the squared log-likelihood statistic for given theta and data."""
        psi, lam = theta
        r2 = 1 / (data['D'][0, 0] * data['D'][1, 1]) * (data['D'][1, 1] * (data['w'][0] - psi * cos(lam))**2 + 
                                                        data['D'][0, 0] * (data['w'][1] - psi * sin(lam))**2)
        return r2

    # Compute the full model MLEs and log-likelihood
    th_full = MLEs(data)
    L_full = -nlogL(th_full[0], th_full[1:], data)
    H_full = j_theta(th_full, data)

    # Check if the full model Hessian is positive definite
    if not is_positive_definite(H_full):
        raise ValueError("Hessian matrix of the full model is not positive definite.")

    # Compute standard deviations from the Hessian matrix
    th_se = np.sqrt(np.diag(inv(H_full)))
    psi_se = th_se[0]

    # Generate grid of psi values if not provided
    if psi is None:
        if th_full[0] - z * psi_se > 0:
            psi = np.linspace(th_full[0] - z * psi_se, th_full[0] + z * psi_se, n_psi)
        else:
            psi = np.linspace(delta, th_full[0] + z * psi_se, n_psi)
    n_psi = len(psi)

    # Initialize arrays to store results
    th_rest = np.tile(th_full, (n_psi, 1))
    L_rest = np.zeros(n_psi)
    J_rest = np.zeros(n_psi)
    r = np.zeros(n_psi)
    q = np.zeros(n_psi)
    q_B = np.zeros(n_psi)
    coef = np.zeros(n_psi)

    # Initialize th_init if not provided
    if th_init is None:
        th_init = th_full[1:]

    # Loop through each psi value and perform restricted optimization
    for j in range(n_psi-1, -1, -1):
        res = minimize(lambda lam: nlogL(psi[j], lam, data), 
                       th_init, 
                       method=method, 
                       bounds=[(low, up)] * len(th_init), 
                       jac=lambda lam: gr_rest(psi[j], lam, data))
        
        if not res.success:
            warnings.warn(f"Optimization did not converge for psi = {psi[j]}")
            continue
        
        th_rest[j, :] = [psi[j]] + list(res.x)
        L_rest[j] = -res.fun
        H_rest = j_theta(th_rest[j, :], data)
        
        if not is_positive_definite(H_rest):
            raise ValueError(f"Hessian matrix of the restricted model is not positive definite for psi = {psi[j]}")
        
        J_rest[j] = H_rest[1, 1]
        
        r[j] = np.sign(th_full[0] - th_rest[j, 0]) * sqrt(2 * (L_full - L_rest[j]))
        q[j] = q_a(th_rest[j, :], data)
        coef[j] = sqrt(det(H_full) / det(H_rest))
        dphi_dth_full = d_phi(th_full)
        dphi_dth_rest = d_phi(th_rest[j, :])
        q_B[j] = q[j] * (det(dphi_dth_rest) / det(dphi_dth_full)) * coef[j]

    # Remove entries where r is zero to avoid division by zero errors
    non_zero_indices = r != 0
    r = r[non_zero_indices]
    q = q[non_zero_indices]
    q_B = q_B[non_zero_indices]
    psi = psi[non_zero_indices]

    # Compute rstar and rstar_B
    rstar = r + np.log(q / r) / r
    rstar_B = r + np.log(q_B / r) / r

    # Compile results into a dictionary
    out = {
        'psi': psi,
        'L_full': L_full,
        'L_rest': L_rest,
        'r': r,
        'rstar': rstar,
        'q': q,
        'coef': coef,
        'q_B': q_B,
        'rstar_B': rstar_B,
        'th_full': th_full,
        'th_rest': th_rest,
        'j_th_th': det(H_full),
        'normal': [th_full[0], psi_se],
        'th_hat': th_full,
        'th_hat_se': th_se
    }

    return out


#############################################################
############### Tangent Exponential model in 6D #############
#############################################################


def tem_6D(psi, nlogL, gr_rest, MLEs, th_init, phi, d_phi, data, delta, z, n_psi, low, up, method):
    """
    Apply the tangent exponential model in six dimensions, a scenario of an unknown velocity.

    Args:
        psi (array): Grid of psi values or None to auto-generate.
        nlogL (callable): Function to compute the negative log-likelihood.
        gr_rest (callable): Function to compute the gradient of the restricted model.
        MLEs (callable): Function to compute Maximum Likelihood Estimates.
        phi (callable): Canonical parameter function.
        d_phi (callable): Derivative of the phi function.
        th_init (array): Initial guess for the parameters.
        data (dict): Dictionary containing observed data and other necessary parameters.
        delta (float): Lower bound for the grid of psi if psi is None.
        z (float): Multiplier for the range of psi around MLE.
        n_psi (int): Number of points in the psi grid.
        low (float): Lower bound for optimization.
        up (float): Upper bound for optimization.
        method (str): Optimization method to use.

    Returns:
        dict: Dictionary containing results of the analysis.
    """
    def nlogL_full(theta):
        return nlogL(theta[0], theta[1:], data)
    
    # Compute the MLEs for the full model
    th_full = MLEs(data)
    L_full = -nlogL_full(th_full)
    
    # Compute Hessian matrix at the MLE
    hessian_func = nd.Hessian(lambda th: nlogL_full(th))
    hessian_full = hessian_func(th_full)
    
    # Compute standard deviations
    th_se = np.sqrt(np.diag(np.linalg.inv(hessian_full)))
    psi_se = th_se[0]

    # Initialize the output dictionary
    out = {}
    out['normal'] = [th_full[0], psi_se]
    out['th_hat'] = th_full
    out['th_hat_se'] = th_se
    out['L_full'] = L_full

    # Generate grid of psi if not provided
    if psi is None:
        psi = np.linspace(th_full[0] - z * psi_se, th_full[0] + z * psi_se, n_psi) if th_full[0] - z * psi_se > 0 else \
              np.linspace(delta, th_full[0] + z * psi_se, n_psi)

    # Initialize arrays for storing results
    out['L_rest'] = np.zeros_like(psi)
    out['J_rest'] = np.zeros_like(psi)
    out['th_rest'] = np.tile(th_full, (len(psi), 1))

    # Compute coefficients used for q and q_B
    dphi_dth_full = d_phi(th_full)
    D_bot = det(dphi_dth_full)
    out['j_th_th'] = det(hessian_full)
    Jeff_prior_full = np.sqrt(det(np.dot(np.dot(dphi_dth_full, data['Omega']), dphi_dth_full.T)))

    out['q'] = np.zeros_like(psi)
    out['q_B'] = np.zeros_like(psi)
    out['D_top'] = np.zeros_like(psi)
    out['psi'] = psi

    # Restricted model log-likelihood and Hessian on the grid of psi
    for i, psi_val in enumerate(psi):
        result = minimize(lambda th: nlogL(psi_val, th, data), th_init, method=method, 
                          bounds=list(zip(low, up)), options={'maxiter': 10000})
        if result.success:
            # Store the results of the optimization
            out['th_rest'][i, :] = np.concatenate([[psi_val], result.x])
            out['L_rest'][i] = -result.fun
            
            # Compute the Hessian matrix for the restricted model
            hessian_rest = nd.Hessian(lambda th: nlogL(psi_val, th, data))(result.x)
            out['J_rest'][i] = det(hessian_rest)
            
            # Compute q and q_B values
            dphi_dth_rest = d_phi(out['th_rest'][i, :])
            coef = det(dphi_dth_rest) / D_bot
            Jeff_prior_rest = np.sqrt(det(np.dot(np.dot(dphi_dth_rest, data['Omega']), dphi_dth_rest.T)))
            dphi_dth_rest[:, 0] = phi(th_full) - phi(out['th_rest'][i, :])
            out['D_top'][i] = det(dphi_dth_rest)
            out['q'][i] = (out['D_top'][i] / D_bot) * np.sqrt(out['j_th_th'] / out['J_rest'][i])
            out['q_B'][i] = out['q'][i] * coef * (Jeff_prior_full / Jeff_prior_rest)

    # Compute r values
    out['r'] = np.sign(out['normal'][0] - out['th_rest'][:, 0]) * np.sqrt(2 * (out['L_full'] - out['L_rest']))

    # Filter out zero r values and corresponding elements
    non_zero_indices = out['r'] != 0
    out['r'] = out['r'][non_zero_indices]
    out['q'] = out['q'][non_zero_indices]
    out['q_B'] = out['q_B'][non_zero_indices]
    out['psi'] = out['psi'][non_zero_indices]

    # Computing rstar and rstar_B
    out['rstar'] = out['r'] + np.log(out['q'] / out['r']) / out['r']
    out['rstar_B'] = out['r'] + np.log(out['q_B'] / out['r']) / out['r']

    return out

