# ############################################################
# # Function to compute likelihood confidence intervals or tests
# # from a Fraser-Reid object-like structure, focusing on p-values
# # and 95% confidence intervals based on Wald, r, and r*.
# ############################################################
#
# # Importing necessary libraries
# from scipy.interpolate import interp1d
# from scipy.stats import norm
# import numpy as np


# def lik_ci(data, conf=[0.025, 0.975], psi0=None):
#     """
#     Compute and return likelihood confidence intervals or tests from a Fraser-Reid object-like structure.
    
#     Args:
#     - data (dict): Dictionary containing a grid of psi and the corresponding values for the pivots: w, r, rstar.
#     - conf (list): List of confidence levels (default is [0.025, 0.975] for 95% confidence interval).
#     - psi0 (float, optional): Value of psi under the null hypothesis.
    
#     Returns:
#     - dict: Contains point estimates, confidence intervals, and possibly p-values if psi0 is provided.
#     """
#     def make_unique(x, y):
#         """
#         Ensure x values are unique by averaging the corresponding y values.
        
#         Args:
#         - x (array-like): Array of x values.
#         - y (array-like): Array of y values.
        
#         Returns:
#         - tuple: Unique x values and their corresponding averaged y values.
#         """
#         unique_x = {}
#         for xi, yi in zip(x, y):
#             if xi in unique_x:
#                 unique_x[xi].append(yi)
#             else:
#                 unique_x[xi] = [yi]
#         unique_x = {k: np.mean(v) for k, v in unique_x.items()}
#         return np.array(list(unique_x.keys())), np.array(list(unique_x.values()))
    
#     # Preprocess data to ensure unique r and rstar values
#     unique_r, unique_psi_r = make_unique(data['r'], data['psi'])
#     unique_rstar, unique_psi_rstar = make_unique(data['rstar'], data['psi'])
    
#     # Fitting interpolation functions using interp1d
#     fit_r = interp1d(unique_r, unique_psi_r, fill_value="extrapolate", kind='linear')
#     fit_rstar = interp1d(unique_rstar, unique_psi_rstar, fill_value="extrapolate", kind='linear')
    
#     # Inverse CDF (Quantile function) for the given confidence levels
#     quantiles = norm.ppf(conf)
    
#     # Confidence intervals based on r and rstar
#     r_lims = fit_r(quantiles)
#     rstar_lims = fit_rstar(quantiles)
    
#     # Point estimates
#     pointEst_z = data['normal'][0]
#     pointEst_r = fit_r(0)
#     pointEst_rstar = fit_rstar(0)

#     results = {
#         'pointEst_z': pointEst_z,
#         'pointEst_r': pointEst_r,
#         'pointEst_rstar': pointEst_rstar,
#         'z_lims': data['normal'][0] - quantiles * data['normal'][1],
#         'r_lims': r_lims,
#         'rstar_lims': rstar_lims
#     }

#     if psi0 is not None:
#         # P-value calculations if psi0 is provided
#         fit0_r = interp1d(data['psi'], data['r'], fill_value="extrapolate", kind='linear')
#         fit0_rstar = interp1d(data['psi'], data['rstar'], fill_value="extrapolate", kind='linear')
#         p0_r = fit0_r(psi0)
#         p0_rstar = fit0_rstar(psi0)
#         p0_w = (data['normal'][0] - psi0) / data['normal'][1]
#         p_values = 1 - norm.cdf([p0_w, p0_r, p0_rstar])
        
#         results['p_values'] = p_values
#         print("P-values for Wald, r, r*: ", p_values)

#     # Print the desired information
#     print("Point estimates: Wald =", pointEst_z, ", r =", pointEst_r, ", r* =", pointEst_rstar)
#     print("95% Confidence Intervals:")
#     print("Wald: [", results['z_lims'][0], ",", results['z_lims'][1], "]")
#     print("r: [", results['r_lims'][0], ",", results['r_lims'][1], "]")
#     print("r*: [", results['rstar_lims'][0], ",", results['rstar_lims'][1], "]")

#     return results

# # Example usage:
# # data = {
# #     'psi': np.linspace(0, 10, 100),
# #     'r': np.sin(np.linspace(-3, 3, 100)),
# #     'rstar': np.cos(np.linspace(-4, 5, 100)),
# #     'normal': [5, 1]
# # }
# # conf = [0.025, 0.975]
# # psi0 = 2
# # results = lik_ci(data, conf, psi0=psi0)
# # print(results)


############################################################
# Function to compute likelihood confidence intervals or tests
# from a Fraser-Reid object-like structure.
############################################################

# Importing necessary libraries
from scipy.interpolate import interp1d
from scipy.stats import norm
import numpy as np

def lik_ci(data, conf_levels, psi0=None):
    """
    Compute and return likelihood confidence intervals or tests from a Fraser-Reid object-like structure.
    
    Args:
    - data (dict): Dictionary containing a grid of psi and the corresponding values for the pivots: w, r, rstar.
    - conf_levels (list): List of significance levels (e.g., [0.01, 0.05, 0.1]).
    - psi0 (float, optional): Value of psi under the null hypothesis.
    
    Returns:
    - dict: Contains point estimates, confidence intervals, and possibly p-values if psi0 is provided.
    """
    # Ensure 'r' and 'rstar' are sorted in increasing order for interpolation
    sorted_indices_r = np.argsort(data['r'])
    sorted_indices_rstar = np.argsort(data['rstar'])
    
    # Interpolation functions using interp1d
    fit_r = interp1d(data['r'][sorted_indices_r], data['psi'][sorted_indices_r], fill_value="extrapolate", kind='linear')
    fit_rstar = interp1d(data['rstar'][sorted_indices_rstar], data['psi'][sorted_indices_rstar], fill_value="extrapolate", kind='linear')
    
    # Compute the quantiles for each significance level
    quantiles = {level: norm.ppf([(1 - level / 2), (level / 2)]) for level in conf_levels}
    
    # Confidence intervals based on r and rstar
    conf_intervals = {}
    for level, q in quantiles.items():
        z_lims = np.clip(data['normal'][0] - q * data['normal'][1], 0, None)
        r_lims = np.clip(fit_r(q), 0, None)
        rstar_lims = np.clip(fit_rstar(q), 0, None)
        conf_intervals[level] = {
            'z_lims': z_lims,
            'r_lims': r_lims,
            'rstar_lims': rstar_lims
        }
    
    # Point estimates
    pointEst_z = data['normal'][0]
    pointEst_r = fit_r(0)
    pointEst_rstar = fit_rstar(0)

    results = {
        'pointEst_z': pointEst_z,
        'pointEst_r': pointEst_r,
        'pointEst_rstar': pointEst_rstar,
        'conf_intervals': conf_intervals
    }

    if psi0 is not None:
        # Evaluate r and rstar at psi0
        r_psi0 = interp1d(data['psi'], data['r'], fill_value="extrapolate", kind='linear')(psi0)
        rstar_psi0 = interp1d(data['psi'], data['rstar'], fill_value="extrapolate", kind='linear')(psi0)
        
        # P-value calculations if psi0 is provided
        p_w = (data['normal'][0] - psi0) / data['normal'][1]
        p_r = r_psi0
        p_rstar = rstar_psi0
        p_values = 1 - norm.cdf([p_w, p_r, p_rstar])
        
        results['p_values'] = p_values
        print("P-values for Wald, r, r*: ", p_values)

    # Print multiple confidence intervals
    for level, intervals in conf_intervals.items():
        print(f"{int((1 - level) * 100)}% Confidence Intervals:")
        print("Wald statistic, z           :", intervals['z_lims'])
        print("Likelihood root, r          :", intervals['r_lims'])
        print("Modified likelihood root, r*:", intervals['rstar_lims'])

    return results

# # Example usage:
# data = {
#     'psi': np.linspace(0, 10, 100),
#     'r': np.sin(np.linspace(-3, 3, 100)),
#     'rstar': np.cos(np.linspace(-4, 5, 100)),
#     'normal': [5, 1]
# }
# conf_levels = [0.01, 0.05, 0.1]  # 99%, 95%, and 90% confidence intervals
# psi0 = 2
# results = lik_ci(data, conf_levels, psi0=psi0)
# print(results)
