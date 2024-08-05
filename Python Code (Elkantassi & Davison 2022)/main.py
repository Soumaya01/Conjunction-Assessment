# Importing standard and third-party libraries
import numpy as np
import os 
import sys

# Ensure the directory containing the custom modules is in the Python path
base_path = '/Users/selkanta/Library/CloudStorage/Dropbox/AFOSR project/Code (Python)'
sys.path.append(base_path)

# Importing custom modules for specific tasks related to 2D encounter projections and statistical analysis
from data_transformation import load_scenario, transform_data_to_ECI
from Inference2DProjection import EncounterProj, nlogL2D, gr_rest2D, j_theta2D, MLE2D, phi2D, d_phi2D
from tem_xD import tem_2D
from Pc_2D import Pc2D_Foster, probabilityEllipse
from plot_TEM import plot_prob, smooth_curves
from lik_ci import lik_ci
from scipy.stats import norm

######################################################################
######################### 2-dimensional setup ########################
######################################################################


# Define the full path to the scenario file
scenario_path = os.path.join(base_path, 'CARA scenarios', 'OmitronTestCase_Test01_HighPc.py')

# Load scenario and transform data to ECI (Earth-Centered Inertial coordinates)
eta, Omega = transform_data_to_ECI(scenario_path=scenario_path)
m = eta[:3]  # Relative position vector
v = eta[3:]  # Relative velocity vector

# Perform 2D transformation for encounter projection
Param2D = EncounterProj(m, v, Omega)
eta_p = Param2D['eta.p']
Omega_p = Param2D['Omega.p']

# Analyze the encounter with the TEM method
c = 1  # Coefficient to adjust the covariance matrix if necessary
data = {'w': eta_p, 'D': c * Omega_p}

# Compute TEM in 2D
out = tem_2D(psi=None, nlogL=nlogL2D, gr_rest=gr_rest2D, j_theta=j_theta2D, MLEs=MLE2D, th_init=None, phi=phi2D, d_phi=d_phi2D, data=data, delta=10, z=5, n_psi=100, low=-np.inf, up=np.inf, method='L-BFGS-B')

# Define quantiles for plotting horizontal lines based on the norm.ppf (percent point function or inverse of cdf)
q = [10**-5, 10**-4, 10**-3, 10**-2, 0.025, 0.05]
alpha = q + [0.5] + [1 - val for val in q]  # Quantiles including median and complementary probabilities

# Set psi0 to be the hypothesized value of psi, for example the HBR (Hard Body Radius) or another appropriate float depending on the context
psi0 = 5

# Plot the significance/pivots
plot_prob(out, psi0, q=alpha, prob=False, xlab=r'$\psi (m)$', rstarB=True)


h = 0.2  # Set your threshold based on the expected smoothness
smoothed_out = smooth_curves(out, h)
plot_prob(smoothed_out, psi0=5, q=alpha, prob=False, xlab=r'$\psi(m)$')

# Compute the p-values using r and rstar

conf_levels = [0.01, 0.05, 0.1]  # 95%, 99%, and 90% confidence intervals
CI = lik_ci(smoothed_out, conf_levels, psi0=5)

# Compute the probability of collision 
HBR = 5
Pc2D_Foster(m, v, cov=Omega, HBR=HBR, HBRType="circle", RelTol=10**-8)
probabilityEllipse(-eta_p[0] / np.sqrt(Omega_p[0, 0]), -eta_p[1] / np.sqrt(Omega_p[1, 1]), HBR / np.sqrt(Omega_p[0, 0]), HBR / np.sqrt(Omega_p[1, 1]), nPart=10**5)

######################################################################
######################### 6-dimensional setup ########################
######################################################################


from tem_xD import tem_6D 
from Inference6D import spher2cart, cart2spher, cos_incl, nlogL6D, gr_rest6D, MLE6D, phi6D, d_phi6D


# Define the primary and secondary object's position and velocity
# Referencing the U.S. and Russian satellite collision event (Case Study B in Elkantassi & Davison 2022)

Primary = np.array([-1457.273246, 1589.568484, 6814.189959])
V_Primary = np.array([-7.001731, -2.439512, -0.926209])
Secondary = np.array([-1457.532155, 1588.932671, 6814.316188])
V_Secondary = np.array([3.578705, -6.172896, 2.200215])

# Calculate relative position and velocity
m = Secondary - Primary
v = V_Secondary - V_Primary

# Convert Cartesian coordinates to spherical coordinates
pol_m = cart2spher(m)
pol_v = cart2spher(v)

# Calculate cosine of inclination
cos_i = cos_incl(pol_m, pol_v)
print("Cosine of the inclination:", cos_i)

# Set variances for position and velocity
sigma2_m = 10**-2
sigma2_v = 10**-3
tau = sigma2_m / sigma2_v
sigma2 = sigma2_v

# Define initial parameter estimates
theta = np.concatenate([[pol_m[0] * np.sqrt(1 - cos_i**2)], pol_m[1:], pol_v])
psi0 = theta[0]

# Setup data dictionary
Omega = np.diag(np.concatenate([np.full(3, tau * sigma2), np.full(3, sigma2)]))
data_orig = {'y': np.concatenate([m, v]), 'Omega': Omega}

# Bounds for optimization
low_bounds = [0, -np.pi, 0, 0, -np.pi]
up_bounds = [np.pi, np.pi, np.inf, np.pi, np.pi]

# Call the 6D TEM model
results = tem_6D(
    psi=None,
    nlogL=nlogL6D,
    gr_rest=gr_rest6D,
    MLEs=MLE6D,
    th_init=theta[1:],
    phi=phi6D,
    d_phi=d_phi6D,
    data=data_orig,
    delta=4 * 10**-3,
    z=5,
    n_psi=100,
    low=low_bounds,
    up=up_bounds,
    method="L-BFGS-B"
)

# Define quantiles for plotting horizontal lines based on the norm.ppf (percent point function or inverse of cdf)
q = [10**-5, 10**-4, 10**-3, 10**-2, 0.025, 0.05]
alpha = q + [0.5] + [1 - val for val in q]  # Quantiles including median and complementary probabilities

# Set psi0 to be the hypothesized value of psi
psi0 = 0.5

# Plot the significance/pivots
plot_prob(results, psi0, q=alpha, prob=False, xlab=r'$\psi (m)$', rstarB=True)
