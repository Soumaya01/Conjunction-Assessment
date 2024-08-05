###########################################################################################
# This script reproduces Figure 2 from Elkantassi & Davison 2022, showing the behavior of 
# the collision probability estimate as a function of the variance for a given hard-body radius (HBR).
# The figure depicts the results for HBR = 5 m (left) and 20 m (right) in case study C, with 
# ψ = 11.92 m, for various scaling factors.
###########################################################################################


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
from joblib import Parallel, delayed

# Ensure the Python interpreter knows where to find your custom modules
sys.path.append('/Users/selkanta/Library/CloudStorage/Dropbox/AFOSR project/Code (Python)')

# Import the function to transform data to ECI coordinates
from data_transformation import transform_data_to_ECI
from Inference2DProjection import EncounterProj
from Pc_2D import Pc2D_Foster, probabilityEllipse

# Define the full path to the scenario file
base_path = '/Users/selkanta/Library/CloudStorage/Dropbox/AFOSR project/Code (Python)'
scenario_path = os.path.join(base_path, 'CARA scenarios', 'OmitronTestCase_Test01_HighPc.py')

def simulate_data(eta_p, Omega_p, c, R, HBR):
    """
    Simulates data for the collision probability estimation.

    Args:
    - eta_p (array): Projected relative position vector.
    - Omega_p (array): Projected covariance matrix.
    - c (float): Scaling factor for the covariance matrix.
    - R (int): Number of Monte Carlo runs.
    - HBR (float): Hard-body radius.

    Returns:
    - list: Simulated probabilities.
    - float: True probability.
    """
    data_D = c * Omega_p
    probabilities = []
    for _ in range(R):
        w = np.random.multivariate_normal(mean=eta_p, cov=data_D)
        prob = probabilityEllipse(x0=-w[0]/np.sqrt(data_D[0, 0]), y0=-w[1]/np.sqrt(data_D[1, 1]),
                                  a=HBR/np.sqrt(data_D[0, 0]), b=HBR/np.sqrt(data_D[1, 1]))
        probabilities.append(prob)
    true_prob = probabilityEllipse(x0=-eta_p[0]/np.sqrt(data_D[0, 0]), y0=-eta_p[1]/np.sqrt(data_D[1, 1]),
                                   a=HBR/np.sqrt(data_D[0, 0]), b=HBR/np.sqrt(data_D[1, 1]))
    return probabilities, true_prob

# Load scenario and transform data to ECI
eta, Omega = transform_data_to_ECI(scenario_path=scenario_path)
m = eta[:3]  # Relative position vector
v = eta[3:]  # Relative velocity vector

# Perform 2D transformation for encounter projection
Param2D = EncounterProj(m, v, Omega)
eta_p = Param2D['eta.p']
Omega_p = Param2D['Omega.p']

# Define the range of scaling factors and hard-body radii to be used
c_values = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1, 2, 3, 4, 5, 6, 8, 9, 10]
HBR_values = [5, 20]
R = 2000  # Number of Monte Carlo runs

# Setup the figure and axes for the plots
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for i, HBR in enumerate(HBR_values):
    ax = axes[i]
    log_probs = []
    true_probs = []
    for c in c_values:
        probs, true_prob = simulate_data(eta_p, Omega_p, c, R, HBR)
        log_probs.append(np.log10(probs))
        true_probs.append(np.log10(true_prob))
    
    # Creating the boxplot
    bp = ax.boxplot(log_probs, patch_artist=True, notch=True, positions=range(len(c_values)), widths=0.6)
    
    # Styling the boxplots
    for patch in bp['boxes']:
        patch.set(facecolor='lightgrey')
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=2)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=2)
    
    # Adding segments for the true probabilities
    for pos, true_prob in zip(range(len(c_values)), true_probs):
        ax.plot([pos - 0.2, pos + 0.2], [true_prob, true_prob], 'b-', linewidth=2)
    
    # Setting the x-ticks and labels
    ax.set_xticks(range(len(c_values)))
    ax.set_xticklabels(c_values)
    ax.set_title(f'HBR = {HBR} m')
    ax.set_xlabel('Scaling Factor c')
    ax.set_ylabel('Log10(Probability)')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
