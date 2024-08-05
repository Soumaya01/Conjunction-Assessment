############################################################################################################
# This script reproduces Figure 3 from Elkantassi & Davison 2022, showing the encounter plane for case study C.
# The figure illustrates the distribution of points where a second object will traverse the plane, 
# and how these points compare to the Hard Body Radius (HBR) in determining collision probabilities.
############################################################################################################


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
from matplotlib.patches import Ellipse

# Ensure the Python interpreter knows where to find your custom modules
sys.path.append('/Users/selkanta/Library/CloudStorage/Dropbox/AFOSR project/Code (Python)')

# Import the function to transform data to ECI coordinates
from data_transformation import transform_data_to_ECI
from Inference2DProjection import EncounterProj
from Pc_2D import Pc2D_Foster, probabilityEllipse

# Define the full path to the scenario file
base_path = '/Users/selkanta/Library/CloudStorage/Dropbox/AFOSR project/Code (Python)'
scenario_path = os.path.join(base_path, 'CARA scenarios', 'OmitronTestCase_Test01_HighPc.py')

# Load scenario and transform data to ECI
eta, Omega = transform_data_to_ECI(scenario_path=scenario_path)
m = eta[:3]  # Relative position vector
v = eta[3:]  # Relative velocity vector

# Perform 2D transformation for encounter projection
Param2D = EncounterProj(m, v, Omega)
eta_p = Param2D['eta.p']
Omega_p = Param2D['Omega.p']

# Function to add ellipses to the plot
def add_ellipse(ax, m=np.array([0, 0]), r=1, D=np.array([1, 1]), label='', **kwargs):
    """
    Adds an ellipse to the matplotlib Axes object.
    
    Args:
    - ax: The matplotlib Axes object.
    - m (array): The center of the ellipse.
    - r (float): The scaling factor for the ellipse axes.
    - D (array): The array containing scaling factors for the ellipse.
    - label (str): The label for the ellipse.
    """
    a = r * D[0]
    b = r * D[1]
    ellipse = Ellipse(xy=m, width=2*a, height=2*b, edgecolor='black', facecolor='none', alpha=1, linewidth=1, **kwargs)
    ax.add_patch(ellipse)
    if label:
        ax.annotate(label, xy=(m[0], m[1] + b), xytext=(0, 10), textcoords='offset points', fontsize=12, ha='center')

# Plot setup for different scaling factors
def plot_samples(c, HBR_values, eta_p, Omega_p, np_samples=10000):
    """
    Plots the samples for different Hard Body Radii (HBR) and scaling factors.
    
    Args:
    - c (float): The scaling factor for the covariance matrix.
    - HBR_values (list): List of Hard Body Radii to plot.
    - eta_p (array): Projected relative position vector.
    - Omega_p (array): Projected covariance matrix.
    - np_samples (int): Number of samples to generate for the plot.
    """
    w = np.random.multivariate_normal(mean=eta_p, cov=c*Omega_p, size=np_samples)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, (ax, HBR) in enumerate(zip(axes, HBR_values)):
        sns.scatterplot(x=w[:, 0], y=w[:, 1], ax=ax, palette="Blues", sizes=(20, 200))
        ax.scatter(eta_p[0], eta_p[1], color='red', s=100, label='Estimated Position')
        ax.quiver(0, 0, eta_p[0], eta_p[1], color='red', scale_units='xy', scale=1, width=0.005, label='Psi Vector')
        ellipse = Ellipse(xy=(0, 0), width=2*HBR, height=2*HBR, edgecolor='navy', facecolor='none', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.add_patch(ellipse)
        buffer = max(HBR, np.max(np.abs(w)) * 1.1)
        ax.set_xlim(-buffer, buffer)
        ax.set_ylim(-buffer, buffer)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_title(f"HBR={HBR} (m)")
    
    plt.tight_layout()
    plt.show()

# Parameters for the plot
HBR_values = [5, 20]
np_samples = 10000

# Plot for c = 0.01
plot_samples(c=0.01, HBR_values=HBR_values, eta_p=eta_p, Omega_p=Omega_p, np_samples=np_samples)

# Plot for c = 1
plot_samples(c=1, HBR_values=HBR_values, eta_p=eta_p, Omega_p=Omega_p, np_samples=np_samples)

# Plot for c = 2
plot_samples(c=2, HBR_values=HBR_values, eta_p=eta_p, Omega_p=Omega_p, np_samples=np_samples)
