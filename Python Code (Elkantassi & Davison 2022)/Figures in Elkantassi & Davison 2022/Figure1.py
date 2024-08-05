###########################################################################################
# This script reproduces Figure 1 from Elkantassi & Davison 2022, depicting the statistical 
# formulation of satellite conjunction in the encounter plane. The figure shows the primary 
# object at the origin and the secondary object's true and observed positions, along with 
# density ellipses.
###########################################################################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm

def add_ellipse(ax, m=np.array([0, 0]), r=1, D=np.array([1, 1]), label='', **kwargs):
    """
    Adds an ellipse to the matplotlib Axes object.
    """
    theta = np.linspace(0, 2 * np.pi, 201)
    a = r * D[0]
    b = r * D[1]
    ellipse = Ellipse(xy=m, width=2*a, height=2*b, edgecolor='black', facecolor='none', alpha=1, linewidth=1, **kwargs)
    ax.add_patch(ellipse)
    if label:
        # Positioning the label at the top of the ellipse
        ax.annotate(label, xy=(m[0], m[1] + b), xytext=(0, 10), textcoords='offset points', fontsize=12, ha='center')

# Setup plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Parameters
mu = np.array([1, 2])
d = 2 * np.array([0.3, 0.6])
y = mu + 2 * np.array([0.3, 0.1])

# Grid for density
x1 = np.linspace(-1, 4, 300)
x2 = np.linspace(-1, 4, 300)
X1, X2 = np.meshgrid(x1, x2)
Z = norm.pdf(X1, mu[0], d[0]) * norm.pdf(X2, mu[1], d[1])

# Plotting the density
for ax in axes:
    ax.imshow(Z, extent=(-1, 4, -1, 4), origin='lower', cmap='Greys', alpha=0.5)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)

    # Points and labels directly at their locations
    ax.plot(mu[0], mu[1], 's', markersize=8)
    ax.text(mu[0], mu[1] + 0.2, r'$\xi$', fontsize=12, ha='center')
    ax.text(mu[0]-1, 0.6, r'$\psi_{\min}$', fontsize=12, ha='center')
    ax.plot(y[0], y[1], 'o', markersize=8)
    ax.text(y[0], y[1] - 0.3, r'$x$', fontsize=12, ha='center')

    # Psi and Lambda lines
    ax.plot([0, mu[0]], [0, mu[1]], 'k--', linewidth=2)
    ax.plot([0, 2], [0, 0], 'k-', linewidth=2)
    ax.annotate(r'$\lambda$', xy=(0.3, 0.1), fontsize=12, ha='center')

    # Adding ellipses with respective annotations
    add_ellipse(ax, m=mu, r=0.6, D=d)
    add_ellipse(ax, m=mu, r=1.5, D=d)
    add_ellipse(ax, r=0.5)
    add_ellipse(ax, r=np.sqrt(5), linestyle='--')

plt.tight_layout()
plt.show()
