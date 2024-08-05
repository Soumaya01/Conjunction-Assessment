# Project Overview

This document serves as a guide for using a collection of Python scripts designed for the analysis of space collision avoidance. The toolkit focuses on computational methods to simulate orbital encounters using real or hypothetical data, transform the data into Earth-Centered Inertial (ECI) coordinates, and compute relative distances and velocities between space objects, along with their covariance matrices. It also projects the dynamics of these encounters onto a 2D encounter plane for simplified examination, computes collision probabilities, evaluates the statistical significance of close encounters (p-values under the null hypothesis of minimal miss distance), and generates detailed visualizations to interpret analytical outcomes effectively.

## Detailed Description of Modules and Functions

To elaborate on each function within the modules and provide a clearer understanding of their roles, we can expand the "Project Overview" section into a more detailed description.

### `data_transformation.py`
- `load_scenario(scenario_path)`: Loads data from specified scenario files, transforming them into a format suitable for further processing.
- `transform_data_to_ECI(scenario_path)`: Converts the loaded scenario data into Earth-Centered Inertial (ECI) coordinates, computing relative positions, velocities, and covariance matrices.

### `Inference2DProjection.py`
- `EncounterProj(m, v, Omega)`: Projects 3D motion vectors (position `m` and velocity `v`) and covariance matrices (`Omega`) onto a 2D plane. This simplification aids in further analysis and visualization.
- `RTN_to_ECI(r, v)`: Transforms position and velocity vectors from Radial, Tangential, Normal (RTN) coordinates to Earth-Centered Inertial (ECI) coordinates. This function provides the transformation matrix necessary for this conversion.
- `nlogL2D(psi, lam, data)`: Calculates the negative log-likelihood for 2D projected data.
- `MLE2D(data)`: Computes Maximum Likelihood Estimates (MLE) for 2D data, providing parameter estimates that best fit the data under the assumed model.
- `gr_rest2D(psi, lam, data)`: Computes the gradient of the likelihood function for a restricted model in 2D, which is useful for optimization and finding maximum likelihood estimates.
- `j_theta2D(theta, data)`: Calculates the Jacobian matrix of the likelihood function for the 2D data, important for understanding how changes in parameters affect the transformed data.
- `phi2D(theta)`: Defines the canonical parameter function for the exponential family in 2D.
- `d_phi2D(theta)`: Provides the derivative of the canonical parameter of the local exponential family.

### `Inference6D.py`
- `spher2cart(v)`: Converts spherical coordinates [r, theta, phi] to Cartesian coordinates [x, y, z].
- `cart2spher(v)`: Converts Cartesian coordinates [x, y, z] to spherical coordinates [r, theta, phi].
- `cos_incl(spher_x, spher_y)`: Computes the cosine of the inclination between two vectors in spherical coordinates.
- `nlogL6D(psi, lam, data)`: Calculates the negative log-likelihood for a 6D setup.
- `gr_rest6D(psi, lam, data)`: Computes the gradient of the restricted negative log-likelihood.
- `MLE6D(data)`: Computes Maximum Likelihood Estimates (MLE) for 6D data.
- `phi6D(theta)`: Defines the canonical parameter for the exponential family in 6D.
- `d_phi6D(theta)`: Computes the Jacobian matrix of the phi function in 6D.

### `Pc_2D.py`
- `probabilityEllipse(x0, y0, a, b, nPart=100)`: Computes the probability of a joint random point (x, y) falling within a described ellipse. The semi-major and semi-minor axes of the ellipse are aligned with the cardinal directions of the plane, see Balch, Martin, and Ferson (2018).
- `Pc2D_Foster(r, v, cov, HBR, HBRType, RelTol)`: Implements an alternative method for calculating 2D collision probabilities, according to the method of Foster.

### `tem_xD.py`
- `tem_2D(...)`, `tem_3D(...)`, `tem_6D(...)`: Applies the tangent exponential model across two, three, and six dimensions. These functions compute first-order pivots like the likelihood root and the Wald statistic, and a third-order pivot which is the modified likelihood root. In two dimensions, the function also offers a Bayesian counterpart of the modified likelihood root using a uniform prior. These pivots are used to test the null hypothesis that the miss distance is below a threshold value, aiding in the assessment of collision risks.

### `plot_TEM.py`
- `plot_prob(...)`: Visualizes the evidence function (1 - significance function) and the pivots as functions of the miss distance parameter. This function provides graphical representations that help interpret the analytical outcomes, making it easier to understand the implications of the modeled encounter scenarios.
- `smooth_curves(out, h)`: Removes singularities from the likelihood and modified likelihood roots (`r` and `rstar`) using smoothing splines for monotonic interpolation. This function enhances the smoothness and accuracy of pivot plots by handling failed fits and ensuring continuity in the displayed curves.

### `lik_ci.py`
- `lik_ci(...)`: Generates likelihood-based confidence intervals for the miss distance using various pivots, supporting the assessments of the statistical significance of findings obtained using `tem_xD.py`.

### Subfolders
- `/CARA scenarios/`: This folder contains predefined scenarios available for analysis. The scenarios are adapted from the open-source NASA repository, which provides data for simulation and testing purposes.

- `/figures/`: This folder is designated for storing output figures generated during the analysis. It includes visualizations cited in the 2023 article by Elkantassi and Davison.

### Scripts
- `main.py`: The main executable script that uses all the above modules to perform the entire analysis process, from data loading and transformation to computing collision probabilities, conducting statistical analysis, and plotting the results.
