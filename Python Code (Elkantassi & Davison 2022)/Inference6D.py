###############################################################################################################################
# Functions needed to call tem_6D, convert between spherical and Cartesian coordinates, compute the cosine of inclination
###############################################################################################################################

# Importing standard and third-party libraries
import numpy as np
from scipy.stats import multivariate_normal
 
def spher2cart(v):
    """
    Converts spherical coordinates to Cartesian coordinates.
    Args:
    - v (array): Spherical coordinates [r, theta, phi]
    
    Returns:
    - array: Cartesian coordinates [x, y, z]
    """
    r, theta, phi = v
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def cart2spher(v):
    """
    Converts Cartesian coordinates to spherical coordinates.
    Args:
    - v (array): Cartesian coordinates [x, y, z]
    
    Returns:
    - array: Spherical coordinates [r, theta, phi]
    """
    x, y, z = v
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])

def cos_incl(spher_x, spher_y):
    """
    Computes the cosine of the inclination between two vectors given in spherical coordinates.
    Args:
    - spher_x (array): Spherical coordinates of the first vector
    - spher_y (array): Spherical coordinates of the second vector
    
    Returns:
    - float: Cosine of the angle of inclination
    """
    sin_theta_x, cos_theta_x, sin_phi_x, cos_phi_x = np.sin(spher_x[1]), np.cos(spher_x[1]), np.sin(spher_x[2]), np.cos(spher_x[2])
    sin_theta_y, cos_theta_y, sin_phi_y, cos_phi_y = np.sin(spher_y[1]), np.cos(spher_y[1]), np.sin(spher_y[2]), np.cos(spher_y[2])
    cos_alpha = (sin_theta_x * cos_phi_x * sin_theta_y * cos_phi_y +
                 sin_theta_x * sin_phi_x * sin_theta_y * sin_phi_y +
                 cos_theta_x * cos_theta_y)
    return cos_alpha

def nlogL6D(psi, lam, data):
    """
    Computes the negative log-likelihood in 6D setup.
    
    Args:
    - psi (float): Parameter psi.
    - lam (array): Nuisance parameters.
    - data (dict): Contains 'y' and 'Omega', the data and its covariance matrix.
    
    Returns:
    - float: Negative log-likelihood value.
    """
    theta = np.concatenate([[psi], lam])
    eta = phi6D(theta)
    
    try:
        log_prob = multivariate_normal.logpdf(data['y'], mean=eta, cov=data['Omega'])
        return -log_prob
    except np.linalg.LinAlgError:
        return np.inf  # Handle the case where the covariance matrix is not invertible


def gr_rest6D(psi, lam, data):
    """
    Gradient of the restricted negative log-likelihood with respect to lambda.
    Args:
    - psi (float): Fixed parameter psi
    - lam (array): Variable parameters lambda
    - data (dict): Contains 'y' and 'Omega', the data and its covariance matrix
    
    Returns:
    - array: Gradient of the restricted negative log-likelihood
    """
    theta = np.concatenate([[psi], lam])
    grad_phi = d_phi6D(theta)[:, 1:]  # Exclude the derivative w.r.t psi
    eta = phi6D(theta)
    diff = data['y'] - eta
    inv_omega = np.linalg.inv(data['Omega'])
    return np.dot(inv_omega, diff).dot(grad_phi)

def MLE6D(data):
    """
    Computes Maximum Likelihood Estimates for 6D data assuming known velocity.
    Args:
    - data (dict): Contains 'y', the data vector
    
    Returns:
    - array: Estimated parameters
    """
    hat_m = data['y'][:3]  # First three entries for position
    hat_v = data['y'][3:]  # Next three entries for velocity
    pol_m = cart2spher(hat_m)
    pol_v = cart2spher(hat_v)
    cos_i = cos_incl(pol_m, pol_v)  
    theta_MLE = np.concatenate([[pol_m[0] * np.sqrt(1 - cos_i**2)], pol_m[1:], pol_v])
    return theta_MLE


def phi6D(theta):
    """
    Computes the canonical parameter of the exponential family in 6D setup.
    Args:
    - theta (array): Parameters array [psi, theta_1, varphi_1, nu, theta_2, varphi_2]

    Returns:
    - array: canonical parameter vector in 6D
    """
    psi, theta_1, varphi_1, nu, theta_2, varphi_2 = theta
    cos_x = (np.sin(theta_1) * np.cos(varphi_1) * np.sin(theta_2) * np.cos(varphi_2) +
             np.sin(theta_1) * np.sin(varphi_1) * np.sin(theta_2) * np.sin(varphi_2) +
             np.cos(theta_1) * np.cos(theta_2))
    rho = psi / np.sqrt(1 - cos_x**2)

    eta = np.array([rho * np.sin(theta_1) * np.cos(varphi_1),
                    rho * np.sin(theta_1) * np.sin(varphi_1),
                    rho * np.cos(theta_1),
                    nu * np.sin(theta_2) * np.cos(varphi_2),
                    nu * np.sin(theta_2) * np.sin(varphi_2),
                    nu * np.cos(theta_2)])
    return eta



def d_phi6D(theta):
    
    """
    Computes the derivative matrix (Jacobian) for the phi function in 6D.
    Args:
    - theta (array): parameter vector [psi, theta_1, varphi_1, nu, theta_2, varphi_2]
    
    Returns:
    - matrix: Derivative matrix (6x6)
    """
    psi, theta_1, varphi_1, nu, theta_2, varphi_2 = theta
    sin1, cos1 = np.sin(theta_1), np.cos(theta_1)
    sin2, cos2 = np.sin(theta_2), np.cos(theta_2)
    sinv1, cosv1 = np.sin(varphi_1), np.cos(varphi_1)
    sinv2, cosv2 = np.sin(varphi_2), np.cos(varphi_2)
   
    cos_x = sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2
    one_minus_sq_cos_x = 1 - cos_x**2
    factor = 1 / np.sqrt(one_minus_sq_cos_x)


    J = np.zeros((6, 6))
   
    # Fill in the first column: partial derivatives with respect to psi
    J[:, 0] = [sin1 * cosv1 * factor,
               sin1 * sinv1 * factor,
               cos1 * factor,
               0, 0, 0
               ]
    # Second column: partial derivatives with respect to theta_1
    J[:, 1] = [ psi * cos1 * cosv1 * factor + psi * sin1 * cosv1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (cos1 * cosv1 * sin2 * cosv2 + cos1 * sinv1 * sin2 * sinv2 - sin1 * cos2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1) 
               , psi * cos1 * sinv1 * factor + psi * sin1 * sinv1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (cos1 * cosv1 * sin2 * cosv2 + cos1 * sinv1 * sin2 * sinv2 - sin1 * cos2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1) 
               ,- psi * sin1 * factor + psi * cos1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (cos1 * cosv1 * sin2 * cosv2 + cos1 * sinv1 * sin2 * sinv2 - sin1 * cos2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1) 
               ,0,0,0]
    
    # Third column: partial derivatives with respect to varphi_1
    J[:, 2] = [- psi * sin1 * sinv1 * factor + psi * sin1 * cosv1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (-sin1 * sinv1 * sin2 * cosv2 + sin1 * cosv1 * sin2 * sinv2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1) 
               ,psi * sin1 * cosv1 * factor + psi * sin1 * sinv1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (-sin1 * sinv1 * sin2 * cosv2 + sin1 * cosv1 * sin2 * sinv2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1)
               ,psi * cos1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (-sin1 * sinv1 * sin2 * cosv2 + sin1 * cosv1 * sin2 * sinv2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1)
               ,0,0,0]


    # Fourth column: partial derivatives with respect to ||nu||
    J[:, 3] = [0,0,0
               ,sin2 * cosv2
               ,sin2 * sinv2
               ,cos2
               ]
    
    # Fifth column: derivatives with respect to theta_2
    J[:, 4] =[psi * sin1 * cosv1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (sin1 * cosv1 * cos2 * cosv2 + sin1 * sinv1 * cos2 * sinv2 - cos1 * sin2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1)
              ,psi * sin1 * sinv1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (sin1 * cosv1 * cos2 * cosv2 + sin1 * sinv1 * cos2 * sinv2 - cos1 * sin2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1)
              ,psi * cos1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (sin1 * cosv1 * cos2 * cosv2 + sin1 * sinv1 * cos2 * sinv2 - cos1 * sin2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1)
              ,nu * cos2 * cosv2
              ,nu * cos2 * sinv2
              ,-nu * sin2]
    
    # Sixth column: derivatives with respect to varphi_2
    J[:, 5] =[psi * sin1 * cosv1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (-sin1 * cosv1 * sin2 * sinv2 + sin1 * sinv1 * sin2 * cosv2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1)
              ,psi * sin1 * sinv1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (-sin1 * cosv1 * sin2 * sinv2 + sin1 * sinv1 * sin2 * cosv2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1)
              ,psi * cos1 * (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) * (-sin1 * cosv1 * sin2 * sinv2 + sin1 * sinv1 * sin2 * cosv2) * (1 - (sin1 * cosv1 * sin2 * cosv2 + sin1 * sinv1 * sin2 * sinv2 + cos1 * cos2) ** 2) ** (-0.3e1 / 0.2e1)
              ,-nu * sin2 * sinv2
              ,nu * sin2 * cosv2,0]
    
    
    return J



