###########################################################################################
#Functions needed to call tem_2D and transform the data for 2D encounter projection analysis.
###########################################################################################


# Importing standard and third-party libraries
import numpy as np
from scipy.linalg import block_diag, norm
from scipy.spatial.transform import Rotation as R


def RTN_to_ECI(r, v):
    """
    Transforms position and velocity vectors from RTN to ECI coordinate system.
    Args:
    - r (array): Position vector in RTN coordinates.
    - v (array): Velocity vector in RTN coordinates.
    Returns:
    - array: Block diagonal transformation matrix converting RTN to ECI.
    """
    rECI = np.array(r)
    vECI = np.array(v)
    R = rECI / norm(rECI)
    W = np.cross(rECI, vECI)
    W = W / norm(W)
    S = np.cross(W, R)
    B = np.column_stack((R, S, W))
    return block_diag(B, B)

def EncounterProj(m, v, Omega):
    """
    Projects motion vectors and covariance into a plane perpendicular to the velocity for 2D analysis.
    Args:
    - m (array): Relative position vector.
    - v (array): Relative velocity vector.
    - Omega (matrix): Covariance matrix of the relative motion.
    Returns:
    - dict: Dictionary with projected eta, Omega, and transformation matrix A.
    """
    v1, v2, v3 = v
    b1 = np.array([0, v3, -v2])
    b2 = np.array([v2**2 + v3**2, -v1*v2, -v1*v3])
    b3 = np.array(v)
    B = np.column_stack((b1, b2, b3))
    N = np.diag(1 / np.linalg.norm(B, axis=0))
    BN = B @ N
    C = BN[:, :-1]
    
    # Eigen decomposition of the projected covariance matrix
    CTC = C.T @ Omega @ C
    eigenvalues, eigenvectors = np.linalg.eig(CTC)
    
    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    
    # Manually calculate eigenvectors
    eigenvectors = []
    for lamb in eigenvalues:
        # Solve (A - λI)v = 0
        matrix = CTC - lamb * np.eye(2)
        # Find the null space (eigenvector corresponding to the eigenvalue)
        vector = np.linalg.svd(matrix)[0][1]  # Extracting the last singular vector
        eigenvectors.append(vector / np.linalg.norm(vector))  # Normalize

    eigenvectors= np.array(eigenvectors).T
    
    V = eigenvectors #x[1]
    DD = eigenvalues #x[0]
    M = C @ V
    eta_p = M.T @ m
    Omega_p = M.T @ Omega[:3, :3] @ M
    out2D = {'eta.p': eta_p, 'Omega.p': Omega_p, 'A': np.column_stack((M, v / np.linalg.norm(v)))}
    return out2D

def nlogL2D(psi, lam, data):
    """
    Computes the negative log-likelihood for 2D data.
    Args:
    - psi (float): Scalar parameter psi.
    - lam (float): Angle parameter lambda.
    - data (dict): Contains observed data.
    Returns:
    - float: Computed negative log-likelihood value.
    """
    mu_1 = psi * np.cos(lam)
    mu_2 = psi * np.sin(lam)
    out = 0.5 * ((data['w'][0] - mu_1)**2 / data['D'][0, 0] + (data['w'][1] - mu_2)**2 / data['D'][1, 1])
    return out

def MLE2D(data):
    """
    Computes the Maximum Likelihood Estimates for 2D data.
    Args:
    - data (dict): Contains observed data.
    Returns:
    - array: Estimated parameters (theta).
    """
    hat_theta = np.array([np.sqrt(data['w'][0]**2 + data['w'][1]**2), np.arctan2(data['w'][1], data['w'][0])])
    return hat_theta

def gr_rest2D(psi, lam, data):
    """
    Gradient of the likelihood function for restricted 2D model.
    Args:
    - psi (float): Scalar parameter psi.
    - lam (float): Angle parameter lambda.
    - data (dict): Contains observed data.
    Returns:
    - float: Gradient at lambda.
    """
    mu_1 = psi * np.cos(lam)
    mu_2 = psi * np.sin(lam)
    gr_lam = mu_2 * (data['w'][0] - mu_1) / data['D'][0, 0] - mu_1 * (data['w'][1] - mu_2) / data['D'][1, 1]
    return gr_lam

def j_theta2D(theta, data):
    """
    Jacobian matrix of the transformation for 2D data.
    Args:
    - theta (array): Parameter vector.
    - data (dict): Contains observed data.
    Returns:
    - matrix: Jacobian matrix.
    """
    psi, lam = theta
    a = np.cos(lam)**2 / data['D'][0, 0] + np.sin(lam)**2 / data['D'][1, 1]
    b = (psi * (np.cos(lam) * data['w'][0] - psi * np.cos(2 * lam)) / data['D'][0, 0] + 
         psi * (np.sin(lam) * data['w'][1] + psi * np.cos(2 * lam)) / data['D'][1, 1])
    c = (np.sin(lam) * data['w'][0] / data['D'][0, 0] - np.cos(lam) * data['w'][1] / data['D'][1, 1] +
         psi * np.sin(2 * lam) * (1 / data['D'][1, 1] - 1 / data['D'][0, 0]))
    J = np.array([[a, c], [c, b]])
    return J

def phi2D(theta):
    """
    Canonical parameter function for the exponential family in 2D.
    Args:
    - theta (array): Parameter vector.
    Returns:
    - array: Canonical parameter vector.
    """
    psi, lam = theta
    return np.array([psi * np.cos(lam), psi * np.sin(lam)])

def d_phi2D(theta):
    """
    Derivative of the phi function for 2D data.
    Args:
    - theta (array): Parameter vector.
    Returns:
    - matrix: Derivative matrix.
    """
    psi, lam = theta
    cg = np.array([[np.cos(lam), -psi * np.sin(lam)], [np.sin(lam), psi * np.cos(lam)]])
    return cg
