######################################################
########### Compute Pc in 2-dimensional setup ########
######################################################

# Importing standard and third-party libraries
import numpy as np
from scipy.linalg import norm, det, inv
from scipy.integrate import dblquad

def Pc2D_Foster(r, v, cov, HBR, HBRType, RelTol):
    """
    Pc2D_Foster - Computes 2D Pc according to Foster method. This function
                  supports three different types of hard body regions: 
                  'circle', 'square', and square equivalent to the area of 
                  the circle ('squareEquArea'). It also handles both 3x3 and
                  6x6 covariances but, by definition, the 2D Pc calculation
                  only uses the 3x3 position covariance.
    
    Syntax:       Pc = Pc2D_Foster(r, v, cov, HBR, RelTol, 'HBRType')
    
    Inputs:
        r - Primary-Secondary object's position vector in ECI coordinates (1x3 numpy array)
        v - Primary-Secondary object's velocity vector in ECI coordinates (1x3 numpy array)
        cov - Combined covariance matrix in ECI coordinate frame (3x3 or 6x6 numpy array)
        HBR - Hard body region
        RelTol - Tolerance used for double integration convergence (usually set to 1e-08)
        HBRType - Type of hard body region. One of: 'circle', 'square', 'squareEquArea'
    
    Outputs:
        Pc - Probability of collision
    """

    # Combined relative position uncertainty
    covcomb = cov[:3, :3]

    # Construct relative encounter frame
    h = np.cross(r, v)

    # Relative encounter frame
    y = v / norm(v)
    z = h / norm(h)
    x = np.cross(y, z)

    # Transformation matrix from ECI to relative encounter plane
    eci2xyz = np.stack([x, y, z], axis=0)

    # Transform combined ECI covariance into xyz
    covcombxyz = eci2xyz @ covcomb @ eci2xyz.T

    # Projection onto xz-plane in the relative encounter frame
    Mxz = np.array([[1, 0, 0], [0, 0, 1]])
    Cp = Mxz @ covcombxyz @ Mxz.T

    # Center of HBR in the relative encounter plane
    x0 = norm(r)
    z0 = 0

    # Inverse of the Cp matrix
    C = inv(Cp)

    # Integrand
    def integrand(z, x):
        return np.exp(-0.5 * (C[0, 0] * x**2 + C[0, 1] * x * z + C[1, 0] * z * x + C[1, 1] * z**2))

    # Depending on the type of hard body region, compute Pc
    AbsTol = 10**-13
    if HBRType == "circle":
        def lower_bound_y(x):
            return -np.sqrt(HBR**2 - (x - x0)**2) if abs(x - x0) <= HBR else 0

        def upper_bound_y(x):
            return np.sqrt(HBR**2 - (x - x0)**2) if abs(x - x0) <= HBR else 0

        xmin, xmax = x0 - HBR, x0 + HBR
        result, error = dblquad(integrand, xmin, xmax, lower_bound_y, upper_bound_y, epsrel=RelTol, epsabs=AbsTol)
    elif HBRType == "square":
        xa, xb = x0 - HBR, x0 + HBR
        result, error = dblquad(integrand, xa, xb, lambda x: z0 - HBR, lambda x: z0 + HBR, epsrel=RelTol, epsabs=AbsTol)
    elif HBRType == "squareEquArea":
        a = (np.sqrt(np.pi) / 2) * HBR
        result, error = dblquad(integrand, x0 - a, x0 + a, lambda x: z0 - a, lambda x: z0 + a, epsrel=RelTol, epsabs=AbsTol)
    else:
        raise ValueError("HBRType is not supported")

    # Probability of collision
    Pc = 1 / (2 * np.pi * np.sqrt(det(Cp))) * result
    return Pc

# =============================================================================
# # Example usage
# r = np.array([7.678, 9.152, 0.564])
# v = np.array([-9926.39283, 9653.043039, 4110.229766])
# cov = np.array([[6900.79792436, -4361.22404216, -2042.72823953],
#                 [-4361.22404216, 3161.73509484, 1361.3455951],
#                 [-2042.72823953, 1361.3455951, 757.7489808]])
# HBR = 5
# RelTol = 1e-8
# HBRType = "circle"
# 
# # Call the function
# Pc = Pc2D_Foster(r, v, cov, HBR, HBRType, RelTol)
# print("Probability of Collision using Foster method:", Pc)
# 
# 
# =============================================================================



def probabilityEllipse(x0, y0, a, b, nPart=100):
    """
    Computes the probability of a joint random point (x,y) falling within a described ellipse,
    where the point (x, y) is governed by a joint unit normal probability distribution.
    The semi-major and semi-minor axes of the ellipse are aligned with the cardinal directions of the plane.

    Inputs:
    a: semi-axis along x-dimension
    b: semi-axis along y-dimension
    x0, y0: coordinates of ellipse center
    nPart: Number of partitions for integration, default 100

    Outputs:
    pro_Ellipse: probability of random point (x, y) falling within the ellipse given the described distribution
    """

    nPart = max(nPart, int(10 * max(a / b, b / a)))  # Ensure sufficient resolution relative to the aspect ratio
    eps = np.linspace(0, 2 * np.pi, nPart + 1)  # Include endpoint to close the loop

    xx = x0 + a * np.cos(eps)
    yy = y0 + b * np.sin(eps)

    rrSqr = xx**2 + yy**2
    rSmallBool = (rrSqr <= 1e-4).astype(int)

    # Calculate the integral using the Trapezoidal Rule
    guts = ((1 - rSmallBool) * (1 - np.exp(-rrSqr / 2)) / (2 * np.pi * rrSqr + rSmallBool) + 
            rSmallBool * (0.5 - rrSqr / 8) / (2 * np.pi)) * (a * b + x0 * b * np.cos(eps) + y0 * a * np.sin(eps))
    pro_Ellipse = np.trapz(guts, dx=2 * np.pi / nPart)  # Use np.trapz for numerical integration

    return pro_Ellipse

# =============================================================================
# 
# # # Example usage
# # x0 = 0
# # y0 = 0
# # a = 1
# # b = 0.5
# # Pc2 = probabilityEllipse(x0, y0, a, b)
# # print("Probability of the point within the ellipse:", Pc2)
# =============================================================================
