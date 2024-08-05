import numpy as np

# Cara Analysis Tool: OmitronTestCase_Test01_HighPc
# MISS_DISTANCE = 11.959493 [m]

# Coordinates and velocities
mu1 = 10**3 * np.array([-1818.269382, 1040.563930, -6772.707308])
nu1 = 10**3 * np.array([-3.609802798, 6.269245755, 1.933211527])
mu2 = 10**3 * np.array([-1818.277060, 1040.554778, -6772.707872])
nu2 = 10**3 * np.array([6.316590032, -3.383797284, -2.177018239])

# Object 1 covariance matrix
CR_R = 1.858000000000000e+01
CT_R = -1.744999999999664e-01
CT_T = 1.190000000000000e+03
CN_R = 2.086000000000027e+00
CN_T = 1.275000000000094e+00
CN_N = 3.391999999999984e+00
CRDOT_R = 0.000000000000000e+00
CRDOT_T = 0.000000000000000e+00
CRDOT_N = 0.000000000000000e+00
CRDOT_RDOT = 0.000000000000000e+00
CTDOT_R = 0.000000000000000e+00
CTDOT_T = 0.000000000000000e+00
CTDOT_N = 0.000000000000000e+00
CTDOT_RDOT = 0.000000000000000e+00
CTDOT_TDOT = 0.000000000000000e+00
CNDOT_R = 0.000000000000000e+00
CNDOT_T = 0.000000000000000e+00
CNDOT_N = 0.000000000000000e+00
CNDOT_RDOT = 0.000000000000000e+00
CNDOT_TDOT = 0.000000000000000e+00
CNDOT_NDOT = 0.000000000000000e+00

C1_RTN = np.array([
    [CR_R, CT_R, CN_R, CRDOT_R, CTDOT_R, CNDOT_R],
    [CT_R, CT_T, CN_T, CRDOT_T, CTDOT_T, CNDOT_T],
    [CN_R, CN_T, CN_N, CRDOT_N, CTDOT_N, CNDOT_N],
    [CRDOT_R, CRDOT_T, CRDOT_N, CRDOT_RDOT, CTDOT_RDOT, CNDOT_RDOT],
    [CTDOT_R, CTDOT_T, CTDOT_N, CTDOT_RDOT, CTDOT_TDOT, CNDOT_TDOT],
    [CNDOT_R, CNDOT_T, CNDOT_N, CNDOT_RDOT, CNDOT_TDOT, CNDOT_NDOT]
])

# Object 2 covariance matrix
CR_R = 1.406000000000001e+02
CT_R = -5.145999999999999e+02
CT_T = 9.417000000000000e+03
CN_R = -2.912999999999997e+01
CN_T = 3.658000000000002e+02
CN_N = 5.071000000000074e+01
CRDOT_R = 0.000000000000000e+00
CRDOT_T = 0.000000000000000e+00
CRDOT_N = 0.000000000000000e+00
CRDOT_RDOT = 0.000000000000000e+00
CTDOT_R = 0.000000000000000e+00
CTDOT_T = 0.000000000000000e+00
CTDOT_N = 0.000000000000000e+00
CTDOT_RDOT = 0.000000000000000e+00
CTDOT_TDOT = 0.000000000000000e+00
CNDOT_R = 0.000000000000000e+00
CNDOT_T = 0.000000000000000e+00
CNDOT_N = 0.000000000000000e+00
CNDOT_RDOT = 0.000000000000000e+00
CNDOT_TDOT = 0.000000000000000e+00
CNDOT_NDOT = 0.000000000000000e+00

C2_RTN = np.array([
    [CR_R, CT_R, CN_R, CRDOT_R, CTDOT_R, CNDOT_R],
    [CT_R, CT_T, CN_T, CRDOT_T, CTDOT_T, CNDOT_T],
    [CN_R, CN_T, CN_N, CRDOT_N, CTDOT_N, CNDOT_N],
    [CRDOT_R, CRDOT_T, CRDOT_N, CRDOT_RDOT, CTDOT_RDOT, CNDOT_RDOT],
    [CTDOT_R, CTDOT_T, CTDOT_N, CTDOT_RDOT, CTDOT_TDOT, CNDOT_TDOT],
    [CNDOT_R, CNDOT_T, CNDOT_N, CNDOT_RDOT, CNDOT_TDOT, CNDOT_NDOT]
])


