import numpy as np

# CARA Analysis Tool: OmitronTestCase_Test02_MaxRadialSigma.cdm
# MISS_DISTANCE = -91.455902 [m]

# Coordinates and velocities
mu1 = 10**3 * np.array([6703.392053, -1969.223265, 3276.274276])
nu1 = 10**3 * np.array([3.527732779, 2.322656571, -5.815904333])
mu2 = 10**3 * np.array([6703.301055, -1969.231944, 3276.271432])
nu2 = 10**3 * np.array([2.538967996, 6.496491661, 5.104787983])

# Object 1 covariance matrix
CR_R = 5.841000000000003e+01
CT_R = -9.881000000000006e+01
CT_T = 1.253000000000000e+03
CN_R = 1.875000000000004e+01
CN_T = 2.552999999999933e+00
CN_N = 6.209000000000003e+01
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
CR_R = 4.355000000000002e+07
CT_R = 1.242000000000000e+08
CT_T = 3.542999999999999e+08
CN_R = 2.325999999999982e+04
CN_T = 6.465999999997421e+04
CN_N = 8.801999999867694e+02
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


