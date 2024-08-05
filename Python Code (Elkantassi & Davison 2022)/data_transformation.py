#################################################################################
# Functions to load scenario data and transform it from RTN to ECI coordinates.
#################################################################################

import numpy as np
from Inference2DProjection import RTN_to_ECI

def load_scenario(scenario_path):
    """ Load scenario configuration. 
    - scenario_path (str): Path to the scenario file containing initial conditions and data.
     # Specify the path to the scenario file. Uncomment the line corresponding to the scenario you wish to analyze.
     # scenario_path = 'CARA scenarios/OmitronTestCase_Test01_HighPc.py'       
     # scenario_path = 'CARA scenarios/OmitronTestCase_Test02_MaxRadialSigma.py' 
     # scenario_path = 'CARA scenarios/OmitronTestCase_Test03_MaxIntrackSigma.py' 
     # scenario_path = 'CARA scenarios/OmitronTestCase_Test04_NonPDCovariance.py' 
     # scenario_path = 'CARA scenarios/OmitronTestCase_Test05_MinMiss.py'        
     # scenario_path = 'CARA scenarios/OmitronTestCase_Test06_MinRelVel.py'      
     # scenario_path = 'CARA scenarios/OmitronTestCase_Test07_HighInclination.py' 

    Returns:
    - tuple: Contains the relative position and velocity vector `eta` and the covariance matrix `Omega`.
    """
    
    try:
        with open(scenario_path, 'r') as file:
            scenario_data = {}
            exec(file.read(), scenario_data)
        return scenario_data
    except FileNotFoundError:
        raise Exception(f"File not found: {scenario_path}")


def transform_data_to_ECI(scenario_path):
    """
    Load data from a specified scenario file, compute relative motion parameters,
    and calculate the covariance matrix of the relative motion in the ECI coordinate frame.
    
    Args:
    - scenario_path (str): Path to the scenario file containing initial conditions and data.
    
    Returns:
    - tuple: Contains the relative position and velocity vector `eta` and the covariance matrix `Omega`.
    """
    # Load the scenario
    data = load_scenario(scenario_path)
    mu1, nu1, mu2, nu2 = data['mu1'], data['nu1'], data['mu2'], data['nu2']
    C1_RTN, C2_RTN = data['C1_RTN'], data['C2_RTN']

    # Compute relative position and velocity
    eta = np.concatenate((mu1 - mu2, nu1 - nu2))

    # Transform RTN to ECI for both objects and compute the ECI covariance matrices
    M1 = RTN_to_ECI(mu1, nu1)
    M2 = RTN_to_ECI(mu2, nu2)
    C1_ECI = M1 @ C1_RTN @ M1.T
    C2_ECI = M2 @ C2_RTN @ M2.T

    # Compute the covariance matrix of the relative motion
    Omega = C1_ECI[:3, :3] + C2_ECI[:3, :3]

    return eta, Omega