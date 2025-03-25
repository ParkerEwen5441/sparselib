import scipy
import sparselib
import numpy as np
import matplotlib.pyplot as plt


def dyn(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return x

def uncertainty(x):
    """
    Initial Uniform uncertainty, independent of dimension
    
    :param      x:     Coordinates
    :type       x:     np.array
    :param      mu:    Optional mean
    :type       mu:    np.array

    :returns:   initial uncertainty at coords
    :rtype:     np.array
    """
    return np.ones((x.shape[0]))

class SolverParams():
	max_level: int = 5
	dim: int = 6
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [uncertainty, dyn, dyn, dyn, dyn, dyn, dyn]


params = SolverParams()
specgal = sparselib.SpectralGalerkin(params, logging=True)