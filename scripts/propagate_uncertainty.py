import numpy as np
import matplotlib.pyplot as plt

import sparselib

def dynamics(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return -np.sin(2 * x)

def uniform_uncertainty(x):
    """
    Initial Uniform uncertainty, independent of dimension
    
    :param      x:     Coordinates
    :type       x:     np.array
    :param      mu:    Optional mean
    :type       mu:    np.array

    :returns:   initial uncertainty at coords
    :rtype:     np.array
    """
    return np.sqrt(np.ones_like(x[:,0]) * 1 / (2 * np.pi))

def normalize(ys, N):
    return np.squeeze(ys / (2 * np.pi * np.sum(ys) / N))

class SolverParams():
	max_level: int = 6
	dim: int = 1
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [uniform_uncertainty, dynamics]


params = SolverParams()
specgal = sparselib.SpectralGalerkin(params)
specgal.solve(t=1.5)

# Plot the propagated uncertainty
N = 1000
xs = np.linspace(params.domain[0], params.domain[1], N)
xs = np.expand_dims(xs, axis=1)

# Grids[0] for uncertainty, grids[1] for vector field
interp = np.real(specgal.container.grids[0].eval(xs))

# Square the half-density if plotting uncertainty
interp = normalize(np.power(interp, 2), N)
plt.ylim(np.minimum(0, np.min(interp)), np.maximum(1, np.max(interp)))

plt.plot(xs, interp)
plt.xlabel("Domain")
plt.ylabel("Probability Density")
plt.show()