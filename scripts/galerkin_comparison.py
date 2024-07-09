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

def uniform_dense(x):
    """
    Initial Uniform uncertainty, independent of dimension
    
    :param      x:     Coordinates
    :type       x:     np.array
    :param      mu:    Optional mean
    :type       mu:    np.array

    :returns:   initial uncertainty at coords
    :rtype:     np.array
    """
    return np.ones_like(x[:,0]) * 1 / (2 * np.pi)

def uniform_sparse(x):
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

def ground_truth(t, x):
	return np.power(np.exp(2*time) * np.power(np.sin(x), 2) + \
                      np.exp(-2*time) * np.power(np.cos(x), 2), -1)

def normalize(ys, N):
	return np.squeeze(ys / np.sum(ys) / N)


class SolverParamsDense():
	max_level: int = 6
	dim: int = 1
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [uniform_dense, dynamics]


class SolverParamsSparse():
	max_level: int = 6
	dim: int = 1
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [uniform_sparse, dynamics]


# Initialize solver parameters
paramsDense = SolverParamsDense()
paramsSparse = SolverParamsSparse()

# Standard Galerkin method
specgalDense = sparselib.SpectralGalerkin(paramsDense)
specgalDense.solve(t=1.5)

# Our sparse method
specgalSparse = sparselib.SpectralGalerkin(paramsSparse)
specgalSparse.solve(t=1.5)

# Evaluate results
N = 1000
xs = np.linspace(paramsDense.domain[0], paramsDense.domain[1], N)
xs = np.expand_dims(xs, axis=1)

# Compute the propagated uncertainty for our proposed sparse, half-density
# method, a standard Galerkin approach, and the ground-truth distribution.
interpDense = normalize(np.real(specgalDense.container.grids[0].eval(xs)), N)
interpSparse = normalize(np.power(np.real(specgalDense.container.grids[0].eval(xs)), 2), N)
gt = normalize(ground_truth(t=1.5, xs), N)


plt.plot(xs, interpSparse)
plt.plot(xs, interpDense)
plt.plot(xs, gt)
plt.xlabel("Domain")
plt.ylabel("Probability Density")
plt.gca().legend(('Half-Densities','standard Galerkin', 'Ground Truth'))
plt.ylim(np.minimum(0, np.min(interpDense)), np.maximum(1, np.max(interpDense)))
plt.show()