import time
import sparselib
import numpy as np
import matplotlib
import s3dlib.surface as s3d
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from matplotlib import cm, colorbar
from scipy.integrate import odeint
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator


def dynamics1(x):
	"""
	Dynamics for x
	
	:param      x:    Coordinates
	:type       x:    np.array
	
	:returns:   vector field at coords
	:rtype:     np.array
	"""
	return 1.0 * np.sin(x[:,2]) + \
			0.2 * np.cos(x[:,1]) + \
			0.5 * np.cos(x[:,0])

def dynamics2(x):
	"""
	Dynamics for x
	
	:param      x:    Coordinates
	:type       x:    np.array
	
	:returns:   vector field at coords
	:rtype:     np.array
	"""
	return 0.5 * np.sin(x[:,2]) + \
			1.0 * np.cos(x[:,1]) + \
			0.5 * np.cos(x[:,1])

def dynamics3(x):
	"""
	Dynamics for x
	
	:param      x:    Coordinates
	:type       x:    np.array
	
	:returns:   vector field at coords
	:rtype:     np.array
	"""
	return 1.0 * np.sin(x[:,2]) + \
			0.5 * np.cos(x[:,1]) + \
			0.5 * np.cos(x[:,2])

def gaussian_uncertainty_sparse(x):
	"""
	Initial Uniform uncertainty, independent of dimension
	
	:param      x:     Coordinates
	:type       x:     np.array
	:param      mu:    Optional mean
	:type       mu:    np.array

	:returns:   initial uncertainty at coords
	:rtype:     np.array
	"""
	mu = np.array([np.pi, np.pi, np.pi])
	cov = np.array([0.2, 0.3, 0.3])

	vals = np.ones((x.shape[0]))
	for d in range(x.shape[1]):
		vals *= 1 / np.sqrt(2 * np.pi * cov[d]) * np.exp(-0.5 * \
					np.power(x[:,d] - mu[d], 2) / cov[d])

	return np.sqrt(vals)

def gaussian_uncertainty_dense(x):
	"""
	Initial Uniform uncertainty, independent of dimension
	
	:param      x:     Coordinates
	:type       x:     np.array
	:param      mu:    Optional mean
	:type       mu:    np.array

	:returns:   initial uncertainty at coords
	:rtype:     np.array
	"""
	mu = np.array([np.pi, np.pi, np.pi])
	cov = np.array([0.2, 0.3, 0.3])

	vals = np.ones((x.shape[0]))
	for d in range(x.shape[1]):
		vals *= 1 / np.sqrt(2 * np.pi * cov[d]) * np.exp(-0.5 * \
					np.power(x[:,d] - mu[d], 2) / cov[d])

	return vals


class SolverParamsSparse():
	max_level: int = 7
	dim: int = 3
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [gaussian_uncertainty_sparse, dynamics1, dynamics2, dynamics3]

class SolverParamsDense():
	max_level: int = 7
	dim: int = 3
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [gaussian_uncertainty_dense, dynamics1, dynamics2, dynamics3]
	dense: bool = True

# paramsSparse = SolverParamsSparse()
paramsDense = SolverParamsDense()

# sparse = np.zeros((8,100))
dense = np.zeros((6,100))
levels = np.arange(1, 6)
for idx, level in enumerate(levels):
	print('At level ', level, '...')
	
	pbar = tqdm(total=100)
	for i in range(0,100):
		# paramsSparse.max_level = level
		paramsDense.max_level = level

		# start = time.time()
		# specgalSparse = sparselib.SpectralGalerkin(paramsSparse, logging=False)
		# specgalSparse.solve(t=1.5)
		# end = time.time()
		# sparse[idx, i] = end - start

		start = time.time()
		specgalDense = sparselib.SpectralGalerkin(paramsDense)
		specgalDense.solve(t=1.5)
		end = time.time()
		dense[idx, i] = end - start
		pbar.update(1)
	pbar.close()


# np.save('sparse_times_3D.npy', sparse)
np.save('dense_times_3D.npy', dense)