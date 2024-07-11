import numpy as np
from tqdm import tqdm
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
	return np.power(np.exp(2*t) * np.power(np.sin(x), 2) + \
					  np.exp(-2*t) * np.power(np.cos(x), 2), -1)

def normalize(ys, N):
	return np.squeeze(ys / (2 * np.pi * np.sum(ys) / N))


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


def Lp_error_vs_time():
	# Initialize solver parameters
	paramsDense = SolverParamsDense()
	paramsSparse = SolverParamsSparse()

	# Standard Galerkin method
	specgalDense = sparselib.SpectralGalerkin(paramsDense)

	# Our sparse method
	specgalSparse = sparselib.SpectralGalerkin(paramsSparse)

	# Evaluate results
	N = 1000
	xs = np.linspace(paramsDense.domain[0], paramsDense.domain[1], N)
	xs = np.expand_dims(xs, axis=1)

	# Compute the propagated uncertainty for our proposed sparse, half-density
	# method, a standard Galerkin approach, and the ground-truth distribution.
	interpDense = normalize(np.real(specgalDense.container.grids[0].eval(xs)), N)
	interpSparse = normalize(np.power(np.real(specgalSparse.container.grids[0].eval(xs)), 2), N)
	gt = normalize(ground_truth(0, xs), N)

	L1sparse = []
	L2sparse = []
	Linfsparse = []

	L1dense = []
	L2dense = []
	Linfdense = []

	L1sparse.append(np.sum(2 * np.pi * np.abs(interpSparse - gt) / N))
	L2sparse.append(np.sqrt(np.sum(2 * np.pi * np.power(interpSparse - gt, 2)) / N))
	Linfsparse.append(np.max(np.abs(interpSparse - gt)))

	L1dense.append(np.sum(2 * np.pi * np.abs(interpDense - gt) / N))
	L2dense.append(np.sqrt(np.sum(2 * np.pi * np.power(interpDense - gt, 2)) / N))
	Linfdense.append(np.max(np.abs(interpDense - gt)))

	total_time = 1.5
	M = 100
	t = 0
	dt = total_time / M

	print("Computing Lp errors ...")
	pbar = tqdm(total=M)
	for i in range(M):
		t += dt
		specgalSparse.solve(dt)
		specgalDense.solve(dt)

		interpDense = normalize(np.real(specgalDense.container.grids[0].eval(xs)), N)
		interpSparse = normalize(np.power(np.real(specgalSparse.container.grids[0].eval(xs)), 2), N)
		gt = normalize(ground_truth(t, xs), N)

		L1sparse.append(np.sum(2 * np.pi * np.abs(interpSparse - gt) / N))
		L2sparse.append(np.sqrt(np.sum(2 * np.pi * np.power(interpSparse - gt, 2)) / N))
		Linfsparse.append(np.max(np.abs(interpSparse - gt)))

		L1dense.append(np.sum(2 * np.pi * np.abs(interpDense - gt) / N))
		L2dense.append(np.sqrt(np.sum(2 * np.pi * np.power(interpDense - gt, 2)) / N))
		Linfdense.append(np.max(np.abs(interpDense - gt)))

		pbar.update(1)
	pbar.close()

	L1sparse = np.array(L1sparse)
	L2sparse = np.array(L2sparse)
	Linfsparse = np.array(Linfsparse)

	L1dense = np.array(L1dense)
	L2dense = np.array(L2dense)
	Linfdense = np.array(Linfdense)

	ts = np.linspace(0, total_time, M+1)

	plt.plot(ts, L1sparse, 'b-')
	plt.plot(ts, L1dense, 'b--')
	plt.plot(ts, L2sparse, 'r-')
	plt.plot(ts, L2dense, 'r--')
	plt.plot(ts, Linfsparse, 'k-')
	plt.plot(ts, Linfdense, 'k--')
	plt.xlabel("Time [s]")
	plt.ylabel("Lp Error")
	plt.gca().legend(('L1 (Ours)','L1 (Galerkin)', 
					  'L2 (Ours)', 'L2 (Galerkin)', 
					  'Linf (Ours)', 'Linf (Galerkin)'))
	plt.show()

def Lp_Error_vs_num_bases():
	N = 1000
	xs = np.linspace(0, 2*np.pi, N)
	xs = np.expand_dims(xs, axis=1)

	L1sparse = []
	L2sparse = []
	Linfsparse = []

	L1dense = []
	L2dense = []
	Linfdense = []

	num_bases = []

	t = 1.0

	pbar = tqdm(total=2)
	for i in range(1, 9):
		# Initialize solver parameters
		paramsDense = SolverParamsDense()
		paramsSparse = SolverParamsSparse()

		paramsSparse.max_level = int(i)
		paramsDense.max_level = int(i)

		# Standard Galerkin method
		specgalDense = sparselib.SpectralGalerkin(paramsDense)
		specgalDense.solve(t)

		# Our sparse method
		specgalSparse = sparselib.SpectralGalerkin(paramsSparse)
		specgalSparse.solve(t)

		interpDense = normalize(np.real(specgalDense.container.grids[0].eval(xs)), N)
		interpSparse = normalize(np.power(np.real(specgalSparse.container.grids[0].eval(xs)), 2), N)
		gt = normalize(ground_truth(t, xs), N)

		L1sparse.append(np.sum(2 * np.pi * np.abs(interpSparse - gt) / N))
		L2sparse.append(np.sqrt(np.sum(2 * np.pi * np.power(interpSparse - gt, 2)) / N))
		Linfsparse.append(np.max(np.abs(interpSparse - gt)))

		L1dense.append(np.sum(2 * np.pi * np.abs(interpDense - gt) / N))
		L2dense.append(np.sqrt(np.sum(2 * np.pi * np.power(interpDense - gt, 2)) / N))
		Linfdense.append(np.max(np.abs(interpDense - gt)))

		num_bases.append(specgalSparse.container.grids[0].N)

		pbar.update(1)
	pbar.close()

	L1sparse = np.array(L1sparse)
	L2sparse = np.array(L2sparse)
	Linfsparse = np.array(Linfsparse)

	L1dense = np.array(L1dense)
	L2dense = np.array(L2dense)
	Linfdense = np.array(Linfdense)

	num_bases = np.array(num_bases)

	plt.plot(num_bases, L1sparse, 'b-')
	plt.plot(num_bases, L1dense, 'b--')
	plt.plot(num_bases, L2sparse, 'r-')
	plt.plot(num_bases, L2dense, 'r--')
	plt.plot(num_bases, Linfsparse, 'k-')
	plt.plot(num_bases, Linfdense, 'k--')
	plt.xlabel("Number of Basis Functions")
	plt.ylabel("Lp Error")
	plt.xscale('log')
	plt.yscale('log')
	plt.gca().legend(('L1 (Ours)','L1 (Galerkin)', 
					  'L2 (Ours)', 'L2 (Galerkin)', 
					  'Linf (Ours)', 'Linf (Galerkin)'))
	plt.show()


Lp_error_vs_time()
Lp_Error_vs_num_bases()