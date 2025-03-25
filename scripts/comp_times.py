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

sparse_files = ['sparse_times_2D.npy', 'sparse_times_3D.npy']
dense_files = ['dense_times_2D.npy', 'dense_times_3D.npy']

# sparse_files = ['sparse_times_2D.npy']
# dense_files = ['dense_times_2D.npy']

colors = [['#05445e', '#fc2e20'],
		  ['#189ab4', '#fd7f20']]
linestyle = ['-', '-']
dense_labels = ['Galerkin', 'Galerkin']
sparse_labels = ['Sparse (Ours)', 'Sparse (Ours)']

for idx, _ in enumerate(sparse_files):
	sparse = np.load(sparse_files[idx])[:5,:]
	dense = np.load(dense_files[idx])[:5,:]

	# Compute the mean and 3-sigma deviation for each case
	sparse_mean = np.mean(sparse, axis=1)
	sparse_std = np.std(sparse, axis=1)
	sparse_upper = sparse_mean + 3 * sparse_std
	sparse_lower = sparse_mean - 3 * sparse_std

	dense_mean = np.mean(dense, axis=1)
	dense_std = np.std(dense, axis=1)
	dense_upper = dense_mean + 3 * dense_std
	dense_lower = dense_mean - 3 * dense_std

	for i in np.arange(dense.shape[0]):
		sparse_lower[i] = np.maximum(sparse_lower[i], np.min(sparse[i,:]))
		dense_lower[i] = np.maximum(dense_lower[i], np.min(dense[i,:]))

		sparse_upper[i] = np.minimum(sparse_upper[i], np.max(sparse[i,:]))
		dense_upper[i] = np.minimum(dense_upper[i], np.max(dense[i,:]))

	# Create the x-axis values (case indices)
	x = np.arange(1, dense.shape[0] + 1)

	# Plot the results
	matplotlib.rcParams.update({'font.size': 18})
	fig, ax = plt.subplots(figsize=(18, 6))
	ax.plot(x, sparse_mean, ls='-', color=colors[1][0], solid_capstyle='round', label=sparse_labels[idx])
	ax.fill_between(x, sparse_lower, sparse_upper, color=colors[1][0], alpha=0.3)
	ax.plot(x, dense_mean, ls='--', color=colors[1][1], solid_capstyle='round', label=dense_labels[idx])
	ax.fill_between(x, dense_lower, dense_upper, color=colors[1][1], alpha=0.3)

	# matplotlib.rcParams.update({'font.size': 18})
	ax.legend()
	ax.set_yscale('log')
	ax.set_xlabel('Maximum Grid Level')
	ax.set_ylabel('Computation Time [s]')
	ax.set_xticks(np.linspace(1, 5, 5))
	fig.tight_layout()

# Show the plot
plt.show()
