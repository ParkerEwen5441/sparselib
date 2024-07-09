import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import pinv, expm

import sparselib

class SpectralGalerkin:
    def __init__(self, params, logging=False):
        """
        Class for spectral Galerkin method.

        ...

        Attributes
        ----------

        Methods
        -------

        """
        self.logging = logging

        self.container = sparselib.GridContainer(params)
        self.container.fit(params)

        self.spgridUncertainty = container.grids[0]
        self.spgridVectorField = container.grids[1:]

        if not os.path.exists(sparselib.config.get_L_matrix_file_path()) and not os.path.exists(sparselib.config.get_A_matrix_file_path()):
            print("Precomputed matrices not found.")
            print("Please run build_matrices() from main.py ...")
            return

        self.compute_galerkin_matrix()

    def compute_galerkin_matrix(self):
        """
        Calculates the galerkin matrix.
        """
        if sparselib.config.should_build_L_matrix():
            L = np.load(sparselib.config.get_L_matrix_file_path())

        # Matrices containing sparse grid coefficients
        D1 = np.zeros((self.spgridUncertainty.N, self.spgridUncertainty.domain.dim))
        if self.spgridUncertainty.domain.dim == 1:
            D1 = self.spgridVectorField[0].weights
        else:
            for l in range(self.spgridUncertainty.domain.dim):
                dl1 = self.spgridVectorField[l].weights
                D1[:, l] = dl1

        self.A = np.load(sparselib.config.get_A_matrix_file_path())
        np.save(sparselib.config.get_D1_matrix_file_path(), D1)
        print('Building B matrix...')
        if sparselib.config.should_build_cpp_B_matrix() and not os.path.exists(sparselib.config.get_cpp_B_matrix_file_path()):
            build_matrices_pybind.build_B_matrix(
                sparselib.config.get_cpp_B_matrix_file_path(),
                sparselib.config.get_D1_matrix_file_path(),
                sparselib.config.get_table1_file_path(),
                sparselib.config.get_table2_file_path(), 
                sparselib.config.get_global_dimwise_indices_file_path(), 
                sparselib.config.get_scaling_file_path(), 
                self.spgridUncertainty.domain.dim, 
                self.spgridUncertainty.N,
            )
        if sparselib.config.should_build_python_B_matrix() and not os.path.exists(sparselib.config.get_python_B_matrix_file_path()):
            print('L sparsity', np.sum(np.abs(L) == 0) / L.size)
            print('D1 sparsity', np.sum(np.abs(D1) == 0) / D1.size)

            B_py = np.einsum('ijkl,kl->ji', L, D1)
            np.save(sparselib.config.get_python_B_matrix_file_path(), B_py)
            print('B_py shape', B_py.shape)

        print('Loading B matrix...')
        self.B = np.load(sparselib.config.get_B_matrix_file_path())
        print('Loaded B matrix...')

    def solve(self, t):
        """
        Propagates the probability density approximation coefficients using the
        spectral Galerkin approach.
        
        :param      t:    Propagation time
        :type       t:    float
        """
        if self.logging:
            print("Propagating...")
            t0 = time.time()

        # Get the weights of the Fourier basis
        coeffs = self.spgridUncertainty.weights

        # Compute Galerkin matrix
        G = -(self.B) / (2 * np.pi)

        # Solve ODE
        new_coeffs = expm(G * t) @ np.array(coeffs)

        # Set new coeffs as spgrid weights
        self.spgridUncertainty.weights = new_coeffs

        if self.logging:
            t1 = time.time()
            print('Propagation time: ', t1 - t0)

    def compare(self, time, M=100):
        N = 1000
        xs = np.linspace(self.spgridUncertainty.domain.bounds[0,0], 
                         self.spgridUncertainty.domain.bounds[0,1], 
                         N)
        xs = np.expand_dims(xs, axis=1)

        interp = np.power(np.real(self.spgridUncertainty.eval(xs)), 2)
        normalize = np.sum(interp) / N
        interp = np.squeeze(interp / normalize)

        gt = np.power(np.exp(2*time) * np.power(np.sin(xs), 2) + \
                      np.exp(-2*time) * np.power(np.cos(xs), 2), -1)
        normalize = np.sum(interp) / N
        gt = np.squeeze(gt / normalize)

        L1s = []
        L2s = []
        Linfs = []

        L1s.append(np.sum(2 * np.pi * np.abs(interp - gt) / N))
        L2s.append(np.sqrt(np.sum(2 * np.pi * np.power(interp - gt, 2)) / N))
        Linfs.append(np.max(np.abs(interp - gt)))

        t = 0
        dt = time / M

        print("Computing Lp errors ...")
        pbar = tqdm(total=M)
        for i in range(M):
            t += dt
            self.propagate(dt)

            interp = np.power(np.real(self.spgridUncertainty.eval(xs)), 2)
            normalize = np.sum(interp) / N
            interp = np.squeeze(interp / normalize)

            gt = np.power(np.exp(2*t) * np.power(np.sin(xs), 2) + \
                          np.exp(-2*t) * np.power(np.cos(xs), 2), -1)
            normalize = np.sum(interp) / N
            gt = np.squeeze(gt / normalize)

            L1s.append(np.sum(2 * np.pi * np.abs(interp - gt) / N))
            L2s.append(np.sqrt(np.sum(2 * np.pi * np.power(interp - gt, 2)) / N))
            Linfs.append(np.max(np.abs(interp - gt)))

            pbar.update(1)
        pbar.close()

        L1s = np.array(L1s)
        L2s = np.array(L2s)
        Linfs = np.array(Linfs)

        ts = np.linspace(0, time, M)

        plt.plot(ts, L1s[1:])
        plt.plot(ts, L2s[1:])
        plt.plot(ts, Linfs[1:])
        plt.gca().legend(('L1','L2', 'Linf'))
        plt.ylabel("Lp Error")
        plt.xlabel("Time [s]")
        plt.show()

        plt.plot(xs, interp)
        plt.plot(xs, gt)
        plt.xlabel("Domain")
        plt.ylabel("Probability Density")
        plt.title("Uncertainty Propagation t=1.5s")
        plt.gca().legend(('Sparse Spectral Approximation','Ground Truth'))
        plt.show()