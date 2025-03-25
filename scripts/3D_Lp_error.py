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


def torusFunc(rtz) :
    r,t,z = rtz
    Z = 0.45 * np.sin(z*np.pi)
    R = r + 0.45 * np.cos(z*np.pi)
    return R,t,Z

def dynamics1(x, c=1):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return -np.sin(2 * x[:,0])

def dynamics2(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return -np.sin(2 * x[:,1])

def dynamics3(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return -np.sin(2 * x[:,2])

def init_uncertainty_sparse(x):
    """
    Initial Uniform uncertainty, independent of dimension
    
    :param      x:     Coordinates
    :type       x:     np.array

    :returns:   initial uncertainty at coords
    :rtype:     np.array
    """
    return np.sqrt(np.ones_like(x[:,0]) / np.power(2 * np.pi, 3))

def init_uncertainty_dense(x):
    """
    Initial Uniform uncertainty, independent of dimension
    
    :param      x:     Coordinates
    :type       x:     np.array

    :returns:   initial uncertainty at coords
    :rtype:     np.array
    """
    return np.ones_like(x[:,0]) / np.power(2 * np.pi, 3)

def exact_solution(x, t):
    vals1 = np.power(np.exp(2*t) * np.power(np.sin(x[:,0]), 2) + \
                      np.exp(-2*t) * np.power(np.cos(x[:,0]), 2), -1)
    vals2 = np.power(np.exp(2*t) * np.power(np.sin(x[:,1]), 2) + \
                      np.exp(-2*t) * np.power(np.cos(x[:,1]), 2), -1)
    vals3 = np.power(np.exp(2*t) * np.power(np.sin(x[:,2]), 2) + \
                      np.exp(-2*t) * np.power(np.cos(x[:,2]), 2), -1)
    return (vals1 * vals2 * vals3) / np.power(2 * np.pi, 3)

class SolverParamsSparse():
	max_level: int = 5
	dim: int = 3
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [init_uncertainty_sparse, dynamics1, dynamics2, dynamics3]


class SolverParamsDense():
    max_level: int = 5
    dim: int = 3
    domain: np.ndarray = np.array([0, 2*np.pi])
    funcs: list = [init_uncertainty_dense, dynamics1, dynamics2, dynamics3]
    dense: bool = True


def Lp_error_vs_time():
    # Standard Galerkin method
    paramsDense = SolverParamsDense()
    specgalDense = sparselib.SpectralGalerkin(paramsDense)

    # Our sparse method
    paramsSparse = SolverParamsSparse()
    specgalSparse = sparselib.SpectralGalerkin(paramsSparse)

    # Evaluate results
    M = 100
    nLevel = [np.linspace(paramsSparse.domain[0], paramsSparse.domain[1], M),
              np.linspace(paramsSparse.domain[0], paramsSparse.domain[1], M),
              np.linspace(paramsSparse.domain[0], paramsSparse.domain[1], M)]
    coordinates = np.array(np.meshgrid(*nLevel)).T.reshape(-1, 3)

    # Compute the propagated uncertainty for our proposed sparse, half-density
    # method, a standard Galerkin approach, and the ground-truth distribution.
    interpSparse = np.power(np.real(specgalSparse.eval(coordinates, container_id=0)), 2)
    interpDense = np.real(specgalDense.eval(coordinates, container_id=0))
    interpExact = exact_solution(coordinates, 0)

    L1sparse = []
    L2sparse = []
    Linfsparse = []

    L1dense = []
    L2dense = []
    Linfdense = []

    L1sparse.append(np.sum(np.abs(interpSparse - interpExact) / interpExact.size))
    L2sparse.append(np.sqrt(np.sum(np.power(interpSparse - interpExact, 2)) / interpExact.size))
    Linfsparse.append(np.max(np.abs(interpSparse - interpExact)))

    L1dense.append(np.sum(np.abs(interpDense - interpExact) / interpExact.size))
    L2dense.append(np.sqrt(np.sum(np.power(interpDense - interpExact, 2)) / interpExact.size))
    Linfdense.append(np.max(np.abs(interpDense - interpExact)))

    total_time = 1.0
    M = 100
    t = 0
    dt = total_time / M

    print("Computing Lp errors ...")
    pbar = tqdm(total=M)
    for i in range(M):
        t += dt
        specgalSparse.solve(dt)
        specgalDense.solve(dt)

        interpSparse = np.power(np.real(specgalSparse.eval(coordinates, container_id=0)), 2)
        interpDense = np.real(specgalDense.eval(coordinates, container_id=0))
        interpExact = exact_solution(coordinates, t)

        L1sparse.append(np.sum(np.abs(interpSparse - interpExact) / interpExact.size))
        L2sparse.append(np.sqrt(np.sum(np.power(interpSparse - interpExact, 2)) / interpExact.size))
        Linfsparse.append(np.max(np.abs(interpSparse - interpExact)))

        L1dense.append(np.sum(np.abs(interpDense - interpExact) / interpExact.size))
        L2dense.append(np.sqrt(np.sum(np.power(interpDense - interpExact, 2)) / interpExact.size))
        Linfdense.append(np.max(np.abs(interpDense - interpExact)))

        pbar.update(1)
    pbar.close()

    L1sparse = np.array(L1sparse)
    L2sparse = np.array(L2sparse)
    Linfsparse = np.array(Linfsparse)

    L1dense = np.array(L1dense)
    L2dense = np.array(L2dense)
    Linfdense = np.array(Linfdense)

    ts = np.linspace(0, total_time, M+1)

    np.save('L1sparse.npy', L1sparse)
    np.save('L2sparse.npy', L2sparse)
    np.save('Linfsparse.npy', Linfsparse)
    np.save('L1dense.npy', L1dense)
    np.save('L2dense.npy', L2dense)
    np.save('Linfdense.npy', Linfdense)

    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(ts, L1sparse, color='#05445e', linestyle='-', marker='^', markevery=10, markersize=10)
    ax.plot(ts, L2sparse, color='#189ab4', linestyle='-', marker='v', markevery=10, markersize=10)
    ax.plot(ts, Linfsparse, color='#75e6da', linestyle='-', marker='*', markevery=10, markersize=10)
    ax.plot(ts, L1dense, color='#fc2e20',  linestyle='--', marker='^', markevery=10, markersize=10)
    ax.plot(ts, L2dense, color='#fd7f20', linestyle='--', marker='v', markevery=10, markersize=10)
    ax.plot(ts, Linfdense, color='#fdb750', linestyle='--', marker='*', markevery=10, markersize=10)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$L^p$-norm Error")
    ax.set_xlim([0, total_time])
    ax.set_ylim([0, 1.0])
    plt.gca().legend(('$L^1$-norm (Ours)',
                      '$L^2$-norm (Ours)', 
                      '$L^\\infty$-norm (Ours)', 
                      '$L^1$-norm (Galerkin)', 
                      '$L^2$-norm (Galerkin)', 
                      '$L^\\infty$-norm (Galerkin)'))
    fig.tight_layout()
    plt.show()


def Lp_Error_vs_num_bases():
    # Initialize solver parameters
    paramsDense = SolverParamsDense()
    paramsSparse = SolverParamsSparse()

    M = 100
    nLevel = [np.linspace(paramsSparse.domain[0], paramsSparse.domain[1], M),
              np.linspace(paramsSparse.domain[0], paramsSparse.domain[1], M),
              np.linspace(paramsSparse.domain[0], paramsSparse.domain[1], M)]
    coordinates = np.array(np.meshgrid(*nLevel)).T.reshape(-1, 3)

    L1sparse = []
    L2sparse = []
    Linfsparse = []

    L1dense = []
    L2dense = []
    Linfdense = []

    num_bases_sparse = []
    num_bases_dense = []

    t = 1.5

    pbar = tqdm(total=4)
    for i in range(1, 5):
        paramsSparse.max_level = int(i)
        paramsDense.max_level = int(i)

        # Standard Galerkin method
        specgalDense = sparselib.SpectralGalerkin(paramsDense)
        specgalDense.solve(t)

        # Our sparse method
        specgalSparse = sparselib.SpectralGalerkin(paramsSparse)
        specgalSparse.solve(t)

        interpSparse = np.power(np.real(specgalSparse.eval(coordinates, container_id=0)), 2)
        interpDense = np.real(specgalDense.eval(coordinates, container_id=0))
        interpExact = exact_solution(coordinates, t)

        L1sparse.append(np.sum(np.abs(interpSparse - interpExact) / interpExact.size))
        L2sparse.append(np.sqrt(np.sum(np.power(interpSparse - interpExact, 2)) / interpExact.size))
        Linfsparse.append(np.max(np.abs(interpSparse - interpExact)))

        L1dense.append(np.sum(np.abs(interpDense - interpExact) / interpExact.size))
        L2dense.append(np.sqrt(np.sum(np.power(interpDense - interpExact, 2)) / interpExact.size))
        Linfdense.append(np.max(np.abs(interpDense - interpExact)))

        num_bases_sparse.append(specgalSparse.container.grids[0].sparseGrid.shape[0])
        num_bases_dense.append(specgalDense.container.grids[0].sparseGrid.shape[0])

        pbar.update(1)
    pbar.close()

    # pbar = tqdm(total=2)
    # for i in range(5, 7): 
    #     paramsSparse.max_level = int(i)
    #     specgalSparse = sparselib.SpectralGalerkin(paramsSparse)
    #     specgalSparse.solve(t)

    #     interpSparse = np.power(np.real(specgalSparse.eval(coordinates, container_id=0)), 2)
    #     interpExact = exact_solution(coordinates, t)

    #     L1sparse.append(np.sum(np.abs(interpSparse - interpExact) / interpExact.size))
    #     L2sparse.append(np.sqrt(np.sum(np.power(interpSparse - interpExact, 2)) / interpExact.size))
    #     Linfsparse.append(np.max(np.abs(interpSparse - interpExact)))

    #     num_bases_sparse.append(specgalSparse.container.grids[0].sparseGrid.shape[0])

    #     pbar.update(1)
    # pbar.close()

    L1sparse = np.array(L1sparse)
    L2sparse = np.array(L2sparse)
    Linfsparse = np.array(Linfsparse)

    L1dense = np.array(L1dense)
    L2dense = np.array(L2dense)
    Linfdense = np.array(Linfdense)

    num_bases_sparse = np.array(num_bases_sparse)
    num_bases_dense = np.array(num_bases_dense)

    np.save('L1sparse.npy', L1sparse)
    np.save('L2sparse.npy', L2sparse)
    np.save('Linfsparse.npy', Linfsparse)
    np.save('L1dense.npy', L1dense)
    np.save('L2dense.npy', L2dense)
    np.save('Linfdense.npy', Linfdense)
    np.save('num_bases_sparse.npy', num_bases_sparse)
    np.save('num_bases_dense.npy', num_bases_dense)

    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(num_bases_sparse, L1sparse, color='#05445e', linestyle='-', marker='^', markersize=10)
    ax.plot(num_bases_sparse, L2sparse, color='#189ab4', linestyle='-', marker='v', markersize=10)
    ax.plot(num_bases_sparse, Linfsparse, color='#75e6da', linestyle='-', marker='*', markersize=10)
    ax.plot(num_bases_dense, L1dense, color='#fc2e20',  linestyle='--', marker='^', markersize=10)
    ax.plot(num_bases_dense, L2dense, color='#fd7f20', linestyle='--', marker='v', markersize=10)
    ax.plot(num_bases_dense, Linfdense, color='#fdb750', linestyle='--', marker='*', markersize=10)
    ax.set_xlabel("Num. of Basis Functions")
    ax.set_ylabel("$L^p$-norm Error")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1, np.max(num_bases_dense)])
    plt.gca().legend(('$L^1$-norm (Ours)', 
                      '$L^2$-norm (Ours)',
                      '$L^\\infty$-norm (Ours)',
                      '$L^1$-norm (Galerkin)', 
                      '$L^2$-norm (Galerkin)', 
                      '$L^\\infty$-norm (Galerkin)'))
    fig.tight_layout()
    plt.show()


Lp_error_vs_time()
Lp_Error_vs_num_bases()