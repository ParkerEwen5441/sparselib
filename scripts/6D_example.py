import sparselib
import numpy as np
import matplotlib
import s3dlib.surface as s3d
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from scipy import interpolate
from matplotlib import cm, colorbar
from scipy.integrate import odeint
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator


K = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.1, 1.0, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 1.0, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 1.0, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 1.0, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 1.0]])

def normalize(ys, N):
    return np.squeeze(ys / (2 * np.pi * np.sum(ys) / N))

def marginalize(specgal, dim):
    """
    Compute marginal distribution of dims, integrating out
    additional dimensions from total probability density function.
    
    :param      dims:       The dimensions to compute marginals for
    :type       dims:       list(int)
    """

    spcoords = specgal.spgridUncertainty.sparseGrid[:,dim]
    coords = np.unique(spcoords)
    N = coords.shape[0]

    vals = specgal.eval(specgal.spgridUncertainty.sparseGrid)
    num_coords = [np.where(spcoords==b)[0].shape[0] for b in coords]

    marginal = [np.sum(vals[np.where(spcoords==b)],axis=0) for b in np.unique(spcoords)]
    marginal = np.power(np.real(marginal / np.asarray(num_coords)), 2)

    return normalize(marginal, N)

def dynamics1(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return x[:,0] + K[0,1] * np.sin(x[:,1] - x[:,0]) \
                  + K[0,2] * np.sin(x[:,2] - x[:,0]) \
                  + K[0,3] * np.sin(x[:,3] - x[:,0]) \
                  + K[0,4] * np.sin(x[:,4] - x[:,0]) \
                  + K[0,5] * np.sin(x[:,5] - x[:,0])

def dynamics2(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return x[:,1] + K[1,0] * np.sin(x[:,0] - x[:,1]) \
                  + K[1,2] * np.sin(x[:,2] - x[:,1]) \
                  + K[1,3] * np.sin(x[:,3] - x[:,1]) \
                  + K[1,4] * np.sin(x[:,4] - x[:,1]) \
                  + K[1,5] * np.sin(x[:,5] - x[:,1])

def dynamics3(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return x[:,2] + K[2,0] * np.sin(x[:,0] - x[:,2]) \
                  + K[2,1] * np.sin(x[:,1] - x[:,2]) \
                  + K[2,3] * np.sin(x[:,3] - x[:,2]) \
                  + K[2,4] * np.sin(x[:,4] - x[:,2]) \
                  + K[2,5] * np.sin(x[:,5] - x[:,2])

def dynamics4(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return x[:,3] + K[3,0] * np.sin(x[:,0] - x[:,3]) \
                  + K[3,1] * np.sin(x[:,1] - x[:,3]) \
                  + K[3,2] * np.sin(x[:,2] - x[:,3]) \
                  + K[3,4] * np.sin(x[:,4] - x[:,3]) \
                  + K[3,5] * np.sin(x[:,5] - x[:,3])

def dynamics5(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return x[:,4] + K[4,0] * np.sin(x[:,0] - x[:,4]) \
                  + K[4,1] * np.sin(x[:,1] - x[:,4]) \
                  + K[4,2] * np.sin(x[:,2] - x[:,4]) \
                  + K[4,3] * np.sin(x[:,3] - x[:,4]) \
                  + K[4,5] * np.sin(x[:,5] - x[:,4])

def dynamics6(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return x[:,5] + K[5,0] * np.sin(x[:,0] - x[:,5]) \
                  + K[5,1] * np.sin(x[:,1] - x[:,5]) \
                  + K[5,2] * np.sin(x[:,2] - x[:,5]) \
                  + K[5,3] * np.sin(x[:,3] - x[:,5]) \
                  + K[5,4] * np.sin(x[:,4] - x[:,5])

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
    # mu = np.linspace(0, 2*np.pi, 6)
    mu = np.array([0.39269908, 1.17809725, 2.35619449, 3.14159265, 4.3196899, 5.10508806, 5.89048623])
    cov = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    vals = np.ones((x.shape[0]))
    for d in range(x.shape[1]):
        vals *= 1 / np.sqrt(2 * np.pi * cov[d]) * np.exp(-0.5 * \
                    np.power(x[:,d] - mu[d], 2) / cov[d])

    return np.sqrt(vals)

class SolverParamsSparse():
	max_level: int = 5
	dim: int = 6
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [gaussian_uncertainty_sparse,
                   dynamics1, dynamics2, dynamics3,
                   dynamics4, dynamics5, dynamics6]

paramsSparse = SolverParamsSparse()
specgalSparse = sparselib.SpectralGalerkin(paramsSparse, logging=True)

# Propoagation time parameters
total_time = 1.0
M = 50
N = np.unique(specgalSparse.spgridUncertainty.sparseGrid[:,0]).shape[0]
t = 0
dt = total_time / M
ts = np.linspace(0, total_time, M+1)

# Results
t1 = marginalize(specgalSparse, dim=0)
t2 = marginalize(specgalSparse, dim=1)
t3 = marginalize(specgalSparse, dim=2)
t4 = marginalize(specgalSparse, dim=3)
t5 = marginalize(specgalSparse, dim=4)
t6 = marginalize(specgalSparse, dim=5)

x = np.linspace(0, 2*np.pi, N)
xs, ys = np.meshgrid(x, ts)

print("Computing coordinates wrt time ...")
pbar = tqdm(total=M)
for i in range(M):
    t += dt
    specgalSparse.solve(dt)
    
    # Marginalize along each dimension
    t1s = marginalize(specgalSparse, dim=0)
    t2s = marginalize(specgalSparse, dim=1)
    t3s = marginalize(specgalSparse, dim=2)
    t4s = marginalize(specgalSparse, dim=3)
    t5s = marginalize(specgalSparse, dim=4)
    t6s = marginalize(specgalSparse, dim=5)

    t1 = np.vstack((t1, t1s))
    t2 = np.vstack((t2, t2s))
    t3 = np.vstack((t3, t3s))
    t4 = np.vstack((t4, t4s))
    t5 = np.vstack((t5, t5s))
    t6 = np.vstack((t6, t6s))
    pbar.update(1)
pbar.close()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Time [s]')
ax1.set_zlabel('Probability Density')
ax1.set_title('Harmonic Oscillation Frequency vs Time of First Mode')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Time [s]')
ax2.set_zlabel('Probability Density')
ax2.set_title('Harmonic Oscillation Frequency vs Time of Second Mode')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Time [s]')
ax3.set_zlabel('Probability Density')
ax3.set_title('Harmonic Oscillation Frequency vs Time of Third Mode')

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_xlabel('Frequency')
ax4.set_ylabel('Time [s]')
ax4.set_zlabel('Probability Density')
ax4.set_title('Harmonic Oscillation Frequency vs Time of Fourth Mode')

fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.set_xlabel('Frequency')
ax5.set_ylabel('Time [s]')
ax5.set_zlabel('Probability Density')
ax5.set_title('Harmonic Oscillation Frequency vs Time of Fifth Mode')

fig6 = plt.figure()
ax6 = fig6.add_subplot(111, projection='3d')
ax6.set_xlabel('Frequency')
ax6.set_ylabel('Time [s]')
ax6.set_zlabel('Probability Density')
ax6.set_title('Harmonic Oscillation Frequency vs Time of Sixth Mode')

xnew, ynew = np.mgrid[0:2*np.pi:100j, 0:1:100j]
tck = interpolate.bisplrep(xs, ys, t1, s=0.1)
t1 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t2, s=0.1)
t2 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t3, s=0.1)
t3 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t4, s=0.1)
t4 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t5, s=0.1)
t5 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t6, s=0.1)
t6 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

ax1.plot_surface(xnew, ynew, t1.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax2.plot_surface(xnew, ynew, t2.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax3.plot_surface(xnew, ynew, t3.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax4.plot_surface(xnew, ynew, t4.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax5.plot_surface(xnew, ynew, t5.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax6.plot_surface(xnew, ynew, t6.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
plt.show()