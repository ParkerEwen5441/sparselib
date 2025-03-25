import sparselib
import matplotlib
import numpy as np
import s3dlib.surface as s3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from matplotlib import cm
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator

def torusFunc(rtz) :
	r,t,z = rtz
	Z = 0.45 * np.sin(z*np.pi)
	R = r + 0.45 * np.cos(z*np.pi)
	return R,t,Z

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

	return np.sqrt(vals)

def marginalize(specgal, half=True):
	"""
	Compute marginal distribution of dims, integrating out
	additional dimensions from total probability density function.
	
	:param      dims:       The dimensions to compute marginals for
	:type       dims:       list(int)
	"""
	# Initialize new 1D sparse grid
	params1D = SolverParams1D()
	spgrid1D = sparselib.SparseGrid(params1D.domain, params1D.max_level, params1D.dim)
	spgrid1D.build()

	M = 20
	nLevel = [np.linspace(params1D.domain[0], params1D.domain[1], M),
			  np.linspace(params1D.domain[0], params1D.domain[1], M)]
	coordinates = np.array(np.meshgrid(*nLevel)).T.reshape(-1, 2)

	interp = np.zeros((coordinates.shape[0],))
	for val in spgrid1D.sparseGrid:
		extended_coordinates = np.hstack((coordinates, val * np.ones((coordinates.shape[0],1))))

		if half:
			interp += np.power(np.real(specgal.eval(extended_coordinates)), 2)
		else:
			interp += np.real(specgal.eval(extended_coordinates))

	return coordinates, interp/spgrid1D.sparseGrid.shape[0]

def plot_torus(ax, zq):
	cm = plt.get_cmap('magma')
	vals = cm(zq)[:,:,:]
	fig = plt.figure(frameon=False)
	plt.axis('off')
	plt.imshow(vals, interpolation='bicubic')
	plt.savefig('prob.png', bbox_inches='tight', pad_inches=0)
	plt.close(fig)

	vmin = np.minimum(0, np.min(zq))
	vmax = np.maximum(1, np.max(zq))

	torus = s3d.CylindricalSurface(6).map_geom_from_op(torusFunc)
	torus.map_color_from_image('prob.png')

	ax.set(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1) )
	ax.xaxis.set_major_locator(LinearLocator(5))
	ax.yaxis.set_major_locator(LinearLocator(5))
	ax.zaxis.set_major_locator(LinearLocator(5))
	minc = torus.bounds['vlim'][0]
	maxc = torus.bounds['vlim'][1]
	ax.add_collection3d(torus)
	  
	fig.tight_layout()
	plt.axis('off')

	return ax

class SolverParams1D():
	max_level: int = 7
	dim: int = 1
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = []

class SolverParamsSparse():
	max_level: int = 7
	dim: int = 3
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [gaussian_uncertainty_sparse, dynamics1, dynamics2, dynamics3]

def surf_plot_change(frame_number, zarray, plot, ax, xq, yq):
	"""
	Animation update function
	
	:param      frame_number:  The frame number
	:type       frame_number:  int
	:param      zarray:        The surface plot values
	:type       zarray:        np.array
	:param      plot:          The matplotlib object
	:type       plot:          mpl Object
	:param      ax:            Plot axis
	:type       ax:            mpl axis Object
	:param      xq:            x coordiantes
	:type       xq:            np.array
	:param      yq:            y coordinates
	:type       yq:            np.array
	"""
	plot[0].clear()
	plot[0] = plot_torus(ax, zarray[:, :, frame_number])

def make_surface_animation():
	"""
	Animates the uncertainty propagation over time t and saves result as
	a gif.

	:param      fps:    Frames per second
	:type       fps:    int
	:param      t  :    Propagation time
	:type       t  :    float
	""" 

	t = 1.5
	fps = 100

	paramsSparse = SolverParamsSparse()
	specgalSparse = sparselib.SpectralGalerkin(paramsSparse, logging=False)

	# Total number of frames
	frames = int(np.ceil(t * fps))

	# Generate coordinates
	M = 100
	nLevel = [np.linspace(paramsSparse.domain[0], paramsSparse.domain[1], M),
			  np.linspace(paramsSparse.domain[0], paramsSparse.domain[1], M)]
	# coordinates = np.array(np.meshgrid(*nLevel)).T.reshape(-1, 2)

	dx = 2 * np.pi / 1024
	xq, yq = np.meshgrid(np.arange(paramsSparse.domain[0], paramsSparse.domain[1], dx),
						 np.arange(paramsSparse.domain[0], paramsSparse.domain[1], dx))

	# Initialize surface plot values
	zarray = np.zeros((xq.shape[0], xq.shape[1], frames))

	# Create figure
	fig = plt.figure(figsize=plt.figaspect(0.75))
	ax = plt.axes(projection='3d')

	pbar = tqdm(total=frames)
	for dt in range(frames):
		# Propagate dt for each frame
		specgalSparse.solve(t=1/fps)
		coordinates, valsSparse = marginalize(specgalSparse)


		# interp = np.power(np.real(specgalSparse.eval(coordinates, container_id=0)), 2)
		vals = griddata(coordinates, valsSparse, (xq, yq), method='cubic')
		zarray[:,:,dt] = (vals / np.sum(vals * (2*np.pi/xq.shape[0])**2))

		pbar.update(1)
	pbar.close()

	# Plot and create animation
	plot = [plot_torus(ax, zarray[:, :, 0])]
	ani = animation.FuncAnimation(fig, surf_plot_change, frames, fargs=(zarray, plot, ax, xq, yq), interval=1000 / 75)
	
	# Show animation and save
	ani.save(filename="3D_example.gif", writer="pillow")
	plt.show()

make_surface_animation()