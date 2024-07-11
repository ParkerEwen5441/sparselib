from .spgrid import SparseGrid

class GridContainer():
	def __init__(self, params, logging=False):
		"""
		Constructs a container for the sparse grid classes.
		Each element of the container contains a single SparseGrid
		class. By cvonvention, the first grid in the container is
		corresponds to the state uncertainty.
		
		:param      params:  The solver parameters
		:type       params:  SolverParams struct
		"""
		# Sparse grids for uncertainty and vector fields
		self.grids = [SparseGrid(params.domain, params.max_level, 
								 params.dim, logging=logging) 
						for _ in range(params.dim + 1)]

		# Build each sparse grid in container
		[grid.build() for grid in self.grids]

	def fit(self, params):
		"""
		Calls the fit functions in each container element.
		
		:param      params:  The solver parameters
		:type       params:  SolverParams struct
		"""

		if len(params.funcs) != len(self.grids):
			raise RuntimeError('There are {} functions defined but {} sparse grids'.format(
				len(params.funcs), len(self.grids)))

		# Fit each sparse grid to the corresponding function
		[grid.fit(f) for grid, f in zip(self.grids, params.funcs)]
