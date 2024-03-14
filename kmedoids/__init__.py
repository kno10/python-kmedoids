"""K-medoids Clustering in Python.

This package implements common k-medoids clustering algorithms,
in decreasing order of performance:

- FasterPAM
- FastPAM (same result as PAM; but faster)
- PAM (the original Partitioning Around Medoids algorithm)
- Alternating (k-means style algorithm, yields results of lower quality)
- BUILD (the initialization of PAM)

Additionally, the package implements clustering algorithms
for direct optimization of the (Medoid) Silhouette,
in decreasing order of performance:

- FasterMSC
- FastMSC (same result as PAMMEDSIL; but faster)
- DynMSC (automatic choice of k; faster than repeated FasterMSC)
- PAMMEDSIL
- PAMSIL

Evaluation measures:

- Silhouette evaluation
- Medoid Silhouette evaluation

References:

| Erich Schubert and Lars Lenssen:
| Fast k-medoids Clustering in Rust and Python
| Journal of Open Source Software 7(75), 4183
| https://doi.org/10.21105/joss.04183 (open access)

| Erich Schubert, Peter J. Rousseeuw
| Fast and Eager k-Medoids Clustering:
| O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
| Information Systems (101), 2021, 101804
| https://doi.org/10.1016/j.is.2021.101804 (open access)

| Erich Schubert, Peter J. Rousseeuw:
| Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
| In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.
| https://doi.org/10.1007/978-3-030-32047-8_16
| Preprint: https://arxiv.org/abs/1810.05691

| Lars Lenssen, Erich Schubert:
| Medoid silhouette clustering with automatic cluster number selection
| Information Systems (120), 2024, 102290
| https://doi.org/10.1016/j.is.2023.102290
| Preprint: https://arxiv.org/abs/2309.03751

| Lars Lenssen, Erich Schubert:
| Clustering by Direct Optimization of the Medoid Silhouette
| In: 15th International Conference on Similarity Search and Applications (SISAP 2022).
| https://doi.org/10.1007/978-3-031-17849-8_15

| Leonard Kaufman, Peter J. Rousseeuw:
| Clustering by means of medoids.
| In: Dodge Y (ed) Statistical Data Analysis Based on the L 1 Norm and Related Methods, 405-416, 1987

| Leonard Kaufman, Peter J. Rousseeuw:
| Finding Groups in Data: An Introduction to Cluster Analysis.
| John Wiley&Sons, 1990, https://doi.org/10.1002/9780470316801

| Peter J. Rousseeuw:
| Silhouettes: A graphical aid to the interpretation and validation of cluster analysis
| Journal of Computational and Applied Mathematics, Volume 20, 1987
| https://doi.org/10.1016/0377-0427(87)90125-7

| Mark Van der Laan, Katherine Pollard, Jennifer Bryan:
| A new partitioning around medoids algorithm.
| In: Journal of Statistical Computation and Simulation, pp 575-584, 2003
| https://doi.org/10.1080/0094965031000136012

"""
__all__ = [
	"pam",
	"fastpam1",
	"fasterpam",
	"fastmsc",
	"fastermsc",
	"dynmsc",
	"alternating",
	"pam_build",
	"silhouette",
	"medoid_silhouette",
	"KMedoidsResult",
	"DynkResult",
]

class KMedoidsResult:
	"""
	K-medoids clustering result

	:param loss: Loss of this clustering (sum of deviations)
	:type loss: float

	:param labels: Cluster assignment
	:type labels: ndarray

	:param medoids: Chosen medoid indexes
	:type medoids: ndarray

	:param n_iter: Number of iterations
	:type n_iter: int

	:param n_swap: Number of swaps performed
	:type n_swap: int
	"""
	def __init__(self, loss, labels, medoids, n_iter=None, n_swap=None):
		self.loss = loss
		self.labels = labels
		self.medoids = medoids
		self.n_iter = n_iter
		self.n_swap = n_swap

	def __repr__(self):
		return f"KMedoidsResult(loss={self.loss}, labels={self.labels}, medoids={self.medoids}, n_iter={self.n_iter}, n_swaps={self.n_swap})"


class DynkResult:
	"""
	K-medoids or Silhouette clustering result with automatic number of clusters

	:param loss: Loss of this clustering (sum of deviations)
	:type loss: float

	:param labels: Cluster assignment
	:type labels: ndarray

	:param medoids: Chosen medoid indexes
	:type medoids: ndarray

	:param n_iter: Number of iterations
	:type n_iter: int

	:param n_swap: Number of swaps performed
	:type n_swap: int

	:param bestk: Best k by Medoid Silhouette
	:type bestk: int

	:param losses: Medoid Silhouette over range of k
	:type losses: ndarray

	:param rangek: range of k
	:type rangek: range
	"""
	def __init__(self, loss, labels, medoids, bestk, losses, rangek, n_iter=None, n_swap=None):
		self.loss = loss
		self.labels = labels
		self.medoids = medoids
		self.n_iter = n_iter
		self.n_swap = n_swap
		self.bestk = bestk
		self.losses = losses
		self.rangek = rangek

	def __repr__(self):
		return f"DynkResult(loss={self.loss}, labels={self.labels}, medoids={self.medoids}, bestk={self.bestk}, losses={self.losses}, rangek={self.rangek}, n_iter={self.n_iter}, n_swaps={self.n_swap})"

def _check_medoids(diss, medoids, init, random_state):
	"""Check the medoids and random_state parameters."""
	import numpy as np, numbers
	import warnings
	if isinstance(medoids, np.ndarray):
		if random_state is not None:
			warnings.warn("Seed will be ignored if initial medoids are given")
		return medoids.astype(np.uintp)
	if isinstance(medoids, int):
		if init.lower() == "build":
			return pam_build(diss, medoids).medoids
		if init.lower() == "first":
			return np.arange(medoids)
		if random_state is None or random_state is np.random:
			random_state = np.random.mtrand._rand
		elif isinstance(random_state, numbers.Integral):
			random_state = np.random.RandomState(random_state)
		if not isinstance(random_state, np.random.RandomState):
			raise ValueError("Pass a numpy random generator, RandomState or integer seed")
		return random_state.choice(diss.shape[0], medoids, False).astype(np.uintp)
	raise ValueError("Specify the number of medoids, or give a numpy array of initial medoids")

def fasterpam(diss, medoids, max_iter=100, init="random", random_state=None, n_cpu=-1):
	"""FasterPAM k-medoids clustering

	This is an accelerated version of PAM clustering, that eagerly
	performs any swap found, and contains the O(k) improvement to find
	the best swaps faster.

	References:

	| Erich Schubert, Peter J. Rousseeuw
	| Fast and Eager k-Medoids Clustering:
	| O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
	| Information Systems (101), 2021, 101804
	| https://doi.org/10.1016/j.is.2021.101804 (open access)

	| Erich Schubert, Peter J. Rousseeuw:
	| Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
	| In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.
	| https://doi.org/10.1007/978-3-030-32047-8_16
	| Preprint: https://arxiv.org/abs/1810.05691

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param medoids: number of clusters to find or existing medoids
	:type medoids: int or ndarray
	:param max_iter: maximum number of iterations
	:type max_iter: int
	:param init: initialization method
	:type init: str, "random", "first" or "build"
	:param random_state: random seed (also used for shuffling the processing order)
	:type random_state: int, RandomState instance or None
	:param n_cpu: number of threads to use (-1: automatic)
	:type n_cpu: int

	:return: k-medoids clustering result
	:rtype: KMedoidsResult
	"""
	import numpy as np, numbers, os
	from .kmedoids import _fasterpam_i32, _fasterpam_i64, _fasterpam_f32, _fasterpam_f64
	from .kmedoids import _rand_fasterpam_i32, _rand_fasterpam_i64, _rand_fasterpam_f32, _rand_fasterpam_f64
	from .kmedoids import _par_fasterpam_i32, _par_fasterpam_i64, _par_fasterpam_f32, _par_fasterpam_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	medoids = _check_medoids(diss, medoids, init, random_state)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if n_cpu == -1 and diss.shape[0] < 1000: n_cpu = 1
		if n_cpu == -1 and os.cpu_count() is not None: n_cpu = os.cpu_count()
		if n_cpu == -1: n_cpu = 1
		assert n_cpu > 0
		if n_cpu > 1:
			seed = None
			if random_state is None or random_state is np.random:
				seed = np.random.mtrand._rand.randint(0, 2147483647)
			elif isinstance(random_state, numbers.Integral):
				seed = int(random_state)
			elif isinstance(random_state, np.random.RandomState):
				seed = random_state.randint(0, 2147483647)
			else:
				raise ValueError("Pass a numpy random generator, state or integer seed")
			if dtype == np.float32:
				return KMedoidsResult(*_par_fasterpam_f32(diss, medoids, max_iter, seed, n_cpu))
			elif dtype == np.float64:
				return KMedoidsResult(*_par_fasterpam_f64(diss, medoids, max_iter, seed, n_cpu))
			elif dtype == np.int32:
				return KMedoidsResult(*_par_fasterpam_i32(diss, medoids, max_iter, seed, n_cpu))
			elif dtype == np.int64:
				return KMedoidsResult(*_par_fasterpam_i64(diss, medoids, max_iter, seed, n_cpu))
		elif random_state is None:
			if dtype == np.float32:
				return KMedoidsResult(*_fasterpam_f32(diss, medoids, max_iter))
			elif dtype == np.float64:
				return KMedoidsResult(*_fasterpam_f64(diss, medoids, max_iter))
			elif dtype == np.int32:
				return KMedoidsResult(*_fasterpam_i32(diss, medoids, max_iter))
			elif dtype == np.int64:
				return KMedoidsResult(*_fasterpam_i64(diss, medoids, max_iter))
		else:
			seed = None
			if random_state is np.random:
				seed = np.random.mtrand._rand.randint(0, 2147483647)
			elif isinstance(random_state, numbers.Integral):
				seed = int(random_state)
			elif isinstance(random_state, np.random.RandomState):
				seed = random_state.randint(0, 2147483647)
			else:
				raise ValueError("Pass a numpy random generator, state or integer seed")
			if dtype == np.float32:
				return KMedoidsResult(*_rand_fasterpam_f32(diss, medoids, max_iter, seed))
			elif dtype == np.float64:
				return KMedoidsResult(*_rand_fasterpam_f64(diss, medoids, max_iter, seed))
			elif dtype == np.int32:
				return KMedoidsResult(*_rand_fasterpam_i32(diss, medoids, max_iter, seed))
			elif dtype == np.int64:
				return KMedoidsResult(*_rand_fasterpam_i64(diss, medoids, max_iter, seed))
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def fastpam1(diss, medoids, max_iter=100, init="random", random_state=None):
	"""FastPAM1 k-medoids clustering

	This is an accelerated version of PAM clustering, that performs the
	same swaps as the original PAM (given the same starting conditions),
	but finds the best swap O(k) times faster.

	References:

	| Erich Schubert, Peter J. Rousseeuw
	| Fast and Eager k-Medoids Clustering:
	| O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
	| Information Systems (101), 2021, 101804
	| https://doi.org/10.1016/j.is.2021.101804 (open access)

	| Erich Schubert, Peter J. Rousseeuw:
	| Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
	| In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.
	| https://doi.org/10.1007/978-3-030-32047-8_16
	| Preprint: https://arxiv.org/abs/1810.05691

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param medoids: number of clusters to find or existing medoids
	:type medoids: int or ndarray
	:param max_iter: maximum number of iterations
	:type max_iter: int
	:param init: initialization method
	:type init: str, "random", "first" or "build"
	:param random_state: random seed if no medoids are given
	:type random_state: int, RandomState instance or None

	:return: k-medoids clustering result
	:rtype: KMedoidsResult
	"""
	import numpy as np
	from .kmedoids import _fastpam1_i32, _fastpam1_i64, _fastpam1_f32, _fastpam1_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	medoids = _check_medoids(diss, medoids, init, random_state)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return KMedoidsResult(*_fastpam1_f32(diss, medoids, max_iter))
		elif dtype == np.float64:
			return KMedoidsResult(*_fastpam1_f64(diss, medoids, max_iter))
		elif dtype == np.int32:
			return KMedoidsResult(*_fastpam1_i32(diss, medoids, max_iter))
		elif dtype == np.int64:
			return KMedoidsResult(*_fastpam1_i64(diss, medoids, max_iter))
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def pam_build(diss, k):
	"""PAM k-medoids clustering -- BUILD only

	This is an implementation of the original PAM (Partitioning Around Medoids)
	clustering algorithm. For improved versions, see the fastpam and fasterpam methods.

	References:

	| Leonard Kaufman, Peter J. Rousseeuw:
	| Clustering by means of medoids.
	| In: Dodge Y (ed) Statistical Data Analysis Based on the L 1 Norm and Related Methods, 405-416, 1987

	| Leonard Kaufman, Peter J. Rousseeuw:
	| Finding Groups in Data: An Introduction to Cluster Analysis.
	| John Wiley&Sons, 1990, https://doi.org/10.1002/9780470316801

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param k: number of clusters to find
	:type k: int

	:return: k-medoids clustering result
	:rtype: KMedoidsResult
	"""
	import numpy as np
	from .kmedoids import _pam_build_i32, _pam_build_i64, _pam_build_f32, _pam_build_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return KMedoidsResult(*_pam_build_f32(diss, k))
		elif dtype == np.float64:
			return KMedoidsResult(*_pam_build_f64(diss, k))
		elif dtype == np.int32:
			return KMedoidsResult(*_pam_build_i32(diss, k))
		elif dtype == np.int64:
			return KMedoidsResult(*_pam_build_i64(diss, k))
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def pam(diss, medoids, max_iter=100, init="build", random_state=None):
	"""PAM k-medoids clustering

	This is an implementation of the original PAM (Partitioning Around Medoids)
	clustering algorithm. For improved versions, see the fastpam and fasterpam methods.

	References:

	| Leonard Kaufman, Peter J. Rousseeuw:
	| Clustering by means of medoids.
	| In: Dodge Y (ed) Statistical Data Analysis Based on the L 1 Norm and Related Methods, pp 405–416, 1987

	| Leonard Kaufman, Peter J. Rousseeuw:
	| Finding Groups in Data: An Introduction to Cluster Analysis.
	| John Wiley&Sons, 1990, https://doi.org/10.1002/9780470316801

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param medoids: number of clusters to find or existing medoids
	:type medoids: int or ndarray
	:param max_iter: maximum number of iterations
	:type max_iter: int
	:param init: initialization method
	:type init: str, "random", "first" or "build"
	:param random_state: random seed if no medoids are given
	:type random_state: int, RandomState instance or None

	:return: k-medoids clustering result
	:rtype: KMedoidsResult
	"""
	import numpy as np
	from .kmedoids import _pam_swap_i32, _pam_swap_i64, _pam_swap_f32, _pam_swap_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	medoids = _check_medoids(diss, medoids, init, random_state)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return KMedoidsResult(*_pam_swap_f32(diss, medoids, max_iter))
		elif dtype == np.float64:
			return KMedoidsResult(*_pam_swap_f64(diss, medoids, max_iter))
		elif dtype == np.int32:
			return KMedoidsResult(*_pam_swap_i32(diss, medoids, max_iter))
		elif dtype == np.int64:
			return KMedoidsResult(*_pam_swap_i64(diss, medoids, max_iter))
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def pammedsil(diss, medoids, max_iter=100, init="build", random_state=None):
	"""PAMMEDSIL clustering

	This is an implementation of the original PAMMEDSIL
	clustering algorithm. For improved versions, see the fastmsc and fastermsc methods.

	References:

	| Mark Van der Laan, Katherine Pollard, Jennifer Bryan:
	| A new partitioning around medoids algorithm.
	| In: Journal of Statistical Computation and Simulation, pp 575-584, 2003
	| https://doi.org/10.1080/0094965031000136012

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param medoids: number of clusters to find or existing medoids
	:type medoids: int or ndarray
	:param max_iter: maximum number of iterations
	:type max_iter: int
	:param init: initialization method
	:type init: str, "random", "first" or "build"
	:param random_state: random seed if no medoids are given
	:type random_state: int, RandomState instance or None

	:return: k-medoids clustering result
	:rtype: KMedoidsResult
	"""
	import numpy as np
	from .kmedoids import _pammedsil_swap_f32, _pammedsil_swap_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	medoids = _check_medoids(diss, medoids, init, random_state)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return KMedoidsResult(*_pammedsil_swap_f32(diss, medoids, max_iter))
		elif dtype == np.float64:
			return KMedoidsResult(*_pammedsil_swap_f64(diss, medoids, max_iter))
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def pamsil(diss, medoids, max_iter=100, init="build", random_state=None):
	"""PAMSIL k-medoids clustering

	This is an implementation of the original PAMSIL.

	References:

	| Mark Van der Laan, Katherine Pollard, Jennifer Bryan:
	| A new partitioning around medoids algorithm.
	| In: Journal of Statistical Computation and Simulation, pp 575-584, 2003
	| https://doi.org/10.1080/0094965031000136012

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param medoids: number of clusters to find or existing medoids
	:type medoids: int or ndarray
	:param max_iter: maximum number of iterations
	:type max_iter: int
	:param init: initialization method
	:type init: str, "random", "first" or "build"
	:param random_state: random seed if no medoids are given
	:type random_state: int, RandomState instance or None

	:return: k-medoids clustering result
	:rtype: KMedoidsResult
	"""
	import numpy as np
	from .kmedoids import _pamsil_swap_f32, _pamsil_swap_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	medoids = _check_medoids(diss, medoids, init, random_state)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return KMedoidsResult(*_pamsil_swap_f32(diss, medoids, max_iter))
		elif dtype == np.float64:
			return KMedoidsResult(*_pamsil_swap_f64(diss, medoids, max_iter))
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def fastmsc(diss, medoids, max_iter=100, init="random", random_state=None):
	"""FastMSC clustering

	This is an accelerated version of PAMMEDSIL clustering, that performs the
	same swaps as the original PAMMEDSIL (given the same starting conditions),
	but finds the best swap O(k^2) times faster.

	References:

	| Lars Lenssen, Erich Schubert:
	| Medoid silhouette clustering with automatic cluster number selection
	| Information Systems (120), 2024, 102290
	| https://doi.org/10.1016/j.is.2023.102290
	| Preprint: https://arxiv.org/abs/2309.03751

	| Lars Lenssen, Erich Schubert:
	| Clustering by Direct Optimization of the Medoid Silhouette
	| In: 15th International Conference on Similarity Search and Applications (SISAP 2022).
	| https://doi.org/10.1007/978-3-031-17849-8_15

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param medoids: number of clusters to find or existing medoids
	:type medoids: int or ndarray
	:param max_iter: maximum number of iterations
	:type max_iter: int
	:param init: initialization method
	:type init: str, "random", "first" or "build"
	:param random_state: random seed if no medoids are given
	:type random_state: int, RandomState instance or None

	:return: k-medoids clustering result
	:rtype: KMedoidsResult
	"""
	import numpy as np
	from .kmedoids import _fastmsc_f32, _fastmsc_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	medoids = _check_medoids(diss, medoids, init, random_state)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return KMedoidsResult(*_fastmsc_f32(diss, medoids, max_iter))
		elif dtype == np.float64:
			return KMedoidsResult(*_fastmsc_f64(diss, medoids, max_iter))
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def fastermsc(diss, medoids, max_iter=100, init="random", random_state=None):
	"""FasterMSC clustering

	This is an accelerated version of PAMMEDSIL clustering, that eagerly
	performs any swap found, and contains the O(k^2) improvement to find
	the best swaps faster.

	References:

	| Lars Lenssen, Erich Schubert:
	| Medoid silhouette clustering with automatic cluster number selection
	| Information Systems (120), 2024, 102290
	| https://doi.org/10.1016/j.is.2023.102290
	| Preprint: https://arxiv.org/abs/2309.03751

	| Lars Lenssen, Erich Schubert:
	| Clustering by Direct Optimization of the Medoid Silhouette
	| In: 15th International Conference on Similarity Search and Applications (SISAP 2022).
	| https://doi.org/10.1007/978-3-031-17849-8_15

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param medoids: number of clusters to find or existing medoids
	:type medoids: int or ndarray
	:param max_iter: maximum number of iterations
	:type max_iter: int
	:param init: initialization method
	:type init: str, "random", "first" or "build"
	:param random_state: random seed if no medoids are given
	:type random_state: int, RandomState instance or None

	:return: k-medoids clustering result
	:rtype: KMedoidsResult
	"""
	import numpy as np
	from .kmedoids import _fastermsc_f32, _fastermsc_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	medoids = _check_medoids(diss, medoids, init, random_state)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return KMedoidsResult(*_fastermsc_f32(diss, medoids, max_iter))
		elif dtype == np.float64:
			return KMedoidsResult(*_fastermsc_f64(diss, medoids, max_iter))
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def dynmsc(diss, medoids, minimum_k=2, max_iter=100, init="random", random_state=None):
	"""DynMSC clustering

	This is a version of FasterMSC with automatic cluster number selection, that
	performs FasterMSC for a minimum k to the number of input medoids and returns
	the clustering with the highest Average Medoid Silhouette.

	References:

	| Lars Lenssen, Erich Schubert:
	| Medoid silhouette clustering with automatic cluster number selection
	| Information Systems (120), 2024, 102290
	| https://doi.org/10.1016/j.is.2023.102290
	| Preprint: https://arxiv.org/abs/2309.03751

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param medoids: maximum number of clusters to find or existing medoids with length of maximum number of clusters to find
	:type medoids: int or ndarray
	:param minimum_k: minimum number of clusters to find
	:type minimum_k: int
	:param max_iter: maximum number of iterations
	:type max_iter: int
	:param init: initialization method
	:type init: str, "random", "first" or "build"
	:param random_state: random seed if no medoids are given
	:type random_state: int, RandomState instance or None

	:return: k-medoids clustering with automatic number of clusters
	:rtype: DynkResult
	"""
	import numpy as np
	from .kmedoids import _dynmsc_f32, _dynmsc_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	medoids = _check_medoids(diss, medoids, init, random_state)

	if medoids.shape[0] < minimum_k:
		raise ValueError("Maximum k should be at least minimum k.")
	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return DynkResult(*_dynmsc_f32(diss, medoids, minimum_k, max_iter))
		elif dtype == np.float64:
			return DynkResult(*_dynmsc_f64(diss, medoids, minimum_k, max_iter))
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def alternating(diss, medoids, max_iter=100, init="random", random_state=None):
	"""Alternating k-medoids clustering (k-means-style algorithm)

	Note: this yields substantially worse results than PAM algorithms on difficult data sets.

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param medoids: number of clusters to find or existing medoids
	:type medoids: int or ndarray
	:param max_iter: maximum number of iterations
	:type max_iter: int
	:param init: initialization method
	:type init: str, "random", "first" or "build"
	:param random_state: random seed if no medoids are given
	:type random_state: int, RandomState instance or None

	:return: k-medoids clustering result
	:rtype: KMedoidsResult
	"""
	import numpy as np
	from .kmedoids import _alternating_i32, _alternating_i64, _alternating_f32, _alternating_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	medoids = _check_medoids(diss, medoids, init, random_state)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return KMedoidsResult(*_alternating_f32(diss, medoids, max_iter))
		elif dtype == np.float64:
			return KMedoidsResult(*_alternating_f64(diss, medoids, max_iter))
		elif dtype == np.int32:
			return KMedoidsResult(*_alternating_i32(diss, medoids, max_iter))
		elif dtype == np.int64:
			return KMedoidsResult(*_alternating_i64(diss, medoids, max_iter))
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def silhouette(diss, labels, samples=False, n_cpu=-1):
	"""Silhouette index for cluster evaluation.

	The Silhouette, proposed by Peter Rousseeuw in 1987, is a popular
	internal evaluation measure for clusterings. Although it is defined on
	arbitary metrics, it is most appropriate for evaluating "spherical"
	clusters, as it expects objects to be closer to all members of its own
	cluster than to members of other clusters.

	References:

	| Peter J. Rousseeuw:
	| Silhouettes: A graphical aid to the interpretation and validation of cluster analysis
	| Journal of Computational and Applied Mathematics, Volume 20, 1987
	| https://doi.org/10.1016/0377-0427(87)90125-7

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param labels: cluster labels (use 0 to k-1, no negative values allowed)
	:type labels: ndarray of int
	:param samples: whether to return individual samples or not
	:type samples: boolean
	:param n_cpu: number of threads to use (-1: automatic)
	:type n_cpu: int

	:return: tuple containing the average silhouette and the individual samples
	:rtype: (float, ndarray)
	"""
	import numpy as np, os
	from .kmedoids import _silhouette_i32, _silhouette_f32, _silhouette_f64
	from .kmedoids import _par_silhouette_i32, _par_silhouette_f32, _par_silhouette_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)
	labels = np.unique(labels, return_inverse=True)[1].astype(np.uintp) # ensure labels are 0..k-1

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if n_cpu == -1 and samples: n_cpu = 1
		if n_cpu == -1: n_cpu = os.cpu_count() or 1
		assert n_cpu > 0
		if n_cpu > 1:
			assert not samples, "samples=true currently may only be used with n_cpu=1"
			if dtype == np.float32:
				return (_par_silhouette_f32(diss, labels, n_cpu), [])
			elif dtype == np.float64:
				return (_par_silhouette_f64(diss, labels, n_cpu), [])
			elif dtype == np.int32:
				return (_par_silhouette_i32(diss, labels, n_cpu), [])
			elif dtype == np.int64:
				raise ValueError("Input of int64 is currently not supported, as it could overflow the float64 used internally when computing Silhouette. Use diss.astype(numpy.float64) if that is acceptable and you have the necessary memory for this copy.")
		else:
			if dtype == np.float32:
				return _silhouette_f32(diss, labels, samples)
			elif dtype == np.float64:
				return _silhouette_f64(diss, labels, samples)
			elif dtype == np.int32:
				return _silhouette_i32(diss, labels, samples)
			elif dtype == np.int64:
				raise ValueError("Input of int64 is currently not supported, as it could overflow the float64 used internally when computing Silhouette. Use diss.astype(numpy.float64) if that is acceptable and you have the necessary memory for this copy.")
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def medoid_silhouette(diss, meds, samples=False):
	"""Medoid Silhouette index for cluster evaluation.

	The Medoid Silhouette is an approximation to the Silhouette index, that
	uses the distance to the cluster medoids instead of the average distance
	to the other cluster members. If every point is assigned to the nearest
	medoid, the Medoid Silhouette of a point reduces to 1-a/b where a is the
	distance to the nearest, and b the distance to the second nearest medoid.
	If b is 0, the Medoid Silhouette is 1.

	This function assumes you already have a distance matrix. It is not necessary
	to compute a distance matrix to evaluate the medoid silhouette -- only the
	distances between points and medoids are necessary. If you do not have a
	distance matrix, simply compute the medoid Silhouette directly, by computing
	(1) the N x k distance matrix to the medoids, (2) finding the two smallest values
	for each data point, and (3) computing the average of 1-a/b on these (with 0/0 as 0).
	This can be implemented in a few lines with numpy easily.

	:param diss: square numpy array of dissimilarities
	:type diss: ndarray
	:param meds: medoid indexes (k distinct values in 0 to n-1)
	:type meds: ndarray of int
	:param samples: whether to return individual samples or not
	:type samples: boolean

	:return: tuple containing the average Medoid Silhouette and the individual samples
	:rtype: (float, ndarray)
	"""
	import numpy as np, os
	from .kmedoids import _medoid_silhouette_i32, _medoid_silhouette_f32, _medoid_silhouette_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)
	if not isinstance(meds, np.ndarray):
		meds = np.array(meds)
	meds = meds.astype(np.uintp)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return _medoid_silhouette_f32(diss, meds, samples)
		elif dtype == np.float64:
			return _medoid_silhouette_f64(diss, meds, samples)
		elif dtype == np.int32:
			return _medoid_silhouette_i32(diss, meds, samples)
		elif dtype == np.int64:
			raise ValueError("Input of int64 is currently not supported, as it could overflow the float64 used internally when computing Silhouette. Use diss.astype(numpy.float64) if that is acceptable and you have the necessary memory for this copy.")
	raise ValueError("Input data not supported. Use a numpy array of floats.")

# This is a hack to make sklearn an optional dependency only:
try:
	from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
	class SKLearnClusterer(BaseEstimator, ClusterMixin, TransformerMixin):
		pass
except ImportError:
	SKLearnClusterer = object  # fallback if sklearn not available

class KMedoids(SKLearnClusterer):
	"""K-Medoids Clustering using PAM, FasterPAM, and FasterMSC (sklearn-compatible API).

	References:

	| Erich Schubert and Lars Lenssen:
	| Fast k-medoids Clustering in Rust and Python
	| Journal of Open Source Software 7(75), 4183
	| https://doi.org/10.21105/joss.04183 (open access)

	| Erich Schubert, Peter J. Rousseeuw:
	| Fast and Eager k-Medoids Clustering
	| O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
	| Information Systems (101), 2021, 101804
	| https://doi.org/10.1016/j.is.2021.101804 (open access)

	| Erich Schubert, Peter J. Rousseeuw:
	| Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
	| In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.
	| https://doi.org/10.1007/978-3-030-32047-8_16
	| Preprint: https://arxiv.org/abs/1810.05691

	| Lars Lenssen, Erich Schubert:
	| Medoid silhouette clustering with automatic cluster number selection
	| Information Systems (120), 2024, 102290
	| https://doi.org/10.1016/j.is.2023.102290
	| Preprint: https://arxiv.org/abs/2309.03751

	| Lars Lenssen, Erich Schubert:
	| Clustering by Direct Optimization of the Medoid Silhouette
	| In: 15th International Conference on Similarity Search and Applications (SISAP 2022).
	| https://doi.org/10.1007/978-3-031-17849-8_15

	| Leonard Kaufman, Peter J. Rousseeuw:
	| Clustering by means of medoids.
	| In: Dodge Y (ed) Statistical Data Analysis Based on the L 1 Norm and Related Methods, 405-416, 1987

	| Leonard Kaufman, Peter J. Rousseeuw:
	| Finding Groups in Data: An Introduction to Cluster Analysis.
	| John Wiley&Sons, 1990, https://doi.org/10.1002/9780470316801

	| Mark Van der Laan, Katherine Pollard, Jennifer Bryan:
	| A new partitioning around medoids algorithm.
	| In: Journal of Statistical Computation and Simulation, pp 575-584, 2003
	| https://doi.org/10.1080/0094965031000136012

	:param n_clusters: The number of clusters to form (maximum number of clusters if `method="dynmsc"`)
	:type n_clusters: int
	:param metric: It is recommended to use 'precomputed', in particular when experimenting with different `n_clusters`.
	    If you have sklearn installed, you may pass any metric supported by `sklearn.metrics.pairwise_distances`.
	:type metric: string, default: 'precomputed'
	:param metric_params: Additional keyword arguments for the metric function.
	:type metric_params: dict, default=None
	:param method: Which algorithm to use
	:type method: string, "fasterpam" (default), "fastpam1", "pam", "alternate", "fastermsc", "fastmsc", "pamsil" or "pammedsil"
	:param init: initialization method
	:type init: string, "random" (default), "first" or "build"
	:param max_iter: Specify the maximum number of iterations when fitting
	:type max_iter: int
	:param random_state: random seed if no medoids are given
	:type random_state: int, RandomState instance or None

	:ivar cluster_centers_: None for 'precomputed'
	:type cluster_centers_: array
	:ivar medoid_indices_: The indices of the medoid rows in X
	:type medoid_indices_: array, shape = (n_clusters,)
	:ivar labels_: Labels of each point
	:type labels_: array, shape = (n_samples,)
	:ivar inertia_: Sum of distances of samples to their closest cluster center
	:type inertia_: float
	"""
	def __init__(
		self,
		n_clusters,
		*,
		metric="precomputed",
		metric_params=None,
		method="fasterpam",
		init="random",
		max_iter=300,
		random_state=None,
	):
		self.n_clusters = n_clusters
		self.metric = metric
		self.metric_params = metric_params
		self.method = method
		self.init = init
		self.max_iter = max_iter
		self.random_state = random_state

	def fit(self, X, y=None):
		"""Fit K-Medoids to the provided data.

		:param X: Dataset to cluster
		:type X: {array-like, sparse matrix}, shape = (n_samples, n_samples)
		:param y: ignored

		:return: self
		"""
		if self.init != "random" and self.init != "first" and self.init != "build":
			raise ValueError(
				f"init={self.init} is not supported. Supported inits "
				f"are 'random', 'first' and 'build'."
			)

		if self.metric != "precomputed":
			from sklearn.metrics.pairwise import pairwise_distances
			Xd	= X
			X = pairwise_distances(X, metric=self.metric)
		if self.method == "fasterpam":
			result = fasterpam(X, self.n_clusters, self.max_iter, self.init, random_state=self.random_state)
		elif self.method == "fastpam1":
			result = fastpam1(X, self.n_clusters, self.max_iter, self.init, random_state=self.random_state)
		elif self.method == "pam":
			result = pam(X, self.n_clusters, self.max_iter, self.init, random_state=self.random_state)
		elif self.method == "fastermsc":
			result = fastermsc(X, self.n_clusters, self.max_iter, self.init, random_state=self.random_state)
		elif self.method == "dynmsc":
			result = dynmsc(X, self.n_clusters, 2, self.max_iter, self.init, random_state=self.random_state)
		elif self.method == "fastmsc":
			result = fastmsc(X, self.n_clusters, self.max_iter, self.init, random_state=self.random_state)
		elif self.method == "pamsil":
			result = pamsil(X, self.n_clusters, self.max_iter, self.init, random_state=self.random_state)
		elif self.method == "pammedsil":
			result = pammedsil(X, self.n_clusters, self.max_iter, self.init, random_state=self.random_state)
		elif self.method == "alternate":
			result = alternating(X, self.n_clusters, self.max_iter, self.init, random_state=self.random_state)
		else:
			raise ValueError(
				f"method={self.method} is not supported. Supported methods "
				f"are 'fasterpam', 'fastpam1', 'pam', 'alternate', "
				f"'fastermsc', 'fastmsc', 'dynmsc', 'pamsil', and 'pammedsil'. "
				f"Recommended values are 'fasterpam' for classic k-medoids and 'fastermsc' for Silhouette optimization."
			)
		self.labels_ = result.labels
		self.medoid_indices_ = result.medoids
		self.inertia_ = float(result.loss)
		if self.metric == "precomputed":
			self.cluster_centers_ = None
		else:
			self.cluster_centers_ = Xd[result.medoids]
		return self

	def predict(self, X):
		"""Predict the closest cluster for each sample in X.

		:param X: New data to predict
		:type X: {array-like, sparse matrix}, shape = (n_samples, n_samples)

		:return: Index of the cluster each sample belongs to
		:rtype: array, shape = (n_query,)
		"""
		if self.metric != "precomputed":
			from sklearn.metrics.pairwise import pairwise_distances_argmin
			Y = self.cluster_centers_
			X = pairwise_distances_argmin(X, Y=Y, metric=self.metric)
		else:
			raise NotImplementedError("This API is not safe to use with precomputed distances. Use the argmin of the distances to the medoids.")
		return self.medoid_indices_[X]

	def transform(self, X):
		"""Transforms X to cluster-distance space.

		:param X: Data to transform
		:type X: {array-like}, shape (n_query, n_features), or (n_query, n_indexed) if metric == 'precomputed'

		:return: X transformed in the new space of distances to cluster centers
		:rtype: {array-like}, shape=(n_query, n_clusters)
		"""
		if self.metric == "precomputed":
			return X[:, self.medoid_indices_]
		else:
			from sklearn.metrics.pairwise import pairwise_distances
			Y = self.cluster_centers_
			return pairwise_distances(X, Y=Y, metric=self.metric, metric_params=self.metric_params)

	def fit_predict(self, X, y=None):
		"""Predict the closest cluster for each sample in X.

		:param X: Input data
		:type X: array-like of shape (n_samples, n_features)
		:param y: Not used, present for API consistency by convention
		:type y: Ignored

		:return: Cluster labels
		:rtype: ndarray of shape (n_samples,)
		"""
		self.fit(X)
		return self.labels_
