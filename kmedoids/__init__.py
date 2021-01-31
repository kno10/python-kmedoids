"""K-medoids Clustering in Python.

This package implements common k-medoids clustering algorithms,
in decreasing order of performance:

- FasterPAM
- FastPAM (same result as PAM; but faster)
- PAM (the original Partitioning Around Medoids algorithm)
- Alternating (k-means style algorithm, yields results of lower quality)
- BUILD (the initialization of PAM)

References
----------
Erich Schubert, Peter J. Rousseeuw
Fast and Eager k-Medoids Clustering:
O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
Under review at Information Systems, Elsevier.
Preprint: https://arxiv.org/abs/2008.05171

Erich Schubert, Peter J. Rousseeuw:
Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.
https://doi.org/10.1007/978-3-030-32047-8_16
Preprint: https://arxiv.org/abs/1810.05691

Leonard Kaufman, Peter J. Rousseeuw:
Clustering by means of medoids.
In: Dodge Y (ed) Statistical Data Analysis Based on the L 1 Norm and Related Methods, 405-416, 1987

Leonard Kaufman, Peter J. Rousseeuw:
Finding Groups in Data: An Introduction to Cluster Analysis.
John Wiley&Sons, 1990, https://doi.org/10.1002/9780470316801
"""
__all__ = [
	"pam",
	"fastpam",
	"fasterpam",
	"alternating",
	"KMedoidsResult"
]
from .kmedoids import KMedoidsResult

def _check_medoids(diss, medoids, init, random_state):
	"""Check the medoids and random_state parameters."""
	import numpy as np, numbers
	if isinstance(medoids, np.ndarray):
		if random_state is not None:
			warnings.warn("Seed will be ignored if initial medoids are given")
		return medoids
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
			raise ValueError("Pass a numpy random generator, state or integer seed")
		return random_state.randint(0, diss.shape[0], medoids)
	raise ValueError("Specify the number of medoids, or give a numpy array of initial medoids")

def fasterpam(diss, medoids, max_iter=100, init="random", random_state=None):
	"""FasterPAM k-medoids clustering

	This is an accelerated version of PAM clustering, that eagerly
	performs any swap found, and contains the O(k) improvement to find
	the best swaps faster.

	References
	----------
	Erich Schubert, Peter J. Rousseeuw
	Fast and Eager k-Medoids Clustering:
	O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
	Under review at Information Systems, Elsevier.
	Preprint: https://arxiv.org/abs/2008.05171

	Erich Schubert, Peter J. Rousseeuw:
	Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
	In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.
	https://doi.org/10.1007/978-3-030-32047-8_16
	Preprint: https://arxiv.org/abs/1810.05691

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
	from .kmedoids import _fasterpam_i32, _fasterpam_i64, _fasterpam_f32, _fasterpam_f64

	if not isinstance(diss, np.ndarray):
		diss = np.array(diss)

	medoids = _check_medoids(diss, medoids, init, random_state)

	if isinstance(diss, np.ndarray):
		dtype = diss.dtype
		if dtype == np.float32:
			return _fasterpam_f32(diss, medoids.astype(np.uint64), max_iter)
		elif dtype == np.float64:
			return _fasterpam_f64(diss, medoids.astype(np.uint64), max_iter)
		elif dtype == np.int32:
			return _fasterpam_i32(diss, medoids.astype(np.uint64), max_iter)
		elif dtype == np.int64:
			return _fasterpam_i64(diss, medoids.astype(np.uint64), max_iter)
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def fastpam1(diss, medoids, max_iter=100, init="random", random_state=None):
	"""FastPAM1 k-medoids clustering

	This is an accelerated version of PAM clustering, that performs the
	same swaps as the original PAM (given the same starting conditions),
	but finds the best swap O(k) times faster.

	References
	----------
	Erich Schubert, Peter J. Rousseeuw
	Fast and Eager k-Medoids Clustering:
	O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
	Under review at Information Systems, Elsevier.
	Preprint: https://arxiv.org/abs/2008.05171

	Erich Schubert, Peter J. Rousseeuw:
	Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
	In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.
	https://doi.org/10.1007/978-3-030-32047-8_16
	Preprint: https://arxiv.org/abs/1810.05691

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
			return _fastpam1_f32(diss, medoids.astype(np.uint64), max_iter)
		elif dtype == np.float64:
			return _fastpam1_f64(diss, medoids.astype(np.uint64), max_iter)
		elif dtype == np.int32:
			return _fastpam1_i32(diss, medoids.astype(np.uint64), max_iter)
		elif dtype == np.int64:
			return _fastpam1_i64(diss, medoids.astype(np.uint64), max_iter)
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def pam_build(diss, k):
	"""PAM k-medoids clustering -- BUILD only

	This is an implementation of the original PAM (Partitioning Around Medoids)
	clustering algorithm. For improved versions, see the fastpam and fasterpam methods.

	References
	----------
	Leonard Kaufman, Peter J. Rousseeuw:
	Clustering by means of medoids.
	In: Dodge Y (ed) Statistical Data Analysis Based on the L 1 Norm and Related Methods, 405-416, 1987

	Leonard Kaufman, Peter J. Rousseeuw:
	Finding Groups in Data: An Introduction to Cluster Analysis.
	John Wiley&Sons, 1990, https://doi.org/10.1002/9780470316801

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
			return _pam_build_f32(diss, k)
		elif dtype == np.float64:
			return _pam_build_f64(diss, k)
		elif dtype == np.int32:
			return _pam_build_i32(diss, k)
		elif dtype == np.int64:
			return _pam_build_i64(diss, k)
	raise ValueError("Input data not supported. Use a numpy array of floats.")

def pam(diss, medoids, max_iter=100, init="build", random_state=None):
	"""PAM k-medoids clustering

	This is an implementation of the original PAM (Partitioning Around Medoids)
	clustering algorithm. For improved versions, see the fastpam and fasterpam methods.

	References
	----------
	Leonard Kaufman, Peter J. Rousseeuw:
	Clustering by means of medoids.
	In: Dodge Y (ed) Statistical Data Analysis Based on the L 1 Norm and Related Methods, pp 405–416, 1987

	Leonard Kaufman, Peter J. Rousseeuw:
	Finding Groups in Data: An Introduction to Cluster Analysis.
	John Wiley&Sons, 1990, https://doi.org/10.1002/9780470316801

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
			return _pam_swap_f32(diss, medoids, max_iter)
		elif dtype == np.float64:
			return _pam_swap_f64(diss, medoids, max_iter)
		elif dtype == np.int32:
			return _pam_swap_i32(diss, medoids, max_iter)
		elif dtype == np.int64:
			return _pam_swap_i64(diss, medoids, max_iter)
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
			return _alternating_f32(diss, medoids.astype(np.uint64), max_iter)
		elif dtype == np.float64:
			return _alternating_f64(diss, medoids.astype(np.uint64), max_iter)
		elif dtype == np.int32:
			return _alternating_i32(diss, medoids.astype(np.uint64), max_iter)
		elif dtype == np.int64:
			return _alternating_i64(diss, medoids.astype(np.uint64), max_iter)
	raise ValueError("Input data not supported. Use a numpy array of floats.")

