.. py:module:: kmedoids

Fast k-medoids clustering in Python
===================================

This package is a wrapper around the fast
`Rust k-medoids package <https://github.com/kno10/rust-kmedoids>`_,
implementing the FasterPAM and FastPAM algorithms
along with the baseline k-means-style and PAM algorithms.
Furthermore, the (Medoid) Silhouette can be optimized by the
FasterMSC, FastMSC, PAMMEDSIL and PAMSIL algorithms.

All algorithms expect a *distance matrix* and the number of clusters as input.
They hence can be used with arbitrary dissimilarity functions.

If you use this code in scientific work, please cite the papers in the :ref:`References`. Thank you.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Installation
============

.. image:: https://badge.fury.io/py/kmedoids.svg
   :alt: PyPI version
   :target: https://pypi.org/project/kmedoids/
.. image:: https://anaconda.org/conda-forge/kmedoids/badges/version.svg
   :alt: Conda Version
   :target: https://anaconda.org/conda-forge/kmedoids
.. image:: https://img.shields.io/conda/pn/conda-forge/kmedoids.svg
   :alt: Conda Platforms
   :target: https://anaconda.org/conda-forge/kmedoids

Installation with pip or conda
------------------------------

Pre-built packages for many Linux, Windows, and OSX systems are available in PyPI and conda-forge can be installed with

- :kbd:`pip install kmedoids` respectively
- :kbd:`conda install -c conda-forge kmedoids`.

On uncommon architectures, you may need to first install Cargo (i.e., the Rust programming language) first, and a subsequent pip install kmedoids will try to compile the package for your CPU architecture and operating system.

Compilation from source
-----------------------

You need to have Rust and Python 3 installed.

Installation uses
`maturin <https://github.com/PyO3/maturin#maturin>`_,
for compiling and installing Rust extensions.
Maturin is best used within a Python virtual environment.

.. code-block:: sh

   # activate your desired virtual environment first
   pip install maturin
   git clone https://github.com/kno10/python-kmedoids.git
   cd python-kmedoids
   # build and install the package:
   maturin develop --release

Integration test to validate the installation.

.. code-block:: sh

   python -m unittest discover tests

This procedure uses the latest git version from https://github.com/kno10/rust-kmedoids.
If you want to use local modifications to the Rust code, you need to provide the source folder of the Rust module in :kbd:`Cargo.toml`
by setting the :kbd:`path=` option of the :kbd:`kmedoids` dependency.

Example
=======

.. code-block:: python

   import kmedoids
   c = kmedoids.fasterpam(distmatrix, 5)
   print("Loss is:", c.loss)

Using the sklearn-compatible API
-------------------

Note that KMedoids defaults to the `"precomputed"` metric, expecting a pairwise distance matrix.
If you have sklearn installed, you can use `metric="euclidean"`.

.. code-block:: python

   import kmedoids
   km = kmedoids.KMedoids(5, method='fasterpam')
   c = km.fit(distmatrix)
   print("Loss is:", c.inertia_)

MNIST (10k samples)
-------------------

.. code-block:: python

	import kmedoids
	import numpy
	from sklearn.datasets import fetch_openml
	from sklearn.metrics.pairwise import euclidean_distances
	X, _ = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
	X = X[:10000]
	diss = euclidean_distances(X)
	start = time.time()
	fp = kmedoids.fasterpam(diss, 100)
	print("FasterPAM took: %.2f ms" % ((time.time() - start)*1000))
	print("Loss with FasterPAM:", fp.loss)
	start = time.time()
	pam = kmedoids.pam(diss, 100)
	print("PAM took: %.2f ms" % ((time.time() - start)*1000))
	print("Loss with PAM:", pam.loss)

Choose the optimal number of clusters
-------------------

.. code-block:: python

    import kmedoids, numpy
    from sklearn.datasets import fetch_openml
    from sklearn.metrics.pairwise import euclidean_distances
    X, _ = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X[:10000]
    diss = euclidean_distances(X)
    bk = kmedoids.bestk(diss, 100)
    print("Optimal number of clusters according to the Medoid Silhouette:", bk.bestk)
    print("Medoid Silhouette over range of k:", bk.losses)
    print("Range of k:", bk.rangek)

Memory Requirements
-------------------

Because the algorithms require a distance matrix as input, you need O(N²) memory to use these implementations. With single precision, this matrix needs 4·N² bytes, so a typical laptop with 8 GB of RAM could handle data sets of over 40.000 instances, but if your computation of the distance matrix incurs copying the matrix, only 30.000 or less may be feasible.

The majority of run time usually is the distance matrix computation, so it is recommended you only compute it once, then experiment with different algorithm settings. Avoid recomputing it repeatedly.

For larger data sets, it is recommended to only cluster a representative sample of the data. Usually, this will still yield sufficient result quality.

Implemented Algorithms
======================

* :ref:`FasterPAM<fasterpam>` (Schubert and Rousseeuw, 2020, 2021)
* :ref:`FastPAM1<fastpam1>` (Schubert and Rousseeuw, 2019, 2021)
* :ref:`PAM<pam>` (Kaufman and Rousseeuw, 1987) with BUILD and SWAP
* :ref:`Alternating<alternating>` (k-means-style approach)
* :ref:`BUILD<build>` (Kaufman and Rousseeuw, 1987)
* :ref:`Silhouette<silhouette>` (Kaufman and Rousseeuw, 1987)
* :ref:`FasterMSC<fastermsc>` (Lenssen and Schubert, 2022)
* :ref:`FastMSC<fastmsc>` (Lenssen and Schubert, 2022)
* :ref:`DynMSC<dynmsc>` (Lenssen and Schubert, 2023)
* :ref:`PAMSIL<pamsil>` (Van der Laan and Pollard, 2003)
* :ref:`PAMMEDSIL<pammedsil>` (Van der Laan and Pollard, 2003)
* :ref:`Medoid Silhouette<medoid_silhouette>` (Van der Laan and Pollard, 2003)

Note that the k-means style "alternating" algorithm yields rather poor result quality
(see Schubert and Rousseeuw 2021 for an example and explanation).

.. _FasterPAM:

FasterPAM
=========

.. autofunction:: fasterpam

.. _FastPAM1:

FastPAM1
========

.. autofunction:: fastpam1

.. _PAM:

PAM
===

.. autofunction:: pam

.. _Alternating:

Alternating k-medoids (k-means style)
=====================================

.. autofunction:: alternating

.. _BUILD:

PAM BUILD
=========

.. autofunction:: pam_build

.. _FasterMSC:

FasterMSC
=========

.. autofunction:: fastermsc

.. _FastMSC:

FastMSC
=========

.. autofunction:: fastmsc

.. _DynMSC:

DynMSC
=========

.. autofunction:: dynmsc

.. _PAMSIL:

PAMSIL
=========

.. autofunction:: pamsil

.. _PAMMEDSIL:

PAMMEDSIL
=========

.. autofunction:: pammedsil

.. _Silhouette:

Silhouette
=========

.. autofunction:: silhouette

.. _MedoidSilhouette:

Medoid Silhouette
=================

.. autofunction:: medoid_silhouette

.. _KMedoidsResult:

k-Medoids result object
=======================

.. autoclass:: KMedoidsResult

.. _KMedoids:

sklearn-compatible API
=====================================

.. autoclass:: KMedoids
   :members: 
   :inherited-members:
   
.. _References:

References
==========

This software has been published in the Journal of Open-Source Software:

     | Erich Schubert and Lars Lenssen:  
     | **Fast k-medoids Clustering in Rust and Python**  
     | Journal of Open Source Software 7(75), 4183  
     | https://doi.org/10.21105/joss.04183 (open access)

For further details on the implemented algorithm FasterPAM, see:

     | Erich Schubert, Peter J. Rousseeuw  
     | **Fast and Eager k-Medoids Clustering:**  
     | **O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms**  
     | Information Systems (101), 2021, 101804  
     | https://doi.org/10.1016/j.is.2021.101804 (open access)

an earlier (slower, and now obsolete) version was published as:

     | Erich Schubert, Peter J. Rousseeuw:  
     | **Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms**  
     | In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.  
     | https://doi.org/10.1007/978-3-030-32047-8_16  
     | Preprint: https://arxiv.org/abs/1810.05691

For further details on medoid Silhouette clustering with automatic cluster number selection (FasterMSC, DynMSC), see:

     | Lars Lenssen, Erich Schubert:
     | **Medoid silhouette clustering with automatic cluster number selection**
     | Information Systems (120), 2024, 102290
     | https://doi.org/10.1016/j.is.2023.102290

an earlier version was published as:

     | Lars Lenssen, Erich Schubert:  
     | **Clustering by Direct Optimization of the Medoid Silhouette**  
     | In: 15th International Conference on Similarity Search and Applications (SISAP 2022).  
     | https://doi.org/10.1007/978-3-031-17849-8_15

This is a port of the original Java code from `ELKI <https://elki-project.github.io/>`_ to Rust.
The `Rust version <https://github.com/kno10/rust-kmedoids>`_ is then wrapped for use with Python.

If you use this code in scientific work, please cite above papers. Thank you.

License: GPL-3 or later
=======================

.. code-block:: text

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

