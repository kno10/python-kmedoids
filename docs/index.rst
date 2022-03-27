.. py:module:: kmedoids

Fast k-medoids clustering in Python
===================================

This package is a wrapper around the fast
`Rust k-medoids package <https://github.com/kno10/rust-kmedoids>`_,
implementing the FasterPAM and FastPAM algorithms
along with the baseline k-means-style and PAM algorithms.

All algorithms expect a *distance matrix* and the number of clusters as input.
They hence can be used with arbitrary dissimilarity functions.

If you use this code in scientific work, please cite the papers in the :ref:`References`. Thank you.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Installation
============

Installation with pip
---------------------

Pre-built packages are on PyPi https://pypi.org/project/kmedoids/ and can be installed with :kbd:`pip install kmedoids`.

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


Implemented Algorithms
======================

* :ref:`FasterPAM<fasterpam>` (Schubert and Rousseeuw, 2020, 2021)
* :ref:`FastPAM1<fastpam1>` (Schubert and Rousseeuw, 2019, 2021)
* :ref:`PAM<pam>` (Kaufman and Rousseeuw, 1987) with BUILD and SWAP
* :ref:`Alternating<alternating>` (k-means-style approach)
* :ref:`BUILD<build>` (Kaufman and Rousseeuw, 1987)
* :ref:`Silhouette<silhouette>` (Kaufman and Rousseeuw, 1987)

Note that the k-means style "alternating" algorithm yields rather poor result quality.

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

Alternating k=medoids (k-means style)
=====================================

.. autofunction:: alternating

.. _BUILD:

PAM BUILD
=========

.. autofunction:: pam_build

.. _Silhouette:

Silhouette
=========

.. autofunction:: silhouette

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

FAQ: Why GPL and not Apache/MIT/BSD?
------------------------------------

Because copyleft software like Linux is what built the open-source community.

Tit for tat: you get to use my code, I get to use your code.
