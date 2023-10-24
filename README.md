# k-Medoids Clustering in Python with FasterPAM

[![PyPI version](https://badge.fury.io/py/kmedoids.svg)](https://pypi.org/project/kmedoids/)
[![Conda Version](https://anaconda.org/conda-forge/kmedoids/badges/version.svg)](https://anaconda.org/conda-forge/kmedoids)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/kmedoids.svg)](https://anaconda.org/conda-forge/kmedoids)

This python package implements k-medoids clustering with PAM and variants of clustering by direct optimization of the (Medoid) Silhouette.
It can be used with arbitrary dissimilarites, as it requires a dissimilarity matrix as input.

This software package has been introduced in JOSS:

> Erich Schubert and Lars Lenssen  
> **Fast k-medoids Clustering in Rust and Python**  
> Journal of Open Source Software 7(75), 4183  
> <https://doi.org/10.21105/joss.04183> (open access)

For further details on the implemented algorithm FasterPAM, see:

> Erich Schubert, Peter J. Rousseeuw  
> **Fast and Eager k-Medoids Clustering:**  
> **O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms**  
> Information Systems (101), 2021, 101804  
> <https://doi.org/10.1016/j.is.2021.101804> (open access)

an earlier (slower, and now obsolete) version was published as:

> Erich Schubert, Peter J. Rousseeuw:  
> **Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms**  
> In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.  
> <https://doi.org/10.1007/978-3-030-32047-8_16>  
> Preprint: <https://arxiv.org/abs/1810.05691>

This is a port of the original Java code from [ELKI](https://elki-project.github.io/) to Rust.
The [Rust version](https://github.com/kno10/rust-kmedoids) is then wrapped for use with Python.

For further details on medoid Silhouette clustering with automatic cluster number selection (FasterMSC, DynMSC), see:

> Lars Lenssen, Erich Schubert:  
> **Medoid silhouette clustering with automatic cluster number selection**  
> Information Systems (120), 2024, 102290
> <https://doi.org/10.1016/j.is.2023.102290>

an earlier version was published as:

> Lars Lenssen, Erich Schubert:  
> **Clustering by Direct Optimization of the Medoid Silhouette**  
> In: 15th International Conference on Similarity Search and Applications (SISAP 2022)  
> <https://doi.org/10.1007/978-3-031-17849-8_15>

If you use this code in scientific work, please cite above papers. Thank you.

## Documentation

Full python documentation is included, and available on
[python-kmedoids.readthedocs.io](https://python-kmedoids.readthedocs.io/en/latest/)

## Installation

### Installation with pip or conda

Pre-built packages for many Linux, Windows, and OSX systems are available
in [PyPI](https://pypi.org/project/kmedoids/) and
[conda-forge](https://anaconda.org/conda-forge/kmedoids)
can be installed with
* `pip install kmedoids` respectively
* `conda install -c conda-forge kmedoids`.

On uncommon architectures, you may need to first
[install Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
(i.e., the Rust programming language) first, and a subsequent
`pip install kmedoids` will try to compile the package for your CPU architecture and operating system.

### Compilation from source

You need to have Python 3 installed.

Unless you already have Rust, [install Rust/Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html).

Installation uses [maturin](https://github.com/PyO3/maturin#maturin) for compiling and installing the Rust extension.
Maturin is best used within a Python virtual environment:
```sh
# activate your desired virtual environment first, then:
pip install maturin
git clone https://github.com/kno10/python-kmedoids.git
cd python-kmedoids
# build and install the package:
maturin develop --release
```
Integration test to validate the installation.
```sh
pip install numpy
python -m unittest discover tests
```

This procedure uses the latest git version from <https://github.com/kno10/rust-kmedoids>.
If you want to use local modifications to the Rust code, you need to provide the source folder of the Rust module in `Cargo.toml`
by setting the `path=` option of the `kmedoids` dependency.

## Example

Given a distance matrix `distmatrix`, cluster into `k = 5` clusters:

```python
import kmedoids
c = kmedoids.fasterpam(distmatrix, 5)
print("Loss is:", c.loss)
```

### Using the sklearn-compatible API

Note that KMedoids defaults to the `"precomputed"` metric, expecting a pairwise distance matrix.
If you have sklearn installed, you can also use `metric="euclidean"` and other distances supported by sklearn.

```python
import kmedoids
km = kmedoids.KMedoids(5, method='fasterpam')
c = km.fit(distmatrix)
print("Loss is:", c.inertia_)
```

### MNIST (10k samples)

```python
import kmedoids, numpy, time
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
```

### Choose the optimal number of clusters

```python
import kmedoids, numpy
from sklearn.datasets import fetch_openml
from sklearn.metrics.pairwise import euclidean_distances
X, _ = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X[:10000]
diss = euclidean_distances(X)
dm = kmedoids.dynmsc(diss, 100)
print("Optimal number of clusters according to the Medoid Silhouette:", dm.bestk)
print("Medoid Silhouette over range of k:", dm.losses)
print("Range of k:", dm.rangek)
```

### Memory Requirements

Because the algorithms require a distance matrix as input, you need O(N²) memory to use these implementations. With single precision, this matrix needs 4·N² bytes, so a typical laptop with 8 GB of RAM could handle data sets of over 40.000 instances, but if your computation of the distance matrix incurs copying the matrix, only 30.000 or less may be feasible.

The majority of run time usually is the distance matrix computation, so it is recommended you only compute it once, then experiment with different algorithm settings. Avoid recomputing it repeatedly.

For larger data sets, it is recommended to only cluster a representative sample of the data. Usually, this will still yield sufficient result quality.

## Implemented Algorithms

* **FasterPAM** (Schubert and Rousseeuw, 2020, 2021)
* FastPAM1 (Schubert and Rousseeuw, 2019, 2021)
* PAM (Kaufman and Rousseeuw, 1987) with BUILD and SWAP
* Alternating optimization (k-means-style algorithm)
* Silhouette index for evaluation (Rousseeuw, 1987)
* **FasterMSC** (Lenssen and Schubert, 2022)
* FastMSC (Lenssen and Schubert, 2022)
* DynMSC (Lenssen and Schubert, 2023)
* PAMSIL (Van der Laan and Pollard, 2003)
* PAMMEDSIL (Van der Laan and Pollard, 2003)
* Medoid Silhouette index for evaluation (Van der Laan and Pollard, 2003)

Note that the k-means-like algorithm for k-medoids tends to find much worse solutions.

## Contributing to `python-kmedoids`

Third-party contributions are welcome. Please use [pull requests](https://github.com/kno10/python-kmedoids/pulls) to submit patches.

## Reporting issues

Please report errors as an [issue](https://github.com/kno10/python-kmedoids/issues) within the repository's issue tracker.

## Support requests

If you need help, please submit an [issue](https://github.com/kno10/python-kmedoids/issues) within the repository's issue tracker.

## License: GPL-3 or later

> This program is free software: you can redistribute it and/or modify
> it under the terms of the GNU General Public License as published by
> the Free Software Foundation, either version 3 of the License, or
> (at your option) any later version.
> 
> This program is distributed in the hope that it will be useful,
> but WITHOUT ANY WARRANTY; without even the implied warranty of
> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
> GNU General Public License for more details.
> 
> You should have received a copy of the GNU General Public License
> along with this program.  If not, see <https://www.gnu.org/licenses/>.
