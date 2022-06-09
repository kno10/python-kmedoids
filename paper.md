---
title: 'Fast k-medoids Clustering in Rust and Python'
tags:
- k-Medoids
- Clustering
- Rust
- Python
date: "2022-01-25"
output: pdf_document
authors:
- name: Erich Schubert
  orcid: 0000-0001-9143-4880
  affiliation: 1
- name: Lars Lenssen
  orcid: 0000-0003-0037-0418
  affiliation: 1
bibliography: rust-kmedoids.bib
affiliations:
- name: TU Dortmund University, Informatik VIII, 44221 Dortmund, Germany
  index: 1
---

# Summary

A popular technique to cluster non-Euclidean data using arbitrary distance
functions or similarities is k-medoids.
The k-medoids problem is NP-hard [@Kariv/Hakimi/79a], hence we need an approximate solution.
The standard algorithm for this is Partitioning Around Medoids [PAM, @Kaufman/Rousseeuw/87a; -@Kaufman/Rousseeuw/90b],
consisting of a greedy initialization (BUILD) followed by a local optimization (SWAP).
Alternatively, a k-means-style alternating optimization can be employed [@DBLP:journals/ibmsj/Maranzana63; @DBLP:journals/eswa/ParkJ09],
but this tends to produce worse results [@DBLP:journals/ior/TeitzB68; @journals/geoana/Rosing79; @DBLP:journals/jmma/ReynoldsRIR06].

FasterPAM [@DBLP:conf/sisap/SchubertR19; -@DBLP:journals/is/SchubertR21] recently introduced a speedup for larger k,
by exploiting redundancies when computing swaps for all k existing medoids.
Originally FasterPAM was implemented in Java and published within the open-source library ELKI [@Schubert/Zimek/2019a].

Here, we introduce the ``kmedoids`` Rust crate (https://github.com/kno10/rust-kmedoids) along with a
Python wrapper package ``kmedoids`` (https://github.com/kno10/python-kmedoids) to make this fast
algorithm easier to employ by researchers in various fields.
We implemented the FasterPAM approach, the original PAM, and the "Alternating" (k-means-style) approach.
The implementation can be used with arbitrary dissimilarities and distances, as it requires a dissimilarity matrix as input.

# Statement of need

To make the recent algorithmic improvements to k-medoids clustering available to a wider audience,
we made an easy to use package available to the Rust and Python communities, to enable researchers
to easily explore k-medoids clustering on their data sets, which so far is not available for example
in the popular package scikit-learn (to which we include a compatible API).

We chose Rust for the core functionality because of its high reliability, security, and performance,
and a Python wrapper for ease of use. Both parts are documented following community best practice
and available online at <https://docs.rs/kmedoids> respectively <https://python-kmedoids.readthedocs.io>.
We tried to keep library dependencies to a minimum, and some dependencies (e.g., rayon for optional parallelization)
can be disabled via the Rust "feature" functionality. For efficiently sharing data from Python to Rust,
we rely on the well-known numpy/ndarray pairing to avoid copying data.

# Performance

The original FasterPAM prototype was implemented in Java and made available as part of the ELKI open-source toolkit [@Schubert/Zimek/2019a].
Java often is not the best choice for a numerically heavy computation,
to a large extent due to memory management; but it usually is still much faster than interpreted ``pure'' Python or R code
(which can shine when used to drive compiled library code written, e.g., in C, Fortran, or Rust).
To demonstrate the benefits of this new Rust implementation, we compare it to the original Java version
(written by the same authors), and also study the additional speedup that can be obtained by parallelization using multiple threads.

We use the first N instances of the well-known MNIST data set.
As the run times are expected to be quadratic in the number of instances, we report run times normalized by N²
averaged over 25 restarts on an AMD EPYC 7302 processor with up to 16 threads for Rust.
Even without parallelization, the FasterPAM in Rust with 4.48 ns per N², is about 4 times less than the original Java FasterPAM implementation with 21.04 ns per N².
We primarily attribute this to being able to use a better memory layout than currently possible in Java
(Project Valhalla's value types may eventually help).
Using two threads in Rust, we achieve a 34% faster calculation with 2.95 ns per N²,
but we see diminishing returns when further increasing the number of threads for this data set size,
caused by the overhead and synchronization cost.
For small data sets, using a single thread appears beneficial, and the Python
wrapper defaults to this for small data sets.

![Results normalized by N² on MNIST data with k=10.\label{fig:example_mnist}](results.png){ width=100% }

# Comparison of Algorithms

Many existing libraries only implement the (worse) alternating algorithm, or the (slower) original PAM algorithm.
We want to show that using this package makes it easy to find better solutions in less time.
In practice, it is feasible to run multiple random restarts of FasterPAM, because the run time of the optimization
is usually smaller than the time needed to compute the (reusable) distance matrix.
Nevertheless, computing the distance matrix needs O(N²) time and memory,
making the algorithm only a good choice for less than 100,000 instances
(for large data sets, it likely is reasonable to use subsampling).

We compare our implementation with alternative k-medoids implementations and algorithms:
``sklearn_extra.cluster.KMedoids`` [v0.2.0, @sklearnextra],
``PyClustering`` [v0.10.1.2, @Novikov/2019],
``biopython`` [v1.79, @Cock/2009],
and ``BanditPAM`` [v3.0.2, @Tiwari/2020].

Our implementations (via the python wrapper) are the fastest for all algorithms (PAM, Alternating, and FasterPAM).
As expected, the "Alternating" algorithm shows a significantly worse loss than PAM and FasterPAM in all implementations;
while PAM has a substantially worse run time than FasterPAM and Alternating.
FasterPAM achieves a similar loss to PAM (the measured differences are due to random initialization) at the shortest run time.

| **implementation**       | **algorithm**      | **language**     | **ns per N²** | **average loss** |
|--------------------------|--------------------|---------------------|------------|------------------|
| ``kmedoids`` &nbsp;      | FasterPAM &nbsp;   | Python, Rust &nbsp; | **5.53**   | 18755648     |
| ``ELKI`` &nbsp;          | FasterPAM &nbsp;   | Java &nbsp;         | 17.81      | **18744453** |
| ``kmedoids`` &nbsp;      | Alternating &nbsp; | Python, Rust &nbsp; | **8.91**   | 19238742     |
| ``ELKI`` &nbsp;          | Alternating &nbsp; | Java &nbsp;         | 12.80      | 19238868     |
| ``sklearn_extra`` &nbsp; | Alternating &nbsp; | Python &nbsp;       | 13.44      | 19238742     |
| ``biopython`` &nbsp;     | Alternating &nbsp; | Python, C &nbsp;    | 13.68      | 19702804     |
| ``kmedoids`` &nbsp;      | PAM &nbsp;         | Python, Rust &nbsp; | **212.34** | 18780639     |
| ``ELKI`` &nbsp;          | PAM &nbsp;         | Java &nbsp;         | 787.55     | 18764896     |
| ``sklearn_extra`` &nbsp; | PAM &nbsp;         | Python &nbsp;       | 1506.52    | 18755237     |
| ``PyClustering`` &nbsp;  | PAM &nbsp;         | Python, C++ &nbsp;  | 76957.64   | 18753892     |

Table: Results on first 10000 MNIST instances with k = 10.

Because BanditPAM cannot handle precomputed distance matrices, we evaluate BanditPAM separately, including the run time for the distance computations.
On average, for MNIST 5000, 10000, 15000, and 20000 samples, BanditPAM was 55 times slower than FasterPAM in Rust.
While BanditPAM claims "almost linear run time" [@Tiwari/2020], whereas FasterPAM has quadratic run time,
BanditPAM appears to have substantial overhead,^[c.f. bug report https://github.com/ThrunGroup/BanditPAM/issues/175]
and a break-even point likely is beyond 500000 samples for MNIST
(a size where the memory consumption of the distance matrix makes a stored-distances approach prohibitive to use).

# Conclusions

We provide a fast Rust implementation of the FasterPAM algorithm,
with optional parallelization, and an easy-to-use Python wrapper.
K-medoids clustering is a useful clustering algorithm in many domains where
the input data is not continuous, and where Euclidean distance is not suitable,
and with these packages, we hope to make this algorithm easier accessible to
data scientists in various fields, while the source code helps researchers in
data mining to further improve clustering algorithms.

# References
