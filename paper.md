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
The standard algorithm for this is Partioning Around Medoids [PAM, @Kaufman/Rousseeuw/87a; -@Kaufman/Rousseeuw/90b],
consisting of a greedy initialization (BUILD) followed by a local optimization (SWAP).
Alternatively, a k-means-style alternating optimization can be employed [@DBLP:journals/ibmsj/Maranzana63; @DBLP:journals/eswa/ParkJ09],
but this tends to produce worse results [@DBLP:journals/ior/TeitzB68; @journals/geoana/Rosing79; @DBLP:journals/jmma/ReynoldsRIR06].

FasterPAM [@DBLP:conf/sisap/SchubertR19; -@DBLP:journals/is/SchubertR21] recently introduced a speedup for larger k,
by exploiting redundancies when computing swaps for all k existing medoids.
Originally FasterPAM was implemented in Java and published within the open-source library ELKI [@Schubert/Zimek/2019a].

Here, we introduce ``kmedoids`` Rust crate (https://github.com/kno10/rust-kmedoids) along with a
Python wrapper package ``kmedoids`` (https://github.com/kno10/python-kmedoids) to make this fast
algorithm easier to employ by researchers in various fields.
We implemented both the FasterPAM approach, the original PAM, and the Alternating (k-means-style) approach.
The implementation can be used with arbitrary dissimilarites, as it requires a dissimilarity matrix as input.

We chose Rust for the core functionality because of its high reliability and security as well as performance,
and a Python wrapper for ease of use. Both parts are documented following community best practice
and available online at (https://docs.rs/kmedoids/) respectively (https://python-kmedoids.readthedocs.io).
We tried to keep library dependencies to a minimum, and some dependencies (e.g., rayon for optional parallelization)
can be disabled via the Rust "feature" functionality. For efficiency of sharing data from Python to Rust,
we rely on the well known numpy/ndarray pairing.

# Performance

The original FasterPAM prototype was implemented in Java and made available as part of the ELKI open-source toolkit [@Schubert/Zimek/2019a].
It is well known that Java often is not the best choice for a numerically heavy computation,
to a large extend due to memory management; but usually still much faster than interpreted Python or R code
(but which can shine when they are used to drive compiled library code written, e.g., in C, Fortran, or Rust).
To demonstrate the benefits of this new Rust implementation, we compare it to the original Java version
(written by the same authors), and the additional speedup that can be obtained by parallelization using multiple threads.

We run 25 restarts on an AMD EPYC 7302 processor, and evaluate the average values.
We use the first N instances of the well known MNIST data set.
As the run times are expected to be quadratic in the number of instances, we report run times normalized by N².
The Java implementation uses a single thread, whereas for Rust we report up to 16 threads on 16 cores.
Even without parallelization, the FasterPAM in Rust with 4.48 ns/N², about 4 times less than the original Java FasterPAM implementation with 21.04 ns/N².
We primarily attribute this to being able to use a better memory layout than currently possible in Java
(project Valhalla's value types may eventually allow reducing this gap).
Using two threads in Rust, we achieve a 34% faster calculation with 2.95 ns/N²,
but as we further increase the number of threads the gains diminish for this data set size,
caused by the overhead to partition the computation and synchronize the work.
It may be possible to further decrease this overhead. For small data sets, using a single
thread often is beneficial, and the Python wrapper will choose this approach
by default for small data sets.

TODO: refresh figure

![Results normalized by N² on MNIST data with k=10.\label{fig:example_mnist}](results.png){ width=100% }

# Comparison of Algorithms

Many existing software libraries for k-medoids only implement the (worse) alternating algorithm, or the (slower) original PAM algorithm.
We want to show that using this package makes it easy to find better solutions in less time.
In practice, it is feasible to run multiple random restarts of FasterPAM, because the run time of the optimization
is usually smaller than the time needed to compute the (reusable) distance matrix.
Nevertheless, computing the distance matrix clearly needs O(N²) time and memory,
making the algorithm only a good choice for less than 100,000 instances
(for large data sets, it likely is reasonable to use subsampling).

TODO: experiments. Only report N=10000 as a table implementation/language/runtime/loss?

TODO: mention BanditPAM only briefly, that it was several times slower, too?

# Benchmark

To benchmark the FasterPAM implementation in Rust, we compare the run time of alternative k-medoids implementations with our Rust implementation.
k-medoids alternating implementation of ``sklearn_extra.cluster.KMedoids``  (https://github.com/scikit-learn-contrib/scikit-learn-extra) in Python,
``PyClustering`` [@Novikov/2019] in Python and C++, and
``BanditPAM`` [@Tiwari/2020] in Python and C++.
We choose a random initialization and precompute a distance matrix with Euclidean distance (L2 norm).
BanditPAM cannot handle precomputed distance matrices, hence we evaluate BanditPAM separately with including of run time for distance computation.

The fastest version without parallel processing is the FasterPAM in Rust with 4.48 ns/N². The original Java FasterPAM implementation took 21.04 ns/N², 4 times longer. Also sklearn-extra is slower with 13.61 ns/N², which corresponds to a speedup factor of 3. With 135257 ns/N² is PyClustering almost 30000 times slower, so \autoref{fig:example_mnist} shows only sklearn-extra and FasterPAM with its variants. 

Since BanditPAM cannot process precomputed distance matrices, here we compare the run time of BanditPAM with that of FasterPAM in Rust including the calculation time for the full distance matrix. BanditPAM for MNIST 5000, 10000, 15000, and 20000 samples was on average 55 times slower than FasterPAM in Rust. Since BanditPAM with its "almost linear run time" [@Tiwari/2020] scales better than FasterPAM with quadratic run time, a break-even point can be estimated to be beyond 500000 samples for MNIST (a size where the memory consumption of the distance matrix makes a stored-distance approach prohibitive to use).

| **samples** | **implementation** | **average loss** | **run time in s** |
|---------|----------------|---------|----------|---------|----------|
|     5000    |         FasterPAM (Rust) / BanditPAM &nbsp; &nbsp;          |    9365549 / **9361719** &nbsp;     |    **1.48** / 133.19     |
|     10000   |         FasterPAM (Rust) / BanditPAM &nbsp; &nbsp;          |    **18741470** / 18742086 &nbsp;   |    **3.49** / 190.62     |
|     15000   |         FasterPAM (Rust) / BanditPAM &nbsp; &nbsp;          |    28283025 / **28261579** &nbsp;   |    **8.21** / 336.29     |
|     20000   |         FasterPAM (Rust) / BanditPAM &nbsp; &nbsp;          |    37619359 / **37575315** &nbsp;   |    **15.46** / 601.49    |



# References

