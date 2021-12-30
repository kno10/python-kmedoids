---
title: 'k-Medoids Clustering in Rust with FasterPAM'
tags:
- k-Medoids
- Clustering
- Rust
date: "07.10.2021"
output: pdf_document
authors:
- name: Erich Schubert
  orcid: 0000-0001-9143-4880
  affiliation: 1
- name: Lars Lenssen
  orcid: 0000-0003-0037-0418
  affiliation: 1
bibliography: rust-kmedoids.bib
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: 
affiliations:
- name: TU Dortmund University, Informatik VIII, 44221 Dortmund, Germany
  index: 1
---

# Summary

A popular technique to cluster non-Euclidean data using arbitrary distance
functions or similarities is k-medoids. The k-medoids problem is NP-hard [@Kariv/Hakimi/79a], hence we need an approximative solution. The best known algorithm for a heuristic
solution with local optimization techniques is Partioning Around Medoids [PAM, @Kaufman/Rousseeuw/87a; -@Kaufman/Rousseeuw/90b], which uses a greedy search, that is significantly faster than an exhaustive search.
FasterPAM [@Schubert/Rousseeuw/2021a] recently introduced a speedup for larger k, by clever caching of partial results. Originally FasterPAM was implemented in Java and published within the open-source library ELKI [@Schubert/Zimek/2019a]. 

We developed the ``rust-kmedoids`` crate (https://github.com/kno10/rust-kmedoids) with implementations of various algorithms of k-medoids clustering, including FasterPAM. It can be used with arbitrary dissimilarites, as it requires a dissimilarity matrix as input. We also provide optional parallelization using rayon.

# Statement of need



# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

# References

