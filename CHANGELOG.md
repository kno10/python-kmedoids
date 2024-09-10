# Changelog

For changes to the main Rust package, please see <https://github.com/kno10/rust-kmedoids/blob/main/CHANGELOG.md>

## kmedoids 0.5.2 (2024-09-10)

- fix clippy warnings
- update to pyo3 0.21, numpy 0.21
- update github action for python 3.13 and musllinux

## kmedoids 0.5.1 (2024-03-14)

- DynMSC: best loss reported incorrectly if best k=2
- add minimum k parameter
- bump rayon version (no changes)
- use pointer-sized np.uintp type for medoids, for wasm32 target
  that should match Rust usize.

## kmedoids 0.5.0 (2023-12-10)

- add DynMSC, Silhouette clustering with optimal number of clusters
- update dependency versions

## kmedoids 0.4.3 (2023-04-20)

- fix silhouette evaluation for k > 2 (in Rust)
- use np.unique in Python wrapper to ensure labels are 0..k

## kmedoids 0.4.2 (2023-03-07)

- fix predict for non-precomputed distances
- add CITATION.cff for github
- update dependency versions

## kmedoids 0.4.1 (2022-09-24)

- drop a leftover println, remove Display/Debug traits
- optimize marginally the MSC loss function computation
- fix return value inconsistency in Python wrapper with n_cpu set

## kmedoids 0.4.0 (2022-09-24)

- add clustering by optimizing the Silhouette: PAMSIL
- add medoid silhouette
- add medoid silhouette clustering: PAMMEDSIL, FastMSC, FasterMSC

## kmedoids 0.3.3 (2022-07-06)

- Improved platform support (prebuilt for manylinux, OSX, Windows) by David Muhr
- Rust: small but fix in PAM BUILD (ignoring the first object)

## kmedoids 0.3.2 (2022-06-25)

- Rust: small bug fix in PAM BUILD (noticable for tiny data sets with large k only)
- Rust: return less than k centers in BUILD if the total deviation already is 0 (less than k unique points)
- documentation improvement and packaging improvements in Python bindings

## kmedoids 0.3.1 (2022-04-05)

- fix missing import of warnings on bad parameters
- use "choice" instead of "randint" in Python initialization code
- no changes to Rust side, so no 0.3.1 of the Rust module

## kmedoids 0.3.0 (2022-03-27)

- add a sklearn compatible API (but keep sklearn an optional dependency)
- improve documentation and installation instructions
- add MNIST example
- add integration tests

## kmedoids 0.2.2 (2022-01-28)

- really fix incorrect call to numpy random for seeding

## kmedoids 0.2.1 (2022-01-28)

- fix incorrect call to numpy random

## kmedoids 0.2.0 (2022-01-07)

- make KMedoidsResult a native Python object, pass a tuple from Rust
- always use i64/f64 loss, even for i32/f32 input
- add random shuffling support
- add parallelization support

## kmedoids 0.1.6 (2021-09-02)

- update reference with published journal version
- update dependency versions (no changes)

