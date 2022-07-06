# Changelog

For changes to the main Rust package, please see <https://github.com/kno10/rust-kmedoids/blob/main/CHANGELOG.md>

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

