# Changelog

For changes to the main Rust package, please see <https://github.com/kno10/rust-kmedoids/blob/main/CHANGELOG.md>

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

