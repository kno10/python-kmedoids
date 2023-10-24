use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use rand::{rngs::StdRng, SeedableRng};
use rayon;
use rustkmedoids;

macro_rules! variant_call {
($name:ident, $variant:ident, $type: ty, $ltype: ty) => {
/// Run $variant k-medoids clustering function for $type precision
///
/// :param dist: distance matrix
/// :type dist: ndarray
/// :param meds: initial medoids
/// :type meds: ndarray
/// :param max_iter: maximum number of iterations
/// :type max_iter: int
/// :return: k-medoids clustering result
/// :rtype: KMedoidsResult
#[pyfunction]
fn $name(dist: PyReadonlyArray2<'_, $type>, meds: PyReadonlyArray1<'_, usize>, max_iter: usize) -> PyResult<Py<PyAny>> {
    assert_eq!(dist.ndim(), 2);
    assert_eq!(dist.shape()[0], dist.shape()[1]);
    let mut meds = meds.to_vec()?;
    let (loss, assi, n_iter, n_swap): ($ltype, _, _, _) = rustkmedoids::$variant(&dist.as_array(), &mut meds, max_iter);
    Python::with_gil(|py| -> PyResult<Py<PyAny>> {
      Ok((loss, PyArray1::from_vec(py, assi), PyArray1::from_vec(py, meds), n_iter, n_swap).to_object(py))
    })
}
}}
variant_call!(fasterpam_f32, fasterpam, f32, f64);
variant_call!(fasterpam_f64, fasterpam, f64, f64);
variant_call!(fasterpam_i32, fasterpam, i32, i64);
variant_call!(fasterpam_i64, fasterpam, i64, i64);
variant_call!(fastpam1_f32, fastpam1, f32, f64);
variant_call!(fastpam1_f64, fastpam1, f64, f64);
variant_call!(fastpam1_i32, fastpam1, i32, i64);
variant_call!(fastpam1_i64, fastpam1, i64, i64);
variant_call!(pam_swap_f32, pam_swap, f32, f64);
variant_call!(pam_swap_f64, pam_swap, f64, f64);
variant_call!(pam_swap_i32, pam_swap, i32, i64);
variant_call!(pam_swap_i64, pam_swap, i64, i64);
variant_call!(pammedsil_swap_f32, pammedsil_swap, f32, f64);
variant_call!(pammedsil_swap_f64, pammedsil_swap, f64, f64);
variant_call!(pamsil_swap_f32, pamsil_swap, f32, f64);
variant_call!(pamsil_swap_f64, pamsil_swap, f64, f64);
variant_call!(fastmsc_f32, fastmsc, f32, f64);
variant_call!(fastmsc_f64, fastmsc, f64, f64);
variant_call!(fastermsc_f32, fastermsc, f32, f64);
variant_call!(fastermsc_f64, fastermsc, f64, f64);

macro_rules! rand_call {
($name:ident, $variant:ident, $type: ty, $ltype: ty) => {
/// Run $variant k-medoids clustering function for $type precision
///
/// :param dist: distance matrix
/// :type dist: ndarray
/// :param meds: initial medoids
/// :type meds: ndarray
/// :param max_iter: maximum number of iterations
/// :type max_iter: int
/// :param seed: random seed for order permutation
/// :type seed: int
/// :return: k-medoids clustering result
/// :rtype: KMedoidsResult
#[pyfunction]
fn $name(dist: PyReadonlyArray2<'_, $type>, meds: PyReadonlyArray1<'_, usize>, max_iter: usize, seed: u64) -> PyResult<Py<PyAny>> {
    assert_eq!(dist.ndim(), 2);
    assert_eq!(dist.shape()[0], dist.shape()[1]);
    let mut meds = meds.to_vec()?;
    let mut rnd = StdRng::seed_from_u64(seed);
    let (loss, assi, n_iter, n_swap): ($ltype, _, _, _) = rustkmedoids::$variant(&dist.as_array(), &mut meds, max_iter, &mut rnd);
    Python::with_gil(|py| -> PyResult<Py<PyAny>> {
      Ok((loss, PyArray1::from_vec(py, assi), PyArray1::from_vec(py, meds), n_iter, n_swap).to_object(py))
    })
}
}}
rand_call!(rand_fasterpam_f32, rand_fasterpam, f32, f64);
rand_call!(rand_fasterpam_f64, rand_fasterpam, f64, f64);
rand_call!(rand_fasterpam_i32, rand_fasterpam, i32, i64);
rand_call!(rand_fasterpam_i64, rand_fasterpam, i64, i64);

macro_rules! par_call {
($name:ident, $variant:ident, $type: ty, $ltype: ty) => {
/// Run $variant k-medoids clustering function for $type precision
///
/// :param dist: distance matrix
/// :type dist: ndarray
/// :param meds: initial medoids
/// :type meds: ndarray
/// :param max_iter: maximum number of iterations
/// :type max_iter: int
/// :param seed: random seed for order permutation
/// :type seed: int
/// :param n_cpu: number of threads to use
/// :type n_cpu: int
/// :return: k-medoids clustering result
/// :rtype: KMedoidsResult
#[pyfunction]
fn $name(dist: PyReadonlyArray2<'_, $type>, meds: PyReadonlyArray1<'_, usize>, max_iter: usize, seed: u64, n_cpu: usize) -> PyResult<Py<PyAny>> {
    assert_eq!(dist.ndim(), 2);
    assert_eq!(dist.shape()[0], dist.shape()[1]);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(n_cpu).build().unwrap();
    let mut meds = meds.to_vec()?;
    let dist = dist.as_array();
    let (loss, assi, n_iter, n_swap): ($ltype, _, _, _) = pool.install(|| {
        let mut rnd = StdRng::seed_from_u64(seed);
        rustkmedoids::$variant(&dist, &mut meds, max_iter, &mut rnd)
    });
    Python::with_gil(|py| -> PyResult<Py<PyAny>> {
      Ok((loss, PyArray1::from_vec(py, assi), PyArray1::from_vec(py, meds), n_iter, n_swap).to_object(py))
    })
}
}}
par_call!(par_fasterpam_f32, par_fasterpam, f32, f64);
par_call!(par_fasterpam_f64, par_fasterpam, f64, f64);
par_call!(par_fasterpam_i32, par_fasterpam, i32, i64);
par_call!(par_fasterpam_i64, par_fasterpam, i64, i64);

macro_rules! pam_build_call {
($name:ident, $type: ty, $ltype: ty) => {
/// Run the PAM BUILD k-medoids clustering function for $type precision
///
/// :param dist: distance matrix
/// :type dist: ndarray
/// :param k: number of clusters
/// :type k: int
/// :return: k-medoids clustering result
/// :rtype: KMedoidsResult
#[pyfunction]
fn $name(dist: PyReadonlyArray2<'_, $type>, k: usize) -> PyResult<Py<PyAny>> {
    assert_eq!(dist.ndim(), 2);
    assert_eq!(dist.shape()[0], dist.shape()[1]);
    let (loss, assi, meds): ($ltype, _, _) = rustkmedoids::pam_build(&dist.as_array(), k);
    Python::with_gil(|py| -> PyResult<Py<PyAny>> {
      Ok((loss, PyArray1::from_vec(py, assi), PyArray1::from_vec(py, meds), 1).to_object(py))
    })
}
}}
pam_build_call!(pam_build_f32, f32, f64);
pam_build_call!(pam_build_f64, f64, f64);
pam_build_call!(pam_build_i32, i32, i64);
pam_build_call!(pam_build_i64, i64, i64);

macro_rules! alternating_call {
($name:ident, $type: ty, $ltype: ty) => {
/// Run the Alternating k-medoids clustering function for $type precision
///
/// :param dist: distance matrix
/// :type dist: ndarray
/// :param meds: initial medoids
/// :type meds: ndarray
/// :param max_iter: maximum number of iterations
/// :type max_iter: int
/// :return: k-medoids clustering result
/// :rtype: KMedoidsResult
#[pyfunction]
fn $name(dist: PyReadonlyArray2<'_, $type>, meds: PyReadonlyArray1<'_, usize>, max_iter: usize) -> PyResult<Py<PyAny>> {
    assert_eq!(dist.ndim(), 2);
    assert_eq!(dist.shape()[0], dist.shape()[1]);
    let mut meds = meds.to_vec()?;
    let (loss, assi, n_iter): ($ltype, _, _) = rustkmedoids::alternating(&dist.as_array(), &mut meds, max_iter);
    Python::with_gil(|py| -> PyResult<Py<PyAny>> {
      Ok((loss, PyArray1::from_vec(py, assi), PyArray1::from_vec(py, meds), n_iter).to_object(py))
    })
}
}}
alternating_call!(alternating_f32, f32, f64);
alternating_call!(alternating_f64, f64, f64);
alternating_call!(alternating_i32, i32, i64);
alternating_call!(alternating_i64, i64, i64);

macro_rules! dynmsc_call {
($name:ident, $type: ty, $ltype: ty) => {
/// Run $variant k-medoids clustering function for $type precision
///
/// :param dist: distance matrix
/// :type dist: ndarray
/// :param meds: initial medoids
/// :type meds: ndarray
/// :param max_iter: maximum number of iterations
/// :type max_iter: int
/// :return: k-medoids clustering result
/// :rtype: DynkResult
#[pyfunction]
fn $name(dist: PyReadonlyArray2<'_, $type>, meds: PyReadonlyArray1<'_, usize>, max_iter: usize) -> PyResult<Py<PyAny>> {
    assert_eq!(dist.ndim(), 2);
    assert_eq!(dist.shape()[0], dist.shape()[1]);
    let mut meds = meds.to_vec()?;
    let maxk = meds.len() + 1;
    let (loss, assi, n_iter, n_swap, best_meds, losses): ($ltype, _, _, _, _, _) = rustkmedoids::dynmsc(&dist.as_array(), &mut meds, max_iter);
    let bestk = best_meds.len();
    Python::with_gil(|py| -> PyResult<Py<PyAny>> {
      Ok((loss, PyArray1::from_vec(py, assi), PyArray1::from_vec(py, best_meds), bestk, PyArray1::from_vec(py, losses), (2..maxk).collect::<Vec<usize>>(), n_iter, n_swap).to_object(py))
    })
}
}}
dynmsc_call!(dynmsc_f32, f32, f64);
dynmsc_call!(dynmsc_f64, f64, f64);

macro_rules! silhouette_call {
($name:ident, $type: ty) => {
/// Run the Silhouette index evaluation for $type precision
///
/// :param dist: distance matrix
/// :type dist: ndarray
/// :param assi: cluster assignment
/// :type assi: ndarray
/// :param samples: return the per-point Silhouette values
/// :type samples: bool
/// :return: Silhouette evaluation result
/// :rtype: pair of Silhouette score and Silhouette coefficients per point
#[pyfunction]
fn $name(dist: PyReadonlyArray2<'_, $type>, assi: PyReadonlyArray1<'_, usize>, samples: bool) -> PyResult<Py<PyAny>> {
    assert_eq!(dist.ndim(), 2);
    assert_eq!(dist.shape()[0], dist.shape()[1]);
    let (sil, sils): (f64, _) = rustkmedoids::silhouette(&dist.as_array(), &assi.to_vec()?, samples);
    Python::with_gil(|py| -> PyResult<Py<PyAny>> {
      Ok((sil, PyArray1::from_vec(py, sils)).to_object(py))
    })
}
}}
silhouette_call!(silhouette_f32, f32);
silhouette_call!(silhouette_f64, f64);
silhouette_call!(silhouette_i32, i32);
// i64 not supported, as the f64 used internally may overflow

macro_rules! par_silhouette_call {
($name:ident, $type: ty) => {
/// Run the Silhouette index evaluation for $type precision
///
/// :param dist: distance matrix
/// :type dist: ndarray
/// :param assi: cluster assignment
/// :type assi: ndarray
/// :param n_cpu: number of cpu cores to use
/// :type n_cpu: int
/// :return: Silhouette evaluation result
/// :rtype: Silhouette score
#[pyfunction]
fn $name(dist: PyReadonlyArray2<'_, $type>, assi: PyReadonlyArray1<'_, usize>, n_cpu: usize) -> PyResult<f64> {
    assert_eq!(dist.ndim(), 2);
    assert_eq!(dist.shape()[0], dist.shape()[1]);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(n_cpu).build().unwrap();
    let dist = dist.as_array();
    let assi = assi.to_vec()?;
    let sil: f64 = pool.install(|| rustkmedoids::par_silhouette(&dist, &assi));
    Ok(sil)
}
}}
par_silhouette_call!(par_silhouette_f32, f32);
par_silhouette_call!(par_silhouette_f64, f64);
par_silhouette_call!(par_silhouette_i32, i32);
// i64 not supported, as the f64 used internally may overflow

macro_rules! medoid_silhouette_call {
($name:ident, $type: ty) => {
/// Run the Medoid Silhouette index evaluation for $type precision
///
/// :param dist: distance matrix
/// :type dist: ndarray
/// :param meds: medoids indexes
/// :type meds: ndarray
/// :param samples: return the per-point Medoid Silhouette values
/// :type samples: bool
/// :return: Medoid Silhouette evaluation result
/// :rtype: pair of Medoid Silhouette score and Medoid Silhouette coefficients per point
#[pyfunction]
fn $name(dist: PyReadonlyArray2<'_, $type>, meds: PyReadonlyArray1<'_, usize>, samples: bool) -> PyResult<Py<PyAny>> {
    assert_eq!(dist.ndim(), 2);
    assert_eq!(dist.shape()[0], dist.shape()[1]);
    let (sil, sils): (f64, _) = rustkmedoids::medoid_silhouette(&dist.as_array(), &meds.to_vec()?, samples);
    Python::with_gil(|py| -> PyResult<Py<PyAny>> {
      Ok((sil, PyArray1::from_vec(py, sils)).to_object(py))
    })
}
}}
medoid_silhouette_call!(medoid_silhouette_f32, f32);
medoid_silhouette_call!(medoid_silhouette_f64, f64);
medoid_silhouette_call!(medoid_silhouette_i32, i32);
// i64 not supported, as the f64 used internally may overflow

#[pymodule]
#[allow(unused_variables)]
fn kmedoids(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("_fasterpam_f32", wrap_pyfunction!(fasterpam_f32, m)?)?;
    m.add("_fasterpam_f64", wrap_pyfunction!(fasterpam_f64, m)?)?;
    m.add("_fasterpam_i32", wrap_pyfunction!(fasterpam_i32, m)?)?;
    m.add("_fasterpam_i64", wrap_pyfunction!(fasterpam_i64, m)?)?;
    m.add("_rand_fasterpam_f32", wrap_pyfunction!(rand_fasterpam_f32, m)?)?;
    m.add("_rand_fasterpam_f64", wrap_pyfunction!(rand_fasterpam_f64, m)?)?;
    m.add("_rand_fasterpam_i32", wrap_pyfunction!(rand_fasterpam_i32, m)?)?;
    m.add("_rand_fasterpam_i64", wrap_pyfunction!(rand_fasterpam_i64, m)?)?;
    m.add("_fastpam1_f32", wrap_pyfunction!(fastpam1_f32, m)?)?;
    m.add("_fastpam1_f64", wrap_pyfunction!(fastpam1_f64, m)?)?;
    m.add("_fastpam1_i32", wrap_pyfunction!(fastpam1_i32, m)?)?;
    m.add("_fastpam1_i64", wrap_pyfunction!(fastpam1_i64, m)?)?;
    m.add("_fastmsc_f32", wrap_pyfunction!(fastmsc_f32, m)?)?;
    m.add("_fastmsc_f64", wrap_pyfunction!(fastmsc_f64, m)?)?;
    m.add("_fastermsc_f32", wrap_pyfunction!(fastermsc_f32, m)?)?;
    m.add("_fastermsc_f64", wrap_pyfunction!(fastermsc_f64, m)?)?;
    m.add("_dynmsc_f32", wrap_pyfunction!(dynmsc_f32, m)?)?;
    m.add("_dynmsc_f64", wrap_pyfunction!(dynmsc_f64, m)?)?;
    m.add("_pam_swap_f32", wrap_pyfunction!(pam_swap_f32, m)?)?;
    m.add("_pam_swap_f64", wrap_pyfunction!(pam_swap_f64, m)?)?;
    m.add("_pam_swap_i32", wrap_pyfunction!(pam_swap_i32, m)?)?;
    m.add("_pam_swap_i64", wrap_pyfunction!(pam_swap_i64, m)?)?;
    m.add("_pam_build_f32", wrap_pyfunction!(pam_build_f32, m)?)?;
    m.add("_pam_build_f64", wrap_pyfunction!(pam_build_f64, m)?)?;
    m.add("_pam_build_i32", wrap_pyfunction!(pam_build_i32, m)?)?;
    m.add("_pam_build_i64", wrap_pyfunction!(pam_build_i64, m)?)?;
    m.add("_pammedsil_swap_f32", wrap_pyfunction!(pammedsil_swap_f32, m)?)?;
    m.add("_pammedsil_swap_f64", wrap_pyfunction!(pammedsil_swap_f64, m)?)?;
    m.add("_pamsil_swap_f32", wrap_pyfunction!(pamsil_swap_f32, m)?)?;
    m.add("_pamsil_swap_f64", wrap_pyfunction!(pamsil_swap_f64, m)?)?;
    m.add("_alternating_f32", wrap_pyfunction!(alternating_f32, m)?)?;
    m.add("_alternating_f64", wrap_pyfunction!(alternating_f64, m)?)?;
    m.add("_alternating_i32", wrap_pyfunction!(alternating_i32, m)?)?;
    m.add("_alternating_i64", wrap_pyfunction!(alternating_i64, m)?)?;
    m.add("_silhouette_f32", wrap_pyfunction!(silhouette_f32, m)?)?;
    m.add("_silhouette_f64", wrap_pyfunction!(silhouette_f64, m)?)?;
    m.add("_silhouette_i32", wrap_pyfunction!(silhouette_i32, m)?)?;
    // not supported: m.add("_silhouette_i64", wrap_pyfunction!(silhouette_i64, m)?)?;
    m.add("_par_fasterpam_f32", wrap_pyfunction!(par_fasterpam_f32, m)?)?;
    m.add("_par_fasterpam_f64", wrap_pyfunction!(par_fasterpam_f64, m)?)?;
    m.add("_par_fasterpam_i32", wrap_pyfunction!(par_fasterpam_i32, m)?)?;
    m.add("_par_fasterpam_i64", wrap_pyfunction!(par_fasterpam_i64, m)?)?;
    m.add("_par_silhouette_f32", wrap_pyfunction!(par_silhouette_f32, m)?)?;
    m.add("_par_silhouette_f64", wrap_pyfunction!(par_silhouette_f64, m)?)?;
    m.add("_par_silhouette_i32", wrap_pyfunction!(par_silhouette_i32, m)?)?;
    // not supported: m.add("_üarsilhouette_i64", wrap_pyfunction!(par_silhouette_i64, m)?)?;
    m.add("_medoid_silhouette_f32", wrap_pyfunction!(medoid_silhouette_f32, m)?)?;
    m.add("_medoid_silhouette_f64", wrap_pyfunction!(medoid_silhouette_f64, m)?)?;
    m.add("_medoid_silhouette_i32", wrap_pyfunction!(medoid_silhouette_i32, m)?)?;
    // not supported: m.add("_medoid_silhouette_i64", wrap_pyfunction!(medoid_silhouette_i64, m)?)?;
    Ok(())
}

