use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
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
    let gil = Python::acquire_gil();
    let py = gil.python();
    Ok((loss, PyArray1::from_vec(py, assi), PyArray1::from_vec(py, meds), n_iter, n_swap).to_object(py))
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
    let gil = Python::acquire_gil();
    let py = gil.python();
    Ok((loss, PyArray1::from_vec(py, assi), PyArray1::from_vec(py, meds), 1).to_object(py))
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
    let gil = Python::acquire_gil();
    let py = gil.python();
    Ok((loss, PyArray1::from_vec(py, assi), PyArray1::from_vec(py, meds), n_iter).to_object(py))
}
}}
alternating_call!(alternating_f32, f32, f64);
alternating_call!(alternating_f64, f64, f64);
alternating_call!(alternating_i32, i32, i64);
alternating_call!(alternating_i64, i64, i64);

macro_rules! silhouette_call {
($name:ident, $type: ty) => {
/// Run the Silhouette index evaluation for $type precision
///
/// :param dist: distance matrix
/// :type dist: ndarray
/// :param assi: cluster assignment
/// :type assi: ndarray
/// :return: silhouette evaluation result
/// :rtype: pair of silhouette score and silhouette coefficients per point
#[pyfunction]
fn $name(dist: PyReadonlyArray2<'_, $type>, assi: PyReadonlyArray1<'_, usize>, samples: bool) -> PyResult<Py<PyAny>> {
    assert_eq!(dist.ndim(), 2);
    assert_eq!(dist.shape()[0], dist.shape()[1]);
    let (sil, sils): (f64, _) = rustkmedoids::silhouette(&dist.as_array(), &assi.to_vec()?, samples);
    let gil = Python::acquire_gil();
    let py = gil.python();
    Ok((sil, PyArray1::from_vec(py, sils)).to_object(py))
}
}}
silhouette_call!(silhouette_f32, f32);
silhouette_call!(silhouette_f64, f64);
silhouette_call!(silhouette_i32, i32);
// i64 not supported, as the f64 used internally may overflow

#[pymodule]
#[allow(unused_variables)]
fn kmedoids(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("_fasterpam_f32", wrap_pyfunction!(fasterpam_f32, m)?)?;
    m.add("_fasterpam_f64", wrap_pyfunction!(fasterpam_f64, m)?)?;
    m.add("_fasterpam_i32", wrap_pyfunction!(fasterpam_i32, m)?)?;
    m.add("_fasterpam_i64", wrap_pyfunction!(fasterpam_i64, m)?)?;
    m.add("_fastpam1_f32", wrap_pyfunction!(fastpam1_f32, m)?)?;
    m.add("_fastpam1_f64", wrap_pyfunction!(fastpam1_f64, m)?)?;
    m.add("_fastpam1_i32", wrap_pyfunction!(fastpam1_i32, m)?)?;
    m.add("_fastpam1_i64", wrap_pyfunction!(fastpam1_i64, m)?)?;
    m.add("_pam_swap_f32", wrap_pyfunction!(pam_swap_f32, m)?)?;
    m.add("_pam_swap_f64", wrap_pyfunction!(pam_swap_f64, m)?)?;
    m.add("_pam_swap_i32", wrap_pyfunction!(pam_swap_i32, m)?)?;
    m.add("_pam_swap_i64", wrap_pyfunction!(pam_swap_i64, m)?)?;
    m.add("_pam_build_f32", wrap_pyfunction!(pam_build_f32, m)?)?;
    m.add("_pam_build_f64", wrap_pyfunction!(pam_build_f64, m)?)?;
    m.add("_pam_build_i32", wrap_pyfunction!(pam_build_i32, m)?)?;
    m.add("_pam_build_i64", wrap_pyfunction!(pam_build_i64, m)?)?;
    m.add("_alternating_f32", wrap_pyfunction!(alternating_f32, m)?)?;
    m.add("_alternating_f64", wrap_pyfunction!(alternating_f64, m)?)?;
    m.add("_alternating_i32", wrap_pyfunction!(alternating_i32, m)?)?;
    m.add("_alternating_i64", wrap_pyfunction!(alternating_i64, m)?)?;
    m.add("_silhouette_f32", wrap_pyfunction!(silhouette_f32, m)?)?;
    m.add("_silhouette_f64", wrap_pyfunction!(silhouette_f64, m)?)?;
    m.add("_silhouette_i32", wrap_pyfunction!(silhouette_i32, m)?)?;
    // not supported: m.add("_silhouette_i64", wrap_pyfunction!(silhouette_i64, m)?)?;
    Ok(())
}

