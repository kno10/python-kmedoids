use pyo3::prelude::*;
use pyo3::class::basic::PyObjectProtocol;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use rustkmedoids;

/// KMedoids clustering result
///
/// :param loss: Loss of this clustering (sum of deviations)
/// :type loss: float
///
/// :param labels: Cluster assignment
/// :type labels: ndarray
///
/// :param medoids: Chosen medoid indexes
/// :type medoids: ndarray
///
/// :param n_iter: Number of iterations
/// :type n_iter: int
///
/// :param n_swap: Number of swaps performed
/// :type n_swap: int
#[pyclass]
#[derive(Debug)]
struct KMedoidsResult {
    /// Loss: sum of distances to the means
    #[pyo3(get)]
    loss: f64,
    /// Assigned cluster labels
    #[pyo3(get)]
    labels: Py<PyArray1<usize>>,
    /// Selected medoids
    #[pyo3(get)]
    medoids: Py<PyArray1<usize>>,
    /// Number of iterations used
    #[pyo3(get)]
    n_iter: usize,
    /// Number of swaps performed
    #[pyo3(get)]
    n_swap: usize,
}
#[pyproto]
impl PyObjectProtocol for KMedoidsResult {
    fn __repr__(&self) -> PyResult<String> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let builtins = PyModule::import(py, "builtins")?;
        let lbl: String = builtins.call_method1("repr", (self.labels.as_ref(py),))?.extract()?;
        let med: String = builtins.call_method1("repr", (self.medoids.as_ref(py),))?.extract()?;
        Ok(format!("KMedoidsResult(loss={},labels={},medoids={},n_iter={},n_swap={})", self.loss, lbl, med, self.n_iter, self.n_swap))
    }
}

macro_rules! variant_call {
($name:ident, $variant:ident, $type: ty) => {
/// Run $variant k-medoids clustering function for $type precision
///
/// :param data: input data
/// :type data: ndarray
/// :param meds: initial medoids
/// :type meds: ndarray
/// :param max_iter: maximum number of iterations
/// :type max_iter: int
/// :return: k-medoids clustering result
/// :rtype: KMedoidsResult
#[pyfunction]
fn $name(data: PyReadonlyArray2<'_, $type>, meds: PyReadonlyArray1<'_, usize>, max_iter: usize) -> PyResult<KMedoidsResult> {
    assert_eq!(data.ndim(), 2);
    assert_eq!(data.shape()[0], data.shape()[1]);
    let mut meds = meds.to_vec()?;
    let (loss, assi, n_iter, n_swap) = rustkmedoids::$variant(&data.as_array(), &mut meds, max_iter);
    let gil = Python::acquire_gil();
    Ok(KMedoidsResult {
        loss : loss as f64,
        labels : PyArray1::from_vec(gil.python(), assi).to_owned(),
        medoids : PyArray1::from_vec(gil.python(), meds).to_owned(),
        n_iter : n_iter,
        n_swap : n_swap
    })
}
}}
variant_call!(fasterpam_f32, fasterpam, f32);
variant_call!(fasterpam_f64, fasterpam, f64);
variant_call!(fasterpam_i32, fasterpam, i32);
variant_call!(fasterpam_i64, fasterpam, i64);
variant_call!(fastpam1_f32, fastpam1, f32);
variant_call!(fastpam1_f64, fastpam1, f64);
variant_call!(fastpam1_i32, fastpam1, i32);
variant_call!(fastpam1_i64, fastpam1, i64);
variant_call!(pam_swap_f32, pam_swap, f32);
variant_call!(pam_swap_f64, pam_swap, f64);
variant_call!(pam_swap_i32, pam_swap, i32);
variant_call!(pam_swap_i64, pam_swap, i64);

macro_rules! pam_build_call {
($name:ident, $type: ty) => {
/// Run the PAM BUILD k-medoids clustering function for $type precision
///
/// :param data: input data
/// :type data: ndarray
/// :param k: number of clusters
/// :type k: int
/// :return: k-medoids clustering result
/// :rtype: KMedoidsResult
#[pyfunction]
fn $name(data: PyReadonlyArray2<'_, $type>, k: usize) -> PyResult<KMedoidsResult> {
    assert_eq!(data.ndim(), 2);
    assert_eq!(data.shape()[0], data.shape()[1]);
    let (loss, assi, meds) = rustkmedoids::pam_build(&data.as_array(), k);
    let gil = Python::acquire_gil();
    Ok(KMedoidsResult {
        loss : loss as f64,
        labels : PyArray1::from_vec(gil.python(), assi).to_owned(),
        medoids : PyArray1::from_vec(gil.python(), meds).to_owned(),
        n_iter : 1,
        n_swap : 0
    })
}
}}
pam_build_call!(pam_build_f32, f32);
pam_build_call!(pam_build_f64, f64);
pam_build_call!(pam_build_i32, i32);
pam_build_call!(pam_build_i64, i64);

macro_rules! alternating_call {
($name:ident, $type: ty) => {
/// Run the Alternating k-medoids clustering function for $type precision
///
/// :param data: input data
/// :type data: ndarray
/// :param meds: initial medoids
/// :type meds: ndarray
/// :param max_iter: maximum number of iterations
/// :type max_iter: int
/// :return: k-medoids clustering result
/// :rtype: KMedoidsResult
#[pyfunction]
fn $name(data: PyReadonlyArray2<'_, $type>, meds: PyReadonlyArray1<'_, usize>, max_iter: usize) -> PyResult<KMedoidsResult> {
    assert_eq!(data.ndim(), 2);
    assert_eq!(data.shape()[0], data.shape()[1]);
    let mut meds = meds.to_vec()?;
    let (loss, assi, n_iter) = rustkmedoids::alternating(&data.as_array(), &mut meds, max_iter);
    let gil = Python::acquire_gil();
    Ok(KMedoidsResult {
        loss : loss as f64,
        labels : PyArray1::from_vec(gil.python(), assi).to_owned(),
        medoids : PyArray1::from_vec(gil.python(), meds).to_owned(),
        n_iter : n_iter,
        n_swap : 0
    })
}
}}
alternating_call!(alternating_f32, f32);
alternating_call!(alternating_f64, f64);
alternating_call!(alternating_i32, i32);
alternating_call!(alternating_i64, i64);

#[pymodule]
#[allow(unused_variables)]
fn kmedoids(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KMedoidsResult>()?;
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
    Ok(())
}

