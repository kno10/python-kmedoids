# kmedoids (https://github.com/kno10/python-kmedoids).

import numpy as np
import kmedoids
import unittest

class Test_kmedoids(unittest.TestCase):
    def test_pam(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        pam_build = kmedoids.pam_build(dist, 2)
        pam_build_rust = kmedoids.kmedoids._pam_build_i32(dist, 2)
        assert pam_build.loss == 11
        assert pam_build.loss == pam_build_rust[0]
        pam_swap = kmedoids.pam(dist, 2)
        pam_swap_rust = kmedoids.kmedoids._pam_swap_i32(dist, pam_build_rust[2], 100)
        assert pam_swap.loss == 9
        assert pam_swap.loss == pam_swap_rust[0]
        assert np.array_equal(pam_swap.medoids, pam_swap_rust[2])

    def test_fasterpam(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        fp = kmedoids.fasterpam(dist, 2, init="build")
        par_fp = kmedoids.fasterpam(dist, 2, init="build", n_cpu = 2)
        fp_rust = kmedoids.kmedoids._fasterpam_i32(dist, fp.medoids, 100)
        assert fp.loss == 9
        assert fp.loss == fp_rust[0]
        assert fp.loss == par_fp.loss
        assert np.array_equal(fp.medoids, par_fp.medoids)
        assert np.array_equal(fp.medoids, fp_rust[2])

    def test_fastpam1(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        fp1 = kmedoids.fastpam1(dist, 2, init="build")
        assert fp1.loss == 9
        fp1_rust = kmedoids.kmedoids._fastpam1_i32(dist, fp1.medoids, 100)
        assert np.array_equal(fp1.medoids, fp1_rust[2])
        assert fp1.loss == fp1_rust[0]

    def test_alternating(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        alt = kmedoids.alternating(dist, 2, init="build")
        alt_rust = kmedoids.kmedoids._alternating_i32(dist, alt.medoids, 100)
        assert np.array_equal(alt.medoids, alt_rust[2])
        assert alt.loss == alt_rust[0]

    def test_silhouette(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        pam = kmedoids.pam(dist, 2)
        sil = kmedoids.silhouette(dist, pam.labels)
        par_sil = kmedoids.silhouette(dist, pam.labels, n_cpu=2)
        sil_rust = kmedoids.kmedoids._silhouette_i32(dist, pam.labels, False)
        assert sil == par_sil
        assert sil == sil_rust[0]

    def test_sklearn_interface(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        kmed = kmedoids.KMedoids(2, method='fasterpam', init='build')
        res_sk = kmed.fit(dist)
        res = kmedoids.fasterpam(dist, 2, init="build")
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)
        kmed = kmedoids.KMedoids(2, method='pam', init='build')
        res_sk = kmed.fit(dist)
        res = kmedoids.pam(dist, 2, init="build")
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)
        kmed = kmedoids.KMedoids(2, method='fastpam1', init='build')
        res_sk = kmed.fit(dist)
        res = kmedoids.fastpam1(dist, 2, init="build")
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)
        kmed = kmedoids.KMedoids(2, method='alternate', init='build')
        res_sk = kmed.fit(dist)
        res = kmedoids.alternating(dist, 2, init="build")
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)


if __name__ == "__main__":
    unittest.main()