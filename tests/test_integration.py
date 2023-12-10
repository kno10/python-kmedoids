# kmedoids (https://github.com/kno10/python-kmedoids).

import numpy as np
import kmedoids
import unittest

class Test_kmedoids(unittest.TestCase):
    def test_pam(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        pam_build = kmedoids.pam_build(dist, 2)
        pam_build_rust = kmedoids.kmedoids._pam_build_i32(dist, 2)
        assert pam_build.loss == 9
        assert pam_build.loss == pam_build_rust[0]
        pam_swap = kmedoids.pam(dist, 2)
        pam_swap_rust = kmedoids.kmedoids._pam_swap_i32(dist, pam_build_rust[2], 100)
        assert pam_swap.loss == 9
        assert pam_swap.loss == pam_swap_rust[0]
        assert np.array_equal(pam_swap.medoids, pam_swap_rust[2])

    def test_fasterpam(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        fp = kmedoids.fasterpam(dist, 2, init='build')
        par_fp = kmedoids.fasterpam(dist, 2, init='build', n_cpu = 2)
        fp_rust = kmedoids.kmedoids._fasterpam_i32(dist, fp.medoids, 100)
        assert fp.loss == 9
        assert fp.loss == fp_rust[0]
        assert fp.loss == par_fp.loss
        assert np.array_equal(fp.medoids, par_fp.medoids)
        assert np.array_equal(fp.medoids, fp_rust[2])

    def test_fastpam1(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        fp1 = kmedoids.fastpam1(dist, 2, init='build')
        assert fp1.loss == 9
        fp1_rust = kmedoids.kmedoids._fastpam1_i32(dist, fp1.medoids, 100)
        assert np.array_equal(fp1.medoids, fp1_rust[2])
        assert fp1.loss == fp1_rust[0]

    def test_alternating(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        alt = kmedoids.alternating(dist, 2, init='build')
        alt_rust = kmedoids.kmedoids._alternating_i32(dist, alt.medoids, 100)
        assert np.array_equal(alt.medoids, alt_rust[2])
        assert alt.loss == alt_rust[0]

    def test_pamsil(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.float32)
        pamsil = kmedoids.pamsil(dist, 2)
        pamsil_rust = kmedoids.kmedoids._pamsil_swap_f32(dist, pamsil.medoids, 100)
        assert pamsil.loss == 0.3137878787878788
        assert pamsil.loss == pamsil_rust[0]
        assert np.array_equal(pamsil.medoids, pamsil_rust[2])

    def test_pammedsil(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.float32)
        pms = kmedoids.pammedsil(dist, 2)
        pms_rust = kmedoids.kmedoids._pammedsil_swap_f32(dist, pms.medoids, 100)
        assert pms.loss == 0.8172727272727273
        assert pms.loss == pms_rust[0]
        assert np.array_equal(pms.medoids, pms_rust[2])

    def test_fastmsc(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.float32)
        fmsc = kmedoids.fastmsc(dist, 2, init='build')
        fmsc_rust = kmedoids.kmedoids._fastmsc_f32(dist, fmsc.medoids, 100)
        assert fmsc.loss == 0.8172727272727273
        assert np.array_equal(fmsc.medoids, fmsc_rust[2])
        assert fmsc.loss == fmsc_rust[0]

    def test_fastermsc(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.float32)
        fmsc = kmedoids.fastermsc(dist, 2, init='build')
        fmsc_rust = kmedoids.kmedoids._fastermsc_f32(dist, fmsc.medoids, 100)
        assert fmsc.loss == 0.8172727272727273
        assert np.array_equal(fmsc.medoids, fmsc_rust[2])
        assert fmsc.loss == fmsc_rust[0]

    def test_dynmsc(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.float32)
        dmsc = kmedoids.dynmsc(dist, 3, init='build')
        dmsc_rust = kmedoids.kmedoids._dynmsc_f32(dist, dmsc.medoids, 100)
        assert dmsc.loss == 0.8761904761904762
        assert np.array_equal(dmsc.medoids, dmsc_rust[2])
        assert dmsc.loss == dmsc_rust[0]

    def test_silhouette(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        pam = kmedoids.pam(dist, 2)
        sil = kmedoids.silhouette(dist, pam.labels, n_cpu=1)
        par_sil = kmedoids.silhouette(dist, pam.labels, n_cpu=2)
        sil_rust = kmedoids.kmedoids._silhouette_i32(dist, pam.labels, False)
        assert sil[0] == par_sil[0]
        assert sil[0] == sil_rust[0]

    def test_medoid_silhouette(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.float32)
        pam = kmedoids.fasterpam(dist, 2)
        sil = kmedoids.medoid_silhouette(dist, pam.labels)
        sil_rust = kmedoids.kmedoids._medoid_silhouette_f32(dist, pam.labels, False)
        assert sil[0] == sil_rust[0]

    def test_sklearn_kmedoids(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.int32)
        kmed = kmedoids.KMedoids(2, method='fasterpam', init='build')
        res_sk = kmed.fit(dist)
        res = kmedoids.fasterpam(dist, 2, init='build')
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)
        kmed = kmedoids.KMedoids(2, method='pam', init='build')
        res_sk = kmed.fit(dist)
        res = kmedoids.pam(dist, 2, init='build')
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)
        kmed = kmedoids.KMedoids(2, method='fastpam1', init='build')
        res_sk = kmed.fit(dist)
        res = kmedoids.fastpam1(dist, 2, init='build')
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)
        kmed = kmedoids.KMedoids(2, method='alternate', init='build')
        res_sk = kmed.fit(dist)
        res = kmedoids.alternating(dist, 2, init='build')
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)

    def test_sklearn_msc(self):
        dist = np.array([[0, 2, 3, 4, 5], [2, 0, 6, 7, 8], [3, 6, 0, 9, 10], [4, 7, 9, 0, 11], [5, 8, 10, 11, 0]], dtype=np.float32)
        kmed = kmedoids.KMedoids(np.array([0,1,2]), method='dynmsc')
        res_sk = kmed.fit(dist)
        res = kmedoids.dynmsc(dist, np.array([0,1,2]))
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)
        kmed = kmedoids.KMedoids(np.array([0,1,2]), method='fastermsc')
        res_sk = kmed.fit(dist)
        res = kmedoids.fastermsc(dist, np.array([0,1,2]))
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)
        kmed = kmedoids.KMedoids(np.array([0,1,2]), method='fastmsc')
        res_sk = kmed.fit(dist)
        res = kmedoids.fastmsc(dist, np.array([0,1,2]))
        assert res_sk.inertia_ == res.loss
        assert np.array_equal(res_sk.labels_, res.labels)
        assert np.array_equal(res_sk.medoid_indices_, res.medoids)

if __name__ == "__main__":
    unittest.main()
