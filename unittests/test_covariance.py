# coding=utf-8
"""
test_covariance.py

Test whether covariance functions
    1) complain about incorrectly shaped inputs
    2) have expected diagonal values of symmetric matrices
    3) interpolate with an expected rate
"""
import numpy as np
import unittest
from odefilters import covariance as cov


class TestCovLinAlg():
    """
    Test whether covariance functions complain about
    incorrectly shaped inputs.
    """
    def test_complains_about_floats(self):
        """Putting floats into covariance functions should raise AssertionErrors"""
        pt1 = np.random.rand()
        pt2 = np.random.rand()
        correct_input_shape = np.random.rand(1, 1)
        with self.assertRaises(AssertionError):
            self.cov(pt1, pt2)
        with self.assertRaises(AssertionError):
            self.cov(correct_input_shape, pt2)
        with self.assertRaises(AssertionError):
            self.cov(pt1, correct_input_shape)

    def test_complains_about_arrays_1d(self):
        """Putting 1d-arrays into covariance functions should raise AssertionErrors"""
        pt1 = np.random.rand(3)
        pt2 = np.random.rand(3)
        correct_input_shape = np.random.rand(3, 1)
        with self.assertRaises(AssertionError):
            self.cov(pt1, pt2)
        with self.assertRaises(AssertionError):
            self.cov(correct_input_shape, pt2)
        with self.assertRaises(AssertionError):
            self.cov(pt1, correct_input_shape)

    def test_complains_about_arrays_3d(self):
        """Putting 3d-arrays into covariance functions should raise AssertionErrors"""
        pt1 = np.random.rand(3, 1, 1)
        pt2 = np.random.rand(3, 1, 1)
        correct_input_shape = np.random.rand(3, 1)
        with self.assertRaises(AssertionError):
            self.cov(pt1, pt2)
        with self.assertRaises(AssertionError):
            self.cov(correct_input_shape, pt2)
        with self.assertRaises(AssertionError):
            self.cov(pt1, correct_input_shape)

    def test_complains_about_inconsistent_dimensions(self):
        """
        Putting ptsets of different dimension into covariance functions
        should raise AssertionErrors
        """
        ptset1 = np.random.rand(20, 3)
        ptset2 = np.random.rand(21, 2)
        with self.assertRaises(AssertionError):
            self.cov(ptset1, ptset2)

    def test_allows_different_numbers_of_points(self):
        """
        Putting ptsets of different size but same dimension into
        covariance functions should be allowed
        """
        ptset1 = np.random.rand(6, 2)
        ptset2 = np.random.rand(17, 2)
        covmat = self.cov(ptset1, ptset2)
        self.assertEqual(covmat.shape[0], 6)
        self.assertEqual(covmat.shape[1], 17)
        covmat_trans = self.cov(ptset2, ptset1)
        self.assertEqual(covmat_trans.shape[1], 6)
        self.assertEqual(covmat_trans.shape[0], 17)

class TestCovInterpolate():
    """
    Compares interpolation error to some expected value
    """
    def test_can_it_interpolate_as_well_as_expected(self):
        """
        Interpolates a function f(x)=2x**2+1 on 10 points and
        compares error to some expected value
        """
        good_ptset = np.linspace(0, 1)
        good_ptset = good_ptset.reshape((len(good_ptset), 1))
        covmat = self.cov(good_ptset, good_ptset)
        rhs = 2*good_ptset**2 + 1        # interpolate f(x) = 2x^2 + 1
        coeff = np.linalg.solve(covmat, rhs)
        evalptset = np.linspace(0, 1, 500)
        evalptset = evalptset.reshape((len(evalptset), 1))
        evalcovmat = self.cov(evalptset, good_ptset)
        approx = evalcovmat.dot(coeff)
        error = np.linalg.norm(approx - (2*evalptset**2 + 1)) / 500.
        self.assertLess(error, self.expected_interpolation_error)




class TestMaternFamily(TestCovInterpolate, TestCovLinAlg):
    """
    Matern family covariances: Gaussian, Exponential, Matern
    For all Matern family covariances we test linear
    algebra and interpolation, hence we collect these tests here.

    Covariance kernels from this family have a 1 on the diagonal.
    """
    def test_symmetric_ptsets_give_expected_diagonal_value(self):
        """Using the same pointset for both inputs should give a 1 on the diagonal"""
        ptset = np.random.rand(5, 1)
        covmat_large = self.cov(ptset, ptset)
        edv = 1.0
        error_large = np.linalg.norm(np.diag(covmat_large) - edv * np.ones(5))
        self.assertLess(error_large, np.finfo(float).eps)
        covmat_small = self.cov(ptset.T, ptset.T)
        error_small = np.linalg.norm(np.diag(covmat_small) - edv * np.ones(1))
        self.assertLess(error_small, np.finfo(float).eps)



class TestGaussCov(TestMaternFamily, unittest.TestCase):
    """
    Carry out all tests with Gaussian covariance kernel
    """
    def setUp(self):
        """Set up parameters"""
        self.cov = cov.gausscov
        self.expected_interpolation_error = 1e-7


class TestExpCov(TestMaternFamily, unittest.TestCase):
    """
    Carry out all tests with exponential covariance kernel
    """
    def setUp(self):
        """Set up parameters"""
        self.cov = cov.expcov
        self.expected_interpolation_error = 1e-5


class TestMaternCov(TestMaternFamily, unittest.TestCase):
    """
    Carry out all tests with Matern covariance kernel
    """
    def setUp(self):
        """Set up parameters"""
        self.cov = cov.materncov
        self.expected_interpolation_error = 1e-5



class TestIBMAuxiliary(unittest.TestCase):

    def test_is_timeseries(self):
        good_ptset = np.random.rand(10)
        self.assertEqual(cov.ptset_is_timeseries(good_ptset), True)

    def test_is_timeseries_fails_wrong_input(self):
        bad_ptset2d = np.random.rand(20, 2)
        bad_ptset1d = np.random.rand(20, 1)
        with self.assertRaises(AssertionError):
            cov.ptset_is_timeseries(bad_ptset2d)
        with self.assertRaises(AssertionError):
            cov.ptset_is_timeseries(bad_ptset1d)

    def test_is_timeseries_fails_negative(self):
        neg_ptset = -1*np.random.rand(10)
        with self.assertRaises(AssertionError):
            cov.ptset_is_timeseries(neg_ptset)

    def test_create_aligned_copies_pass(self):
        ptset1 = np.random.rand(10)
        ptset2 = np.random.rand(14)
        copies1, copies2 = cov.create_aligned_copies(ptset1, ptset2)
        self.assertEqual(copies1.shape[0], 10)
        self.assertEqual(copies1.shape[1], 14)
        self.assertEqual(copies1.shape, copies2.shape)


class TestIBMFamily(unittest.TestCase):
    """
    Test Family of IBM Covariances: BM, IBM, IBM_CovD, ...
    and associated functions.
    """
    def test_ibm_covd_q3(self):
        """
        Compute output for k(1, 2) and k(2, 1) manually and assert it coincides with output.
        """
        t1 = 1*np.ones(1)
        t2 = 2*np.ones(1)
        sig = 1.2345
        manual_12 = sig**2 * 49.0 / 720.0
        manual_21 = sig**2 * 111.0 / 720.0
        res_12 = cov.ibm_covd_q3(t1, t2, diffconst=sig)[0, 0]
        self.assertAlmostEqual(res_12, manual_12, places=15)
        res_21 = cov.ibm_covd_q3(t2, t1, diffconst=sig)[0, 0]
        self.assertAlmostEqual(res_21, manual_21, places=15)

    def test_ibm_covd_q3_fails_wrong_input(self):
        """
        Raise AssertionError for inputs of shape (N,) and (N, 2).
        """
        wrong1 = 1*np.ones((1, 2))
        wrong2 = 2*np.ones((1, 1))
        correct = np.ones(1)
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q3(correct, wrong1)
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q3(wrong1, correct)
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q3(correct, wrong2)
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q3(wrong2, correct)


    def test_ibm_covd_q2(self):
        """
        Compute output for k(1, 2) and k(2, 1) manually and assert it coincides with output.
        """
        t1 = 1*np.ones(1)
        t2 = 2*np.ones(1)
        sig = 1.2345
        manual_12 = sig**2 * 7.0 / 24.0
        manual_21 = sig**2 * 17.0 / 24.0
        res_12 = cov.ibm_covd_q2(t1, t2, diffconst=sig)[0, 0]
        self.assertAlmostEqual(res_12, manual_12, places=15)
        res_21 = cov.ibm_covd_q2(t2, t1, diffconst=sig)[0, 0]
        self.assertAlmostEqual(res_21, manual_21, places=15)

    def test_ibm_covd_q2_fails_wrong_input(self):
        """
        Raise AssertionError for inputs of shape (N,) and (N, 2).
        """
        wrong1 = 1*np.ones((1, 2))
        wrong2 = 2*np.ones(1)
        correct = np.ones((1, 1))
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q2(correct, wrong1)
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q2(wrong1, correct)
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q2(correct, wrong2)
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q2(wrong2, correct)


    def test_ibm_covd_q1(self):
        """
        Compute output for k(1, 2) and k(2, 1) manually and assert it coincides with output.
        """
        t1 = 1*np.ones(1)
        t2 = 2*np.ones(1)
        sig = 1.2345
        manual_12 = sig**2 * 1./2.
        manual_21 = sig**2 * 3./2.
        res_12 = cov.ibm_covd_q1(t1, t2, diffconst=sig)[0, 0]
        self.assertAlmostEqual(res_12, manual_12, places=15)
        res_21 = cov.ibm_covd_q1(t2, t1, diffconst=sig)[0, 0]
        self.assertAlmostEqual(res_21, manual_21, places=15)

    def test_ibm_covd_q1_fails_wrong_input(self):
        """
        Raise AssertionError for inputs of shape (N,) and (N, 2).
        """
        wrong1 = 1*np.ones((1, 2))
        wrong2 = 2*np.ones(1)
        correct = np.ones((1, 1))
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q1(correct, wrong1)
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q1(wrong1, correct)
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q1(correct, wrong2)
        with self.assertRaises(AssertionError):
            cov.ibm_covd_q1(wrong2, correct)

    def test_q_in_valid_range(self):
        """
        Check q_in_valid_range() for q=1,2,3 (True)
        and for q=0,4 (Error)
        """
        self.assertEqual(cov.q_in_valid_range(1), True)
        self.assertEqual(cov.q_in_valid_range(2), True)
        self.assertEqual(cov.q_in_valid_range(3), True)
        with self.assertRaises(AssertionError):
            cov.q_in_valid_range(0)
        with self.assertRaises(AssertionError):
            cov.q_in_valid_range(4)

    def test_ibm_covd(self):
        t1 = 1*np.ones(1)
        t2 = 2*np.ones(1)
        sig = 1.2345
        manual_12_q1 = sig**2 * 1./2.
        manual_12_q2 = sig**2 * 7.0 / 24.0
        manual_12_q3 = sig**2 * 49.0 / 720.0
        res_12_q1 = cov.ibm_covd(t1, t2, diffconst=sig, q=1)[0, 0]
        res_12_q2 = cov.ibm_covd(t1, t2, diffconst=sig, q=2)[0, 0]
        res_12_q3 = cov.ibm_covd(t1, t2, diffconst=sig, q=3)[0, 0]
        self.assertAlmostEqual(res_12_q1, manual_12_q1, places=15)
        self.assertAlmostEqual(res_12_q2, manual_12_q2, places=15)
        self.assertAlmostEqual(res_12_q3, manual_12_q3, places=15)


    def test_ibmcov_q3(self):
        """
        Compute output for k(1, 2) manually and assert it coincides with output.
        """
        t1 = 1*np.ones(1)
        t2 = 2*np.ones(1)
        sig = 1.2345
        manual = sig**2 * (1./252. + 27./720.)
        res = cov.ibmcov_q3(t1, t2, diffconst=sig)[0, 0]
        self.assertAlmostEqual(res, manual, places=15)

        
    def test_test_ibmcov_q3_fails_wrong_input(self):
        """
        Raise AssertionError for inputs of shape (N,) and (N, 2).
        """
        wrong1 = 1*np.ones((1, 2))
        wrong2 = 2*np.ones(1)
        correct = np.ones((1, 1))
        with self.assertRaises(AssertionError):
            cov.ibmcov_q3(correct, wrong1)
        with self.assertRaises(AssertionError):
            cov.ibmcov_q3(wrong1, correct)
        with self.assertRaises(AssertionError):
            cov.ibmcov_q3(correct, wrong2)
        with self.assertRaises(AssertionError):
            cov.ibmcov_q3(wrong2, correct)


    def test_ibmcov_q2(self):
        """
        Compute output for k(1, 2) manually and assert it coincides with output.
        """
        t1 = 1*np.ones(1)
        t2 = 2*np.ones(1)
        sig = 1.2345
        manual = sig**2 * (1./20. + 5./24.)
        res = cov.ibmcov_q2(t1, t2, diffconst=sig)[0, 0]
        self.assertAlmostEqual(res, manual, places=15)

        
    def test_test_ibmcov_q2_fails_wrong_input(self):
        """
        Raise AssertionError for inputs of shape (N,) and (N, 2).
        """
        wrong1 = 1*np.ones((1, 2))
        wrong2 = 2*np.ones(1)
        correct = np.ones((1, 1))
        with self.assertRaises(AssertionError):
            cov.ibmcov_q2(correct, wrong1)
        with self.assertRaises(AssertionError):
            cov.ibmcov_q2(wrong1, correct)
        with self.assertRaises(AssertionError):
            cov.ibmcov_q2(correct, wrong2)
        with self.assertRaises(AssertionError):
            cov.ibmcov_q2(wrong2, correct)


    def test_ibmcov_q1(self):
        """
        Compute output for k(1, 2) manually and assert it coincides with output.
        """
        t1 = 1*np.ones(1)
        t2 = 2*np.ones(1)
        sig = 1.2345
        manual = sig**2 * (1./3. + 1./2.)
        res = cov.ibmcov_q1(t1, t2, diffconst=sig)[0, 0]
        self.assertAlmostEqual(res, manual, places=15)

        
    def test_test_ibmcov_q1_fails_wrong_input(self):
        """
        Raise AssertionError for inputs of shape (N,) and (N, 2).
        """
        wrong1 = 1*np.ones((1, 2))
        wrong2 = 2*np.ones(1)
        correct = np.ones((1, 1))
        with self.assertRaises(AssertionError):
            cov.ibmcov_q1(correct, wrong1)
        with self.assertRaises(AssertionError):
            cov.ibmcov_q1(wrong1, correct)
        with self.assertRaises(AssertionError):
            cov.ibmcov_q1(correct, wrong2)
        with self.assertRaises(AssertionError):
            cov.ibmcov_q1(wrong2, correct)


    def test_ibmcov(self):
        t1 = 1*np.ones(1)
        t2 = 2*np.ones(1)
        sig = 1.2345
        manual_q1 = sig**2 * (1./3. + 1./2.)
        manual_q2 = sig**2 * (1./20. + 5./24.)
        manual_q3 = sig**2 * (1./252. + 27./720.)
        res_q1 = cov.ibmcov(t1, t2, diffconst=sig, q=1)[0, 0]
        res_q2 = cov.ibmcov(t1, t2, diffconst=sig, q=2)[0, 0]
        res_q3 = cov.ibmcov(t1, t2, diffconst=sig, q=3)[0, 0]
        self.assertAlmostEqual(res_q1, manual_q1, places=15)
        self.assertAlmostEqual(res_q2, manual_q2, places=15)
        self.assertAlmostEqual(res_q3, manual_q3, places=15)


    def test_bmcov(self):
        """
        Compute output for k(1, 2) manually and assert it coincides with output.
        """
        t1 = 1*np.ones(1)
        t2 = 2*np.ones(1)
        sig = 1.2345
        manual = sig**2 * 1.
        res = cov.bmcov(t1, t2, diffconst=sig)[0, 0]
        self.assertAlmostEqual(res, manual, places=15)

        
    def test_bmcov_fails_wrong_input(self):
        """
        Raise AssertionError for inputs of shape (N,) and (N, 2).
        """
        wrong1 = 1*np.ones((1, 2))
        wrong2 = 2*np.ones(1)
        correct = np.ones((1, 1))
        with self.assertRaises(AssertionError):
            cov.bmcov(correct, wrong1)
        with self.assertRaises(AssertionError):
            cov.bmcov(wrong1, correct)
        with self.assertRaises(AssertionError):
            cov.bmcov(correct, wrong2)
        with self.assertRaises(AssertionError):
            cov.bmcov(wrong2, correct)


    def test_ibm_dcovd(self):
        t1 = 1*np.ones(1)
        t2 = 2*np.ones(1)
        sig = 1.2345
        manual_q1 = sig**2 * 1.0
        manual_q2 = sig**2 * (1./3. + 1./2.)
        manual_q3 = sig**2 * (1./20. + 5./24.)
        res_q1 = cov.ibm_dcovd(t1, t2, diffconst=sig, q=1)[0, 0]
        res_q2 = cov.ibm_dcovd(t1, t2, diffconst=sig, q=2)[0, 0]
        res_q3 = cov.ibm_dcovd(t1, t2, diffconst=sig, q=3)[0, 0]
        self.assertAlmostEqual(res_q1, manual_q1, places=15)
        self.assertAlmostEqual(res_q2, manual_q2, places=15)
        self.assertAlmostEqual(res_q3, manual_q3, places=15)

