# coding=utf-8
"""
test_filters.py

We test two things;
    1) does every function exit successfully
    2) (regressiontest) do we get an expected result?
"""

import unittest
import numpy as np
from difflikelihoods import filters
from difflikelihoods import statespace

class TestFilter(unittest.TestCase):
    """
    """
    def test_instantiation_impossible(self):
        """
        Test passes if Filter object cannot be instantiated.
        """
        ssm = statespace.IBM(q=2, dim=1)
        with self.assertRaises(TypeError):
            filters.Filter(ssm)

class TestKalmanFilter(unittest.TestCase):
    """
    Tests whether the KalmanFilter has successful  __repr__,
    filter(), initialise(), predict() and update() methods.
    """
    def setUp(self):
        """
        Sets up KalmanFilter object.
        """
        mat = np.eye(1)
        ssm = statespace.InvariantSSM(mat, 0.001*mat, mat, 0.001*mat, np.zeros(1), mat)
        self.data = ssm.sample_trajectory(nsteps=10)
        self.filt = filters.KalmanFilter(ssm)
        np.random.seed(1)

    def test_repr(self):
        """
        Test passes if KalmanFilter().__repr__ does something.
        """
        self.filt

    def test_filter(self):
        """
        Test passes if KalmanFilter().filter(self.data)
        replicates the data more or less.
        """
        m, unc = self.filt.filter(self.data)
        discrep = np.abs(m[-1] - self.data[-1])
        self.assertLess(discrep, 0.1)

    def test_initialise(self):
        """
        Test passes if KalmanFilter().initialse()
        returns init_mean and init_covar.
        """
        m, c = self.filt.initialise()
        self.assertEqual(m[0], 0.)
        self.assertEqual(c[0], 1.)

    def test_predict(self):
        """
        Test passes if first prediction returns (0, 1.001).
        This is a regression test, which is, due to a lack of better
        framework, is masked as a unittest.
        """
        m, c = self.filt.initialise()
        predict1, predict2 = self.filt.predict(m, c)
        self.assertEqual(predict1[0], 0.)
        self.assertEqual(predict2[0], 1.001)

    def test_update(self):
        """
        Test passes if first update returns (1.5..., 0.0...).
        This is a regression test, which is, due to a lack of better
        framework, is masked as a unittest.
        """
        m, c = self.filt.initialise()
        mpred, cpred = self.filt.predict(m, c)
        up1, up2 = self.filt.update(mpred, cpred, self.data[1])
        self.assertAlmostEqual(up1[0], 1.57214229, places=8)
        self.assertAlmostEqual(up2[0, 0], 0.000999, places=8)
