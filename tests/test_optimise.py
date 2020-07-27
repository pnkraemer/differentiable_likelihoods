# coding=utf-8
"""
test_optimise.py
"""
import unittest
import numpy as np
from difflikelihoods import optimise

class TestOptimiseQuadraticFunction(unittest.TestCase):
    """
    Minimises simple quadratic functions with known minimum.
    Tests pass if minimum is close to known minimum and
    number of iterations as calibrated once in the beginning.
    """
    def test_1d(self):
        """Minimise f(x) = x**2"""
        fct = lambda x: x**2
        der = lambda x: 2*x
        xmin, numit = optimise.minimise(fun=fct, jac=der, x0=10.0)
        self.assertAlmostEqual(xmin, 0.0, places=6)
        self.assertEqual(numit, 4)      # for x0=10.0

    def test_2d(self):
        """Minimise f(x) = x_0**2 + x_1**2"""
        fct = lambda x: x[0]**2 + x[1]**2
        der = lambda x: np.array([2*x[0], 2*x[1]])
        x0 = np.array([-20.1, 12.0])
        xmin, numit = optimise.minimise(fun=fct, jac=der, x0=x0)
        self.assertAlmostEqual(xmin[0], 0.0, places=6)
        self.assertAlmostEqual(xmin[1], 0.0, places=6)
        self.assertEqual(numit, 8)      # for x0=10.0


class TestOptimiseGaussian(unittest.TestCase):
    """
    Maximise f(x) = exp(-x**2) with minimum at x=0.
    """
    def test_gaussian(self):
        """
        Test passes if minimum is found
        at x=0 after 10 steps using x0=2.0.
        """
        fct = lambda x: -np.exp(-x**2)
        der = lambda x: 2*x * np.exp(-x**2)
        xmin, numit = optimise.minimise(fun=fct, jac=der, x0=2.0)
        self.assertAlmostEqual(xmin[0], 0.0, places=6)
        self.assertEqual(numit, 10)      # for x0=2.0

