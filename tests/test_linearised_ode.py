# coding=utf-8
"""
test_linearised_ode.py

We test each class in the linearised_ode.py module:
    * LinearisedODE
    * LinearODE
    * SigmoidLogGrowth
    * LotkaVolterra

for initialisation and evaluation (standard and vectorised).
"""

import unittest
import numpy as np
from difflikelihoods import linearised_ode as linode


class TestLinearisedODE(unittest.TestCase):
    """
    Tests LinearisedODE module. As an abstract method, it should
    not be possible to be initialised.
    """
    def test_init(self):
        """
        Passes if initialisation of abstract LinearisedODE object
        raises 'TypeError: Can't instantiate abstract class ODE ...'.
        """
        with self.assertRaises(TypeError):
            linode.LinearisedODE(t0=0.0, tmax=1.0, params=0.0,
                                 initval=2.1, initval_unc=1.1)


class TestLinearODE(unittest.TestCase):
    """
    We check whether a linear ODE can be set up and evaluated
    without complaints, in both standard and vectorised way.
    Further tests, whether non-scalar inputs are rejected.
    """

    def setUp(self):
        """
        Test passes if scalar LinearODE can be successfully initialised.
        """
        self.lin_ode = linode.LinearODE(t0=0., tmax=1., params=3.,
                                        initval=0.1, initval_unc=2.1)

    def modeval_scalar(self):
        """
        Test passes if scalar LinearODE can be successfully evaluated.
        """
        evaluation = self.lin_ode.modeval(1., 2.124)
        self.assertEqual(evaluation, 3.*2.124)

    def modeval_vect_scalar(self):
        """
        Test passes if linear ODE allows vectorised evaluation.
        """
        evaluation = self.lin_ode.modeval(0.1, np.random.rand(20))
        self.assertEqual(len(evaluation.shape), 1)
        self.assertEqual(evaluation.shape[0], 20)

    def test_multidim_init_fails_params(self):
        """
        Test passes if matrix-valued parameter is rejected.
        """
        params = np.eye(2)
        with self.assertRaises(TypeError):
            linode.LinearODE(t0=0., tmax=1., params=params,
                             initval=0.1, initval_unc=2.1)

    def test_multidim_init_fails_initval(self):
        """
        Test passes if multidimensional initial value is rejected
        """
        initval = np.ones(2)
        with self.assertRaises(TypeError):
            linode.LinearODE(t0=0., tmax=1., params=1.0,
                             initval=initval, initval_unc=2.1)

    def test_is_linearised(self):
        """
        Test passes if lin_ode.is_linearised is True which is True
        if LinearODE is a subclass of LinearisedODE.
        """
        self.assertEqual(self.lin_ode.is_linearised, True)


class TestLogisticODE(unittest.TestCase):
    """
    We check whether a logistic ODE can be set up and evaluated with
    scalar parameters without complaints and
    whether vectorised evaluation works.
    """
    def setUp(self):
        """
        Test passes if SLG can be initialised.
        """
        self.log_ode = linode.LogisticODE(t0=0., tmax=1., params=[3., 1.5],
                                          initval=0.1, initval_unc=2.1)

    def test_modeval(self):
        """
        Test passes if SLG can be evaluated and
        if evaluation matches expected values.
        """
        check_this = self.log_ode.modeval(1., 2.124)
        self.assertAlmostEqual(check_this,
                               3.*2.124 * (1 - 2.124/2.), places=15)

    def test_vect_eval(self):
        """
        Test passes if SLG allows vectorised evaluation.
        """
        evaluation = self.log_ode.modeval(0.1, np.random.rand(20))
        self.assertEqual(len(evaluation.shape), 1)
        self.assertEqual(evaluation.shape[0], 20)

    def test_init_wrong_number_params(self):
        """
        Test passes if SLG initialisation complains about
        wrong number of parameters; 3 (wrong) instead of 2 (right).
        """
        params = [0.2, 3.0, 1.0]
        with self.assertRaises(TypeError):
            linode.LogisticODE(t0=0., tmax=1., params=params,
                               initval=0.1, initval_unc=2.1)

    def test_is_linearised(self):
        """
        Test passes if log_ode.is_linearised is True which is True
        if LogisticODE is a subclass of LinearisedODE.
        """
        self.assertEqual(self.log_ode.is_linearised, True)


class TestLotkaVolterra(unittest.TestCase):
    """
    Test Lotka-Volterra ODE for initialisation, model evaluation and
    for vectorised evalation.
    """
    def setUp(self):
        t0, tmax = 0., 1.
        params = [2., 1., 2., 1.]
        initval, initval_unc = [1., 1.], 0.1
        self.lotka = linode.LotkaVolterra(t0, tmax, params,
                                          initval, initval_unc)

    def test_is_linearised(self):
        """
        Test passes if lotka.is_linearised is True which is True
        if LotkaVolterra is subclass of LinearisedODE.
        """
        self.assertEqual(self.lotka.is_linearised, True)

    def test_modeval_parts(self):
        """
        Test passes if Lotka-Volterra successfully evaluates modeval_parts.
        """
        self.lotka.modeval_parts(0., self.lotka.initval)

    def test_modeval(self):
        """
        Test passes if Lotka-Volterra successfully evaluates modeval
        and if results coincides with inner product of
        params and modeval_parts.
        """
        rhs_parts = self.lotka.modeval_parts(0., self.lotka.initval)
        modeval_comparison = self.lotka.params @ rhs_parts
        modeval = self.lotka.modeval(0., self.lotka.initval)
        self.assertAlmostEqual(modeval[0], modeval_comparison[0], places=16)
        self.assertAlmostEqual(modeval[1], modeval_comparison[1], places=16)

    def test_modeval_vect(self):
        """
        Test passes if Lotka-Volterra successfully evaluates modeval
        in a vectorised way and if results coincide with
        inner product of params and modeval_parts.
        """
        evalpts = np.random.rand(10, 2)
        evaluation = self.lotka.modeval(0., evalpts)
        self.assertEqual(evaluation.shape[0], 10)
        self.assertEqual(evaluation.shape[1], 2)

    def test_modeval_parts_vect(self):
        """
        Test passes if lotka.modeval_parts allows vectorised evaluation.
        """
        evalpts = np.random.rand(10, 2)
        evaluation_parts = self.lotka.modeval_parts(0., evalpts)
        self.assertEqual(evaluation_parts.shape[0], 4)
        self.assertEqual(evaluation_parts.shape[1], 10)
        self.assertEqual(evaluation_parts.shape[2], 2)

    def test_init_wrong_number_params_3(self):
        """
        Test passes if LotkaVolterra initialisation complains about
        wrong number of parameters; 3 instead of 4.
        """
        t0, tmax = 0., 1.
        params = [0.2, 3.0, 1.0]
        initval, initval_unc = [1., 1.], 0.1
        with self.assertRaises(TypeError):
            linode.LotkaVolterra(t0, tmax, params,
                                 initval, initval_unc)

    def test_init_wrong_number_params_5(self):
        """
        Test passes if LotkaVolterra initialisation complains about
        wrong number of parameters; 5 instead of 4.
        """
        t0, tmax = 0., 1.
        params = [0.2, 3.0, 1.0, 1., 2.]
        initval, initval_unc = [1., 1.], 0.1
        with self.assertRaises(TypeError):
            linode.LotkaVolterra(t0, tmax, params,
                                 initval, initval_unc)

    def test_init_wrong_initval_1(self):
        """
        Test passes if LotkaVolterra initialisation complains about
        wrong number of parameters; 1 instead of 2.
        """
        t0, tmax = 0., 1.
        params = [0.2, 3.0, 1.0, 2.0]
        initval, initval_unc = [1.], 0.1
        with self.assertRaises(TypeError):
            linode.LotkaVolterra(t0, tmax, params,
                                 initval, initval_unc)

    def test_init_wrong_initval_3(self):
        """
        Test passes if LotkaVolterra initialisation complains about
        wrong number of parameters; 3 instead of 2.
        """
        t0, tmax = 0., 1.
        params = [0.2, 3.0, 1.0, 2.0]
        initval, initval_unc = [1., 2., 3.], 0.1
        with self.assertRaises(TypeError):
            linode.LotkaVolterra(t0, tmax, params,
                                 initval, initval_unc)

