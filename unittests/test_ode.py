# coding=utf-8
"""
test_ode.py

We test each class in the ode.py module
    * ODE
    * CustomODE
    * LinearODE
    * MatrixLinearODE
    * FitzHugh-Nagumo
    * Res2Bod
for initialisation, standard evaluation and
vectorised evaluation.
"""

import unittest
import numpy as np
from odefilters import ode


# pylint: disable=E0110
class TestODE(unittest.TestCase):
    """
    ODE object should be impossible to initialise,
    since it is an abstract class.
    """
    def test_init(self):
        """
        Passes if initialisation of abstract ODE object
        raises 'TypeError: Can't instantiate abstract class ODE ...'.
        """
        with self.assertRaises(TypeError):
            ode.ODE(t0=0.0, tmax=1.0, initval=2.1, initval_unc=1.1)


class RHSObject():
    """
    RHS object as linear ODE for
    test of CustomODE class.
    """
    def __init__(self, param=1.123):
        """
        Initialise with parameter p=1.123
        """
        self.param = param

    def __repr__(self):
        """
        Represent as RHSObject.
        """
        return "RHSObject(param=%r)" % (self.param)

    def modeval(self, __, loc):
        """
        Evaluate f(x) = p * x
        """
        return self.param * loc

    def modeval_without_t(self, loc):
        """
        Evaluate f(x) = p * x
        """
        return self.param * loc


class TestCustomODEObject(unittest.TestCase):
    """
    Similarly to ../examples/custom_ode.py, we initialise the RHS with
    an object and evaluate before and
    after a change of parameters.
    """
    def setUp(self):
        """
        Set up CustomODE with rhs-function for further tests.
        This setup is a test itself, it passes
        if CustomODE can be successfully initialised.
        """
        self.rhs_obj = RHSObject(param=1.123)
        self.custode = ode.CustomODE(t0=0.0, tmax=1.0,
                                     modeval=self.rhs_obj.modeval,
                                     initval=2.34, initval_unc=0.0)

    def test_modeval_obj(self):
        """
        Test passes if CustomODE with object can
        be successfully evaluated and
        if evaluation matches expected values.
        """
        self.assertEqual(self.custode.modeval(1., 2.0), 1.123 * 2.0)

    def test_paramchange_obj(self):
        """
        Test passes if changing the parameters of the
        RHS object changes the evaluation of the CustomODE
        model and if evaluation matches expected values.
        """
        self.rhs_obj.param = 100.1
        self.assertEqual(self.custode.modeval(1., 2.0), 100.1 * 2.0)

    def test_warning_raised(self):
        """
        Test passes if initialising a CustomODE object
        with a rhs-function that does not take (t, x) as inputs
        raises a TypeError.
        """
        with self.assertRaises(TypeError):
            ode.CustomODE(t0=0.0, tmax=1.0,
                          modeval=self.rhs_obj.modeval_without_t,
                          initval=2.34, initval_unc=0.0)


def rhs_fct(__, loc):
    """
    RHS function as linear ODE for test
    of CustomODE class.

    Evaluates f(x) = p * loc
    """
    return 1.123 * loc


def rhs_fct_without_t(loc):
    """
    RHS function as linear ODE for test
    of CustomODE class.

    Evaluates f(x) = p * x
    """
    return 1.123 * loc


class TestCustomODEFct(unittest.TestCase):
    """
    Similarly to ../examples/custom_ode.py, we initialise the RHS with
    a function and evaluate normally and in a vectorised way.
    """
    def setUp(self):
        """
        Set up CustomODE with rhs-function for further tests.
        This setup is a test itself, it passes
        if CustomODE can be successfully initialised.
        """
        self.custode = ode.CustomODE(t0=0.0, tmax=1.0, modeval=rhs_fct,
                                     initval=2.34, initval_unc=0.0)

    def test_modeval_fct(self):
        """
        Test passes if CustomODE with function can
        be successfully evaluated and
        if evaluation matches expected values.
        """
        self.assertEqual(self.custode.modeval(0., 2.0), 1.123 * 2.0)

    def test_warning_raised(self):
        """
        Test passes if initialising a CustomODE object
        with a rhs-function that does not take (t, x) as inputs
        raises a TypeError.
        """
        with self.assertRaises(TypeError):
            ode.CustomODE(t0=0.0, tmax=1.0,
                          modeval=rhs_fct_without_t,
                          initval=2.34, initval_unc=0.0)

class TestLinearODE(unittest.TestCase):
    """
    We check whether a linear ODE can be set up and evaluated with
    scalar parameters,  whether vectorised evaluation works as well
    as whether it cannot be initialised with wrong values.

    Note:
        The linear ODE as given is defined with element-wise
        multiplication and not with matrix-vector multiplication.
    """
    def setUp(self):
        """
        Set up LinearODE for further tests.
        This setup is a test itself, it passes
        if LinearODE can be successfully initialised.
        """
        ode.LinearODE(t0=0., tmax=1., params=3., initval=0.1, initval_unc=2.1)

    def test_modeval_scalar(self):
        """
        Test passes if scalar LinearODE can be successfully evaluated.
        """
        lin_ode = ode.LinearODE(t0=0., tmax=1., params=3.152,
                                initval=0.1, initval_unc=2.1)
        self.assertEqual(lin_ode.modeval(1., 2.124), 3.152*2.124)

    def test_modeval_vect_scalar(self):
        """
        Test passes if linear ODE allows vectorised evaluation.
        """
        lin_ode = ode.LinearODE(t0=0., tmax=1., params=3.,
                                initval=0.1, initval_unc=2.1)
        evaluation = lin_ode.modeval(0.1, np.random.rand(20))
        self.assertEqual(len(evaluation.shape), 1)
        self.assertEqual(evaluation.shape[0], 20)  # pylint: disable=E1136

    def test_init_multidim_fails_param(self):
        """
        Test fails if LinearODE cannot be initialised with
        anything but a scalar parameter.
        """
        params = np.eye(2)
        with self.assertRaises(TypeError):
            ode.LinearODE(t0=0., tmax=1., params=params,
                          initval=0.1, initval_unc=2.1)

    def test_init_multidim_fails_initval(self):
        """
        Test fails if LinearODE cannot be initialised with
        anything but a scalar initial value.
        """
        initval = np.eye(2)
        with self.assertRaises(TypeError):
            ode.LinearODE(t0=0., tmax=1., params=1.0,
                          initval=initval, initval_unc=2.1)


class TestMatrixLinearODE(unittest.TestCase):
    """
    We check whether a multidimensional linear ODE can be set up and
    evaluated with matrix parameters,  whether vectorised evaluation
    works as well as whether it cannot be initialised with wrong values.
    """
    def setUp(self):
        """
        Set up MatrixLinearODE for further tests.
        This setup is a test itself, it passes
        if MatrixLinearODE can be successfully initialised.
        """
        initval = np.ones(2)
        params = np.eye(2)
        self.matode = ode.MatrixLinearODE(t0=0., tmax=1., params=params,
                                          initval=initval, initval_unc=2.1)

    def test_modeval_multidim(self):
        """
        Test passes if multidimensional linear ODE can be
        successfully evaluated and
        if evaluation matches expected values.
        """
        params = np.eye(2)
        xeval = np.random.rand(2)
        discrepancy = np.linalg.norm(self.matode.modeval(0., xeval)
                                     - params @ xeval)
        self.assertEqual(discrepancy, 0.0)

    def test_modeval_vect_multidim(self):
        """
        Test passes if linear ODE allows vectorised evaluation.
        """
        evalpts = np.random.rand(20, 2)
        evaluation = self.matode.modeval(0., evalpts)
        self.assertEqual(len(evaluation.shape), 2)
        self.assertEqual(evaluation.shape[0], 20)  # pylint: disable=E1136
        self.assertEqual(evaluation.shape[1], 2)  # pylint: disable=E1136

    def test_init_scalar_fails(self):
        """
        Test passes if matrix linear ODE cannot be initialised with
        scalar parameter
        """
        with self.assertRaises(TypeError):
            ode.MatrixLinearODE(t0=0., tmax=1., params=2.0,
                                initval=0.1, initval_unc=2.1)

    def test_init_array_fails(self):
        """
        Test passes if matrix linear ODE cannot be initialised with
        1d-array parameter
        """
        param = np.array([1., 2.])
        with self.assertRaises(TypeError):
            ode.MatrixLinearODE(t0=0., tmax=1., params=param,
                                initval=0.1, initval_unc=2.1)

    def test_init_array_fails_initval(self):
        """
        Test passes if matrix linear ODE cannot be initialised with
        1d-array parameter
        """
        param = np.eye(2)
        initval = 1.0
        with self.assertRaises(TypeError):
            ode.MatrixLinearODE(t0=0., tmax=1., params=param,
                                initval=initval, initval_unc=2.1)


class TestFitzHughNagumo(unittest.TestCase):
    """
    Test whether FHN can be initialised, evaluated
    and whether modeval allows vectorised evaluation.
    """
    def setUp(self):
        """
        Set up FHN ode for further tests.
        This setup is a test itself,
        it passes if FHN can be successfully initialised.
        """
        params = [0., 0.2, 0.2, 3.0]
        self.fhn = ode.FitzHughNagumo(t0=0.1, tmax=2.1, params=params,
                                      initval=[-1., 1.], initval_unc=1.0)

    def test_init_wrong_number_params(self):
        """
        Test passes if FHN initialisation complains about
        wrong number of parameters.
        """
        params = [0.2, 3.0]
        with self.assertRaises(TypeError):
            ode.FitzHughNagumo(t0=0.1, tmax=2.1, params=params,
                               initval=[-1., 1.], initval_unc=1.0)

    def test_init_wrong_initval(self):
        """
        Test passes if FHN initialisation complains about
        wrong dimension of initial values.
        """
        params = [0.2, 3.0, 3.0]
        initval_wrong = [1., 2., 2.]
        with self.assertRaises(TypeError):
            ode.FitzHughNagumo(t0=0.1, tmax=2.1, params=params,
                               initval=initval_wrong, initval_unc=1.0)

    def test_modeval(self):
        """
        Test passes if FHN can be successfully evaluated and
        if evaluation matches expected values.
        """
        evaluation = self.fhn.modeval(0., np.array([1., 1.]))
        discrepancy = np.linalg.norm(evaluation - np.array([-1./3, 1./3]))
        self.assertAlmostEqual(discrepancy, 0., places=15)

    def test_modeval_vect(self):
        """
        Test passes if FHN allows vectorised evaluation.
        """
        evalpts = np.random.rand(10, 2)
        evaluation = self.fhn.modeval(0., evalpts)
        self.assertEqual(evaluation.shape[0], 10)
        self.assertEqual(evaluation.shape[1], 2)


class TestRes2Bod(unittest.TestCase):
    """
    Test Restricted Two-Body Problem for initialisation,
    evaluation and vectorised evaluation.

    Note:
        A useful sanitycheck is solve_res2bod.py.
        The resulting trajectory should be periodic.
        Warning: this takes roughly 20 seconds.
    """
    def setUp(self):
        """
        Set up R2B ode for further tests.
        This setup is a test itself,
        it passes if R2B can be successfully initialised.
        """
        self.r2b = ode.Res2Bod(t0=0., tmax=1.)

    def test_modeval(self):
        """
        Test passes if R2B can be successfully evaluated.
        """
        evaluation = self.r2b.modeval(0., np.random.rand(4))
        self.assertEqual(evaluation.shape[0], 4)

    def test_modeval_vect(self):
        """
        Test passes if R2B allows vectorised evaluation.
        """
        evalpts = np.random.rand(10, 4)
        evaluation = self.r2b.modeval(0., evalpts)
        self.assertEqual(evaluation.shape[0], 10)
        self.assertEqual(evaluation.shape[1], 4)
