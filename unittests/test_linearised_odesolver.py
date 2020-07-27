"""
test_linearised_odesolver.py

We test the LinearisedODESolver class in the linearised_odesolver
module. First, we repeat the tests from test_odesolver.py then we
test specialised variants.
"""
import unittest
import numpy as np
from odefilters import statespace as stsp
from odefilters import linearised_odesolver as linsolver
from odefilters import linearised_ode as linode
from odefilters import ode as standard_ode


class TestScalarODE(unittest.TestCase):
    """
    Test whether first few means and covariances coincide with Prop. 1.
    in Schober/Särkkä/Hennig paper for a 1d ODE with one parameter.
    """
    def setUp(self):
        """
        Set up linear ode and solve with stepsize h = 0.1.
        """
        prior = stsp.IBM(q=1, dim=1)
        self.solver = linsolver.LinearisedODESolver(prior, filtertype="kalman")
        self.ode = linode.LinearODE(t0=0.0, tmax=1.2, params=2.0, initval=4.1)
        self.h = 0.1
        output = self.solver.solve(self.ode, self.h)
        self.tsteps, self.means, self.stdevs, __, __ = output

    def test_first_few_iterations(self):
        """
        Test whether first few means and covariances coincide with Prop. 1.
        in Schober/Särkkä/Hennig paper.
        """
        self.check_mean_t0()
        self.check_stdevs_t0()
        self.check_mean_t1()
        self.check_stdevs_t1()

    def check_mean_t0(self):
        """
        Expect: m(t0) = (y0, z0) where z0=f(y0).
        """
        y0 = self.ode.initval
        z0 = self.ode.modeval(t=0.0, x=y0)
        mean_at_t0 = self.means[0][0]
        self.assertEqual(mean_at_t0[0], y0)
        self.assertEqual(mean_at_t0[1], z0)

    def check_stdevs_t0(self):
        """
        Expect: C(t0) = 0, hence stdevs equal to zero.
        """
        stdev_at_t0 = self.stdevs[0][0]
        self.assertEqual(stdev_at_t0[0], 0.0)
        self.assertEqual(stdev_at_t0[1], 0.0)

    def check_mean_t1(self):
        """
        Expect: m(t0) = (y0 + h/2*(z0 + z1), z1).
        """
        y0 = self.ode.initval
        z0 = self.ode.modeval(t=0.0, x=y0)
        z1 = self.ode.modeval(t=0.0, x=(y0 + self.h * z0))
        mean_at_t1 = self.means[1][0]
        self.assertEqual(mean_at_t1[0], y0 + 0.5 * self.h * (z0 + z1))
        self.assertEqual(mean_at_t1[1], z1)

    def check_stdevs_t1(self):
        """
        Expect: C(t1) = (sigma**2 h**3/12, 0; 0, 0).
        """
        stdev_at_t1 = self.stdevs[1][0]
        diffconst = self.solver.filt.ssm.diffconst       # digging deep
        self.assertAlmostEqual(stdev_at_t1[0],
                               np.sqrt(self.h**3 * diffconst / 12.0),
                               places=12)
        self.assertEqual(stdev_at_t1[1], 0.0)


class TestWrongInputs(unittest.TestCase):
    """
    Check whether the ODESolver is 'robust' against some common wrong inputs:
    spelling mistakes, nonexistant keys and wrong formats.
    """
    def setUp(self):
        """
        Create a working prior,
        working filtertype and working evalparam.
        """
        self.working_2d_prior = stsp.IBM(q=1, dim=2)
        self.working_filtertype = "kalman"

    def test_wrong_filterkeys(self):
        """
        What if instead of 'kalman',
        one enters other things as filterkeys.
        """
        self.check_nonexistant_filterkey()
        self.check_particle_filter_not_implemented()
        self.check_custom_filter_not_implemented()

    def check_nonexistant_filterkey(self):
        """
        Entering any filterkey other than 'kalman', 'particle'
        or 'custom' raises AssertionError.
        """
        with self.assertRaises(NameError):
            linsolver.LinearisedODESolver(self.working_2d_prior,
                                          filtertype="rubbish")

    def check_particle_filter_not_implemented(self):
        """
        Particle filter should be an option, yet raise an error.
        """
        with self.assertRaises(NotImplementedError):
            linsolver.LinearisedODESolver(self.working_2d_prior,
                                          filtertype="particle")

    def check_custom_filter_not_implemented(self):
        """
        Custom filter should be an option, yet raise an error.
        """
        with self.assertRaises(NotImplementedError):
            linsolver.LinearisedODESolver(self.working_2d_prior,
                                          filtertype="custom")

    def test_inconsistent_prior_and_ssm(self):
        """
        Prior uses dim=2, so a scalar ODE should raise an AssertionError.
        """
        solver = linsolver.LinearisedODESolver(self.working_2d_prior,
                                               self.working_filtertype)
        wrong_dimensional_ode = linode.LinearODE(t0=0., tmax=1.,
                                                 params=1.123, initval=1.)
        with self.assertRaises(AssertionError):
            solver.solve(wrong_dimensional_ode, stepsize=0.1)


class TestLinearisationScalar(unittest.TestCase):
    """
    Tests output format of LinearisedODESolver for scalar ODEs.
    Among others, we test whether the rhs_parts output has shape
    (ntsteps, npar=1, ndim=1).
    """
    def setUp(self):
        """
        Set up linear ODE and linearised ODESolver.
        """
        prior = stsp.IBM(q=1, dim=1)
        solver = linsolver.LinearisedODESolver(prior, filtertype="kalman")
        ode = linode.LinearODE(t0=0.0, tmax=1.2, params=2.0, initval=4.1)
        h = 0.1
        output = solver.solve(ode, h)
        __, __, __, self.rhs_parts, self.uncert = output

    def test_output_format_rhs_parts(self):
        """
        self.rhs_parts should have shape (ntsteps, nparams=1, ndim=1)
        to be compliant with linearisation module.
        """
        self.assertEqual(len(self.rhs_parts.shape), 3)
        self.assertEqual(self.rhs_parts.shape[1], 1)
        self.assertEqual(self.rhs_parts.shape[2], 1)

    def test_output_format_uncerts(self):
        """
        self.uncert should have shape (ntsteps, ndim=1) to be compliant with
        linearisation module.
        """
        self.assertEqual(len(self.uncert.shape), 2)
        self.assertEqual(self.uncert.shape[1], 1)


class TestLinearisation2d(unittest.TestCase):
    """
    Tests output format of LinearisedODESolver for 2d ODEs.
    Among others, we test whether the rhs_parts output has shape
    (ntsteps, npar=4, ndim=2), where m is the number of parameters.
    """
    def setUp(self):
        """
        Set up LotkaVolterra ODE and LinearisedODESolver.
        """
        prior = stsp.IBM(q=1, dim=2)
        solver = linsolver.LinearisedODESolver(prior, filtertype="kalman")
        params = [0.1, 0.2, 0.3, 0.4]
        ode = linode.LotkaVolterra(t0=0.0, tmax=1.2,
                                   params=params, initval=np.ones(2))
        h = 0.1
        output = solver.solve(ode, h)
        __, __, __, self.rhs_parts, self.uncert = output

    def test_output_format_rhs_parts(self):
        """
        self.rhs_parts should have shape (ntsteps, npar=4, ndim=2)
        to be compliant with linearisation module.
        """
        self.assertEqual(len(self.rhs_parts.shape), 3)
        self.assertEqual(self.rhs_parts.shape[1], 4)
        self.assertEqual(self.rhs_parts.shape[2], 2)

    def test_output_format_uncerts(self):
        """
        self.uncert should have shape (ntsteps, ndim=2) to
        be compliant with linearisation module.
        """
        self.assertEqual(len(self.uncert.shape), 2)
        self.assertEqual(self.uncert.shape[1], 2)


class TestRejectNonLinearisedODE(unittest.TestCase):
    """
    Tests whether linsolver.solve cannot be used with an ODE that is not
    linearised
    """
    def setUp(self):
        """
        Set up linearised ODESolver and Res2Bod as a non-linearised ODE.
        """
        prior = stsp.IBM(q=1, dim=4)
        self.solver = linsolver.LinearisedODESolver(prior, filtertype="kalman")
        self.ode = standard_ode.Res2Bod(t0=0.0, tmax=1.2)

    def test_reject_res2bod(self):
        """
        Using linsolver with res2bod should raise an attribute error
        """
        h = 0.1
        with self.assertRaises(AttributeError):
            self.solver.solve(self.ode, h)
