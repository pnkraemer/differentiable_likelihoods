# coding=utf-8
"""
test_odesolver.py

Tests include:
    * First few means and covars coincide with
      the proof of Proposition 1 in [1]
    * Wrong inputs raise errors

Reference for TestScalarODE:
[1] A probabilistic model for the numerical solution of initial value problems,
    M. Schober and S. Särkkä and P. Hennig,
    Statistics and Computing (2019) 29:99-122
"""

import unittest
import numpy as np
from odefilters import statespace as stsp
from odefilters import odesolver as oso
from odefilters import ode


class TestScalarODE(unittest.TestCase):
    """
    Test whether the mean and covariance output of
    applying the solver to a SCALAR ODE coincide
    with the ones in the proof of Proposition 1 in [1]; see p. 108.
    """
    def setUp(self):
        """Setup odesolver and solve a scalar ode"""
        prior = stsp.IBM(q=1, dim=1)
        self.solver = oso.ODESolver(prior, filtertype="kalman")
        self.ode = ode.LinearODE(t0=0.0, tmax=1.2, params=2.0, initval=4.1)
        self.h = 0.1
        self.tsteps, self.means, self.stdevs = self.solver.solve(
            self.ode, stepsize=self.h)

    def test_first_few_iterations(self):
        """
        Test whether first few means and covariances coincide with Prop. 1.
        """
        self.check_mean_t0()
        self.check_stdevs_t0()
        self.check_mean_t1()
        self.check_stdevs_t1()

    def check_mean_t0(self):
        """Expect: m(t0) = (y0, z0) where z0=f(y0)"""
        y0 = self.ode.initval
        z0 = self.ode.modeval(t=0.0, x=y0)
        mean_at_t0 = self.means[0][0]
        self.assertEqual(mean_at_t0[0], y0)
        self.assertEqual(mean_at_t0[1], z0)

    def check_stdevs_t0(self):
        """Expect: C(t0) = 0, hence stdevs equal to zero"""
        stdev_at_t0 = self.stdevs[0][0]
        self.assertEqual(stdev_at_t0[0], 0.0)
        self.assertEqual(stdev_at_t0[1], 0.0)

    def check_mean_t1(self):
        """Expect: m(t0) = (y0 + h/2*(z0 + z1), z1)"""
        y0 = self.ode.initval
        z0 = self.ode.modeval(t=0.0, x=y0)
        z1 = self.ode.modeval(t=0.0, x=(y0 + self.h * z0))
        mean_at_t1 = self.means[1][0]
        self.assertEqual(mean_at_t1[0], y0 + 0.5 * self.h * (z0 + z1))
        self.assertEqual(mean_at_t1[1], z1)

    def check_stdevs_t1(self):
        """Expect: C(t1) = (sigma**2 h**3/12, 0; 0, 0)"""
        stdev_at_t1 = self.stdevs[1][0]
        sigmasquared = self.solver.filt.ssm.diffconst       # digging deep
        self.assertAlmostEqual(stdev_at_t1[0],
                               np.sqrt(self.h**3 * sigmasquared / 12.0),
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
            oso.ODESolver(self.working_2d_prior, filtertype="rubbish")

    def check_particle_filter_not_implemented(self):
        """Particle filter should be an option, yet raise an error."""
        with self.assertRaises(NotImplementedError):
            oso.ODESolver(self.working_2d_prior, filtertype="particle")

    def check_custom_filter_not_implemented(self):
        """Custom filter should be an option, yet raise an error."""
        with self.assertRaises(NotImplementedError):
            oso.ODESolver(self.working_2d_prior, filtertype="custom")

    def test_inconsistent_prior_and_ssm(self):
        """
        Prior uses dim=2, so a scalar ODE should raise an AssertionError.
        """
        solver = oso.ODESolver(self.working_2d_prior, self.working_filtertype)
        wrong_dimensional_ode = ode.LinearODE(t0=0., tmax=1.,
                                              params=1.123, initval=1.)
        with self.assertRaises(AssertionError):
            solver.solve(wrong_dimensional_ode, stepsize=0.1)

def traj_passes_pt(traj, pt, threshold=1e-10):
    """
    Returns true if trajectory crosses a point within
    radius 1e-10. 
    """
    return np.amin(np.abs(traj- pt)) < threshold


class TestFHNPassesPoint(unittest.TestCase):
    """
    Tests whether a solution to solving the FHN model
    crosses a certain benchmark point.

    In fact this is more of a regression/integrationtest, however,
    due to a lack of framework for these we include them
    as a unittest.
    """
    def setUp(self):
        pars = [0., 0.2, 0.2, 3.0]
        y0 = np.array([-1, 1])
        ivp = ode.FitzHughNagumo(t0=0., tmax=5., params=pars, initval=y0)
        ibm_prior = stsp.IBM(q=1, dim=2)
        solver = oso.ODESolver(ibm_prior)
        self.t, mean, stdev = solver.solve(ivp, stepsize=0.05)
        self.m1 = oso.get_trajectory(mean, 0, 0)
        self.m2 = oso.get_trajectory(mean, 1, 0)

    def test_solution(self):
        """
        Checks whether the trajectory of the FHN solution crosses
        the benchmark point array([1.85..., 0.06...]) within a
        radius of 1e-3
        """
        x_benchmark = -1.85284439
        y_benchmark = -0.65490004
        thresh = 1e-3
        self.assertEqual(traj_passes_pt(self.m1, x_benchmark, thresh), True)
        self.assertEqual(traj_passes_pt(self.m2, y_benchmark, thresh), True)



class TestRes2BodPassesPoint(unittest.TestCase):
    """
    Tests whether a solution to solving the Res2Bod model
    crosses a certain benchmark point.

    In fact this is more of a regressiontest, however,
    due to a lack of framework for these we include them
    as a unittest.
    """
    def setUp(self):
        ivp = ode.Res2Bod(t0=0., tmax=0.011)
        ibm_prior = stsp.IBM(q=2, dim=4)
        solver = oso.ODESolver(ibm_prior)
        self.t, mean, stdev = solver.solve(ivp, stepsize=0.0001)
        self.m1 = oso.get_trajectory(mean, 0, 0)
        self.m2 = oso.get_trajectory(mean, 1, 0)

    def test_solution(self):
        """
        Checks whether the trajectory of the Res2Bod solution crosses
        the benchmark point array([0.98..., -0.01...]) within a
        radius of 1e-10.
        """
        x_benchmark = 0.986122388809302
        y_benchmark = -0.014199005665214861
        thresh = 1e-10
        self.assertEqual(traj_passes_pt(self.m1, x_benchmark, thresh), True)
        self.assertEqual(traj_passes_pt(self.m2, y_benchmark, thresh), True)


    def test_solution_controlvar(self):
        """
        Controlvariate to the test_solution() test. For q < 2 or
        stepsize > 0.0001 the precision should not be achieved anymore. 
        """
        ivp = ode.Res2Bod(t0=0., tmax=0.011)
        ibm_prior = stsp.IBM(q=1, dim=4)
        solver = oso.ODESolver(ibm_prior)
        self.t, mean, stdev = solver.solve(ivp, stepsize=0.0001)
        self.m1 = oso.get_trajectory(mean, 0, 0)
        self.m2 = oso.get_trajectory(mean, 1, 0)
        x_benchmark = 0.986122388809302
        y_benchmark = -0.014199005665214861
        thresh = 1e-10
        self.assertEqual(traj_passes_pt(self.m1, x_benchmark, thresh), False)
        self.assertEqual(traj_passes_pt(self.m2, y_benchmark, thresh), False)
