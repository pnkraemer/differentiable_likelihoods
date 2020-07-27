"""
test_linearisation.py

We test the linearisation.py module. 

Besides checking that some common wrong inputs raise errors,
we do sanitychecks regarding the equivalence of the linearisation
and a filter output.

Todo:
    * Refactor these tests into some more elegant design. At the moment,
    this is a lot of copy and pase.

Tests include:
    * Make linearisation.compute_linearisation complain about non-scalar
      uncerts (third element of derivative_data)
    * Test linear ODE for one and two evaluation points
    * Test logistic ODE for one and two evaluation points
    * Test lotka-volterra ODE for one and two evaluation points
"""

import unittest
import numpy as np
from odefilters import covariance as cov
from odefilters import odesolver
from odefilters import linearised_odesolver as linsolve
from odefilters import linearised_ode as linode
from odefilters import statespace
from odefilters import linearisation



class TestWrongInputs(unittest.TestCase):
    """
    Tests whether linearisation.compute_linearisation complains about
    input arguments in the wrong shape.
    """
    def test_uncert_not_scalar(self):
        """
        We test whether the uncertainty (third element of derivative_data)
        is only accepted as a scalar.
        """
        # Set Model Parameters
        odeparam = 1.
        y0, y0_unc = 1.0, 0 
        t0, tmax = 0.0, 1.25

        # Set Method Parameters
        q = 1
        h = 0.1

        # Set up and solve ODE
        ibm = statespace.IBM(q=q, dim=1)
        solver = linsolve.LinearisedODESolver(ibm)
        ivp = linode.LinearODE(t0, tmax, odeparam, y0, y0_unc)
        tsteps, means, __, rhs_parts, should_not_work = solver.solve(ivp, stepsize=h)
        self.mean = odesolver.get_trajectory(means, 0, 0)

        # Set up BM and IBM covariance matrices
        evalpt = np.array(tsteps[[-1]])
        with self.assertRaises(TypeError):
            derdat = (tsteps, rhs_parts, should_not_work)
            linearisation.compute_linearisation(ssm=ibm, initial_value=y0,
                                                derivative_data=derdat,
                                                prdct_tsteps=evalpt)


class TestSanityCheckLinearOneData(unittest.TestCase):
    """
    We check whether the mean as given out by the ODE-Filter
    coincides with certain GP regression for ONE datapoint.
    Based on: Linear ODE (one-dim, one parameter) and one evalpt.
    """
    def setUp(self):
        """
        Set up linear ODE (i.e. one-dim, one parameter) and one evalpt.
        """
        # Set Model Parameters
        odeparam = 1.
        y0, y0_unc = 1.0, 0 
        t0, tmax = 0.0, 1.25

        # Set Method Parameters
        q = 1
        h = 0.1

        # Set up and solve ODE
        ibm = statespace.IBM(q=q, dim=1)
        solver = linsolve.LinearisedODESolver(ibm)
        ivp = linode.LinearODE(t0, tmax, odeparam, y0, y0_unc)
        tsteps, means, __, rhs_parts, uncerts = solver.solve(ivp, stepsize=h)
        self.mean = odesolver.get_trajectory(means, 0, 0)

        # Set up BM and IBM covariance matrices
        evalpt = np.array([tsteps[-1]])
        derdat = (tsteps, rhs_parts, 0.)

        const, jacob = linearisation.compute_linearisation(
            ssm=ibm, initial_value=y0,
            derivative_data=derdat, prdct_tsteps=evalpt)

        # Compute GP Estimation of filter mean at t=tmax
        self.postmean = const + np.dot(jacob[:, 0], odeparam)


    def test_equivalence_to_filter_output(self):
        """
        Check whether filter output coincides with linearised version
        up to threshold 1e-12.
        """
        error = np.linalg.norm(self.postmean - self.mean[-1])/np.linalg.norm(self.mean[-1])
        self.assertLess(error, 1e-12)


class TestSanityCheckLogisticOneData(unittest.TestCase):
    """
    We check whether the mean as given out by the ODE-Filter
    coincides with certain GP regression for ONE datapoint.
    Based on: logistic ODE (one-dim, two parameters) and one evalpt.
    """
    def setUp(self):
        """
        Set up logistic ODE (i.e. one-dim, two parameters) and one evalpt.
        """
        # Set Model Parameters
        odeparam = np.array([1, 2])
        y0, y0_unc = 1.0, 0 
        t0, tmax = 0.0, 1.25

        # Set Method Parameters
        q = 1
        h = 0.1

        # Set up and solve ODE
        ibm = statespace.IBM(q=q, dim=1)
        solver = linsolve.LinearisedODESolver(ibm)
        ivp = linode.LogisticODE(t0, tmax, odeparam, y0, y0_unc)
        tsteps, means, __, rhs_parts, uncerts = solver.solve(ivp, stepsize=h)
        self.mean = odesolver.get_trajectory(means, 0, 0)

        # Set up BM and IBM covariance matrices
        evalpt = np.array(tsteps[[-1]])
        derdat = (tsteps, rhs_parts, 0.)

        const, jacob = linearisation.compute_linearisation(
            ssm=ibm, initial_value=y0,
            derivative_data=derdat, prdct_tsteps=evalpt)

        # Compute GP Estimation of filter mean at t=tmax
        self.postmean = const + np.dot(jacob, odeparam)


    def test_equivalence_to_filter_output(self):
        """
        Check whether filter output coincides with linearised version
        up to threshold 1e-12.
        """
        error = np.linalg.norm(self.postmean - self.mean[-1])/np.linalg.norm(self.mean[-1])
        self.assertLess(error, 1e-12)


class TestSanityCheckLotkaVolterraOneData(unittest.TestCase):
    """
    We check whether the mean as given out by the ODE-Filter
    coincides with certain GP regression for ONE datapoint.
    Based on: Lotka-Volterra ODE (two-dim, four parameters) and one evalpt.
    """
    def setUp(self):
        """
        Set up Lotka-Volterra ODE (i.e. two-dim, four parameter) and one evalpt.
        """
        # Set Model Parameters
        odeparam = np.array([0, 1, 1, 2])
        y0, y0_unc = np.ones(2), 0 * np.ones(2)
        t0, tmax = 0.0, 1.25

        # Set Method Parameters
        q = 1
        h = 0.1

        # Set up and solve ODE
        ibm = statespace.IBM(q=q, dim=len(y0))
        solver = linsolve.LinearisedODESolver(ibm)
        ivp = linode.LotkaVolterra(t0, tmax, odeparam, y0, y0_unc)
        tsteps, means, __, rhs_parts, uncerts = solver.solve(ivp, stepsize=h)
        self.mean = odesolver.get_trajectory_multidim(means, [0, 1], 0)

        # Set up BM and IBM covariance matrices
        evalpt = np.array(tsteps[[-1]])
        derdat = (tsteps, rhs_parts, 0.)

        const, jacob = linearisation.compute_linearisation(
            ssm=ibm, initial_value=y0,
            derivative_data=derdat, prdct_tsteps=evalpt)

        # Compute GP Estimation of filter mean at t=tmax
        postmean = const + np.dot(jacob, odeparam)
        self.postmean = postmean.reshape((2,))


    def test_equivalence_to_filter_output(self):
        """
        Check whether filter output coincides with linearised version
        up to threshold 1e-12.
        """
        error = np.linalg.norm(self.postmean - self.mean[-1])/np.linalg.norm(self.mean[-1])
        self.assertLess(error, 1e-12)


class TestSanityCheckLinearMultipleData(unittest.TestCase):
    """
    We check whether the mean as given out by the ODE-Filter
    coincides with certain GP regression for ONE datapoint.
    Based on: linear ODE (one-dim, one parameter) and two evalpts.
    """
    def setUp(self):
        """
        Set up linear ODE (i.e. one-dim, one parameter) and
        multiple (two) evalpts.
        """
        # Set Model Parameters
        odeparam = 1.
        y0, y0_unc = 1.0, 0 
        t0, tmax = 0.0, 1.25

        # Set Method Parameters
        q = 1
        h = 0.1

        # Set up and solve ODE
        ibm = statespace.IBM(q=q, dim=1)
        solver = linsolve.LinearisedODESolver(ibm)
        ivp = linode.LinearODE(t0, tmax, odeparam, y0, y0_unc)
        tsteps, means, __, rhs_parts, uncerts = solver.solve(ivp, stepsize=h)
        self.mean = odesolver.get_trajectory(means, 0, 0)

        # Set up BM and IBM covariance matrices
        evalpt = np.array(tsteps[[-1, -10]])
        derdat = (tsteps, rhs_parts, 0.)

        const, jacob = linearisation.compute_linearisation(
            ssm=ibm, initial_value=y0,
            derivative_data=derdat, prdct_tsteps=evalpt)

        # Compute GP Estimation of filter mean at t=tmax
        self.postmean = const + np.dot(jacob[:, 0], odeparam)


    def test_equivalence_to_filter_output(self):
        """
        Check whether filter output coincides with linearised version
        up to threshold 1e-12.
        """
        error = np.linalg.norm(self.postmean - self.mean[[-1, -10]])/np.linalg.norm(self.mean[-1])
        self.assertLess(error, 1e-12)


class TestSanityCheckLogisticMultipleData(unittest.TestCase):
    """
    We check whether the mean as given out by the ODE-Filter
    coincides with certain GP regression for ONE datapoint.
    Based on: logistic ODE (one-dim, two parameters) and two evalpts.
    """
    def setUp(self):
        """
        Set up logistic ODE (i.e. one-dim, two parameters) and
        multiple (two) evalpts.
        """
        # Set Model Parameters
        odeparam = np.array([1, 2])
        y0, y0_unc = 1.0, 0 
        t0, tmax = 0.0, 1.25

        # Set Method Parameters
        q = 1
        h = 0.1

        # Set up and solve ODE
        ibm = statespace.IBM(q=q, dim=1)
        solver = linsolve.LinearisedODESolver(ibm)
        ivp = linode.LogisticODE(t0, tmax, odeparam, y0, y0_unc)
        tsteps, means, __, rhs_parts, uncerts = solver.solve(ivp, stepsize=h)
        self.mean = odesolver.get_trajectory(means, 0, 0)

        # Set up BM and IBM covariance matrices
        evalpt = np.array(tsteps[[-1, -10]])
        derdat = (tsteps, rhs_parts, 0.)

        const, jacob = linearisation.compute_linearisation(
            ssm=ibm, initial_value=y0,
            derivative_data=derdat, prdct_tsteps=evalpt)

        # Compute GP Estimation of filter mean at t=tmax
        self.postmean = const + np.dot(jacob, odeparam)


    def test_equivalence_to_filter_output(self):
        """
        Check whether filter output coincides with linearised version
        up to threshold 1e-12.
        """
        error = np.linalg.norm(self.postmean - self.mean[[-1, -10]])/np.linalg.norm(self.mean[-1])
        self.assertLess(error, 1e-12)


class TestSanityCheckLotkaVolterraMultipleData(unittest.TestCase):
    """
    We check whether the mean as given out by the ODE-Filter
    coincides with certain GP regression for ONE datapoint.
    Based on: lotka-volterra ODE (two-dim, four parameters) and two evalpts.
    """
    def setUp(self):
        """
        Set up Lotka-Volterra ODE (i.e. two-dim, four parameter) and
        multiple (two) evalpts.
        """
        # Set Model Parameters
        odeparam = np.array([0, 1, 1, 2])
        y0, y0_unc = np.ones(2), 0 * np.ones(2)
        t0, tmax = 0.0, 1.25

        # Set Method Parameters
        q = 1
        h = 0.1

        # Set up and solve ODE
        ibm = statespace.IBM(q=q, dim=len(y0))
        solver = linsolve.LinearisedODESolver(ibm)
        ivp = linode.LotkaVolterra(t0, tmax, odeparam, y0, y0_unc)
        tsteps, means, __, rhs_parts, uncerts = solver.solve(ivp, stepsize=h)
        self.mean = odesolver.get_trajectory_multidim(means, [0, 1], 0)

        # Set up BM and IBM covariance matrices
        evalpt = np.array(tsteps[[-1, -10]])
        derdat = (tsteps, rhs_parts, 0.)

        const, jacob = linearisation.compute_linearisation(
            ssm=ibm, initial_value=y0,
            derivative_data=derdat, prdct_tsteps=evalpt)

        # Compute GP Estimation of filter mean at t=tmax
        postmean = const + np.dot(jacob, odeparam)
        self.postmean = postmean.reshape((2, 2))


    def test_equivalence_to_filter_output(self):
        """
        Check whether filter output coincides with linearised version
        up to threshold 1e-12.
        """
        error = np.linalg.norm(self.postmean - self.mean[[-1, -10]])/np.linalg.norm(self.mean[-1])
        self.assertLess(error, 1e-12)
