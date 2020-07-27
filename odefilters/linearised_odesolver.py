# coding=utf-8
"""
linearised_odesolver.py

Contains a version of odefilter.odesolver.ODESolver that
works with odefilter.linearised_ode.LinearisedODE objects.
These objects are special in the way that their RHS function
is linear w.r.t. parameters,
    f(t, x) = sum_i=1^m p_i f_i(t, x)
Further, it returns the rhs_parts as returned by 
odefilter.LinearisedODE.modeval_parts.

Note:
    This connection is the reason for the name LinearisedODESolver,
    whereas something along the lines of ODESolverWithData may have
    been more correct.
"""

import numpy as np
from odefilters import filters
from odefilters import odesolver


class LinearisedODESolver(odesolver.ODESolver):
    """
    ODESolver which works with linearised ODEs in the form of 
    linearised_ode.LinearisedODE and returns evaluations
    of each f_1, ..., f_N.

    Overwrites:
        LinearisedODESolver.solve
        LinearisedODESolver.evaluate_model
    Creates:
        LinearisedODESolver.allocate_rhsvals
    """
    def solve(self, ode, stepsize):
        """
        Solves given ODE with given stepsize.

        Args:
            ode:      
            stepsize:   stepsize of discretisation; usually known as 'h'
        Returns:
            tsteps:     shape (ntsteps,), timesteps; N is number of timesteps
            means:      shape (ntsteps, ndim, q+1), mean estimates of states
            stdevs:     shape (ntsteps, ndim, q+1), standard deviations of means
            rhs_parts:  evaluations of ode.modeval_parts,
                        shape (ntsteps, nparams, ndim),
            uncerts:    shape (ntsteps, ndim); for now, equal to zero.
        """
        assert(self.ode_matches_ssm(ode) is True)
        if hasattr(ode, 'is_linearised') is False or ode.is_linearised is False:
            raise AttributeError("Please enter an InvProblemODE instance")
        self.discretise_statespacemodel(stepsize)
        tsteps, means, covars = self.make_arrays(ode, stepsize)
        all_rhs_parts, uncertvals = self.allocate_all_rhs_parts(ode, stepsize)
        meanpred, covarpred = self.initialise(ode)
        for i, time in enumerate(tsteps):
            data, rhs_parts, uncert = self.evaluate_model(ode, meanpred, covarpred, time)
            all_rhs_parts[i], uncertvals[i] = rhs_parts, uncert
            means[i], covars[i] = self.filt.update(meanpred, covarpred, data, uncert)
            meanpred, covarpred = self.filt.predict(means[i], covars[i])
        means, stdevs = self.organise(means, covars)
        return tsteps, means, stdevs, all_rhs_parts, uncertvals

    def allocate_all_rhs_parts(self, ode, stepsize):
        """
        Allocates arrays for all_rhs_parts and uncertvals.

        Args:
            ode:        ode.ODE object
            stepsize:   scalar
        Returns:
            rhs_parts:  (N, m, d) shaped array
            uncerts:    (N, d) shaped array
        """
        nsteps = int(np.ceil((ode.tmax - ode.t0) / stepsize)) + 1
        nparams = self.get_dimension(ode.params)
        ndim = self.get_dimension(ode.initval)
        rhs_parts = np.zeros((nsteps, nparams, ndim))
        uncerts = np.zeros((nsteps, ndim))
        return rhs_parts, uncerts

    def get_dimension(self, scal_or_arr):
        """
        Returns dimension of sth that is either a scalar (i.e. d=1)
        or a np.array.

        Args:
            scal_or_arr:    scalar or np.array
        Returns:
            dim:            either 1 or len(np.array)
        """
        if np.isscalar(scal_or_arr) is True:
            dim = 1
        else:
            dim = len(scal_or_arr)
        return dim

    def evaluate_model(self, ode, meanpred, covarpred, time):
        """
        Evaluates model at time t based on predictive mean and covariance.
        Computes y=E[f] and r=E[f**2]-y**2 via strategy defined in self.evalpars

        Args:
            meanpred: mean prediction, shape (d*(q+1),)
            covarpred: covariance prediction, shape (d*(q+1), d*(q+1))
            time: current timestep
        Returns:
            data: shape (d,), approximation to y
            rhs_parts: shape (N, m, d) evaluation of [f_1, f_2, ...]
            uncert: shape (d,), approximation to r
        """
        reduced_mean, reduced_covar = self.extract_coordinates(meanpred, covarpred)
        data = ode.modeval(time, reduced_mean)
        rhs_parts = ode.modeval_parts(time, reduced_mean)
        uncert = 0 * data
        return data, rhs_parts, uncert

