# coding=utf-8
"""
odesolver.py

ODE-solver class based on Bayesian filtering. 

References:
    [1] 'Probabilistic Solutions to ODEs as Non-Linear Bayesian
        Filtering: A New Perspective',
        F. Tronarp, H. Kersting, S. Särkkä and P. Hennig
    [2] 'Active Uncertainty Calibration in Bayesian ODE Solvers',
        H. Kersting and P. Hennig

Example:
    >>> ibm = statespace.IBM(q=2, dim=1)
    >>> solver = odesolver.ODESolver(ibm, filtertype="kalman")
    >>> lin_ode = ode.LinearODE(t0=0.1, tmax=2.0, params=1.0, initval=0.9)
    >>> t, m, c = solver.solve(lin_ode, stepsize=0.01)

Todo:
    Make solver return covariances instead of standard deviations. This
    is only relevant for uncertain inital values
    (otherwise the correlations) are zero.
"""


import numpy as np
from odefilters import filters
from abc import ABC


class AbstractODESolver(ABC):
    """
    An ODE Solver is essentially a Filter with 
    different initialisation and where a data stream is replaced
    by function evaluation.
    """
    def __init__(self, filt):
        """
        Initialise the ODE-solver with a filter
        (which contains the state-space model prior).
        Interface is provided through odesolver.ODESolver
        class below.

        Args:
            filt:   filters.Filter object.
        """
        if isinstance(filt, filters.Filter) is False:
            raise TypeError("Please enter a Filter object.")
        self.filt = filt
        self.dim = filt.ssm.dim
        self.q = filt.ssm.q

    def __repr__(self):
        """
        Represent ODESolver with input information.
        Relies on __repr__ of SSM and Filter.
        """
        return "ODESolver(filter=%r) "% (self.filt)

    def solve(self, ode, stepsize):
        """
        Solves given ODE with given stepsize.

        Args:
            ode:        ODE object, see ode.py
            stepsize:   stepsize of discretisation; usually known as 'h'
        Returns:
            tsteps:     shape (N,), N is number of timesteps
            means:      shape (N, d, q+1), contains mean estimates of states
            stdevs:     shape (N, d, q+1), contains std.-dev. of means
        """
        assert(self.ode_matches_ssm(ode) is True)
        self.discretise_statespacemodel(stepsize)
        tsteps, means, covars = self.make_arrays(ode, stepsize)
        meanpred, covarpred = self.initialise(ode)
        for i, time in enumerate(tsteps):
            data, uncert = self.evaluate_model(ode, meanpred, covarpred, time)
            means[i], covars[i] = self.filt.update(meanpred, covarpred,
                                                   data, uncert)
            meanpred, covarpred = self.filt.predict(means[i], covars[i])
        means, stdevs = self.organise(means, covars)
        return tsteps, means, stdevs

    def ode_matches_ssm(self, ode):
        """
        Checks compliance of ode-dim and ssm-dim.
        """
        if(np.isscalar(ode.initval) is True):
            dim = 1
        else:
            dim = len(ode.initval)
        assert(dim == self.filt.ssm.dim),\
            "The ODE and prior state space model have different dimensions"
        return True

    def discretise_statespacemodel(self, *args):
        """
        Wrapper for IBM.discretise.
        """
        self.filt.ssm.discretise(*args)

    def make_arrays(self, ode, stepsize):
        """
        Allocates arrays for tsteps, means and covars
        depending on time-discretisation.

        Args:
            ode:        ode.ODE object.
            stepsize:   stepsize of discretisation; usually known as 'h'
        Returns:
            tsteps:     timesteps (t0, t0+h, t0+2h, ..., tmax); shape (N,)
            means:      zero mean array, shape (N, d*(q+1))
            covars:     zero covariance array, shape (N, d*(q+1), d*(q+1))
        """
        t0 = ode.t0
        tmax = ode.tmax
        tsteps = np.arange(t0, tmax + stepsize, stepsize)
        means = np.zeros((len(tsteps), self.dim * (self.q + 1)))
        covars = np.zeros((len(tsteps), self.dim * (self.q + 1),
                           self.dim * (self.q + 1)))
        return tsteps, means, covars

    def initialise(self, ode):
        """
        Initialise ODE-filter iterations via
        'correcting' initial mean and covariance from prior ssm
        with respect to initial values from ode.

        Args:
            ode:    dictionary; see ode.py
        Returns:
            mean:   mean initialisation, shape (d*(q+1),)
            covar:  covariance initialisation, shape (d*(q+1), d*(q+1))
        """
        y0, y0_unc = self.read_initial_values(ode)
        self.obsmat = self.compute_obsmat()
        init_mean, init_covar = self.filt.initialise()
        res = self.filt.compute_residual(y0, self.obsmat, init_mean)
        kgain = self.filt.compute_kalmangain(init_covar, self.obsmat,
                                             meascovar=y0_unc)
        mean, covar = self.filt.compute_correction(init_mean, init_covar,
                                                   res, self.obsmat, kgain)
        return mean, covar

    def read_initial_values(self, ode):
        """
        Extracts initial value and---more importantly---uncertainty with
        respect to the initial value (if this key exists)
        """
        y0 = ode.initval
        y0_unc = ode.initval_unc
        return y0, y0_unc

    def compute_obsmat(self):
        """
        Computes projection matrix H_0.
        Note: the projection matrix H_1 is an attribute of
              the state space, not of the solver!

        Returns:
            h0:     shape (d, d*(1+1))
        """
        h0_1d = np.eye(self.q + 1)[:, 0].reshape((1, self.q + 1))
        h0 = np.kron(np.eye(self.dim), h0_1d)
        return h0

    def evaluate_model(self, ode, meanpred, covarpred, time):
        """
        Evaluates model at time t based on predictive mean and covariance.
        Computes y=E[f] and r=E[f**2]-y**2.

        Args:
            meanpred:   mean prediction, shape (d*(q+1),)
            covarpred:  covariance prediction, shape (d*(q+1), d*(q+1))
            time:       current timestep
        Returns:
            data:       shape (d,), approximation to y
            uncert:     shape (d,), approximation to r
        """
        reduced_mean, reduced_covar = self.extract_coordinates(meanpred, covarpred)
        data = ode.modeval(time, reduced_mean)
        uncert = 0 * data
        return data, uncert

    def extract_coordinates(self, mean, covar):
        """
        Extracts current mean and covariance estimate of the state y(t) via
            m -> H_0 m
            C -> H_0 C H_0^T
        and sets offdiagonal values of C to zero. 
        The reason for making the dimensions independent is that
        otherwise, the covariance matrix may be indefinite.
        The result does not seem to be changed. 

        Args:
            mean:       shape (d*(q+1),)
            covar:      shape (d*(q+1), d*(q+1))
        Returns:
            meanex:     extracted mean, shape (d,)
            covarex:    extracted covariance, diagonal of H_1 C H_1^T,
                        shape (d, d)
        """
        meanex = self.obsmat.dot(mean)
        covarex_full = self.obsmat.dot(covar).dot(self.obsmat.T)
        covarex = self.remove_offdiagonal(covarex_full)
        return meanex, covarex

    def remove_offdiagonal(self, mtrx):
        """Sets all but the elements on the diagonal to zero."""
        return np.diag(np.diag(mtrx))

    def organise(self, means, covars):
        """
        Turns (N, d*(q+1)) shaped arrays into
        (N, d, q+1) arrays via
            (m_1^1, ..., m_(q+1)^1, m_1^2, ..., m_(q+1)^2, ...)
            -> ((m_1^1, ..., m_(q+1)^1),
                (m_1^2, ..., m_(q+1)^2),
                ...)
        """
        stdevs = self.extract_stdevs(covars)
        means = means.reshape((len(means), self.dim, self.q + 1), order='C')
        stdevs = stdevs.reshape((len(stdevs), self.dim, self.q + 1), order='C')
        return means, stdevs

    def extract_stdevs(self, covars):
        """
        Return sqrt of diagonal entries at each timestep of
        covariance time series.
        """
        # print(covars)
        stdevs = np.sqrt(np.abs(np.diagonal(covars, axis1=1, axis2=2)))
        return stdevs


class ODESolver(AbstractODESolver):
    """
    Interface for AbstractODESolver class.

    Usage is much more convenient with inputs 'ssm' and 'filtertype'
    instead of the 'filt', however, algorithmically speaking, the ODESolver
    should only have a filter key. {ODESolver has a {Filter has a SSM}}
    instead of {ODESolver has a {{SSM}, {Filter has a SSM}}} because
    otherwise, the ssm key is duplicated.

    Example:
        >>> ibm = statespace.IBM(q=2, dim=1)
        >>> solver = odesolver.ODESolver(ibm, filtertype="kalman")
        >>> lin_ode = ode.LinearODE(t0=0.1, tmax=2.0, params=1.0, initval=0.9)
        >>> t, m, c = solver.solve(lin_ode, stepsize=0.01)
    """
    def __init__(self, ssm, filtertype="kalman"):
        """
        Initialise the ODE-solver with a prior (statespace) model
        and the desired kind of filter.

        Args:
            ssm:        state space model
            filtertype: keyword that decides which filter's methods
                        should be applied. Default is 'kalman'
        """
        filt = self.make_filter(ssm, filtertype)
        AbstractODESolver.__init__(self, filt)

    def make_filter(self, ssm, filtertype):
        """
        Assigns the correct filter according to input 'filtertype'.

        Args:
            ssm:        state space model
            filtertype: keyword that decides which filter's methods
                        should be applied.
        Returns:
            filt:       filter object initialised with statespace
                        model; usually KalmanFilter object.
        """
        if filtertype == "kalman":
            filt = filters.KalmanFilter(ssm)
        elif filtertype == "particle":
            raise NotImplementedError(
                "Particle filter implementation is future work.")
        elif filtertype == "EKF":
            raise NotImplementedError(
                "EKF implementation is future work.")
        elif filtertype == "UKF":
            raise NotImplementedError(
                "UKF implementation is future work.")
        elif filtertype == "custom":
            raise NotImplementedError(
                "Custom filter implementation is future work.")
        else:
            raise NameError(
                "Please enter a valid filter key:\
{'kalman'(, 'particle', 'EKF', 'UKF', 'custom')}")
        return filt


def get_trajectory(states_or_stdevs, idx_dim, idx_q):
    """
    This function remembers which of the many dimensions of state and stdevs
    belong to spatial dimensions and which belong to q

    Args:
        states_or_stdevs:   return value of solve(), either states or stdevs
        idx_dim:            which spatial dimension shall be returned
        idx_q:              which q value shall be returned
    Returns:
        traj:               trajectory of demanded indices
    """
    assert(idx_dim < states_or_stdevs.shape[1]), \
        "Spatial coordinate index out of bounds"
    assert(idx_q < states_or_stdevs.shape[2]), \
        "Derivative coordinate index out of bounds"
    traj = states_or_stdevs[:, idx_dim, idx_q]
    return traj

def get_trajectory_ddim(states_or_stdevs, d, idx_q):
    """
    This function remembers which of the many dimensions of state and stdevs
    belong to spatial dimensions and which belong to q. It returns all spatial
    dimensions for all q in idx_q.

    Args:
        states_or_stdevs: return value of solve(), either states or stdevs
        d: amount of spatial dimensions
        idx_q: which q value shall be returned
    Returns:
        traj: trajectory of demanded indices
    """
    assert(d-1 < states_or_stdevs.shape[1]), \
        "Spatial coordinate index out of bounds"
    assert(idx_q < states_or_stdevs.shape[2]), \
        "Derivative coordinate index out of bounds"
    traj = states_or_stdevs[:, list(range(d)), idx_q]
    return traj

def get_trajectory_multidim(states_or_stdevs, idx_dims, idx_q):
    """
    This function remembers which of the many dimensions of state and stdevs
    belong to spatial dimensions and which belong to q

    Args:
        states_or_stdevs: return value of solve(), either states or stdevs
        idx_dims: list of spatial dimensions that shall be returned
        idx_q: which q value shall be returned
    Returns:
        traj: trajectory of demanded indices
    """
    assert(all(i < states_or_stdevs.shape[1] for i in idx_dims)), \
        "Spatial coordinate index out of bounds"
    assert(idx_q < states_or_stdevs.shape[2]), \
        "Derivative coordinate index out of bounds"
    traj = states_or_stdevs[:, idx_dims, idx_q]
    return traj

# END OF FILE
