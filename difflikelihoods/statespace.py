"""
statespace.py

Reference:
    Chapter 4 of https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf

Example:
    >>> import numpy as np
    >>> import statespace
    >>> tm = np.eye(2)
    >>> var = 0.1 * np.eye(2)
    >>> im = np.zeros(2)
    >>> rw = statespace.InvariantSSM(tm, var, tm, var, im, var)
    >>> traj = rw.sample_trajectory(nsteps=50)
    >>> print(traj)
"""
import numpy as np
from difflikelihoods import auxiliary as aux
from abc import ABC, abstractmethod


class SSM(ABC):
    """
    Linear Gaussian state space model with Gaussian prior:
        X_0 ~ N(m_0, C_0)
        X_(t+1) = A_t X_t + q_t,    q_t ~ N(0, Q_t)
        Y_(t+1) = H_(t+1) X_(t+1) + r_(t+1), r_(t+1) ~ N(0, R_(t+1))
    where we call
        A_t: transition matrix (transmat)
        Q_t: transition covariance matrix (transcovar)
        H_t: measurement matrix (measmat)
        R_t: measurement covariance matrix (meascovar)
    as well as the prior distribution
        m_0: initial mean (init_mean)
        C_0: initial covariance (init_covar)
    """

    @abstractmethod
    def sample_trajectory(self, nsteps):
        """
        Samples N steps of state space model.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_transmat(self, *args):
        """
        Either computes or returns transition matrix.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_transcovar(self, *args):
        """
        Either computes or returns covariance matrix of transition map.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_measmat(self, *args):
        """
        Either computes or returns measurement matrix.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_meascovar(self, *args):
        """
        Either computes or returns covariance matrix of measurement map
        or an approximation thereof.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError


class InvariantSSM(SSM):
    """
    Time invariant linear Gaussian state space model:
        X_(t+1) = A X_t + Q,    X_0 ~ N(m_0, C_0)
        Y_(t+1) = H X_(t+1) + R
    and thus "self.get_*" return existing matrices.
    """

    def __init__(self, transmat, transcovar, measmat, meascovar, init_mean, init_covar):
        """
        Initialises invariant state space model.

        Args:
            transmat: transition matrix (A), shape (n, n)
            transcovar: covariance of transition map (Q), shape (n, n)
            measmat: measurement matrix (H), shape (l, n)
            meascovar: covariance of measurement map (R), shape (l, l)
            init_mean: covariance of measurement map (R), shape (n,)
            init_covar: covariance of measurement map (R), shape (n, n)
        """
        assert (
            self.inputs_are_arrays(
                transmat, transcovar, measmat, meascovar, init_mean, init_covar
            )
            is True
        )
        self.transmat = transmat
        self.transcovar = transcovar
        self.measmat = measmat
        self.meascovar = meascovar
        self.init_mean = init_mean
        self.init_covar = init_covar

    def __repr__(self):
        """
        Represent IBM object as e.g. IBM(q=2, dim=1).
        """
        d = len(self.transmat)
        return "statespace.InvariantSSM() instance in d=%r" % d

    def inputs_are_arrays(
        self, transmat, transcovar, measmat, meascovar, init_mean, init_covar
    ):
        """
        Checks whether inputs are suitable numpy arrays.

        Args:
            transmat: shape (n, n)
            transcovar: shape (n, n)
            measmat: shape (l, n)
            meascovar: shape (l, l)
            init_mean: shape (n,)
            init_covar: shape (n, n)
        Raises:
            AssertionError: if any np.array does not have the correct dimension
        Returns:
            True: if all shapes are as expected
        """
        assert (
            self.input_is_array(transmat, 2) is True
        ), "Please input a 2d array for transmat"
        assert (
            self.input_is_array(transcovar, 2) is True
        ), "Please input a 2d array for transcovar"
        assert (
            self.input_is_array(measmat, 2) is True
        ), "Please input a 2d array for measmat"
        assert (
            self.input_is_array(meascovar, 2) is True
        ), "Please input a 2d array for meascovar"
        assert (
            self.input_is_array(init_mean, 1) is True
        ), "Please input a 1d array for init_mean"
        assert (
            self.input_is_array(init_covar, 2) is True
        ), "Please input a 2d array for init_covar"
        return True

    def input_is_array(self, arr, dim):
        """
        Checks whether array is a numpy.ndarray type of dimension dim.

        Args:
            arr: np.ndarray, shape (a_1, ..., a_dim)
            dim: expected dim. of shape
        Raises:
            AssertionError: if either the type or the dimension is wrong
        Returns:
            True: if everything is as expected
        Examples:
            input_is_array(np.random.rand(2,3), 1): False
            input_is_array(np.random.rand(2,3), 2): True
            input_is_array(np.random.rand(2,3), 3): False
        """
        assert isinstance(arr, np.ndarray), "Please enter a %ud shaped array" % dim
        assert len(arr.shape) == dim, "Please enter a %ud shaped array" % dim
        return True

    def sample_trajectory(self, nsteps):
        """
        Sample a trajectory from the state space for 
        a time-invariante linear Gaussian state space model

        Args:
            nsteps: number of timesteps of sample trajectory
        Returns:
            obs: observation array, shape (nsteps, l)
                 where l is the dimensionality of the observations
        """
        traj, obs = self._alloc_arrays(nsteps)
        traj[0] = self._get_first_state()
        obs[0] = self._observe(traj[0])
        for i in range(1, nsteps):
            traj[i] = self._iterate_state(traj[i - 1])
            obs[i] = self._observe(traj[i])
        return obs

    def _alloc_arrays(self, nsteps):
        """Preallocates arrays for trajectory and observations.

        Args:
            nsteps: number of timesteps
        Returns:
            traj: shape (nsteps, ndim)
            obs: shape (nsteps, nobs)
        """
        traj = np.zeros((nsteps, len(self.init_mean)))
        obs = np.zeros((nsteps, len(self.measmat)))
        return traj, obs

    def _get_first_state(self):
        """
        Returns sample of state trajectory at time zero.
        """
        firststate = np.random.multivariate_normal(self.init_mean, self.init_covar)
        return firststate

    def _iterate_state(self, currstate):
        """
        Iterates states: x->Ax + q, q ~ N(0, Q)
        """
        zeromean = np.zeros(len(self.init_mean))
        noise = np.random.multivariate_normal(zeromean, self.transcovar)
        state = self.transmat.dot(currstate) + noise
        return state

    def _observe(self, currstate):
        """
        Computes observation: x->Hx + r, r ~ N(0, R)
        """
        zeromean = np.zeros(len(self.measmat))
        noise = np.random.multivariate_normal(zeromean, self.meascovar)
        obs = self.measmat.dot(currstate) + noise
        return obs

    def get_transmat(self, *args):
        """
        Returns transition matrix.
        Overwrites superclass method.
        """
        return self.transmat

    def get_transcovar(self, *args):
        """
        Returns covariance matrix of the transition map.
        Overwrites superclass method.
        """
        return self.transcovar

    def get_measmat(self, *args):
        """
        Returns measurement matrix.
        Overwrites superclass method.
        """
        return self.measmat

    def get_meascovar(self, *args):
        """
        Returns covariance matrix of the measurement map.
        Overwrites superclass method.
        """
        return self.meascovar

    def precompute_matrices(self, *args):
        """
        Precomputes all four matrices A, Q, H, R.
        Not required for instances of this class but for instances of subclasses.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError


class IBM(InvariantSSM):
    """
    Discretised state space model with integrated Brownian motion (IBM)
    prior as dynamic model of order q,
        X_0 ~ N(m_0, C_0)
        X_(t+1) = A(h) X_t + q,     q ~ N(0, Q(h)),
        Y_(t+1) = H X_(t+1)
    where h is the stepsize of the discretisation. Observations are generated
    by projecting onto the first (not zeroth) coordinate, which---in the context
    of an ODE solver---is usually to be understood as a projection onto
    the first derivative. This is unrelated to the IBM dynamics.

    Note:
        We term "Brownian motion" instead of "Wiener process" to avoid
        confusion with initial value problems as
        they share similar abbreviations.

    Todo:
        Encapsulate the measurement model into a different class,
        while maintaining same usage. IBM is only A(h) and q,
        the rest is a measurement model and should be able to
        be varied.
    """

    def __init__(self, q, dim, diffconst=1.0):
        """
        Initialise IBM(q) model. Here, n = q+1 and l = 1.

        Args:
            q: order of integration
            dim: spatial dimension of---here---the ode
            diffconst: amplitude of Brownian motion prior; known as 'sigma**2'
        Computes:
            init_mean: zero mean in shape (d(q+1),),
            init_covar: identity covariance in shape (d(q+1), d(q+1))
        """
        self.q = q
        self.dim = dim
        self.diffconst = diffconst
        self.init_mean = self._compute_init_mean()
        self.init_covar = self._compute_init_covar()
        self.measmat = self._compute_measmat()
        self.meascovar = self._compute_meascovar()

    def __repr__(self):
        """
        Represent IBM object as e.g. IBM(q=2, dim=1).
        """
        return "SSM(q=%r, dim=%r)" % (self.q, self.dim)

    def _compute_init_mean(self):
        """
        Creates initial mean array m_0 for state space model. 

        Returns:
            init_mean: shape (d(q+1),)
        """
        init_mean = np.zeros(self.dim * (self.q + 1))
        return init_mean

    def _compute_init_covar(self):
        """
        Creates initial covariance matrix C_0 for state space model.

        Returns:
            init_covar: initial covariance, shape (d(q+1), d(q+1))
        """
        init_covar_1d = np.eye(self.q + 1)
        init_covar = self._kronecker(init_covar_1d)
        return init_covar

    def _compute_measmat(self):
        """
        Creates H_1.
        Note: the projection matrix H_0 is an attribute of the odesolver,
        not of the state space!

        Returns:
            h1: Matrix which projects onto first coordinate (count from zero),
                shape (1, n)
        """
        h1_1d = np.eye(self.q + 1)[:, 1].reshape((1, self.q + 1))
        h1 = self._kronecker(h1_1d)
        return h1

    def _compute_meascovar(self):
        """
        Return R = 0.
        For use in odesolver module, R is determined by model evaluation.

        Returns:
            R: measurement variance equal to zero, shape (d, d)
        """
        meascovar_1d = np.zeros((1, 1))
        meascovar = self._kronecker(meascovar_1d)
        return meascovar

    def discretise(self, stepsize):
        """
        Discretises IBM prior of state space model with given stepsize.
        That is, the transition matrix (A) and its covariance (Q) are computed,
            h -> A(h)
            h -> Q(h)
        Args:
            stepsize: stepsize of IBM discretisation, often referred to as 'h'
        """
        self.transmat = self._compute_transmat(stepsize)
        self.transcovar = self._compute_transcovar(stepsize)

    def _compute_transmat(self, h):
        """
        Computes state transition matrix A(h).

        Args:
            h: stepsize
        Returns:
            ah: transition matrix A(h), shape (d(q+1), d(q=1))
        """
        ah_1d = np.diag(np.ones(self.q + 1), 0)
        for i in range(self.q):
            offdiagonal = h ** (i + 1) / np.math.factorial(i + 1) * np.ones(self.q - i)
            ah_1d += np.diag(offdiagonal, i + 1)
        ah = self._kronecker(ah_1d)
        return ah

    def _compute_transcovar(self, h):
        """
        Computes Q(h).

        Args:
            h: stepsize
        Returns:
            qh: covariance Q(h) of transition, shape (d(q+1), d(q=1))
        """
        qh_1d = np.zeros((self.q + 1, self.q + 1))
        for i in range(self.q + 1):
            for j in range(self.q + 1):
                idx = 2 * self.q + 1 - i - j
                nominator = self.diffconst ** 2 * h ** idx
                denominator = (
                    idx * np.math.factorial(self.q - i) * np.math.factorial(self.q - j)
                )
                qh_1d[i, j] = nominator / denominator
        qh = self._kronecker(qh_1d)
        return qh

    def _kronecker(self, mtrx):
        """
        Computes I_d x M,
        i.e. creates a blockdiagonal matrix with d copies of M for each block

        Args:
            mtrx: np.ndarray of shape (m, m)
        Returns:
            kronmtrx: blockdiagonal matrix, each diagonal block is mtrx; shape (md, md)
        """
        eye_d = np.eye(self.dim)
        kronmtrx = np.kron(eye_d, mtrx)
        return kronmtrx
