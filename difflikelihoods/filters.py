# coding=utf-8
"""
filters.py

Reference:
    Chapter 4 of https://users.aalto.fi/~ssarkka/pub/cu
    p_book_online_20131111.pdf
Note:
    We named this module filters (with an s) instead of
    filter to avoid collision with the existing filter()
    function that turns lists and a condition in to a truth array.
    Please keep this in mind while importing this module.

Example: Filtering the samples of a random walk
    >>> import statespace as ssm
    >>> import filters
    >>> import numpy as np
    >>> rw = ssm.InvariantSSM(transmat=np.eye(2), transcovar=0.1*np.eye(2),
                              measmat=np.eye(2), meascovar=0.1*np.eye(2),
                              init_mean=np.zeros(2), init_covar=0.1*np.eye(2))
    >>> dataset = rw.sample_trajectory(nsteps=25)
    >>> kfilt = filters.KalmanFilter(rw)
    >>> m, c = kfilt.filter(dataset)
    >>> print(m, c)
"""
import numpy as np
from difflikelihoods import auxiliary as aux
from abc import ABC, abstractmethod

class Filter(ABC):
    """
    Filter class. Intended to be used as superclass for
    all kinds of filtering algorithms.
    This includes particle filters, Kalman filters and the like. 
    What unifies these filters is that they
        a) depend on a state space model (linear/nonlinear)
        b) can do "filter()", "initialise()", "update()" and "predict()"
    Methods of this class are intended to be overwritten by subclasses.
    """
    def __init__(self, statespacemodel, *args):
        """
        Initialises a filter with a state space model.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def filter(self, dataset, *args):
        """
        Filters 'true' states from a dataset using the classic
            initialise->update->predict->update->predict->...
        scheme.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def initialise(self, *args):
        """
        Initialises filtering iteration.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args):
        """
        Predicts values at the next timestep.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, *args):
        """
        Computes update step of filtering algorithm.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError


class KalmanFilter(Filter):
    """
    Kalman filter class. Refer to Theorem 4.2 in book above.
    The methods initialise(), update() and
    predict() overwrite the superclass' methods.
    """
    def __init__(self, ssm):
        """
        Initialise a Kalman filter with a state space model.

        Args:
            ssm:    state space model object, see statespace.py
        """
        self.ssm = ssm

    def __repr__(self):
        """
        Returns e.g.
            >>> print(kfilt)
            KalmanFilter(ssm=IBM(q=2, dim=1))
        """
        return "KalmanFilter(ssm=%r)" % self.ssm

    def filter(self, dataset):
        """
        Apply filter to a dataset which is based on the statespacemodel.

        Args:
            dataset:    dataset to be filtered, must correspond to self.ssm; 
                        shape (M, l) with M observations of dimension (l, 1) each.
        Returns:
            means:      means of estimated trajectory,
                        shape (M, n), i.e. with M states of shape (n,)
            covars:     covariances of estimated trajectory, shape (M, n, n)
        """
        means, covars = self._init_arrays(len(dataset), len(self.ssm.init_mean))
        means[0], covars[0] = self.initialise()
        for i in range(1, len(dataset)):
            pmean, pcovar = self.predict(means[i-1], covars[i-1])
            means[i], covars[i] = self.update(pmean, pcovar, dataset[i])
        return means, covars

    def _init_arrays(self, nrows, ncols):
        """
        Allocate storage for means and covariances.
        """
        means = np.zeros((nrows, ncols))
        covars = np.zeros((nrows, ncols, ncols))
        return means, covars

    def initialise(self):
        """
        Initialise Kalman filter return value at zero
        with mean aka mode aka most likely value.
        """
        init_mean = self.ssm.init_mean
        init_covar = self.ssm.init_covar
        return init_mean, init_covar

    def predict(self, mean, covar, *optional):
        """
        Perform the prediction step of Kalman filtering,
            m -> A m
            C -> A C A^T + Q.
        Args:
            mean:       current mean estimate of the states, shape (n,)
            covar:      current covariance estimate, shape (n, n)
            *optional:  optional arguments to be passed to
                        state space model methods, e.g. stepsize
        Returns:
            mpred:      mean prediction, shape (n,)
            cpred:      covariance prediction, shape (n, n)
        """
        transmat = self.ssm.get_transmat(*optional)
        transcovar = self.ssm.get_transcovar(*optional)
        mpred = transmat.dot(mean)
        cpred = transmat.dot(covar).dot(transmat.T) + transcovar
        return mpred, cpred

    def update(self, mpred, cpred, data, uncert=None, *args):
        """
        Perform the update step of Kalman filtering,
            m = m_p + K (y - H m_p),
            C = (I - K H) C_p,
        where 
            K_t = C_p H^T (H C_p H^T + R)^{-1}
        is the Kalman gain and m_p and C_p are proposed mean and covariance.

        Args:
            mpred:  current mean prediction, shape (n,)
            cpred:  current covariance prediction, shape (n, n)
            data:   current value of y, shape (n, 1)
            uncert: current measurement uncertainty ('R'), shape (m, m).
                    If default, the standard matrix is used.
            *args:  optional arguments to be passed to
                    the state space model methods, e.g. stepsize
        Returns:
            mean:   current mean estimate of the states, shape (n,)
            covar:  current covariance estimate, shape (n, n)
        """
        meascovar = self._set_meascovar(uncert, *args)
        measmat = self.ssm.get_measmat(*args)
        res = self.compute_residual(data, measmat, mpred)
        kgain = self.compute_kalmangain(cpred, measmat, meascovar)
        mean, covar = self.compute_correction(mpred, cpred,
                                              res, measmat, kgain)
        return mean, covar

    def _set_meascovar(self, uncert, *optional):
        """
        Reacts to whether measurement covariance input is custom or default.

        Args:
            uncert:     measurement covariance; known as 'R'.
                        If 'default', measurement covariance is read from
                        state space model, if not 'default', input is used.
            *optional:  optional arguments to be passed to
                        the state space model methods, e.g. stepsize
        """
        if uncert is None:
            meascovar = self.ssm.get_meascovar(*optional)
        else:
            meascovar = uncert
        return meascovar

    def compute_residual(self, data, measmat, mpred):
        """
        Computes the residual of the current filter estimate,
            res = y - H m_p.

        Args:
            data:       current data value, shape (l,),
                        often referred to as 'y'
            measmat:    measurement matrix, shape (l, n)
            mpred:      current mean prediction, shape (n,)
        Returns:
            res:        residual, shape (l,)
        """
        res = data - measmat.dot(mpred)
        return res

    def compute_kalmangain(self, cpred, measmat, meascovar):
        """
        Computes the Kalman gain,
            K = C_p H^T (H C_p H^T + R)^{-1}.

        Args:
            cpred:      current covariance prediction, shape (n, n)
            measmat:    measurement matrix, shape (l, n)
            meascovar:  measurement variance, shape (l, l), known as 'R'
        Returns:
            kgain:      Kalman gain, shape (n, l)
        """
        s = measmat.dot(cpred).dot(measmat.T) + meascovar
        s_inv = np.linalg.inv(s)
        kgain = cpred.dot(measmat.T).dot(s_inv)
        return kgain

    def compute_correction(self, mpred, cpred, res, measmat, kgain):
        """
        Computes update of mean and covariance using the Kalman gain,
            m = m_prop + K res
            C = (I - K H) C_p

        Args:
            mpred:      current mean prediction, shape (n,)
            cpred:      current covariance prediction, shape (n, n)
            res:        'error' of current prediction, shape (l,)
            measmat:    measurement matrix, shape (l, n)
            kgain:      Kalman gain matrix, shape (n, l)
        Returns:
            mean:       new mean estimate, shape (n,)
            covar:      new covariance estiamte, shape (n, n)
        """
        n = len(mpred)
        updatemat = np.eye(n) - kgain.dot(measmat)
        mean = mpred + kgain.dot(res)
        covar = updatemat.dot(cpred)
        return mean, covar

# END OF FILE
