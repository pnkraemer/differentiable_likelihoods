# coding=utf-8
"""
ode.py

ODE class and some common initial value problems:
    * Linear ODEs in 1-d and n-d
    * Logistic ODE
    * FitzHugh-Nagumo model
    * Restricted Two-Body problem
as well as a custom ODE class. The standard implementations
are suitable for vectorised evaluation of the rhs functions,
which is a necessary performance criterion for, e.g., quadrature.

Note:
    We use the name 'ode' instead of 'ivp' to avoid confusion with
    the state space model 'integrated Wiener process',
    resp. 'integrated Brownian motion' and the
    abbreviation I(nverse)P(roblem).

Reference:
    Hairer, E., Norsett, S.P., Wanner, G. Solving Ordinary
    Differential Equations I: Nonstiff Problems, Springer, 2009

Example:
    >>> lin_ode = ode.LinearODE(t0=0.1, tmax=2.0, params=1.0, y0=0.9)
    >>> print(lin_ode)
"""

from abc import ABC, abstractmethod
import numpy as np


class ODE(ABC):
    """
    Basic ODE class. Standalone, this is meaningless
    as it does not contain a model function.
    It rather is a collection of attributes that all
    initial value problems share: t0, tmax, y0, y0_unc.

    For usage, refer to CustomODE() or to the specialised
    ODE classes such as LinearODE().
    """

    def __init__(self, t0, tmax, initval, initval_unc=0.0):
        """
        Initialises basic ODE attributes.

        Args:
            t0:             initial time, scalar.
            tmax:           final time, scalar.
            initval:        initial value (y0), scalar or np.ndarray
            initval_unc:    uncertainty of initial value,
                            scalar or np.ndarray
        """
        self.t0 = t0
        self.tmax = tmax
        self.initval = initval
        self.initval_unc = initval_unc

    def __repr__(self):
        """
        Represent ODE object and its parameters, e.g.:
            ODE(t0=0.0, tmax=1.0, initval=2.0, initval_unc=0.0)
        """
        return "ODE(t0=%r, tmax=%r, initval=%r, initval_unc=%r)" % (
            self.t0,
            self.tmax,
            self.initval,
            self.initval_unc,
        )

    @abstractmethod
    def modeval(self, t, x):
        """
        Evaluates model function f.
        Intended to be overwritten by subclasses.

        Args:
            t:  time step, scalar, optional.
            x:  space variable, scalar or array.

        Example:
            >>> some_ode.modeval(x=1)
            >>> some_ode.modeval(t=0, x=1)

        Note:
            CustomODE object allows custom arguments with a
            Warning issued if it deviates from the (t, x) standard.
        """
        raise NotImplementedError("Model evaluation is not implemented")


class CustomODE(ODE):
    """
    ODE with custom model evaluation function.
    See e.g. ./examples/custom_ode.py for help
    with usage.

    Example:
        >>> def rhs(t, x): return x * (1-x)
        >>> custode = ode.CustomODE(t0=0., tmax=1., modeval=rhs, initval=0.)
        >>> print(custode)
    """

    def __init__(self, t0, tmax, modeval, initval, initval_unc=0.0):
        """
        Initialises basic ODE attributes.

        Args:
            t0:             initial time, scalar.
            tmax:           final time, scalar.
            modeval:        model evaluation function, callable
            initval:        initial value (y0), scalar or np.ndarray
            initval_unc:    uncertainty of initial value,
                            scalar or np.ndarray
        """
        if isinstance(initval, list) is True:  # hotfix
            initval = np.array(initval)
        check_compliance_with_odesolver(modeval, t0, initval)
        self.custom_modeval = modeval
        ODE.__init__(self, t0, tmax, initval, initval_unc)

    def __repr__(self):
        """
        Represent CustomODE object and its parameters.

        Example:
            >>> def rhs(t, x): return x * (1-x)
            >>> custode = ode.CustomODE(t0=0., tmax=1., modeval=rhs, initval=0.)
            >>> print(custode)
            CustomODE(t0=0.0, tmax=1.0,
                      modeval=<function rhs at 0x7f5e4638cf28>,
                      initval=0.0, initval_unc=0.0)
        """
        return (
            "CustomODE(t0=%r, tmax=%r, modeval=%r, initval=%r,\
initval_unc=%r)"
            % (self.t0, self.tmax, self.custom_modeval, self.initval, self.initval_unc)
        )

    def modeval(self, t, x):
        """
        Evaluates custom model function f.
        By using this method, we make sure that the (t, x) input
        format is ensured.

        Args:
            *inputs:    inputs are directed to custom function without
                        any checks. If not in (t, x) shape, a warning
                        has been raised at initialisation already.

        Example:
            >>> def rhs(t, x): return x * (1-x)
            >>> custode = ode.CustomODE(t0=0., tmax=1.,
                                        modeval=rhs, initval=0.)
            >>> custode.modeval(0.2, 4.0)
            -12.0
            >>> custode.modeval(0.2, np.array([0.1, 4.0]))
            array([0.09, -12.])
        """
        return self.custom_modeval(t, x)


def check_compliance_with_odesolver(modeval, t0, initval):
    """
    Checks if modeval function can be evaluated
    as modeval(t, x). If not, a TypeError is raised.
    """
    try:
        modeval(t0, initval)
    except TypeError:
        raise TypeError(
            "%r cannot handle (t, x) inputs \
and is thus incompatible with ODESolver.solve()"
            % modeval
        )


class LinearODE(ODE):
    """
    Linear ODE
        x'(t) = p*x(t), x(0) = x0

    Example:
        >>> linode = ode.LinearODE(t0=0., tmax=1., params=0.24, initval=0.)
        >>> print(linode)
    """

    def __init__(self, t0, tmax, params, initval, initval_unc=0.0):
        """
        Initialises basic ODE attributes.

        Args:
            t0:             initial time, scalar.
            tmax:           final time, scalar.
            params:         parameter p, scalar or (n, n) shaped np.ndarray 
            initval:        initial value (y0), scalar or (n,) shaped 
                            np.ndarray
            initval_unc:    uncertainty of initial value,
                            scalar or np.ndarray
        Raises:
            TypeError:      if params is not a scalar
            TypeError:      if initval is not a scalar
        """
        if np.isscalar(params) is False:
            raise TypeError("Please enter a scalar parameter")
        if np.isscalar(initval) is False:
            raise TypeError("Please enter a scalar initial value")
        self.params = params
        ODE.__init__(self, t0, tmax, initval, initval_unc)

    def __repr__(self):
        """
        Represent LinearODE object and its parameters.

        Example:
            >>> linode = ode.LinearODE(t0=0., tmax=1., params=0.24, initval=0.)
            >>> print(linode)
            LinearODE(t0=0.0, tmax=1.0, params=1.0,
                      initval=2.0, initval_unc=0.0)
        """
        return (
            "LinearODE(t0=%r, tmax=%r, params=%r, initval=%r,\
initval_unc=%r)"
            % (self.t0, self.tmax, self.params, self.initval, self.initval_unc)
        )

    def modeval(self, t, x):
        """
        Computes f(t, x) = self.params * x.

        Args:
            t:  time-coordinate, ignored.
                Only here for consistency with ODE class.
            x:  spatial coordinate x either scalar or array,
                the latter implies vectorised output
        Returns:
            Evaluation of f(t, x) = self.params * x.

        Example:
            >>> linode = ode.LinearODE(t0=0., tmax=1., params=0.24, initval=0.)
            >>> linode.modeval(0.1, 0.2)
            0.048
            >>> linode.modeval(0.1, np.array(0.2, 0.5))
            array([0.048, 0.12 ])
        """
        return np.dot(x, self.params)  # weird order for vectorised eval.


class MatrixLinearODE(ODE):
    """
    Matrix-valued linear ODE
        x'(t) = Ax(t), x(0) = x0,
    where A is a (d, d) matrix and x0 is a (d,) array.

    Example:
        >>> params = np.eye(2)
        >>> initval = np.zeros(2)
        >>> matode = ode.MatrixLinearODE(t0=0., tmax=1.,
                                         params=params, initval=initval)
    """

    def __init__(self, t0, tmax, params, initval, initval_unc=0.0):
        """
        Initialises basic ODE attributes.

        Args:
            t0:             initial time, scalar.
            tmax:           final time, scalar.
            params:         parameter p, np.ndarray
            initval:        initial value (y0), scalar or np.ndarray
            initval_unc:    uncertainty of initial value,
                            scalar or np.ndarray
        Raises:
            TypeError:      if params is not an ndarray of shape (d, d)
            TypeError:      if initval is not an ndarray of shape (d,)
        """
        if isinstance(params, np.ndarray) is False or len(params.shape) != 2:
            raise TypeError("Please enter a (d, d) shaped  parameter")
        if isinstance(initval, np.ndarray) is False or len(initval.shape) != 1:
            raise TypeError("Please enter a (d,) shaped initial value")
        self.params = params
        ODE.__init__(self, t0, tmax, initval, initval_unc)

    def __repr__(self):
        """
        Represent MatrixLinearODE object and its parameters.

        Example:
            >>> params = np.eye(2)
            >>> initval = np.zeros(2)
            >>> matode = ode.MatrixLinearODE(t0=0., tmax=1.,
                                             params=params, initval=initval)
            >>> print(matode)
            MatrixLinearODE(t0=0.0, tmax=1.0,
                            params=array([[1., 0.], [0., 1.]]),
                            initval=array([0., 0.]), initval_unc=0.0)
        """
        return (
            "MatrixLinearODE(t0=%r, tmax=%r, params=%r, initval=%r,\
initval_unc=%r)"
            % (self.t0, self.tmax, self.params, self.initval, self.initval_unc)
        )

    def modeval(self, t, x):
        """
        Computes f(t, x) = self.params @ x.

        Args:
            *t:  time-coordinate, ignored.
                Only here for consistency with ODE class.
            x:  spatial coordinate x either scalar or array,
                the latter implies vectorised output
        Returns:
            Evaluation of f(t, x) = self.params * x.

        Example:
            >>> linode = ode.LinearODE(t0=0., tmax=1., params=0.24, initval=0.)
            >>> linode.modeval(0.1, 0.2)
            0.048
            >>> linode.modeval(0.1, np.array(0.2, 0.5))
            array([0.048, 0.12 ])
        """
        return np.dot(x, self.params)  # weird order for vectorised eval.


class FitzHughNagumo(ODE):
    """
    FitzHugh-Nagumo model ODE. A 2d system of ODEs,
        x'(t) = x(t) - x(t)**3/3 - y(t) + I.
        y'(t) = 1/c * (x(t) + a - b*y(t))

    Reference:
        https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model

    Example:
        >>> params = [0., 1., 2., 3.]
        >>> initval = np.array([1., 1.])
        >>> fhn = ode.FitzHughNagumo(t0=0., tmax=1.,
                                     params=params, initval=initval)
    """

    def __init__(self, t0, tmax, params, initval, initval_unc=0.0):
        """
        Initialises basic ODE attributes.

        Args:
            t0:             initial time, scalar.
            tmax:           final time, scalar.
            params:         parameters [c0, c1, c2, c3], list or np.ndarray
            initval:        initial value [y0_1, y0_2], list or np.ndarray
            initval_unc:    uncertainty of initial value,
                            scalar or np.ndarray
        Raises:
            AssertionError: if number of parameters is not 3.
        """
        if len(params) != 4:
            raise TypeError("Please enter 4 parameters [c0, c1, c2, c3]")
        if len(initval) != 2:
            raise TypeError("Please enter a 2d initial value")
        self.params = np.array(params)
        initval = np.array(initval)
        ODE.__init__(self, t0, tmax, initval, initval_unc)

    def __repr__(self):
        """
        Represent ODE object and its parameters.

        Example:
            >>> params = [0., 1., 2., 3.]
            >>> initval = np.array([1., 1.])
            >>> fhn = ode.FitzHughNagumo(t0=0., tmax=1.,
                                         params=params, initval=initval)
            >>> print(fhn)
            FitzHughNagumo(t0=0.0, tmax=1.0, params=array([1., 2., 3.]),
                           initval=array([1., 1.]), initval_unc=0.0)
        """
        return (
            "FitzHughNagumo(t0=%r, tmax=%r, params=%r, initval=%r,\
initval_unc=%r)"
            % (self.t0, self.tmax, self.params, self.initval, self.initval_unc)
        )

    def modeval(self, t, x):
        """
        Computes f(t, x) = array([x1 - x1**3/3 - x2 + c_0,
                                  1/c_3 * (x1 + c_1 - c_2 * x2)]).

        Args:
            t:  time-coordinate, ignored.
                Only here for consistency with ODE class.
            x:  spatial coordinate x either scalar or array,
                the latter implies vectorised output
        Returns:
            Evaluation of f(t, x).

        Example:
            >>> params = [0., 1., 2., 3.]
            >>> initval = np.array([1., 1.])
            >>> fhn = ode.FitzHughNagumo(t0=0., tmax=1.,
                                         params=params, initval=initval)
            >>> fhn.modeval(.1, np.array([1, 1]))
            array([5., -0.66666667])
            >>> fhn.modeval(.1, np.ones((3, 2)))
            array([[5., -0.66666667],
                   [5., -0.66666667],
                   [5., -0.66666667]])
        """
        if len(x.shape) == 2:  # potentially vectorised evaluation
            x = x.T
        par0, par1, par2, par3 = self.params
        return np.array(
            [
                x[0] - x[0] ** 3.0 / 3.0 - x[1] + par0,
                1.0 / par3 * (x[0] + par1 - par2 * x[1]),
            ]
        ).T


class Res2Bod(ODE):
    """
    Restricted Two-Body Problem.

    Be careful with changing initial values and parameters,
    as only with the given choices it is guaranteed to be
    have a periodic orbit.

    Example:
        >>> r2b = ode.Res2Bod(t0=0, tmax=2)
        >>> print(r2b)

    Reference:
        p.129f. in Hairer/Norsett/Wanner book; see module docstring.
    """

    def __init__(self, t0, tmax):
        """
        Initialises ODE with t0 and tmax only.
        The other parameters are to sensitive
        (any deviation destroys periodicity),
        hence we do it automatically.

        Args:
            t0:     initial time, scalar.
            tmax:   final time, scalar.
        """
        initval = np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224])
        initval_unc = np.zeros(initval.shape)
        ODE.__init__(self, t0, tmax, np.array(initval), initval_unc)
        self.params = 0.012277471

    def __repr__(self):
        """
        Represent ODE object and its parameters.

        Example:
            >>> r2b = ode.Res2Bod(t0=0, tmax=2)
            >>> print(r2b)
            Res2Bod(t0=0, tmax=2, params=0.012277471,
                    initval=array([0.994, 0., 0., -2.00158511]),
                    initval_unc=array([0., 0., 0., 0.]))
        """
        return (
            "Res2Bod(t0=%r, tmax=%r, params=%r, initval=%r,\
initval_unc=%r)"
            % (self.t0, self.tmax, self.params, self.initval, self.initval_unc)
        )

    def modeval(self, t, x):
        """
        Evaluates RHS of restricted two-body problem.

        Example:
            >>> r2b.modeval(0., np.ones(4))
            array([1., 1., 2.65279959, -1.35511519])
            >>> r2b.modeval(0., np.ones((2, 4)))
            array([[1., 1., 2.65279959, -1.35511519],
                   [1., 1., 2.65279959, -1.35511519]])
        """
        if len(x.shape) == 2:  # potentially vectorised evaluation
            x = x.T
        coord3 = self.modeval_f1(x)
        coord4 = self.modeval_f2(x)
        return np.array([x[2], x[3], coord3, coord4]).T

    def modeval_f1(self, xarr):
        """
        Computes first coordinate of RHS function.
        """
        mu, mup = self.params, 1 - self.params
        const1, const2 = self.get_normalising_const(xarr)
        return (
            xarr[0]
            + 2 * xarr[3]
            - mup * (xarr[0] + mu) / const1
            - mu * (xarr[0] - mup) / const2
        )

    def modeval_f2(self, xarr):
        """
        Computes second coordinate of RHS function.
        """
        mu, mup = self.params, 1 - self.params
        const1, const2 = self.get_normalising_const(xarr)
        return xarr[1] - 2 * xarr[2] - mup * xarr[1] / const1 - mu * xarr[1] / const2

    def get_normalising_const(self, xarr):
        """
        Computes normalising constants D1 and D2 (see book).
        """
        const1 = self.get_d1(xarr)
        const2 = self.get_d2(xarr)
        return const1, const2

    def get_d1(self, xarr):
        """
        Returns normalising constant d1 =((x1 + mu)**2 + x2**2)**3/2
        """
        mu = self.params
        return ((xarr[0] + mu) ** 2 + xarr[1] ** 2) ** (1.5)

    def get_d2(self, xarr):
        """
        Returns normalising constant d2 = ((x1 + mu')**2 + x2**2)**3/2
        """
        mup = 1 - self.params
        return ((xarr[0] - mup) ** 2 + xarr[1] ** 2) ** (1.5)


# END OF FILE
