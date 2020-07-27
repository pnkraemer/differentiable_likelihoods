# coding=utf-8
"""
linearised_ode.py

Implementation of a class of ODEs, where the right-hand side function
depends linearly on the parameters,
    math::
        \\displaymath f(t, x) = \\sum_{i=1} p_{i} f_i(t, x).

Further description of this class of ODEs:
    Gorbach, Bauer, Buhmann:
    "Scalable Variational Inference for Dynamical Systems", 2017
"""

from abc import ABC, abstractmethod
import numpy as np
from difflikelihoods import ode


class LinearisedODE(ode.ODE, ABC):
    """
    ODE class with a linearised right-hand side function.
    That is, the right-hand side function is linear with respect to
    the parameter,
        math::
            \\displaymath \\sum_{i=1} p_{i} f_i(t, x).
    The interface is identical to the ode.ODE class. Here, we have the
    additional method modeval_parts(), which evaluates each f_i
    separately.

    Attributes:
        is_linearised (True):   Indicates whether an ODE instance is a
                                LinearisedODE instance.
    """

    is_linearised = True

    def __init__(self, t0, tmax, params, initval, initval_unc=0.0):
        """
        Initialises linearised ODE.

        Args:
            t0:             initial time, scalar.
            tmax:           final time, scalar.
            params:         parameter p, scalar or np.ndarray
            initval:        initial value (y0), scalar or np.ndarray
            initval_unc:    uncertainty of initial value,
                            scalar or np.ndarray
        """
        initval = initval * np.ones(1)
        ode.ODE.__init__(self, t0, tmax, initval, initval_unc)
        self.params = params

    def __repr__(self):
        """
        Represent LinearisedODE object and its parameters, e.g.:
            LinearisedODE(t0=0.0, tmax=1.0, params=1.0,
                          initval=2.0, initval_unc=0.0)
        """
        return (
            "LinearisedODE(t0=%r, tmax=%r, params=%r, initval=%r, \
initval_unc=%r)"
            % (self.t0, self.tmax, self.params, self.initval, self.initval_unc)
        )

    def modeval(self, t, x):
        """
        Returns (rhs_parts.T @ self.params).T instead of
        self.params @ rhs_parts in order to be able to multiply
        a (2,) array with a (2, 3, 4) array to obtain a (3, 4) array
        which is important for vectorised evaluations.
        """
        rhs_parts = self.modeval_parts(t, x)
        return self.rhs_parts_into_eval(rhs_parts)

    def rhs_parts_into_eval(self, rhs_parts):
        """
        Computes inner product of params and rhs_parts.
        """
        return np.dot(rhs_parts.T, self.params).T

    @abstractmethod
    def modeval_parts(self, t, x):
        """
        Returns array [f_1(t, x), ..., f_N(t, x)].
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError


class LinearODE(LinearisedODE):
    """
    Linear ODE
        x'(t) = p x(t), x(0) = x0
    in the LinearisedODE formulation.
    Interface intentionally similar to the one of ode.LinearODE.
    """

    def __init__(self, t0, tmax, params, initval, initval_unc=0.0):
        """
        Initialises LinearODE object.

        Args:
            t0:             initial time, scalar.
            tmax:           final time, scalar.
            params:         parameter p, scalar or np.ndarray
            initval:        initial value (y0), scalar or np.ndarray
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
        params_array = params * np.ones(1)
        LinearisedODE.__init__(self, t0, tmax, params_array, initval, initval_unc)
        self.params = params

    def __repr__(self):
        """
        Represent LinearODE object and its parameters, e.g.:
            LinearODE(t0=0.0, tmax=1.0, params=1.0,
                      initval=2.0, initval_unc=0.0)
        """
        return (
            "LinearODE(t0=%r, tmax=%r, params=%r, initval=%r, \
initval_unc=%r) in linearised form"
            % (self.t0, self.tmax, self.params, self.initval, self.initval_unc)
        )

    def modeval_parts(self, t, x):
        """
        Returns f(t, x) = x, where x is an array.
        """
        return np.array(x)


class LogisticODE(LinearisedODE):
    """
    Sigmoid logistig growth function,
        x'(t) = r * (x(t) * (1 - x(t)/k))
    """

    def __init__(self, t0, tmax, params, initval, initval_unc=0.0):
        """
        Initialises basic ODE attributes.

        Args:
            t0:             initial time, scalar.
            tmax:           final time, scalar.
            params:         parameters [r, r/k], list or array.
                            Already in necessary form for linearisation.
            initval:        initial value (y0), scalar or np.ndarray
            initval_unc:    uncertainty of initial value,
                            scalar or np.ndarray
        Raises:
            AssertionError: if number of parameters is not 2.
        """
        if len(params) != 2:
            raise TypeError("Please enter two parameters [r, k]")
        if params[0] <= 0 or params[1] <= 0:
            raise TypeError("Please parameters strictly greater than zero.")
        self.params = params
        LinearisedODE.__init__(self, t0, tmax, self.params, initval, initval_unc)

    def __repr__(self):
        """
        Represent ODE object and its parameters.
            SigmoidLogGrowth(t0=0.0, tmax=1.0, params=array([1., 2.]),
                             initval=2.0, initval_unc=0.0)
        """
        return (
            "SigmoidLogGrowth(t0=%r, tmax=%r, params=%r, initval=%r, \
initval_unc=%r) in linearised form"
            % (self.t0, self.tmax, self.params, self.initval, self.initval_unc)
        )

    def modeval_parts(self, t, x):
        """
        Return f(t, x) = r * (x * (1 - x/k))
                       = r*x - (r/k)*x**2

        Args:
            t:  time-coordinate, ignored. Only here for consistency reasons.
            x:  spatial coordinate x
        """
        f_1 = self.modeval_part_1(t, x)
        f_2 = self.modeval_part_2(t, x)
        return np.array([f_1, f_2])

    def modeval_part_1(self, t, x):
        """
        Returns f_1(x) = x.
        """
        return x

    def modeval_part_2(self, t, x):
        """
        Returns f_2(x) = -x**2.
        """
        return -(x ** 2)


class LotkaVolterra(LinearisedODE):
    """
    Lotka-Volterra equations with right-hand side
        f(t, x) = (p_1*x_1(t) - p2*x_1(t)*x_2(t),
                   p_3*x_1(t)*x_2(t) - p_4*x_2(t))
    which allows linearised evaluation.

    Attributes:
        x_1:    number of prey (rabbits)
        x_2:    number of predators (foxes)
        p:      [p_1, ..., p_4] interaction between species

    Reference:
        https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
    """

    def __init__(self, t0, tmax, params, initval, initval_unc=0.0):
        if len(params) != 4:
            raise TypeError("Please enter 4 parameters [p_1, p_2, p_3, p_4]")
        if len(initval) != 2:
            raise TypeError("Please enter a 2d initial value")
        LinearisedODE.__init__(self, t0, tmax, params, initval, initval_unc)

    def __repr__(self):
        """
        Represent LotkaVolterra object and its parameters, e.g.:
            LotkaVolterra(t0=0.0, tmax=1.0, params=[1.2, 2.1, 3.4, 4.3],
                          initval=[1.0, 2.0], initval_unc=0.0)
        """
        return (
            "LotkaVolterra(t0=%r, tmax=%r, params=%r, initval=%r, \
initval_unc=%r) in linearised form"
            % (self.t0, self.tmax, self.params, self.initval, self.initval_unc)
        )

    def modeval_parts(self, t, x):
        """
        Evaluates linearised version of Lotka-Volterra.
        f(t, x) = p_1 f_1(t, x) + ... + p_4 f_4(t, x)
        """
        f_1 = self.modeval_part_1(t, x)
        f_2 = self.modeval_part_2(t, x)
        f_3 = self.modeval_part_3(t, x)
        f_4 = self.modeval_part_4(t, x)
        return np.array([f_1, f_2, f_3, f_4])

    def modeval_part_1(self, t, x):
        """
        Returns f_1(t, x) = array(x_1, 0).
        """
        out = np.zeros(x.shape)
        out[0] = x[0]
        return out

    def modeval_part_2(self, t, x):
        """
        Returns f_2(t, x) = array(-x_1*x_2, 0).
        """
        out = np.zeros(x.shape)
        out[0] = -x[0] * x[1]
        return out

    def modeval_part_3(self, t, x):
        """
        Returns f_3(t, x) = array(0, x_1*x_2).
        """
        out = np.zeros(x.shape)
        out[1] = x[0] * x[1]
        return out

    def modeval_part_4(self, t, x):
        """
        Returns f_4(t, x) = array(0, -x_2).
        """
        out = np.zeros(x.shape)
        out[1] = -x[1]
        return out


class FitzHughNagumoPsi3(LinearisedODE):
    """
    FitzHugh-Nagumo model ODE. A 2d system of ODEs,
        x'(t) = x(t) - x(t)**3/3 - y(t)
        y'(t) = 1/c * (x(t) + a - b*y(t))

    Reference:
        https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model

    Example:
        >>> params = [01., 2.]
        >>> initval = np.array([1., 1.])
    Nota bene: We assume the parameter c = Psi to be fixed at 3.
    """

    def __init__(self, t0, tmax, params, initval, initval_unc=0.0):
        if len(params) != 2:
            raise TypeError("Please enter 1 parameters [a, b]")
        if len(initval) != 2:
            raise TypeError("Please enter a 2d initial value")
        LinearisedODE.__init__(self, t0, tmax, params, initval, initval_unc)
        self.params = [1.0] + params  # nb: the first parameter always has to be 1.

    def modeval_parts(self, t, x):
        """
        """
        f_1 = self.modeval_part_1(t, x)
        f_2 = self.modeval_part_2(t, x)
        f_3 = self.modeval_part_3(t, x)
        return np.array([f_1, f_2, f_3])

    def modeval_part_1(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        x_1, x_2 = x
        # out[0] = 3*x_1 - x_1**3 - 3*x_2   # this was the previous line
        out[0] = x_1 - x_1 ** 3 / 3.0 - x_2
        out[1] = x_1 / 3.0
        return out

    def modeval_part_2(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        x_1, x_2 = x
        out[0] = 0.0
        out[1] = 1.0 / 3.0
        return out

    def modeval_part_3(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        x_1, x_2 = x
        out[0] = 0.0
        out[1] = -x_2 / 3.0
        return out


class SignalTransduction(LinearisedODE):
    """
    """

    def __init__(self, t0, tmax, params, initval, initval_unc=0.0):
        if len(params) != 5:
            raise TypeError("Please enter 5 parameters [k_1, ..., k_4, v]")
        if len(initval) != 5:
            raise TypeError("Please enter a 5d initial value")
        LinearisedODE.__init__(self, t0, tmax, params, initval, initval_unc)

    def modeval_parts(self, t, x):
        """
        """
        f_1 = self.modeval_part_1(t, x)
        f_2 = self.modeval_part_2(t, x)
        f_3 = self.modeval_part_3(t, x)
        f_4 = self.modeval_part_4(t, x)
        f_5 = self.modeval_part_5(t, x)
        return np.array([f_1, f_2, f_3, f_4, f_5])

    def modeval_part_1(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        x_1, x_2, x_3, x_4, x_5 = x
        out[0] = -x_1
        out[1] = x_1
        return out

    def modeval_part_2(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        x_1, x_2, x_3, x_4, x_5 = x
        out[0] = -x_1 * x_3
        out[2] = -x_1 * x_3
        out[3] = x_1 * x_3
        return out

    def modeval_part_3(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        x_1, x_2, x_3, x_4, x_5 = x
        out[0] = x_4
        out[2] = x_4
        out[3] = -x_4
        return out

    def modeval_part_4(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        x_1, x_2, x_3, x_4, x_5 = x
        out[3] = -x_4
        out[4] = x_4
        return out

    def modeval_part_5(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        x_1, x_2, x_3, x_4, x_5 = x
        out[2] = x_5
        out[4] = -x_5
        return out


class GUiY(LinearisedODE):
    """
    """

    def __init__(self, t0, tmax, params, initval, initval_unc=0.0):
        if len(params) != 10:
            raise TypeError("Please enter 5 parameters [k_1, ..., k_4, v]")
        if len(initval) != 9:
            raise TypeError("Please enter a 5d initial value")
        LinearisedODE.__init__(self, t0, tmax, params, initval, initval_unc)

    def modeval_parts(self, t, x):
        """
        """
        f_1 = self.modeval_part_1(t, x)
        f_2 = self.modeval_part_2(t, x)
        f_3 = self.modeval_part_3(t, x)
        f_4 = self.modeval_part_4(t, x)
        f_5 = self.modeval_part_5(t, x)
        f_6 = self.modeval_part_6(t, x)
        f_7 = self.modeval_part_7(t, x)
        f_8 = self.modeval_part_8(t, x)
        f_9 = self.modeval_part_9(t, x)
        f_10 = self.modeval_part_10(t, x)
        return np.array([f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10])

    def modeval_part_1(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        # x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x  # maybe we will need this if x[0],... doesn't work
        out[0] = -x[7] * x[0]
        out[5] = x[7] * x[0]
        out[7] = -x[7] * x[0]
        return out

    def modeval_part_2(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        # x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x  # maybe we will need this if x[0],... doesn't work
        out[0] = x[5]
        out[5] = -x[5]
        out[7] = x[5]
        return out

    def modeval_part_3(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        # x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x  # maybe we will need this if x[0],... doesn't work
        out[1] = -x[8] * x[1]
        out[6] = x[8] * x[1]
        out[8] = -x[8] * x[1]
        return out

    def modeval_part_4(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        # x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x  # maybe we will need this if x[0],... doesn't work
        out[1] = x[6]
        out[6] = -x[6]
        out[8] = x[6]
        return out

    def modeval_part_5(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        # x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x  # maybe we will need this if x[0],... doesn't work
        out[3] = x[6] * x[4]
        out[4] = -x[6] * x[4]
        out[6] = -x[6] * x[4]
        return out

    def modeval_part_6(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        # x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x  # maybe we will need this if x[0],... doesn't work
        out[3] = -x[3]
        out[4] = x[3]
        out[6] = x[3]
        return out

    def modeval_part_7(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        # x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x  # maybe we will need this if x[0],... doesn't work
        out[2] = x[8] * x[4]
        out[4] = -x[8] * x[4]
        out[8] = -x[8] * x[4]
        return out

    def modeval_part_8(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        # x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x  # maybe we will need this if x[0],... doesn't work
        out[2] = -x[2]
        out[4] = x[6]
        out[8] = x[2]
        return out

    def modeval_part_9(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        # x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x  # maybe we will need this if x[0],... doesn't work
        out[5] = x[8] - x[7]
        out[6] = x[8] - x[7]
        return out

    def modeval_part_10(self, t, x):
        """
        """
        out = np.zeros(x.shape)
        # x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x  # maybe we will need this if x[0],... doesn't work
        out[7] = x[8] - x[7]
        out[8] = x[7] - x[8]
        return out
