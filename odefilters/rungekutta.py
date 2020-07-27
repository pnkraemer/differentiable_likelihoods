"""
rungekutta.py

Wrappers for scipy's rungekutta methods
to be used in conjuction with inverse problems
"""

import scipy.integrate
from abc import ABC, abstractmethod

class RungeKutta(ABC):
    """
    Behaves like an ODESolver object.
    """
    def __init__(self):
        """
        """
        pass

    @abstractmethod
    def solve(ode, tol):
        """
        Solves ODE up to tolerance tol.
        """
        raise NotImplementedError


class RungeKutta23(RungeKutta):
    """
    """
    pass
    def solve(ode, tol):
        """
        """
        fun, t0, y0, t_bound = ode.modeval, ode.t0, ode.initval, ode.tmax
        