"""
"""

from abc import ABC, abstractmethod


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
        return "ODE(t0=%r, tmax=%r, initval=%r, initval_unc=%r)" \
            % (self.t0, self.tmax, self.initval, self.initval_unc)

    @abstractmethod
    def modeval(self, t, x):
        """
        Evaluates model function f.
        Intended to be overwritten by subclasses.
        """
        raise NotImplementedError("Model evaluation is not implemented")



class CustomODE(ODE):
    """
    ODE with custom model evaluation function.
    See e.g. ./examples/custom_ode.py for help
    with usage.
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
        self.custom_modeval = modeval
        ODE.__init__(self, t0, tmax, initval, initval_unc=0.0)

    def __repr__(self):
        """
        Represent CustomODE object and its parameters, e.g.:
            CustomODE(t0=0.0, tmax=1.0,
                      modeval=<function <lambda> at 0x7f5e4638cf28>,
                      initval=0.0, initval_unc=0.0)
        """
        return "CustomODE(t0=%r, tmax=%r, modeval=%r, initval=%r, initval_unc=%r)" \
            % (self.t0, self.tmax, self.custom_modeval, self.initval, self.initval_unc)

    def modeval(self, t, x):
        """
        Evaluates custom model function f.
        """
        return self.custom_modeval(t, x)




def logistic_rhs(t, x, r=2., k=2.):
    """
    RHS evaluation of logistic ODE,
    returns
        f(t, x) = r * x * (1 - x/k)
    """
    return r * x * (1. - x / k)


def get_logistic(t0, tmax, y0, y0_unc, r=2., k=2.):
    """
    """
    logistic_ode = CustomODE(t0, tmax, modeval=logistic_rhs,
                             initval=y0, initval_unc=y0_unc)
    return logistic_ode




if __name__ == "__main__":

    log_ode = get_logistic(t0=0., tmax=1., y0=0., y0_unc=1.)
    print(log_ode)
    
