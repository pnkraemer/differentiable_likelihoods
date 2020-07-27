"""
logdensity.py

Contains objects representing probability distributions
of the form 
    p(z) = exp(-E(z))/Z
via representing E = E(z). These densities are used, for
instance, for Bayesian inference and can be used
within the sampling.py module which does MCMC sampling. 
"""

from abc import ABC, abstractmethod


class LogDensity(ABC):
    """
    """
    def __init__(self, logdens, loggrad=None, loghess=None):
        """
        """
        self.logdens = logdens
        self.loggrad = loggrad
        self.loghess = loghess
        self.has_gradient = bool(loggrad is not None)
        self.has_hessian = bool(loghess is not None)

    def __repr__(self):
        """
        """
        return "LogDensity(%r, %r, %r)" % (self.logdens, self.loggrad, self.loghess)

    def eval(self, loc):
        """
        """
        return self.logdens(loc)

    def gradeval(self, loc):
        """
        """
        if self.loggrad is None:
            raise TypeError("Gradient is not available")
        return self.loggrad(loc)


    def hesseval(self, loc):
        """
        """
        if self.loghess is None:
            raise TypeError("Hessian is not available")
        return self.loghess(loc)