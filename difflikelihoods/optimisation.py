"""
optimisation.py

Zeroth, first and second order minimisation.

So far, we are restricted to the dependency of the LogDensity
object which has a function evaluation as well as evaluations of
higher order objectives.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from odefilters import logdensity


import numpy as np




def minimise_rs(logpdf, nsamps, initval, lrate):
    """
    Convenience function for first order minimisation.
    """
    logdens = logdensity.LogDensity(logpdf)
    rsmini = RandomSearch(logdens)
    return rsmini.minimise(nsamps, initval, lrate)



def minimise_gd(logpdf, loggrad, nsamps, initval, lrate):
    """
    Convenience function for first order minimisation.
    """
    logdens = logdensity.LogDensity(logpdf, loggrad)
    gdmini = GradientDescent(logdens)
    return gdmini.minimise(nsamps, initval, lrate)



def minimise_newton(logpdf, loggrad, loghess, nsamps, initval, lrate):
    """
    Convenience function for second order minimisation.
    """
    logdens = logdensity.LogDensity(logpdf, loggrad, loghess)
    newtonmini = NewtonMethod(logdens)
    return newtonmini.minimise(nsamps, initval, lrate)













# Data structure denoting the current iteration: (x, p(x), p'(x), p''(x))
Iteration = namedtuple("Iteration", "loc objeval gradeval hesseval")



class Optimiser(ABC):
    """
    """
    def __init__(self, logobj):
        """
        logobj behaves like a LogDensity object.
        """
        self.logobj = logobj

    def minimise(self, niter, initval, lrate):
        """
        """
        ndim = len(initval)
        traj = np.zeros((niter, ndim))
        objectives = np.zeros(niter)
        curriter = self.evaluate_logobj(initval)
        traj[0], objectives[0] = curriter.loc, curriter.objeval
        for idx in range(1, niter):
            curriter = self.iterate(curriter, lrate)
            traj[idx], objectives[idx] = curriter.loc, curriter.objeval
        return traj, objectives


    def evaluate_logobj(self, loc):
        """
        """
        logdenseval = self.logobj.eval(loc)
        if self.logobj.has_gradient:
            gradeval = self.logobj.gradeval(loc)
        else:
            gradeval = 0
        if self.logobj.has_hessian:
            hesseval = self.logobj.hesseval(loc)
        else:
            hesseval = 0
        return Iteration(loc, logdenseval, gradeval, hesseval)

    @abstractmethod
    def iterate(self, curriter, lrate):
        """
        """
        raise NotImplementedError("Not implemented---ABSTRACT METHOD")


class RandomSearch(Optimiser):
    """
    """
    def __init__(self, logobj):
        """
        """
        Optimiser.__init__(self, logobj)

    def iterate(self, curriter, lrate):
        """
        """
        this_loc, objval = curriter.loc, curriter.objeval
        sample = np.random.randn(len(curriter.loc))
        newloc = this_loc + lrate * sample/np.linalg.norm(sample)
        newobjval = self.logobj.eval(newloc)
        if newobjval <= objval:
            return Iteration(newloc, newobjval, 0., 0.)
        else:
            return curriter



class GradientDescent(Optimiser):
    """
    """
    def __init__(self, logobj):
        """
        """
        Optimiser.__init__(self, logobj)

    def iterate(self, curriter, lrate):
        """
        """
        this_loc, descent = curriter.loc, curriter.gradeval
        new_iterate = this_loc - lrate*descent
        return self.evaluate_logobj(new_iterate)
    

class NewtonMethod(Optimiser):
    """
    """
    def __init__(self, logobj):
        """
        """
        Optimiser.__init__(self, logobj)

    def iterate(self, curriter, lrate):
        """
        """
        this_loc, grad, hess = curriter.loc, curriter.gradeval, curriter.hesseval
        descent = np.linalg.solve(hess, grad)
        new_iterate = this_loc - lrate*descent
        return self.evaluate_logobj(new_iterate)

