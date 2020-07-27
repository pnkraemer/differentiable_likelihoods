# coding=utf-8
"""
optimise.py

This module contains a gradient descent optimisation algorithm
based on scipy.optimize.minimize().

Example in 1d:
    >>> import optimise
    >>> fct = lambda x: x**2
    >>> der = lambda x: 2*x
    >>> xmin, numit = optimise.minimise(fun=fct, jac=der, x0=10.0)
    >>> print(xmin, numit)

Example in 2d:
    >>> import numpy as np
    >>> import optimise
    >>> fct = lambda x: x[0]**2 + x[1]**2
    >>> der = lambda x: np.array([2*x[0], 2*x[1]])
    >>> x0 = np.array([-20.1, 12.0])
    >>> xmin, numit = optimise.minimise(fun=fct, jac=der, x0=x0)
    >>> print(xmin, numit)
"""
import numpy as np
import scipy.optimize as sco

def minimise(fun, jac, x0, acc=1e-06):
    """
    Minimises a function fun: R^d -> R, d >=1, with CG algorithm.
    Wrapper for scipy.optimise.minimise(..., method='CG').

    Args:
        fun:    function to be minimised, callable
        jac:    Jacobian of function, callable
        x0:     starting point
        acc:    desired accuracy for termination, default is 1e-06

    Returns:
        xmin:   approximation of minimimum
        numit:  number of iterations (function evaluations)
    """
    output = sco.minimize(fun, x0, method='CG', jac=jac, tol=acc)
    xmin = output['x']
    numit = output['nfev']
    return xmin, numit

# END OF FILE
