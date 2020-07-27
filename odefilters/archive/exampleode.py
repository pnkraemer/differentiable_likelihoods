# coding=utf-8
"""
exampleode.py

Some common initial value problems based on the ODE class:
    * Linear ODEs
    * Sigmoid logistic growth function
    * FitzHugh-Nagumo model
    * Restricted Two-Body problem
    * ...
as well as a custom ODE class.

Note:
    We use the name 'ode' instead of 'ivp' to avoid confusion with
    the state space model 'integrated Wiener process',
    resp. 'integrated Brownian motion' and the abbreviation I(nverse)P(roblem).

Reference:
    Hairer, E., Norsett, S.P., Wanner, G. Solving Ordinary
    Differential Equations I: Nonstiff Problems, Springer, 2009

Example:
    >>> lin_ode = ode.LinearODE(t0=0.1, tmax=2.0, params=1.0, y0=0.9)
    >>> print(lin_ode)
"""

import numpy as np
from odefilters.ode import *




