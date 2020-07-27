# coding=utf-8
"""
sanitycheck_filter_gp.py

We assert that the filter mean m(t_*) at time t_*
can be obtained via GP regression based on 
integrated Brownian motion covariance kernels.

DISCLAIMERS:
    * A sanity check for the covariances at the 
    final point has still to be conducted (so
    far the covariances do not match for some
    reason, even though they should from an
    anlaytical perspective.)
    See Figure 12.2 on p. 267 in Applied SDE Book.
    * q > 1 breaks the equivalence due to the
    ill-conditioning of the IBM matrices
"""

import numpy as np
from odefilters import covariance as cov
from odefilters import odesolver
from odefilters import linearised_odesolver as linsolve
from odefilters import linearised_ode as ode
from odefilters import statespace


# Set Model Parameters
odeparam = 2*np.random.rand()
y0, y0_unc = 2*np.random.rand(), 0.0
#t0, tmax = 0.1 + np.random.rand(), 1.1 + np.random.rand()
t0, tmax = 0.1, 1.1 + np.random.rand()

# Set Method Parameters
q = 1
h = np.random.choice([0.1, 0.01, 0.001])

# Set up and solve ODE
ibm = statespace.IBM(q=q, dim=1)
solver = linsolve.LinearisedODESolver(ibm)
ivp = ode.LinearODE(t0, tmax, odeparam, y0, y0_unc)
tsteps, means, __, rhs_parts, __ = solver.solve(ivp, stepsize=h)
mean = odesolver.get_trajectory(means, 0, 0)

# Set up BM and IBM covariance matrices
tsteps = tsteps
evalpt = np.array([tsteps[-1]])
kdell = cov.ibm_covd(evalpt, tsteps, q=q)
dellkdell = cov.ibm_dcovd(tsteps, tsteps, q=q)
kernel_prefactor = kdell.dot(np.linalg.inv(dellkdell))

# Compute GP Estimation of filter mean at t=tmax
data = rhs_parts[:, 0, 0] - y0
jacob = kernel_prefactor.dot(data) + y0*(evalpt - t0)
postmean = y0 + jacob*odeparam

# Display Error and Parameters
error = np.linalg.norm(postmean - mean[-1])/np.linalg.norm(mean[-1])
print("\nModel parameters:")
print("\ttmax = %.1f\n\tt0 = %.1f\n\ty0 = %.1f\n\ttheta = %.1f\n"  % (tmax, t0, y0, odeparam))
print("Method parameters:")
print("\tq = %u\n\th = %.1e"  % (q, h))
print("\nRelative Error of GP and Filter:\n\tr = %.1e\n" % error)

# Raise problem if error is not tiny
if error > 1e-10:
    print("------------------------------------------------------------------")
    print("ATTENTION: This does not seem to be a sensible equivalence anymore")
    print("------------------------------------------------------------------\n")
