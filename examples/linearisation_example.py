"""
linearisation_example.py

We assert that the filter mean m(t_*) at time t_*
can be obtained via GP regression based on 
integrated Brownian motion covariance kernels.

In other words, we assert that the linearisation.py module
does what it is expected to do. A very similar script
is used as unittest; see ../unittests/test_linearisation.py
"""

import numpy as np
from odefilters import covariance as cov
from odefilters import odesolver
from odefilters import linearised_odesolver as linsolve
from odefilters import linearised_ode as linode
from odefilters import statespace
from odefilters import linearisation


# Set Model Parameters
odeparam = np.array([0, 1, 1, 2])
y0, y0_unc = np.ones(2), 0 * np.ones(2)
t0, tmax = 0.0, 1 + np.random.rand()

# Set Method Parameters
q = 1
h = 0.05

# Set up and solve ODE
ibm = statespace.IBM(q=q, dim=len(y0))
solver = linsolve.LinearisedODESolver(ibm)
ivp = linode.LotkaVolterra(t0, tmax, odeparam, y0, y0_unc)
tsteps, means, __, rhs_parts, uncerts = solver.solve(ivp, stepsize=h)
mean = odesolver.get_trajectory_multidim(means, [0, 1], 0)

# Set up BM and IBM covariance matrices
evalpt = np.array(tsteps[[-1, -20]])
derdat = (tsteps, rhs_parts, 0.)

const, jacob = linearisation.compute_linearisation(
    ssm=ibm, initial_value=y0,
    derivative_data=derdat, prdct_tsteps=evalpt)

# Compute GP Estimation of filter mean at t=tmax
postmean = const + np.dot(jacob, odeparam)#jacob*odeparam
postmean = postmean.reshape((2, 2))


# Display Error and Parameters
error = np.linalg.norm(postmean - mean[[-1, -20]])/np.linalg.norm(mean[-1])
print("\nModel parameters:")
print("\ttmax = %.1f\n\tt0 = %.1f\n\ty0 = (%.1f, %.1f)\n\t\n"  % (tmax, t0, y0[0], y0[1]))
print("Method parameters:")
print("\tq = %u\n\th = %.1e"  % (q, h))
print("\nRelative Error of GP and Filter:\n\tr = %.1e\n" % error)

# Raise problem if error is not tiny
if error > 1e-10:
    print("------------------------------------------------------------------")
    print("ATTENTION: This does not seem to be a sensible equivalence anymore")
    print("------------------------------------------------------------------\n")
