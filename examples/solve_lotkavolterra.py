# coding=utf-8
"""
solve_lotkavolterra.py

We replicate the visualisation from 
    https://de.mathworks.com/help/matlab/math/nume
    rical-integration-of-differential-equations.html

We check the compliance of linearised_ode.py with odesolver.py
by solving the Lotka-Volterra model.
If the output looks as in the link above and a little uncertainty
is visible, the test passes.
"""

import numpy as np
from difflikelihoods import linearised_ode as linode
from difflikelihoods import odesolver 
from difflikelihoods import statespace
import matplotlib.pyplot as plt

# Solve ODE
stsp = statespace.IBM(q=2, dim=2)
solver = odesolver.ODESolver(ssm=stsp, filtertype="kalman")
t0, tmax = 0., 15.
lotka = linode.LotkaVolterra(t0=t0, tmax=tmax, params=[1.0, 0.01, 0.02, 1.0],
                             initval=np.array([20.0, 20.0]), initval_unc=1)
h = 0.01
t, m, s = solver.solve(lotka, stepsize=h)

# Extract relevant trajectories
mtraj1 = odesolver.get_trajectory(m, 0, 0)
munc1 = odesolver.get_trajectory(s, 0, 0)
mtraj2 = odesolver.get_trajectory(m, 1, 0)
munc2 = odesolver.get_trajectory(s, 1, 0)

# Plot solution
plt.title("Lotka-Volterra Phase Plane Plot (h=%r)" % h)
plt.plot(mtraj1, mtraj2, color="darkslategray", linewidth=1.5,
        label="Trajectory")
plt.plot(mtraj1[0], mtraj2[0], 'o', label="Starting point at t0=%r" % t0)
plt.plot(mtraj1[-1], mtraj2[-1], 'o', label="Endpoint at tmax=%r" % tmax)
plt.fill_between(mtraj1, mtraj2 + 3*munc2, mtraj2 - 3*munc2,
                 color="darkslategray", alpha=0.125)
plt.fill_betweenx(mtraj2, mtraj1 + 3*munc1, mtraj1 - 3*munc1,
                  color="darkslategray", alpha=0.125)
plt.xlabel("Prey")
plt.ylabel("Predators")
plt.legend()
plt.show()




# END OF FILE
