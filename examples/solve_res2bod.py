# coding=utf-8
"""
solve_res2bod.py

Checks whether odesolver works up to high accuracy
via solving the restricted two-body problem.
This might take 20-30 seconds.
If the resulting orbit is periodic, the solution is correct.
"""

import numpy as np
import matplotlib.pyplot as plt

from difflikelihoods import ode
from difflikelihoods import odesolver
from difflikelihoods import statespace

# Solve ODE
stsp = statespace.IBM(q=2, dim=4)
solver = odesolver.ODESolver(ssm=stsp, filtertype="kalman")
r2b = ode.Res2Bod(t0=0, tmax=32)
t, m, s = solver.solve(r2b, stepsize=0.0001)

# Extract relevant trajectories
mtraj1 = odesolver.get_trajectory(m, 0, 0)
mtraj2 = odesolver.get_trajectory(m, 1, 0)

# Plot solution
plt.title("Restricted Two-Body Problem")
plt.plot(mtraj1, mtraj2, label="Trajectory")
plt.plot(mtraj1[0], mtraj2[0], "o", label="Starting point at t=t0")
plt.plot(mtraj1[-1], mtraj2[-1], "o", label="Endpoint at t=tmax")
plt.legend()
plt.show()


# END OF FILE
