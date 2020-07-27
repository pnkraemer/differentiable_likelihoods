# coding=utf-8
"""
solve_fhn.py

Checking the compliance of ode.py, statespace.py and odesolver.py
by solving the FitzHugh-Nagumo model.
If the output looks 'proper' and a little uncertainty is visible, the test passes.
"""

import numpy as np
from difflikelihoods import ode
from difflikelihoods import odesolver 
from difflikelihoods import statespace
import matplotlib.pyplot as plt

# Solve ODE
stsp = statespace.IBM(q=1, dim=2)
solver = odesolver.ODESolver(ssm=stsp, filtertype="kalman")
fhn = ode.FitzHughNagumo(t0=0., tmax=15., params=[0., 0.08, 0.07, 1.25], initval=np.array([1.0, .0]), initval_unc=1e-24)
t, m, s = solver.solve(fhn, stepsize=0.01)

# Extract relevant trajectories
mtraj1 = odesolver.get_trajectory(m, 0, 0)
munc1 = odesolver.get_trajectory(s, 0, 0)
mtraj2 = odesolver.get_trajectory(m, 1, 0)
munc2 = odesolver.get_trajectory(s, 1, 0)

# Plot solution
plt.title("FitzHugh-Nagumo Model with Uncertainty")
plt.plot(mtraj1, mtraj2, color="darkslategray", linewidth=1.5, label="Trajectory")
plt.plot(mtraj1[0], mtraj2[0], 'o', label="Starting point at t=t0")
plt.plot(mtraj1[-1], mtraj2[-1], 'o', label="Endpoint at t=tmax")
plt.fill_between(mtraj1, mtraj2 + 3*munc2, mtraj2 - 3*munc2, color="darkslategray", alpha=0.125)
plt.fill_betweenx(mtraj2, mtraj1 + 3*munc1, mtraj1 - 3*munc1, color="darkslategray", alpha=0.125)
plt.legend()
plt.show()





# END OF FILE
