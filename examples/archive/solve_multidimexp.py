# coding=utf-8
"""
solve_multidimexp.py

Checking the compliance of ode.py, statespace.py and odesolver.py
by solving the multidimexp model. 
If the output looks 'proper' (that is, the trajectories are somewhat
close to the truthand the relative error is proportional
to the stepsize, the test passes.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spl

from odefilters import odesolver
from odefilters import ode
from odefilters import statespace


# Model parameters
d = 3
t0, tmax = 0., 1.
y0, y0_unc = 0.1*np.ones(d), 0. * np.ones(d)
randommtrx = np.eye(3) + 0.1*np.random.rand(d, d) # too random: sol. difficult
ivp = ode.MatrixLinearODE(t0, tmax, randommtrx,  y0, y0_unc)

# Method parameters
q = 1
stsp = statespace.IBM(q=q, dim=d)
solver = odesolver.ODESolver(ssm=stsp, filtertype="kalman")
h = 0.1

# Solve ODE and extract relevant trajectories 
t, m, s = solver.solve(ivp, stepsize=h)
idx_dims = list(range(d)) 
mtraj = odesolver.get_trajectory_multidim(m, idx_dims, 0)
munc = odesolver.get_trajectory_multidim(s, idx_dims, 0)

# True solution
true_solution = np.zeros((len(t), d))
for idx, tpt in enumerate(t):
	matrix_exponential = spl.expm((tpt-t0)*randommtrx) 
	true_solution[idx] = matrix_exponential @ y0 

# Compute error
error = np.linalg.norm(true_solution - mtraj) / np.sqrt(len(mtraj))
print('The relativ error is %.0e! (h=%.0e)' % (error, h))

# Plot solution
fig, axs = plt.subplots(3,1)
for j in range(d):
	axs[j].plot(t, true_solution[:, j], label="Truth")
	axs[j].plot(t, mtraj[:, j], label="Approximation") 
	axs[j]. plot(t, mtraj[:, j] + 2*munc[:, j], 'k--')
	axs[j]. plot(t, mtraj[:, j] - 2*munc[:, j], 'k--')
plt.legend()
plt.show()

# # END OF FILE