# coding=utf-8
"""

DISCLAIMER:
    * I am not sure about the readability of this script.
"""

import numpy as np
import matplotlib.pyplot as plt


from odefilters import odesolver
from odefilters import linearised_odesolver as linsolver
from odefilters import linearised_ode as linode
from odefilters import statespace
from odefilters import inverseproblem as ip


np.seterr(all='raise') # debugging: all warnings are errors


def create_data(solver, ivp, thetatrue, stepsize, ivpvar):
    """
    Create artificial data for the inverse problem.
    """
    noise = np.sqrt(ivpvar)*np.random.randn(4).reshape((2, 2))
    ivp.params = thetatrue
    tsteps, m, __, __, __ = solver.solve(ivp, stepsize)
    means = odesolver.get_trajectory_ddim(m, 2, 0)
    evalpts = np.array([1., 2.])
    evalidcs = [list(tsteps).index(evalpt) for evalpt in evalpts]
    data = means[evalidcs] + noise # this is 'wrong' noise
    ipdata = ip.InvProblemData(evalpts, data, ivpvar)
    return ipdata


np.random.seed(1)

# Set Model Parameters
initial_value = np.array([20, 20])
initial_time, end_time = 0., 5.
ivpvar = 0.1 # this is not a lot, the number of prey is in [1, 100]
thetatrue = np.array([0.5, 0.01, 0.02, 0.75])
ivp = linode.LotkaVolterra(initial_time, end_time, params=thetatrue, initval=initial_value)

# Set Method Parameters
h = (end_time - initial_time)/100
solver = linsolver.LinearisedODESolver(statespace.IBM(q=1, dim=2))
ipdata = create_data(solver, ivp, thetatrue, 1e-04, ivpvar)
iplklhd = ip.InvProblemLklhd(ipdata, ivp, solver, h, with_jacob=True)

# Prepare a grid
delta = 0.05
xpts, ypts = np.arange(delta, 1., delta), np.arange(delta, 1., delta)
xmesh, ymesh = np.meshgrid(xpts, ypts)

# Evaluate likelihood and gradient at each gridpoint
lklgrid = np.zeros(xmesh.shape)
gradgrid1, gradgrid2 = np.zeros(xmesh.shape), np.zeros(xmesh.shape)
for xidx, __ in enumerate(xmesh):
    for yidx, __ in enumerate(ymesh.T):
        this_theta = np.array([xmesh[xidx, yidx], 0.01, 0.02, ymesh[xidx, yidx]])
        lklgrid[xidx, yidx] = iplklhd.lklhdeval(this_theta)
        gradvl = iplklhd.gradeval(this_theta)
        print(this_theta, gradvl)
        gradgrid1[xidx, yidx] = gradvl[0]
        gradgrid2[xidx, yidx] = gradvl[3]

# # Make a particle follow a trajectory through the parameter space
# traj = np.zeros((500, 2))
# lrate = 0.01
# traj[-1] = np.array([0.7, 0.8])
# for idx, __ in enumerate(traj):
#     this_theta = np.array([traj[idx-1][0], 0.01, 0.02, traj[idx-1][1]])
#     iplklhd.lklhdeval(this_theta)
#     gradvl = iplklhd.gradeval(this_theta)
#     traj[idx] = traj[idx-1] + lrate*gradvl[[0, 3]]

# Draw result
plt.style.use("seaborn-whitegrid")
plt.title("Gradient Estimates, h=%r" % h)
CS = plt.contour(xmesh, ymesh, lklgrid, cmap='Oranges')
plt.clabel(CS, inline=True, fmt='%.0e')
plt.quiver(xmesh, ymesh, 5e-6*gradgrid1, 5e-6*gradgrid2, scale=1, width=0.15*1e-2, alpha=0.5)
# plt.plot(traj[:, 0], traj[:, 1], '-', color='blue', alpha=0.75, label="Trajectory of some flow")
# plt.scatter(traj[0, 0], traj[0, 1], marker='*', s=400, color='blue', label="Starting point")
# plt.scatter(traj[-1, 0], traj[-1, 1], marker='*', s=400, color='black', label="End point")
# plt.legend()
plt.show()