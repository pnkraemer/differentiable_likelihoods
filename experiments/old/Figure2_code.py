# coding=utf-8
"""
"""

import numpy as np
import matplotlib.pyplot as plt

from odefilters import odesolver
from odefilters import linearised_odesolver as linsolver
from odefilters import linearised_ode as linode
from odefilters import statespace
from odefilters import inverseproblem as ip

    

def create_data(solver, ivp, thetatrue, stepsize, ivpvar):
    """
    Create artificial data for the inverse problem.
    """
    ivp.params = thetatrue
    tsteps, m, __, __, __ = solver.solve(ivp, stepsize)
    means = odesolver.get_trajectory_ddim(m, 2, 0)
    evalpts = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
    assert(np.prod(np.in1d(evalpts, tsteps))==1), print(evalpts[np.in1d(evalpts, tsteps)==False])
    evalidcs = [list(tsteps).index(evalpt) for evalpt in evalpts]
    data = np.array([means[evalidx] + np.sqrt(ivpvar)*np.random.randn()
                     for evalidx in evalidcs])
    ipdata = ip.InvProblemData(evalpts, data, ivpvar)
    return ipdata


# Set Model Parameters
initial_value = np.array([20, 20])
initial_time, end_time = 0., 10.
ivpvar = 1e-3
thetatrue = np.array([0.5, 0.05, 0.05, 0.5])
ivp = linode.LotkaVolterra(initial_time, end_time, params=thetatrue, initval=initial_value)

# Set Method Parameters
h_for_data = (end_time - initial_time)/1000
h1 = (end_time - initial_time)/50
h2 = (end_time - initial_time)/400
h3 = (end_time - initial_time)/100
solver = linsolver.LinearisedODESolver(statespace.IBM(q=1, dim=2))
ipdata = create_data(solver, ivp, thetatrue, h_for_data, ivpvar)
iplklhd = ip.InvProblemLklhd(ipdata, ivp, solver, h1, with_jacob=True)
iplklhd2 = ip.InvProblemLklhdClassic(ipdata, ivp, solver, h1, with_jacob=True)
iplklhd3 = ip.InvProblemLklhd(ipdata, ivp, solver, h2, with_jacob=True)
iplklhd4 = ip.InvProblemLklhdClassic(ipdata, ivp, solver, h2, with_jacob=True)

# iplklhd5 = ip.InvProblemLklhd(ipdata, ivp, solver, h3, with_jacob=True)
# iplklhd6 = ip.InvProblemLklhdClassic(ipdata, ivp, solver, h3, with_jacob=True)


# Draw a grid and compute gradient approximations
delta = 0.00125
xpts = np.arange(0.48, 0.52, delta)
ypts = np.arange(0.048, 0.052, 0.1*delta)
X, Y = np.meshgrid(xpts, ypts)
lklgrid1 = np.zeros(X.shape)
lklgrid2 = np.zeros(X.shape)
lklgrid3 = np.zeros(X.shape)
lklgrid4 = np.zeros(X.shape)
# lklgrid5 = np.zeros(X.shape)
# lklgrid6 = np.zeros(X.shape)

for i in range(len(X)):
    for j in range(len(X.T)):
        this_theta = np.array([X[i, j], Y[i, j], 0.05, 0.5])
        lklgrid1[i, j] = (-iplklhd.potenteval(this_theta))
        lklgrid2[i, j] = (-iplklhd2.potenteval(this_theta))
        lklgrid3[i, j] = (-iplklhd3.potenteval(this_theta))
        lklgrid4[i, j] = (-iplklhd4.potenteval(this_theta))
        # lklgrid5[i, j] = (-iplklhd5.potenteval(this_theta))
        # lklgrid6[i, j] = (-iplklhd6.potenteval(this_theta))

lklgrid1[lklgrid1 < -40] = -40
lklgrid2[lklgrid2 < -40] = -40
lklgrid3[lklgrid3 < -40] = -40
lklgrid4[lklgrid4 < -40] = -40
# lklgrid5[lklgrid5 < -40] = -40
# lklgrid6[lklgrid6 < -40] = -40


# print(lklgrid1)
# print(np.amin(lklgrid1))
# print(np.amax(lklgrid1))

# print(lklgrid2)
# print(np.amin(lklgrid2))
# print(np.amax(lklgrid2))

# print(lklgrid4)
# print(np.amin(lklgrid4))
# print(np.amax(lklgrid4))
# from scipy.ndimage.filters import gaussian_filter
# import scipy.ndimage


# perc = 0.05
# lklgrid1 = (gaussian_filter(lklgrid1, perc * np.abs(np.amax(lklgrid1) - np.amin(lklgrid1))))
# lklgrid2 = (gaussian_filter(lklgrid2, perc * np.abs(np.amax(lklgrid2) - np.amin(lklgrid2))))
# lklgrid3 = (gaussian_filter(lklgrid3, perc * np.abs(np.amax(lklgrid3) - np.amin(lklgrid3))))
# lklgrid4 = (gaussian_filter(lklgrid4, perc * np.abs(np.amax(lklgrid3) - np.amin(lklgrid3))))
# lklgrid5 = gaussian_filter(lklgrid5, perc * np.abs(np.amax(lklgrid5) - np.amin(lklgrid5)))
# lklgrid6 = gaussian_filter(lklgrid6, perc * np.abs(np.amax(lklgrid6) - np.amin(lklgrid6)))

# lklgrid1[lklgrid1 < -40] = 0
# lklgrid2[lklgrid2 < -40] = 0
# lklgrid3[lklgrid3 < -40] = 0
# lklgrid4[lklgrid4 < -40] = 0
# lklgrid5[lklgrid5 < -40] = 0
# lklgrid6[lklgrid6 < -40] = 0

# lklgrid1[lklgrid1 > -40] = 1
# lklgrid2[lklgrid2 > -40] = 1
# lklgrid3[lklgrid3 > -40] = 1
# lklgrid4[lklgrid4 > -40] = 1
# lklgrid5[lklgrid5 > -40] = 1
# lklgrid6[lklgrid6 > -40] = 1

# lklgrid1[lklgrid1 < 1e-14] = 0
# lklgrid2[lklgrid2 < 1e-14] = 0
# lklgrid3[lklgrid3 < 1e-14] = 0
# lklgrid4[lklgrid4 < 1e-14] = 0
# lklgrid5[lklgrid5 < 1e-14] = 0
# lklgrid6[lklgrid6 < 1e-14] = 0


# print(np.amax(lklgrid1), np.amin(lklgrid1))
# print(np.amax(lklgrid2), np.amin(lklgrid2))
# print(np.amax(lklgrid3), np.amin(lklgrid3))
# print(np.amax(lklgrid4), np.amin(lklgrid4))
# print(np.amax(lklgrid5), np.amin(lklgrid5))
# print(np.amax(lklgrid6), np.amin(lklgrid6))

# lklgrid1 = np.exp(lklgrid1)
# lklgrid2 = np.exp(lklgrid2)
# lklgrid3 = np.exp(lklgrid3)
# lklgrid4 = np.exp(lklgrid4)

# print(lklgrid1.shape)
# lklgrid1 = lklgrid1[lklgrid1[] > 0]
# print(lklgrid1.shape)

# Visualise results

plt.style.use("./icmlstyle.mplstyle")
# plt.rc('font', size=25)                # controls default text sizes
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(3.3, 2.25))

colormap = 'coolwarm'
_alpha = 0.75

ax[0][0].set_title("Large steps", fontsize=6.5)
ax[0][0].set_ylabel("Aware, eq. (9)", fontsize=6.5)
ax[0][0].contourf(X, Y, (lklgrid1), cmap=colormap, alpha=_alpha)
ax[0][0].plot(thetatrue[0], thetatrue[1], marker='x', markersize=5, mew=1.5, color='black', zorder=2)
ax[0][0].grid(False)


ax[0][1].set_title("Small steps", fontsize=6.5)
ax[0][1].contourf(X, Y, lklgrid3, cmap=colormap, alpha=_alpha)
ax[0][1].plot(thetatrue[0], thetatrue[1], marker='x', markersize=5, mew=1.5,color='black', zorder=2)
ax[0][1].grid(False)



ax[1][0].set_ylabel("Unaware, eq. (6)", fontsize=6.5)
ax[1][0].contourf(X, Y, lklgrid2, cmap=colormap, alpha=_alpha)
ax[1][0].plot(thetatrue[0], thetatrue[1], marker='x', markersize=5, mew=1.5,color='black', zorder=2)
ax[1][0].grid(False)

ax[1][1].contourf(X, Y, lklgrid4, cmap=colormap, alpha=_alpha)
ax[1][1].plot(thetatrue[0], thetatrue[1], marker='x', markersize=5, mew=1.5, color='black', zorder=2)
ax[1][1].grid(False)


for ax_ in ax.flatten():
    ax_.spines["top"].set_visible(True)    
    ax_.spines["right"].set_visible(True)    
    ax_.spines["bottom"].set_visible(True)    
    ax_.spines["left"].set_visible(True)    


# plt.xlim((2.3, 2.7))
# plt.ylim((2.3, 2.7))
plt.xticks([0.49, 0.51])
plt.yticks([0.049, 0.051])
plt.tight_layout()
plt.savefig("./figures/figure2_contours")

plt.show()