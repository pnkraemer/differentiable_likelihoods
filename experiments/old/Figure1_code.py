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

from odefilters.sampling import metropolishastings_pham, metropolishastings_plang, metropolishastings_rw


from odefilters.optimisation import minimise_newton, minimise_gd, minimise_rs


def hamiltonian(nsamps, iplklhd, init_state, stepsize, nsteps, ninits):
    """
    Wrapper for metropolishastings_pham()
    using the InvProblemLklhd objects.
    """
    samples, probs, ratio = metropolishastings_pham(iplklhd.potenteval, iplklhd.gradeval, iplklhd.hesseval,
        nsamps, init_state, stepsize, nsteps, ninits)
    print("HAMILTONIAN")
    print("ratio",  ratio)
    return samples, probs



def langevin(nsamps, iplklhd, init_state, stepsize, ninits):
    """
    Wrapper for metropolishastings_plang()
    using the InvProblemLklhd objects.
    """
    samples, probs, ratio = metropolishastings_plang(iplklhd.potenteval, iplklhd.gradeval, iplklhd.hesseval,
        nsamps, init_state, stepsize, ninits)
    print("Langevin")
    print("ratio",  ratio)

    return samples, probs



def randomwalk(nsamps, iplklhd, init_state, stepsize, ninits):
    """
    Wrapper for metropolishastings_rw()
    using the InvProblemLklhd objects.
    """
    samples, probs, ratio = metropolishastings_rw(iplklhd.potenteval, nsamps, init_state, stepsize, ninits)
    print("RW")
    print("ratio",  ratio)
    return samples, probs





def create_data(solver, ivp, thetatrue, stepsize, ivpvar):
    """
    Create artificial data for the inverse problem.
    """
    ivp.params = thetatrue
    tsteps, m, __, __, __ = solver.solve(ivp, stepsize)
    means = odesolver.get_trajectory_ddim(m, 5, 0)
    evalpts = np.array([1., 2., 4., 5., 7., 10., 15., 20., 30., 40., 50., 60., 80., 100.])
    assert(np.prod(np.in1d(evalpts, tsteps))==1), print(evalpts[np.in1d(evalpts, tsteps)==False])
    noise = np.sqrt(ivpvar)*np.random.randn(len(evalpts)*5).reshape((len(evalpts), 5))
    evalidcs = [list(tsteps).index(evalpt) for evalpt in evalpts]
    data = means[evalidcs] + noise # this is 'wrong' noise
    ipdata = ip.InvProblemData(evalpts, data, ivpvar)
    return ipdata




    
np.random.seed(123)

def create_data(solver, ivp, thetatrue, stepsize, ivpvar):
    """
    Create artificial data for the inverse problem.
    """
    ivp.params = thetatrue
    tsteps, m, __, __, __ = solver.solve(ivp, stepsize)
    means = odesolver.get_trajectory_ddim(m, 1, 0)
    evalpts = 2.0 * np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert(np.prod(np.in1d(evalpts, tsteps))==1), print(evalpts[np.in1d(evalpts, tsteps)==False])
    evalidcs = [list(tsteps).index(evalpt) for evalpt in evalpts]
    data = np.array([means[evalidx] + np.sqrt(ivpvar)*np.random.randn()
                     for evalidx in evalidcs])
    ipdata = ip.InvProblemData(evalpts, data, ivpvar)
    return ipdata


# Set Model Parameters
initial_value = 0.25
initial_time, end_time = 0., 2.
ivpvar = 1e-12
thetatrue = np.array([2.5, 2.5])
ivp = linode.LogisticODE(initial_time, end_time, params=thetatrue, initval=initial_value)

# Set Method Parameters
h_for_data = (end_time - initial_time)/1000
h1 = (end_time - initial_time)/400
h2 = (end_time - initial_time)/400
h3 = (end_time - initial_time)/400
solver = linsolver.LinearisedODESolver(statespace.IBM(q=1, dim=1))
ipdata = create_data(solver, ivp, thetatrue, h_for_data, ivpvar)
iplklhd = ip.InvProblemLklhd(ipdata, ivp, solver, h1, with_jacob=True)
iplklhd2 = ip.InvProblemLklhdClassic(ipdata, ivp, solver, h1, with_jacob=True)
iplklhd3 = ip.InvProblemLklhd(ipdata, ivp, solver, h2, with_jacob=True)
iplklhd4 = ip.InvProblemLklhdClassic(ipdata, ivp, solver, h2, with_jacob=True)

# iplklhd5 = ip.InvProblemLklhd(ipdata, ivp, solver, h3, with_jacob=True)
# iplklhd6 = ip.InvProblemLklhdClassic(ipdata, ivp, solver, h3, with_jacob=True)

niter = 12
init_theta = np.array([2.97, 2.68])
# samples_ham, __ = langevin(niter, iplklhd3, init_theta, stepsize=0.75, ninits=25)

# samples_rw, __ = randomwalk(niter, iplklhd3, init_theta, stepsize=1e-02, ninits=25)

samples_ham, obj_newton = minimise_newton(iplklhd3.potenteval, iplklhd3.gradeval, iplklhd3.hesseval, niter, init_theta, lrate=0.5)
samples_rw, obj_newton = minimise_rs(iplklhd3.potenteval, niter, init_theta, lrate=0.05)

print(samples_ham)
print(samples_rw)

# Draw a grid and compute gradient approximations
delta = 0.0125
xpts = np.arange(2.3, 2.7, delta)
ypts = np.arange(2.3, 2.7, delta)
X, Y = np.meshgrid(xpts, ypts)
lklgrid1 = np.zeros(X.shape)
lklgrid2 = np.zeros(X.shape)
lklgrid3 = np.zeros(X.shape)
lklgrid4 = np.zeros(X.shape)
# lklgrid5 = np.zeros(X.shape)
# lklgrid6 = np.zeros(X.shape)

for i in range(len(X)):
    for j in range(len(X.T)):
        this_theta = np.array([X[i, j], Y[i, j]])
        # lklgrid1[i, j] = (-iplklhd.potenteval(this_theta))
        # lklgrid2[i, j] = (-iplklhd2.potenteval(this_theta))
        if this_theta[0] >= this_theta[1] - 0.05 or this_theta[1] >= this_theta[0] - 0.05:
            lklgrid3[i, j] = (-iplklhd3.potenteval(this_theta))
        # lklgrid4[i, j] = (-iplklhd4.potenteval(this_theta))
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
from scipy.ndimage.filters import gaussian_filter
# import scipy.ndimage


perc = 0.04
# lklgrid1 = (gaussian_filter(lklgrid1, perc * np.abs(np.amax(lklgrid1) - np.amin(lklgrid1))))
# lklgrid2 = (gaussian_filter(lklgrid2, perc * np.abs(np.amax(lklgrid2) - np.amin(lklgrid2))))
lklgrid3 = (gaussian_filter(lklgrid3, perc * np.abs(np.amax(lklgrid3) - np.amin(lklgrid3)))) # for nicer visualisatio
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



# print(samples_ham, samples_rw)
# Visualise results
from matplotlib import ticker

plt.style.use("./icmlstyle.mplstyle")
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True)

colormap = 'coolwarm'
_alpha = 0.75

fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharey=True)

# ax.set_title("PHMC vs RW (proj.)", size='large')
# ax1.set_title("PDF")
# ax1.contourf(X, Y, lklgrid3, cmap=colormap, alpha=_alpha)


# ax2.set_title("PHMC")
# ax2.scatter(samples_ham[:, 0], samples_ham[:, 1], cmap=colormap)

# ax3.set_title("RW")
# ax3.scatter(samples_rw[:, 0], samples_rw[:, 1], bins=0.4/delta, cmap=colormap)

ax1.plot(samples_rw[:, 0], samples_rw[:, 1], color="#6699CC", ls='None',ms=5,  marker='^', alpha=0.75,  label="Likelihood-free")
ax1.plot(samples_ham[:, 0], samples_ham[:, 1], color="darkorange",ls='None', ms=5, marker='d', alpha=0.75, label="Our proposal")

cs = ax1.contour(X, Y, (lklgrid3), linewidths=0.65, linestyles="solid", levels=4, colors="black", alpha=0.8)
#ax1.scatter(thetatrue[0], thetatrue[1], marker='x', color='black', zorder=2, label="Truth")
ax1.plot(init_theta[0], init_theta[1], linestyle="None", marker='s',  ms=6, markerfacecolor="None",
         markeredgecolor='black', markeredgewidth=.5, label="Initial state")
plt.legend(loc="lower right")
#fig.colorbar(cs)
# plt.xticks([])
# plt.yticks([])
plt.xlim((2.35, 3.15))
plt.ylim((2.15, 2.75))
# plt.xticks([2.4, 2.6])
# plt.yticks([2.4, 2.6])



ax1.spines["top"].set_visible(True)    
ax1.spines["right"].set_visible(True)    


plt.savefig("./figures/figure1_firstpage")
plt.show()