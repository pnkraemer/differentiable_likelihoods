# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from odefilters import odesolver
from odefilters import linearised_odesolver as linsolver
from odefilters import linearised_ode as linode
from odefilters import statespace
from odefilters import inverseproblem as ip
from odesolvers.optimisation import minimise_newton, minimise_gd, minimise_rs

def create_data(solver, ivp, thetatrue, stepsize, ivpvar):
    """
    Create artificial data for the inverse problem.
    """
    ivp.params = thetatrue
    tsteps, m, __, __, __ = solver.solve(ivp, stepsize)
    means = odesolver.get_trajectory_ddim(m, 2, 0)
    evalpts = np.array([1., 2., 3., 4., 5.])
    evalpts = np.arange(.5, 5., 5/10)
    assert(np.prod(np.in1d(evalpts, tsteps))==1), print(evalpts[np.in1d(evalpts, tsteps)==False])
    noise = np.sqrt(ivpvar)*np.random.randn(len(evalpts)*2).reshape((len(evalpts), 2))
    evalidcs = [list(tsteps).index(evalpt) for evalpt in evalpts]
    data = means[evalidcs] + noise # this is 'wrong' noise
    ipdata = ip.InvProblemData(evalpts, data, ivpvar)
    return ipdata


np.random.seed(2)

# Set Model Parameters
initial_value = np.array([20, 20])
initial_time, end_time = 0., 5.
ivpvar = 1e-10
thetatrue = np.array([1.0, 0.1, 0.1, 1.0])
ivp = linode.LotkaVolterra(initial_time, end_time, params=thetatrue, initval=initial_value)

# Set Method Parameters
h_for_data = (end_time - initial_time)/10000
h = (end_time - initial_time)/100
solver = linsolver.LinearisedODESolver(statespace.IBM(q=1, dim=2))
ipdata = create_data(solver, ivp, thetatrue, h_for_data, ivpvar)
iplklhd = ip.InvProblemLklhd(ipdata, ivp, solver, h, with_jacob=True)

# Make a particle follow a trajectory through the parameter space
niter = 100
init_theta = np.array([.8, .2, .05, 1.1])

traj_newton, obj_newton = minimise_newton(iplklhd.potenteval, iplklhd.gradeval, iplklhd.hesseval, niter, init_theta, lrate=0.5)
error_newton = np.sqrt(np.sum(np.abs(traj_newton - thetatrue)**2/(thetatrue**2) ,axis=-1))

traj_gd, obj_gd = minimise_gd(iplklhd.potenteval, iplklhd.gradeval, niter, init_theta, lrate=1e-9)
error_gd = np.sqrt(np.sum(np.abs(traj_gd - thetatrue)**2/(thetatrue**2) ,axis=-1))

traj_rs, obj_rs = minimise_rs(iplklhd.potenteval, niter, init_theta, lrate=1e-2)
error_rs = np.sqrt(np.sum(np.abs(traj_rs - thetatrue)**2/(thetatrue**2) ,axis=-1))


benchmark = np.average(np.sqrt(np.diag(iplklhd.ipvar)))
stdev = np.sqrt(ivpvar)


# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(32, 18))
# fig.suptitle("Lotka Volterra")

# ax1.set_title("Error ||thetatrue - x_i||")
# ax1.semilogy(error_rs, linewidth=4,  label="Random Search")
# ax1.semilogy(error_gd, linewidth=4,  label="GD")
# ax1.semilogy(error_newton, linewidth=4,  label="Newton")
# # ax1.semilogy(benchmark * np.ones(error_newton.shape), '--', linewidth=4, label="sqrt(P + sig^2) (%.1e)" % benchmark, color='darkred')
# # ax1.semilogy(stdev * np.ones(error_newton.shape), ':', linewidth=4, label="IP noise sig (%.1e)" % np.sqrt(ivpvar), color='gray')

# plt.title("Negative Log-Likelihoods")
# plt.semilogy(obj_rs, linewidth=4, label="Random Search")
# plt.semilogy(obj_gd, linewidth=4, label="GD")
# plt.semilogy(obj_newton, linewidth=4,  label="Newton")
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()

# plt.show()





print("Newton guess:", traj_newton[-1])
print("GD guess:", traj_gd[-1])
print("RS guess:", traj_rs[-1])
print("Truth:", thetatrue)
print("Init:", init_theta)


plt.style.use("./icmlstyle.mplstyle")

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


ax1.set_xlabel("Iteration")
ax1.set_ylabel("Neg. log-likelihood")
# ax1.set_ylim((1e-310, 1e10))
# plt.semilogy(probs_rw, ':', label="RW")
mark_every = 5
ax1.semilogy((obj_gd), markevery=mark_every, color="gray", ls="-", marker="^", label="GD", alpha=0.8)
ax1.semilogy((obj_newton), markevery=mark_every, color="#999933", ls="-", marker="d", label="NWT", alpha=0.8)
ax1.semilogy((obj_rs), markevery=mark_every, color="#cc6677",   ls="-", marker="s", label="RS", alpha=0.7)
# ax1.set_yticks([1e-300, 1e-200, 1e-100, 1e-0])
# ax1.legend()

ax2.set_xlabel("Iteration")
ax2.set_ylabel("Rel. Error")
# plt.semilogy(probs_rw, ':', label="RW")
ax2.semilogy(np.abs((traj_gd - thetatrue[np.newaxis, :])/thetatrue[np.newaxis, :]).mean(axis=1),  markevery=mark_every, color="gray", ls="-", marker="^", label="GD", alpha=0.8)
ax2.semilogy(np.abs((traj_newton - thetatrue[np.newaxis, :])/thetatrue[np.newaxis, :]).mean(axis=1), markevery=mark_every, color="#999933", ls="-", marker="d", label="NWT", alpha=0.8)
ax2.semilogy(np.abs((traj_rs - thetatrue[np.newaxis, :])/thetatrue[np.newaxis, :]).mean(axis=1), markevery=mark_every, color="#cc6677",  ls="-", marker="s", label="RS", alpha=0.7)

# ax1.spines['bottom'].set_position(('outward', 5))
# ax2.spines['bottom'].set_position(('outward', 5))
ax1.set_title("a", loc="left", fontweight='bold', ha='right')
ax2.set_title("b", loc="left", fontweight='bold', ha='right')

ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.6))
# plt.figlegend(["GD", "NWT", "RS"], loc="lower center", borderaxespad=-0.5, ncol=3)
ax1.minorticks_off()
ax2.minorticks_off()

plt.tight_layout()
plt.savefig("./figures/figure4_optim_left")

plt.show()







