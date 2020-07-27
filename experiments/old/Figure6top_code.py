# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from odefilters import odesolver
from odefilters import linearised_odesolver as linsolver
from odefilters import linearised_ode as linode
from odefilters import statespace
from odefilters import inverseproblem as ip
from odesolvers.optimisation import minimise_newton, minimise_gd, minimise_rs

d_guiy = 9

def create_data(solver, ivp, thetatrue, stepsize, ivpvar):
    """
    Create artificial data for the inverse problem.
    """
    ivp.params = thetatrue
    tsteps, m, __, __, __ = solver.solve(ivp, stepsize)
    means = odesolver.get_trajectory_ddim(m, d_guiy, 0)
    evalpts = np.array([h, 1., 2., 4., 5., 7., 10., 15., 20., 30., 40., 50., 60., 80., 100.])
    assert(np.prod(np.in1d(evalpts, tsteps))==1), print(evalpts[np.in1d(evalpts, tsteps)==False])
    noise = np.sqrt(ivpvar)*np.random.randn(len(evalpts) * d_guiy).reshape((len(evalpts), d_guiy))
    evalidcs = [list(tsteps).index(evalpt) for evalpt in evalpts]
    data = means[evalidcs] + noise # this is 'wrong' noise
    ipdata = ip.InvProblemData(evalpts, data, ivpvar)
    return ipdata

# np.random.seed(2)

# set model parameters 

# Set Model Parameters
initial_value = np.ones([d_guiy])
initial_time, end_time = 0., 100.
ivpvar = 1e-5    # maybe change this later?
thetatrue = np.array([ 0.1,  0.01,  0.4,  0.01,  0.3, 0.01,  0.7,  0.01,  0.1,  0.2])
ivp = linode.GUiY(initial_time, end_time, params=thetatrue, initval=initial_value)

# Set Method Parameters
h_for_data = (end_time - initial_time)/10000
h = (end_time - initial_time)/2000
solver = linsolver.LinearisedODESolver(statespace.IBM(q=1, dim=d_guiy))
ipdata = create_data(solver, ivp, thetatrue, h_for_data, ivpvar)
iplklhd = ip.InvProblemLklhd(ipdata, ivp, solver, h, with_jacob=True)

# Make a particle follow a trajectory through the parameter space
niter = 100
init_theta = 1.2 *  np.array([ 0.1,  0.0,  0.4,  0.0,  0.3, 0.0,  0.7,  0.0,  0.1,  0.2]) + np.random.rand(10)*0.01


traj_gd, obj_gd = minimise_gd(iplklhd.potenteval, iplklhd.gradeval, niter, init_theta, lrate=1e-7)
error_gd = np.sqrt(np.sum(np.abs(traj_gd - thetatrue)**2/(thetatrue + 1e-6)**2 ,axis=-1)) # + 1e-6 to avoid division by zero

traj_newton, obj_newton = minimise_newton(iplklhd.potenteval, iplklhd.gradeval, iplklhd.hesseval, niter, init_theta, lrate=1.)
error_newton = np.sqrt(np.sum(np.abs(traj_newton - thetatrue)**2/(thetatrue + 1e-6)**2 ,axis=-1))  # + 1e-6 to avoid division by zero

traj_rs, obj_rs = minimise_rs(iplklhd.potenteval, niter, init_theta, lrate=1e-3)
error_rs = np.sqrt(np.sum(np.abs(traj_rs - thetatrue)**2/(thetatrue + 1e-6)**2 ,axis=-1))

benchmark = np.average(np.sqrt(np.diag(iplklhd.ipvar)))
stdev = np.sqrt(ivpvar)

print("Init:", init_theta)
print("Newton guess:", traj_newton[-1], (np.abs((traj_newton[-1] - thetatrue[:]))/np.abs(thetatrue[:])))
print("Truth:", thetatrue)
print("GD guess:", traj_gd[-1])
print("RS guess:", traj_rs[-1])



print(traj_gd.shape, thetatrue.shape)




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
ax2.semilogy((np.abs((traj_gd - thetatrue[np.newaxis, :]))/np.abs(thetatrue[np.newaxis, :])).mean(axis=1),  markevery=mark_every, color="gray", ls="-", marker="^", label="GD", alpha=0.8)
ax2.semilogy((np.abs((traj_newton - thetatrue[np.newaxis, :]))/np.abs(thetatrue[np.newaxis, :])).mean(axis=1), markevery=mark_every, color="#999933", ls="-", marker="d", label="NWT", alpha=0.8)
ax2.semilogy((np.abs((traj_rs - thetatrue[np.newaxis, :]))/np.abs(thetatrue[np.newaxis, :])).mean(axis=1), markevery=mark_every, color="#cc6677",  ls="-", marker="s", label="RS", alpha=0.7)

# ax1.spines['bottom'].set_position(('outward', 5))
# ax2.spines['bottom'].set_position(('outward', 5))

ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.6))
# plt.figlegend(["GD", "NWT", "RS"], loc="lower center", borderaxespad=-0.5, ncol=3)
ax1.minorticks_off()
ax2.minorticks_off()
ax1.set_title("a", loc="left", fontweight='bold', ha='right')
ax2.set_title("b", loc="left", fontweight='bold', ha='right')

plt.tight_layout()
plt.savefig("./figures/figure6_optim_left")

plt.show()




