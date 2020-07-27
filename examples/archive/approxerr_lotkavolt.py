# coding=utf-8
"""

approxerr_lotkavolt.py

We compare the convergence behaviour of zeroth, first and second order
methods applied to the negative log-likelihood of a Gaussian distribution.

RESULT:
    * The second order method seems to be able
    to do anything that the first order method
    can do and usually it does it better.
    * I conjecture that far away from the truth 
    the inverse Hessian has troubles because the
    sensitivity in the Jacobian of m_par(t) appears
    in a quadratic way. However, that is hard to falsify
    because far away from the truth, neither works
    well---at least for fairly low precisions, etc.
    * The random search is just there to show how insanely
    difficult this inverse problem is, optimisation-wise
    (the likelihood is zero almost everywhere, so what would
    a reasonable acceptance step be?). I find it not surprising
    that it doesnt converge at all.

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


####################################
# Debugging: all warnings are errors
# If active, loglog plots don't work
# for some weird reason.
####################################
# np.seterr(all='raise')

def metropolishastings(nsamps, iplklhd, init_state, pwidth, sampler):
    """
    Wrapper for metropolishastings_nd()
    using the InvProblemLklhd objects in 1d.

    Args:
        nsamps:     number of states to be sampled
        iplklhd:    InvProblemLklhd object
        init_state: initial state, np.ndarray of shape (1,)
        pwidth:     proposal width
        sampler:    Use random walk or Langevin for proposals,
                    options: {"rw", "lang"}. Default is "rw".
    Raises:
        Warning:    if acceptance ratio is not in interval (0.5, 0.7)
    Returns:
        states:     N states of Markov chain according to MH, shape (N,)
        probs:      (potentially unnormalised) probabilities of states
    """
    if sampler == "lang":
        samples, probs, ratio = metropolishastings_nd(nsamps,
            iplklhd.lklhdeval, init_state, pwidth, sampler, iplklhd.gradeval)
    else:
        samples, probs, ratio = metropolishastings_nd(nsamps,
            iplklhd.lklhdeval, init_state, pwidth, sampler)
    if ratio < 0.5 or ratio > 0.7:
        print("Careful: ratio of %s is off (ratio=%.2f)" % (sampler, ratio))
    return samples, probs


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
ivpvar = .1
thetatrue = np.array([1.0, 0.1, 0.1, 1.0])
ivp = linode.LotkaVolterra(initial_time, end_time, params=thetatrue, initval=initial_value)

# Set Method Parameters
h_for_data = (end_time - initial_time)/10000
h = (end_time - initial_time)/500
solver = linsolver.LinearisedODESolver(statespace.IBM(q=1, dim=2))
ipdata = create_data(solver, ivp, thetatrue, h_for_data, ivpvar)
iplklhd = ip.InvProblemLklhd(ipdata, ivp, solver, h, with_jacob=True)

# Make a particle follow a trajectory through the parameter space
niter = 200
init_theta = np.array([0.8, .2, .05, 1.1])

# 1. Second order optimisation (Newton)
traj = np.zeros((niter, 4))
traj[0] = init_theta
lklhds = np.zeros(len(traj))
gradvls = np.zeros((len(traj), 4))
invhessprods = np.zeros((len(traj), 4))
invhessprod = 0
lrate = 0.5
for idx in range(1, len(traj)):
    this_theta = traj[idx-1]
    lklhds[idx-1] = iplklhd.lklhdeval(this_theta)
    gradvl = iplklhd.gradeval(this_theta)
    hessvl = iplklhd.hesseval(this_theta)
    invhessprod = np.linalg.solve(hessvl, gradvl)
    traj[idx] = traj[idx-1] - lrate*invhessprod
    invhessprods[idx-1] = invhessprod
error_hess = np.sqrt(np.sum(np.abs(traj - thetatrue)**2/(thetatrue**2),axis=-1))
last_iter = np.round(traj[-1], 3)
hessnorms = np.sqrt(np.sum(np.abs(invhessprods[:-1])**2,axis=-1))

# 2. First order optimisation (GD)
traj_grad = np.zeros((niter, 4))
traj_grad[0] = init_theta
lklhds_grad = np.zeros(len(traj))
gradvls_grad = np.zeros((len(traj), 4))
gradvl = 0
for idx in range(1, len(traj_grad)):
    this_theta_grad = traj_grad[idx-1]
    lklhds_grad[idx-1] = iplklhd.lklhdeval(this_theta_grad)
    gradvl_grad = iplklhd.gradeval(this_theta_grad)
    traj_grad[idx] = traj_grad[idx-1] - 1e-7*gradvl_grad
    gradvls_grad[idx-1] = gradvl_grad
error_grad = np.sqrt(np.sum(np.abs(traj_grad - thetatrue)**2/(thetatrue**2),axis=-1))
last_iter_grad = np.round(traj_grad[-1], 3)
gradnorms = np.sqrt(np.sum(np.abs(gradvls_grad[:-1])**2,axis=-1))

# 3. Zeroth order optimisation (Random search; RS)
traj_rs = np.zeros((niter, 4))
lklhds_rs = np.zeros(len(traj))
traj_rs[0] = init_theta
lrate = np.linalg.norm(init_theta - thetatrue) # it could hardly be better
lklhd_proposal, lklhd_old_state = 0, 0
for idx in range(1, len(traj_rs)):
    old_state = traj_rs[idx-1]
    sample = np.random.randn(4)
    proposal = old_state + lrate * sample/np.linalg.norm(sample)
    lklhd_proposal = iplklhd.lklhdeval(proposal)
    if lklhd_proposal >= lklhd_old_state:   # >= to accept if both are zero
        old_state = proposal
        lklhd_old_state = lklhd_proposal
    traj_rs[idx] = old_state
    lklhds_rs[idx] = lklhd_old_state
error_rs = np.sqrt(np.sum(np.abs(traj_rs - thetatrue)**2/(thetatrue**2), axis=-1))
last_iter_rs = np.round(traj_rs[-1], 3)



# Visualisation of the Results
plt.style.use('bmh')
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True)
fig.suptitle("Truth: %r, Init: %r | h=%.1e" % (list(thetatrue), list(init_theta), h))
benchmark = np.sqrt(np.trace(iplklhd.ipvar))/len(iplklhd.ipvar) * np.ones(hessnorms.shape)


ax1.set_title("H^{-1}df\nGuess %r" % list(last_iter))
ax1.semilogy(hessnorms, label="||H^{-1}df||", color='darkblue')
# ax1.semilogy(lklhds[:-1], label="Likelihoods") # comment out if values distort the plot
ax1.semilogy(error_hess, label="Rel. Error", color='darkorange')
ax1.semilogy(benchmark, '--', label="sqrt(tr(P + sig^2))/N", color='black')
ax1.legend()

ax2.set_title("df\nGuess %r" % list(last_iter_grad))
ax2.semilogy(gradnorms, label="||df||", color='darkblue')
# ax2.semilogy(lklhds_grad[:-1], label="Likelihoods") # comment out if values distort the plot
ax2.semilogy(error_grad, label="Rel. Error", color='darkorange')
ax2.semilogy(benchmark, '--', label="sqrt(tr(P + sig^2))/N", color='black')
ax2.legend()

ax3.set_title("Random Search\nGuess %r" % list(last_iter_rs))
# ax3.semilogy(lklhds_rs[:-1], label="Likelihoods") # comment out if values distort the plot
ax3.semilogy(error_rs, label="Rel. Error", color='darkorange')
ax3.semilogy(benchmark, '--', label="sqrt(tr(P + sig^2))/N", color='black')
ax3.legend()

plt.show()
