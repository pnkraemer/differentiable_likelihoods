# coding=utf-8
"""
Check the performance of Langevin MCMC
and Random Walk Metropolis-Hastings
for a simple inverse problem.

DISCLAIMER:
    * I am not sure about the readability of this script.
"""

import numpy as np
from odefilters import covariance as cov
from odefilters import odesolver
from odefilters import linearised_odesolver as linsolver
from odefilters import linearised_ode as linode
from odefilters import statespace
import code
from odefilters.sampling import metropolishastings_nd 
import matplotlib.pyplot as plt
import functools as ft

def metropolishastings(nsamps, iplklhd, init_state, pwidth, sampler):
    """
    Wrapper for metropolishastings_nd()
    using the InvProblemLklhd objects in 2d.

    Args:
        nsamps:     number of states to be sampled
        iplklhd:    InvProblemLklhd object
        init_state: initial state, np.ndarray of shape (2,)
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
    # return samples[:, 0], probs[:, 0]
    return samples, probs


class InvProblemLklhd():
    """
    Likelihood object for ODE-based inverse problems in 1d.

    Args/Attributes:
        data:       measurement of inverse problem
        solver:     ODESolver object, see odesolver.py
        ivp:        ODE (dict), see ode.py
        stepsize:   stepsize for solving ODE
        kprefact:   kernel-prefactor matrix for computation of Jacobians
        with_jacob: Do we compute Jacobians or not? Default is False.
    Further attributes:
        mean:       solver-based guess for data, used for both gradient
                    evaluations and pdf evaluations. Stored as attribute
                    bc. we only want to solve the ODE once.
        vari:       ivp-based and solver-based uncertainty,
                    used together with self.mean
        jacob:      Guess for jacobian, corresponds to mean and vari but
                    exclusively used for gradient evaluations
    Methods:
        lklhdeval:  evaluate likelihood function of inverse problem
        forwardsolve: evaluate forward map of inverse problem
        gradeval:   evaluate gradient of negative log-likelihood
    """
    def __init__(self, data, solver, ivp, stepsize, kprefact, with_jacob=False):
        self.kernel_prefact = kprefact
        self.solver = solver
        self.ivp = ivp
        self.stepsize = stepsize
        self.data = data
        self.with_jacob = with_jacob

    def lklhdeval(self, theta):
        """
        Evaluate likelihood function of inverse problem.

        Args:
            theta:  ODE-parameter we want to solve with
        Returns:
            lklhd:  evaluation of likelihood function at theta
        """
        self.mean, self.vari, self.jacob = self.forwardsolve(theta)
        lklhd = gaussian1d(self.data, self.mean, self.vari)
        return np.array([lklhd])

    def forwardsolve(self, theta):
        """
        Solves ODE, returns mean and uncertainty
        as well as Jacobian.

        Args:
            theta:  ODE-parameter we want to solve with
        Returns:
            mean:   filter mean at time end_time
            vari:   ivp-variance plus filter variance at time end_time
            jacob:  Jacobian estimate at theta.
                    Only computed if Jacobians are specified through
                    with_jacob = True. Otherwise None.
        """
        if len(theta.shape) == 2:
            theta = theta[0]
        ivp.params = theta
        __, m, v, r, __ = self.solver.solve(self.ivp, self.stepsize)
        mean = odesolver.get_trajectory(m, 0, 0)[-1]
        unc = odesolver.get_trajectory(v, 0, 0)[-1]
        vari = ivpnoise + unc
        y = r[:, :, 0]- self.ivp.initval[:, np.newaxis]
        # print(y.shape)
        if self.with_jacob is True:
            # print(self.kernel_prefact.dot(y).shape)
            jacob = self.kernel_prefact.dot(y) + self.ivp.initval*(self.ivp.tmax - self.ivp.t0)
        else:
            jacob = None 
        return mean, vari, jacob

    def gradeval(self, theta):
        """
        Evaluate gradient of negative log likelihood.

        Args:
            theta:  ODE-parameter we want to solve with
        Returns:
            grad:   evaluation of gradient at theta
        """
        assert(self.with_jacob is True), "Please set with_jacob to True."
        grad = self.jacob * np.abs(self.mean - self.data) / self.vari
        print(grad)
        return grad


def gaussian1d(z, m, p):
    """Evaluates 1d Gaussian PDF"""
    return np.exp(-0.5*np.abs(z - m)**2/p)/np.sqrt(2*np.pi*p)


def prepare_jacobian(evalpt, tsteps, q):
    """
    Prepares everything necessary for 
    the computation of the Jacobian.
    """
    kdell = cov.ibm_covd(evalpt, tsteps, q)
    dellkdell = cov.ibm_dcovd(tsteps, tsteps, q)
    kernel_prefactor = kdell.dot(np.linalg.inv(dellkdell))
    return kernel_prefactor


def create_data(solver, ivp, thetatrue, stepsize, ivpnoise):
    """
    Create artificial data for inverse problem.
    """
    noise = np.sqrt(ivpnoise)*np.random.randn()
    ivp.params = thetatrue
    tsteps, m, __, __, __ = solver.solve(ivp, stepsize)
    evalpt = np.array([tsteps[-1]])
    means = odesolver.get_trajectory(m, 0, 0)
    data = means[-1] + noise
    return evalpt, data

import matplotlib.pyplot as plt
import sys
np.random.seed(1)

# Set Model Parameters
initial_value = 0.1
initial_time = 0.01
end_time = 1.1
ivpnoise = 0.01
thetatrue = np.array([1., 4.])
thetatrue_kindof = np.array([1., 1./4.])


# Set Method Parameters
q = 1
h = 0.2
nsamps = 60
init_theta = np.array([1., 1.])
ibm = statespace.IBM(q=q, dim=1)
solver = linsolver.LinearisedODESolver(ibm)

# Create Data and Jacobian
ivp = linode.LogisticODE(initial_time, end_time, params=thetatrue, initval=initial_value, initval_unc=0.0)
evalpt, data = create_data(solver, ivp, thetatrue_kindof, stepsize=1e-04, ivpnoise=ivpnoise)
tsteps, m, __, __, __ = solver.solve(ivp, stepsize=h)
kernel_prefactor = prepare_jacobian(evalpt, tsteps, q=q)

# m = odesolver.get_trajectory(m, 0, 0)
# plt.plot(tsteps, m)
# plt.show()
# sys.exit()

# Sample states from Markov chain
ivp.params = init_theta
lklhd_grad = InvProblemLklhd(data, solver, ivp, h, kernel_prefactor, with_jacob=True)
lklhd_nograd = InvProblemLklhd(data, solver, ivp, h, kernel_prefactor)
lang_thetas, lang_guesses = metropolishastings(nsamps, lklhd_grad, init_theta, 0.002, "lang")
rw_thetas, rw_guesses = metropolishastings(nsamps, lklhd_nograd, init_theta, 0.002, "rw")


# Plot states
plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey="all")
fig.suptitle("Metropolis-Hastings for a Simple Inverse Problem", fontsize=16)


# Plot Langevin States
ax1.set_title("Langevin")
ax1.scatter(lang_thetas[:, 0], lang_thetas[:, 1], marker='.')
ax1.scatter(thetatrue_kindof[0], thetatrue_kindof[1], marker='x')
# ax1.hist(lang_thetas, density=True, alpha=0.1)
# ax1.set_xlim((-1, 2))
# ax1.set_ylim((-1, 3))


# Plot RW States
ax2.set_title("Random Walk")
ax2.scatter(rw_thetas[:, 0], rw_thetas[:, 1], marker='.')
ax2.scatter(thetatrue_kindof[0], thetatrue_kindof[1], marker='x')
# ax2.hist(rw_thetas, density=True, alpha=0.1, label="Histogram of N=%u Samples" % len(rw_thetas))
# ax2.set_xlim((-1, 2))
# ax2.set_ylim((-1, 3))

plt.show()
