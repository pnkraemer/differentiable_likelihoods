# coding=utf-8
"""
Check the performance of Langevin MCMC
and Random Walk Metropolis-Hastings
for a simple inverse problem.

DISCLAIMER:
    * I am not sure about the readability of this script.
"""

import numpy as np
import scipy.linalg as scl

from odefilters import covariance as cov
from odefilters import odesolver
from odefilters import linearised_odesolver as linsolver
from odefilters import linearised_ode as linode
from odefilters import statespace
from odefilters import linearisation
from odefilters import auxiliary as aux

import code
from odefilters.sampling import metropolishastings_nd 
import matplotlib.pyplot as plt
import functools as ft

np.seterr(all='raise')      # debugging: all warnings are errors

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
    def __init__(self, evalpt, data, solver, ivp, stepsize, kprefact, with_jacob=False):
        self.kernel_prefact = kprefact
        self.solver = solver
        self.ivp = ivp
        self.stepsize = stepsize
        self.data = data
        self.with_jacob = with_jacob
        self.evalpt = evalpt

    def prioreval(self, theta):
        # if np.amin(theta) > 0:# and np.amax(theta) < 4.5:
        #     return 1
        # else:
        #     return 0
        return 1

    def lklhdeval(self, theta):
        """
        Evaluate likelihood function of inverse problem.

        Args:
            theta:  ODE-parameter we want to solve with
        Returns:
            lklhd:  evaluation of likelihood function at theta
        """
        self.mean, self.vari, self.jacob = self.forwardsolve(theta)
        lklhd = gaussianmd(self.data, self.mean, self.vari)
        return np.array([lklhd]) * self.prioreval(theta)


    def gradeval(self, theta):
        """
        Evaluate gradient of NEGATIVE LOG LIKELIHOOD.

        Args:
            theta:  ODE-parameter we want to solve with
        Returns:
            grad:   evaluation of gradient at theta
        """
        assert(self.with_jacob is True), "Please set with_jacob to True."
        # print(self.mean.shape, self.vari)
        v = np.diag(self.vari.flatten())
        m = (self.mean.flatten())
        z = (self.data.flatten())
        grad = np.dot(self.jacob.T, np.linalg.solve(v, (m - z)))
        # print(0.1 * grad[:, 0])
        # return 0.1 * grad[:, 0] * self.prioreval(theta)
        return 0.2 * grad * self.prioreval(theta)

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
        # if isinstance(theta, np.ndarray) is True:
        if len(theta.shape) == 2:
            theta = theta[0]
        self.ivp.params = theta
        tsteps, m, v, rhs_parts, __ = self.solver.solve(self.ivp, self.stepsize)
        mean1 = odesolver.get_trajectory_ddim(m, 2, 0)
        mean = return_corresponding_index(tsteps, self.evalpt, mean1)
        unc1 = odesolver.get_trajectory_ddim(v, 2, 0)
        unc = return_corresponding_index(tsteps, self.evalpt, unc1)

        vari = (ivpnoise + unc)
        if self.with_jacob is True:
            jacob = linearisation.compute_jacobian(self.evalpt, tsteps, self.kernel_prefact, rhs_parts)
        else:
            jacob = None 
        return mean, vari, jacob


def gaussianmd(z, m, p):
    """
    Evaluates md Gaussian PDF with mean m and variance p at point z.
    """
    # print(z, z.flatten())
    z = z.flatten()
    m = m.flatten()
    p = np.diag(p.flatten())
    pt = z.reshape((1, len(z.flatten())))
    meanvec = m
    covar = p
    return aux.gaussian_pdf(pt, meanvec.flatten(), covar)

def gaussian1d(z, m, p):
    """
    Evaluates 1d Gaussian PDF with mean m and variance p at point z.
    """
    if np.abs(z - m)**2/p > 100:
        return 0.
    else:
        return np.exp(-0.5*np.abs(z - m)**2/p)/np.sqrt(2*np.pi*p)


def compute_kernel_prefactor(evalpt, tsteps, q):
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
    evalpt = np.array([1., 2., 3., 4., 5.])
    assert(np.in1d(evalpt, tsteps).prod()== 1)
    a1 = list(tsteps).index(evalpt[0])
    a2 = list(tsteps).index(evalpt[1])
    a3 = list(tsteps).index(evalpt[2])
    a4 = list(tsteps).index(evalpt[3])
    a5 = list(tsteps).index(evalpt[4])
    means = odesolver.get_trajectory_ddim(m, 2, 0)
    data = means[[a1, a2, a3, a4, a5]] + noise
    return evalpt, data


# class LotkaLike(linode.LinearisedODE):
#     """
#     LotkaVolterra with 2 parameters instead of 4.
#     Second and third are fixed, first and fourth are flexible.
#     Behaves like LV but isnt.
#     """
#     def __init__(self, t0, tmax, fixparams, flexparams, initval, initval_unc=0.0):
#         self.fixparams = fixparams
#         self.flexparams = flexparams
#         params = np.array([flexparams[0], fixparams[0], fixparams[1], flexparams[1]])
#         # self.params = params
#         self.lotka = linode.LotkaVolterra(t0, tmax, params, initval, initval_unc)
#         linode.LinearisedODE.__init__(self, t0, tmax, params, initval, initval_unc)


#     def modeval(self, t, x):
#         """
#         Returns (rhs_parts.T @ self.params).T instead of
#         self.params @ rhs_parts in order to be able to multiply
#         a (2,) array with a (2, 3, 4) array to obtain a (3, 4) array
#         which is important for vectorised evaluations.
#         """
#         rhs_parts = self.modeval_parts(t, x)
#         return self.rhs_parts_into_eval(rhs_parts)


#     def rhs_parts_into_eval(self, rhs_parts):
#         """
#         Computes inner product of params and rhs_parts.
#         """
#         return np.dot(rhs_parts.T, self.lotka.params).T


#     def modeval_parts(self, t, x):
#         # print(self.params)
#         self.lotka.params = np.array([self.flexparams[0], self.fixparams[0], self.fixparams[1], self.flexparams[1]])
#         return self.lotka.modeval_parts(t, x)





np.random.seed(1)

# Set Model Parameters
initial_value = np.array([20, 20])
initial_time = 0.
end_time = 5
ivpnoise = 0.0000
thetatrue = np.array([1, 0.01, 0.02, 1])



# Set Method Parameters
q = 1
h = (end_time - initial_time)/100
nsamps = 1
init_theta = np.array([1.5, 0.05, 0.2, 2.])
ibm = statespace.IBM(q=q, dim=2)
solver = linsolver.LinearisedODESolver(ibm)
pwidth = 0.006


# ntsteps = ((end_time - initial_time)/h) + 1
# evalidcs = [-int(ntsteps/2)-1, -1]


# Create Data and Jacobian
ivp = linode.LotkaVolterra(initial_time, end_time, params=thetatrue, initval=initial_value, initval_unc=0.0)
print(ivp.params)
evalpt, data = create_data(solver, ivp, thetatrue, 1e-04, ivpnoise)

# print(evalpt, data)
tsteps, __, __, __, __ = solver.solve(ivp, stepsize=h)
# evalpt = np.array(tsteps[[-1]])
# ntsteps = ((end_time - initial_time)/h) + 1
# evalidcs = [-int(ntsteps/2)-1, -1]

# print(evalpt)
kernel_prefactor = linearisation.compute_kernel_prefactor(ibm, 0., tsteps, evalpt)

print()
# Sample states from Markov chain
ivp.params = init_theta
lklhd_grad = InvProblemLklhd(evalpt, data, solver, ivp, h, kernel_prefactor, with_jacob=True)
lklhd_nograd = InvProblemLklhd(evalpt, data, solver, ivp, h, kernel_prefactor)
# lang_thetas, lang_guesses = metropolishastings(nsamps, lklhd_grad, init_theta, pwidth, "lang")
# rw_thetas, rw_guesses = metropolishastings(nsamps, lklhd_nograd, init_theta, pwidth, "rw")




# Draw a grid

# Draw a grid
delta = 0.0001
xpts = np.arange(0.009, 0.011, delta)
ypts = np.arange(0.019, 0.021, delta)
X, Y = np.meshgrid(xpts, ypts)
lklgrid = np.zeros(X.shape)
gradgrid1 = np.zeros(X.shape)
gradgrid2 = np.zeros(X.shape)
diffquot1 = np.zeros(X.shape)
diffquot2 = np.zeros(X.shape)
for i in range(len(X)):
    for j in range(len(X.T)):
        this_theta = np.array([1., X[i, j], Y[i, j], 1])
        lklgrid[i, j] = lklhd_grad.lklhdeval(this_theta)
        gradvl = lklhd_grad.gradeval(this_theta)
        gradgrid1[i, j] = np.sign(gradvl[1])#*np.log10(np.abs((gradvl[0])**4))
        gradgrid2[i, j] = np.sign(gradvl[2])#*np.log10(np.abs((gradvl[1])**4))
        print(this_theta[[0, 2]], gradgrid1[i, j], gradgrid2[i, j])



# diffquot2[-1, :] = lklgrid[0, :]
# diffquot1[:, -1] = lklgrid[:, 0]


lklgrid += np.finfo(float).eps**2

thresh = 0

for i in range(1, len(X)-1):
    for j in range(1, len(X.T)-1):
        if lklgrid[i+1, j] < thresh or lklgrid[i-1, j] < thresh:
            diffquot2[i, j] = 0
        else:
            diffquot2[i, j] = -0.5*(-np.log(lklgrid[i+1, j]) + np.log(lklgrid[i-1, j]))/delta
        if lklgrid[i, j+1] < thresh or lklgrid[i, j-1] < thresh:
            diffquot1[i, j] = 0
        else:
            diffquot1[i, j] = -0.5*(-np.log(lklgrid[i, j+1]) + np.log(lklgrid[i, j-1]))/delta






# diffquot2[0:2, :] = 0
# diffquot2[:, 0:2] = 0
# diffquot1[0:2, :] = 0
# diffquot1[:, 0:2] = 0
# gradgrid1[0:2, :] = 0
# gradgrid1[:, 0:2] = 0
# gradgrid2[0:2, :] = 0
# gradgrid2[:, 0:2] = 0







# X, Y = np.meshgrid(xpts, ypts)
# gradgrid1 = np.zeros(X.shape)
# gradgrid2 = np.zeros(X.shape)


# Evaluate gradients again for coarser stepsize
# h = (end_time - initial_time)/100
# ivp = linode.LotkaVolterra(initial_time, end_time, params=thetatrue, initval=initial_value, initval_unc=0.0)
# print(ivp.params)
# evalpt, data = create_data(solver, ivp, thetatrue, 1e-04, ivpnoise)

# # print(evalpt, data)
# tsteps, __, __, __, __ = solver.solve(ivp, stepsize=h)
# # evalpt = np.array(tsteps[[-1]])
# # ntsteps = ((end_time - initial_time)/h) + 1
# # evalidcs = [-int(ntsteps/2)-1, -1]

# # print(evalpt)
# kernel_prefactor = linearisation.compute_kernel_prefactor(ibm, 0., tsteps, evalpt)

# print()
# # Sample states from Markov chain
# ivp.params = init_theta
# lklhd_grad2 = InvProblemLklhd(evalpt, data, solver, ivp, h, kernel_prefactor, with_jacob=True)

# for i in range(len(X)):
#     for j in range(len(X.T)):
#         this_theta = np.array([X[i, j], Y[i, j], 0.02, 1])
#         dummy = lklhd_grad2.lklhdeval(this_theta)
#         gradvl = lklhd_grad2.gradeval(this_theta)
#         gradgrid1[i, j] = -1*gradvl[0]
#         gradgrid2[i, j] = -1*gradvl[1]



# gradgrid1 = np.sign(gradgrid1)
# gradgrid2 = np.sign(gradgrid2)
diffquot1 = np.sign(diffquot1)
diffquot2 = np.sign(diffquot2)





# gradgrid1[0:2, :] = 0
# gradgrid1[:, 0:2] = 0
# gradgrid2[0:2, :] = 0
# gradgrid2[:, 0:2] = 0

# gradgrid1[-1, :] = 0
# gradgrid1[:, -1] = 0
# gradgrid2[-1, :] = 0
# gradgrid2[:, -1] = 0




# plt.scatter(lang_thetas[:, 0], lang_thetas[:, 1], c=range(len(lang_thetas)), cmap='Reds')
plt.style.use("seaborn-whitegrid")
# plt.title("N = %r Samples; LANG (blue) and RW (red)" % nsamps)
# plt.scatter(rw_thetas[:, 0], rw_thetas[:, 1], c=range(len(rw_thetas)), vmin=0, vmax=0.5*len(rw_thetas), cmap='Reds')
# plt.scatter(lang_thetas[:, 0], lang_thetas[:, 1], c=range(len(rw_thetas)), vmin=0, vmax=0.5*len(lang_thetas), cmap='Blues')
# plt.scatter(thetatrue[1], thetatrue[2], marker='*', s=400, color='green', label="True value")
# plt.scatter(init_theta[0], init_theta[1], marker='*', s=400, color='gray', label="Starting point")
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=2)
plt.title("Sign gradient estimates")
CS = plt.contour(X, Y, lklgrid, cmap='Greens')
plt.clabel(CS, inline=True, fmt='%.1f')
plt.quiver(X, Y, gradgrid1, gradgrid2, width=0.125*1e-2, alpha=0.75)
# plt.quiver(X, Y, 0.1*diffquot1, 0.1*diffquot2, scale=1, color='orange', width=0.125*1e-2, alpha=0.75)

# plt.xlim((0.5, 4.5))
# plt.ylim((0.5, 4.5))
plt.show()