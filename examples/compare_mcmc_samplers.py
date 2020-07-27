# coding=utf-8
"""
compare_mcmc_samplers.py

Compares performance of Metropolis-Adjusted-Langevin-Algorithm
against random walk samplers for Gaussian mixture models.
"""

import numpy as np
from odefilters.sampling import metropolishastings_nd
import matplotlib.pyplot as plt


def mixturemodel(mean1, prob1, mean2, prob2):
    """
    Returns Gaussian mixture model as a mix of two distributions,
        p(x) = a_1 p_1(x) + a_2 p_2(x), a_1 + a_2 = 1.
    Each variance is set to 1.

    Reference:
        https://en.wikipedia.org/wiki/Mix
        ture_model#Gaussian_mixture_model

    Args:
        mean1: mean of first Gaussian distribution
        prob1: a_1
        mean2: mean of second Gaussian distribution
        prob2: a_2
    Raises:
        Error: if a_1 + a_2 != 1
    Returns:
        dens:  PDF of p(x)
        grad:  grad of -log(p(x))
    """
    assert(prob1 + prob2 == 1), "a_1 + a_2 != 1"
    dens1, grad1 = gaussiangradient(mean1)
    dens2, grad2 = gaussiangradient(mean2)
    dens = lambda x: prob1*dens1(x) + prob2*dens2(x)
    grad = lambda x: prob1*grad1(x) + prob2*grad2(x)
    return dens, grad


def gaussiangradient(mean):
    """
    Returns std. Gaussian PDF and
    gradient of negative log-density.
    Covariances are set to 1.0.

    Args:
        mean: mean of Gaussian
    Returns:
        dens: PDF of Gaussian
        grad: Gradient of negative log-density
    """
    dens = lambda x: np.exp(-(x-mean).dot((x-mean).T)/2.)/np.sqrt(2*np.pi)
    grad = lambda x: (x-mean)/np.sqrt(2*np.pi)
    return dens, grad


# Compute densities and gradient of negative log-posterior
shift = 1
dens, gradient = gaussiangradient(shift)

# Pick parameters for MCMC
nsamps = 500
init_state = np.array([20.0])
pwidth = 0.75

# Compute Markov chains
states_rw, __, acc_rw = metropolishastings_nd(nsamps, dens, init_state, pwidth, sampler="rw")
states_lang, __, acc_lang = metropolishastings_nd(nsamps, dens, init_state, pwidth, sampler="lang", grad=gradient)

# Evaluate Gaussian pdf
xval = np.linspace(-4, 4, 500)
yval = np.exp(-(xval-shift)**2/2.)/np.sqrt(2*np.pi)

# Plot Results
plt.title("Langevin vs. Metropolis-Hastings, N=%u states, x0=%.1f" % (nsamps, init_state))
plt.hist(states_rw, bins=50, density=True, alpha=0.5, label="Random Walk")
plt.hist(states_lang, bins=50, density=True, alpha=0.5, label="Langevin")
plt.plot(xval, yval, label="Std-Normal Density")
plt.legend()
plt.show()

# END OF FILE
