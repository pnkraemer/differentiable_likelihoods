"""
efficient_sampling.py

This module contains sampling algorithms for Hamiltonian
Monte Carlo and Metropolis-Hastings (Langevin and random walk)

Reference for Metropolis-Hastings:
    https://arxiv.org/pdf/1801.02309.pdf

Reference for Hamiltonian:
    Chapter 30 of
    "Information theory, inference and learning algorithms", David Mackay.

Example:
    >>> import numpy as np
    >>> from sampling import metropolishastings_nd
    >>> x0 = np.array([1.0])
    >>> gausspdf = lambda x: np.exp(-0.5*x @ x.T)/np.sqrt(2*np.pi)
    >>> states, ratio = metropolishastings_nd(nsamps=100, statlogpdf=gausspdf, init_state=x0)
    >>> print(states, ratio)

Todo:
    * Preconditioning for MALA
    * Make Hamiltonian MCMC part of the metropolis hastings samplers
    * Unittest Hamiltoniam MCMC
    * What I call here trapezoidal rule: is it a leapfrog?
    * I am not sure whether the sign of langevin sampler is wrong (in sample_langevin())
    * Is there a 1d as well as a 2d unittest for Langevin?
    * Rewrite everything in terms of only considering stationary distributions
    of the form p(z) = exp(-E(z)) and evaluations of E(z). Check that this can
    still be used reliably for any distributions. Working with E(z) only
    instead of exp(-E(z)) will increase numerical stability/precision in areas
    where E(z) is large. (Python thinks---understandably---that
    np.exp(-1000)=0) holds. Further, np.exp(-40) is below machine precision.)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class MCMCState:
    """
    An MCMC-State is a triple (x, l(x), g(x)) with
        x:      spatial location
        l(x):   evaluation of stationary PDF at x_i
        g(x):   evaluation of negative log-pdf,
                    f(s) = -grad(log(p(f))
    """
    state:      np.ndarray
    loglkldeval:   np.ndarray
    gradeval:   np.ndarray

# ######################################################################
# Hamiltonian Monte Carlo
# ######################################################################

def hamiltonian_nd(nsamps, eval_ptntl, init_state, stepsize,
                   ntraps, eval_gradptntl):
    """
    Implementation strongly inspired by p. 388 in 
    "Information theory, inference and learning algorithms"
    by David MacKay.

    Note 1:
        This version is merely ad-hoc, that is, soon it will be made
        part of the metropolishastings_nd function. 

    Note 2:
        Algorithms assumes that kinetic term is
            K(momentum) = 0.5 * momentum.T @ momentum

    Args:
        eval_ptntl:      callable, evaluate Potential E(x) = -log(l(x))
        eval_gradptntl:  callable, evaluate gradient of Potential
        nsamps:          int, number of samples
        ntraps:          int, number of trapez. steps for solving diffew
        stepsize:        float, stepsize for diffeq
        init_state:      shape (d,), initial state of the Markov chain
    Returns:
        states:          shape (nsamps, d), nsamps states of Markov chain.
    """
    assert(init_state_compliant(init_state) is True)
    currstate = init_currstate(init_state, eval_ptntl, eval_gradptntl)
    ndim = currstate.state.shape[0]
    states = np.zeros((nsamps, ndim))
    probs = np.zeros(nsamps)    
    states[0] = currstate.state
    accepted = 0
    for idx in range(nsamps): 
        momtm = np.random.randn(ndim)
        curr_hamilt = evaluate_hamiltonian(momtm, currstate)
        momtm, proposal = compute_dynamics(momtm, currstate, ntraps,
                                           stepsize, eval_gradptntl, eval_ptntl)
        # print("\t\tproposal:", proposal.state)
        prop_hamilt = evaluate_hamiltonian(momtm, proposal)
        if hamiltonian_accepted(prop_hamilt, curr_hamilt):
            currstate = proposal
            accepted += 1
        states[idx], probs[idx] = currstate.state, currstate.loglkldeval
    return states, probs, accepted/nsamps





def compute_dynamics(momentum, currstate, ntraps, stepsize, gradE, findE):
    """
    Approximates solution to Hamiltonian dynamics ODE,
        x' = d_p H, p' = -d_x H
    via trapezoidal rule.

    Args:
        momentum:   shape (ndim,)
        currstate:  MCMCState object, current state
        ntraps:     number of steps of trapezoidal rule
        stepsize:   stepsize for trapezoidal rule
        gradE:      callable, evaluate gradient of potential
                    (potential is negative log likelihood)
        findE:      callable, evaluate potential
                    (potential is negative log likelihood)
    Returns:
        momentum:   shape (ndim,), updated momentum after ntraps iterations
        proposal:   MCMCState object, updated proposal
    """
    proposal = MCMCState(currstate.state, 0, currstate.gradeval)
    for tau in range(ntraps):
        momentum, proposal = compute_next_trapstep(momentum, proposal,
                                                   stepsize, findE, gradE)
    proposal.loglkldeval = np.exp(-findE(proposal.state))
    return momentum, proposal


def compute_next_trapstep(momentum, proposal, stepsize, findE, gradE):
    """
    Computes next trapezoidal step. 
        1. Half step in momentum
        2. Step in spatial variable
        3. Find new gradient
        4. Half step in momentum

    Args:
        momentum:   shape (ndim,)
        proposal:   MCMCState object, current proposal
        stepsize:   stepsize used for trapezoidal rule
        gradE:      callable, evaluate gradient of potential
                    (negative log likelihood)
    Returns:
        momentum:   momentum after trapezoidal step
        proposal:   updated proposal
    """
    momentum = momentum - stepsize * proposal.gradeval / 2
    proposal.state = proposal.state + stepsize * momentum
    findE(proposal.state)
    proposal.gradeval = gradE(proposal.state)
    momentum = momentum - stepsize * proposal.gradeval / 2
    return momentum, proposal


def evaluate_hamiltonian(momentum, state):
    """
    Evaluates H(x, p) = K(p) + E(x), K(p) = p.T @ p / 2

    Args:
        momentum:   shape (ndim,); this is 'p'
        state:      MCMCState, we need the lklhd-evaluation
    Returns:
        scalar, H(x, p) according to formula above
    """
    return 0.5 * momentum.T @ momentum + np.exp(-state.loglkldeval)


def hamiltonian_accepted(hamilt_new, hamilt):
    """
    Checks whether the acceptance condition for HMC is met,
        H_new < H or u < exp(-H_new) / exp(-H).

    Args:
        hamilt_new: value of Hamiltonian after dynamics
        hamilt:     value of Hamiltonian before dynamics
    Returns:
        True of False (is acceptance condition met?)
    """
    hamilt_diff = hamilt_new - hamilt
    # print(hamilt_diff)
    if hamilt_diff < 0 or np.random.rand() < hamilt_diff:
        # print("Accet")
        return True
    else:
        return False


# ######################################################################
# Metropolis Hastings
# ######################################################################

def metropolishastings_nd(nsamps, statlogpdf, init_state, pwidth=0.5,
                          sampler="rw", grad=None):
    """
    Samples Markov chain in N dimensions
    according to the Metropolis-Hastings algorithm.

    Terminology, e.g.:
        https://theclevermachine.wordpress.com/2012/10/
        20/mcmc-the-metropolis-hastings-sampler/

    Args:
        nsamps:     number of states to be sampled
        statlogpdf:    PDF of stationary distribution,
                    callable: statlogpdf(x)
        init_state: initial state, np.ndarray of shape (d,).
                    This controls the dimension of the output.
        pwidth:     proposal width, default is 0.5
        sampler:    Use random walk or Langevin for proposals,
                    options: {"rw", "lang"}. Default is "rw".
        grad:       gradient of negative log-posterior density,
                    optional. Default is None.
    Returns:
        states:     N states of Markov chain according to MH, shape (N, d)
        probs:      (potentially unnormalised) probabilities of states,
                    shape (N,)
        ratio:      ratio of accepted states vs. total number of states.
    """
    assert(init_state_compliant(init_state) is True)
    proposalfct = turn_sampler_into_function(sampler)
    states = np.zeros((nsamps, len(init_state)))    
    probs = np.zeros(nsamps)    
    accepted = 0
    currstate = init_currstate(init_state, statlogpdf, grad)
    for idx in range(nsamps):
        proposal, corrfact = proposalfct(currstate, statlogpdf, pwidth, grad)
        # print("\t\tProposal:", proposal.state)
        currstate, is_accept = accept_or_reject(proposal, currstate, corrfact)
        # print(currstate.loglkldeval)
        states[idx], probs[idx] = currstate.state, np.exp(-currstate.loglkldeval)
        if is_accept is True:
            accepted += 1
    ratio = accepted/nsamps
    return states, probs, ratio


def init_state_compliant(init_state):
    """
    Checks whether init_state is compliant with an Nd algorithm.
    That is, whether init_state is an (d,) np.ndarray.
    """
    assert(isinstance(init_state, np.ndarray)), \
        "Please enter init_state of shape (d,)"
    assert(len(init_state.shape) == 1), \
        "Please enter init_state of shape (d,)"
    return True


def turn_sampler_into_function(sampler):
    """
    Interprets 'sampler' keyword and returns corresponding proposal kernel.

    Args:
        sampler:        either "rw" or "lang"
    Raises:
        AssertionError: if neither "rw" nor "lang"
    Returns:
        propkern:       proposal kernel, either
                        propkern_rw() or propkern_lang()
    """
    assert(sampler in ["rw", "lang"]), \
        "Please enter a sampler strategy in {'rw', 'lang'}"
    if sampler == "rw":
        propkern = propkern_rw
    elif sampler == "lang":
        propkern = propkern_lang
    return propkern


def init_currstate(init_state, statlogpdf, grad):
    """
    Initialises two instances of the MCMCState data structure
    and fills one of them with initial values.
    If a gradient is None, the gradient value is initialised with zero.
    """
    if grad is not None:
        currstate = MCMCState(init_state, statlogpdf(init_state), grad(init_state))
        # print("cstate", currstate)
    else:
        currstate = MCMCState(init_state, statlogpdf(init_state), 0)
    return currstate


def accept_or_reject(proposal, currstate, corrfact):
    """
    Performs acceptance-rejection step
    of Metropolis-Hastings algorithm.

    Args:
        statlogpdf:    PDF of stationary distribution,
                    callable: statlogpdf(x)
        proposal:   proposal z, MCMCState object
        currstate:  last state x_i of Markov chain,
                    MCMCState object
        corrfact:   correction factor q(x_i, z)/q(z, x_i)
    Returns:
        state:      either accepted proposal or currstate,
                    MCMCState object
        is_accept:  boolean, is the proposal accepted or not
    """
    accprob = get_accprob(proposal, currstate, corrfact)
    # print("accprob", accprob)
    # print("currloglkld", currstate.loglkldeval)
    if accprob < np.random.rand():
        state = currstate
        is_accept = False
    else:
        state = proposal
        is_accept = True
    return state, is_accept


def get_accprob(proposal, currstate, corrfact):
    """
    Computes acceptance probability of Metropolis-Hastings
        A(x_i, z) = min(1, p(z)/p(x_i) * corrfact)
    The minimum is not required later on, hence we ignore it.

    Note:
        If the likelihood of the current state is numerically zero,
        we accept the proposal by default; this turned out to be useful
        for Metropolis-Hastings with initial values in regions with
        extremely low probability as otherwise, possibly many NaNs
        are introduced.

    Args:
        proposal:   proposal z, MCMCState object
        currstate:  current state x_i of Markov chain,
                    MCMCState object
        corrfact:   correction factor q(x_i,z)/q(z,x_i)
    Returns:
        accprob:    p(z)/p(x_i) * corrfact
    """
    if np.exp(-currstate.loglkldeval) > 0:      # 
        accprob = corrfact * np.exp(-(proposal.loglkldeval-currstate.loglkldeval))           # HERE IS A NUMERICAL PROBLEM!!!!
    else:
        accprob = 1
    # # print(currstate.loglkldeval, corrfact)
    # accprob = corrfact * np.exp(-(proposal.loglkldeval-currstate.loglkldeval))           # HERE IS A NUMERICAL PROBLEM!!!!
    # print(proposal.loglkldeval, currstate.loglkldeval, corrfact, accprob)
    return accprob

def propkern_rw(currstate, statlogpdf, pwidth, *ignore):
    """
    Proposal kernel for RANDOM WALK.

    Reference:
        https://bookdown.org/rdpeng/ad
        vstatcomp/metropolis-hastings.html

    Args:
        currstate:  current state x_i of Markov chain,
                    MCMCState object
        statlogpdf:    PDF of stationary distribution,
                    callable: statlogpdf(x)
        pwidth:     proposal width s**2
        *ignore:    allows to ignore additional
                    arguments, e.g. the gradient
    Returns:
        proposal:       sampled proposal (MCMCState),
                    (z, p(z), 0), z ~ N(x_i, s**2*I_d)
        corrfact:   correction factor equal to 1.0
                    (RW is a symmetric proposal distribution)
    """
    propstate = sample_randomwalk(currstate.state, pwidth)
    proposal = MCMCState(propstate, statlogpdf(propstate), 0.)
    corrfact = 1.0
    return proposal, corrfact


def sample_randomwalk(mean, var):
    """
    Samples from N(mean, var * I_d).

    Args:
        mean: np.ndarray, shape (d,)
        var: scalar
    Returns:
        sample from N(mean, var * I_d), shape (d,).
    """
    return mean + np.sqrt(var) * np.random.randn(len(mean))


def propkern_lang(currstate, statlogpdf, pwidth, gradient):
    """
    Proposal kernel for LANGEVIN.

    Reference:
        https://en.wikipedia.org/wiki/M
        etropolis-adjusted_Langevin_algorithm

    Args:
        currstate:  current state x_i of Markov chain,
                    MCMCState object
        statlogpdf:    PDF of stationary distribution,
                    callable: statlogpdf(x)
        pwidth:     proposal width s**2
        gradient:   gradient of negative log-likelihood
                    callable: gradient(x)
    Returns:
        proposal:   sampled proposal (MCMCState),
                    (z, p(z), df(z))
        corrfact:   correction factor q(x_i,z)/q(z,x_i)
    """
    assert(gradient is not None), \
        "Please enter a gradient of the negative log-posterior dist."
    prop = sample_langevin(currstate, pwidth)
    proposal = MCMCState(prop, statlogpdf(prop), gradient(prop))
    corrfact = compute_corrfact(currstate, proposal, pwidth)
    return proposal, corrfact


def sample_langevin(currstate, pwidth):
    """
    Samples from N(x_i - h*df(x_i), 2*h).

    Args:
        currstate:  current state x_i of Markov chain,
                    MCMCState object (i.e. includes df(x_i))
        pwidth:     proposal width h
    Returns:
        sample:     sample from N(x_i - h*df(x_i), 2*h)
    """
    dim = len(currstate.state)
    gvl = currstate.gradeval
    noise = np.random.randn(dim)
    # print("LV Dynamics:", pwidth * np.linalg.norm(gvl))
    # print("LV Lklhd:", currstate.loglkldeval)
    # print("LV Noise:", np.sqrt(2*pwidth) *np.linalg.norm(noise))
    newmean = currstate.state - pwidth * gvl#/np.linalg.norm(gvl)
    sample = newmean + np.sqrt(2*pwidth) * noise
    return sample


def compute_corrfact(currstate, proposal, pwidth):
    """
    Computes correction factor q(x_i,z)/q(z,x_i)

    Args:
        currstate:  current state x_i of Markov chain,
                    MCMCState object
        proposal:   proposal z, MCMCState object
        pwidth:     proposal width h
    Returns:
        corrfact:   correction factor q(x_i,z)/q(z,x_i)
    """
    nomin = kernel_langevin(currstate, proposal, pwidth)
    denom = kernel_langevin(proposal, currstate, pwidth)
    # print("correction", nomin - denom)
    # if nomin-denom > 250:
    #     corrfact = 0
    # else:
    corrfact = nomin/denom
    print(nomin - denom)
    # corrfact = np.exp(-(nomin - denom))/(np.sqrt(2*np.pi*2*pwidth))
    print("corrfact", corrfact)
    return corrfact

def kernel_langevin(state1, state2, pwidth):
    """
    Evaluates density q(state1, state2) of proposal kernel
    Q(x, -) = N(x - h*df(x), 2*h) at 2 MCMCState objects
    for Langevin proposal distribution.

    Args:
        state1: MCMCState object
        state2: MCMCState object
        pwidth: proposal width h
    Returns:
        evaluation: evaluation of q(state1, state2)
    """
    dist = np.linalg.norm(state1.state - (state2.state-pwidth*state2.gradeval))
    scale = 1.0/(np.sqrt(2*np.pi*2*pwidth))
    # if dist**2/(2*2*pwidth) > 100:
    #     evaluation = 0
    # else:
    #     evaluation = scale * np.exp(-dist**2/(2*2*pwidth))
    return np.exp(-dist**2/(2*2*pwidth))*scale                                             # CAREFUL: CHANGED -dist to dist here
    # return evaluation

# END OF FILE