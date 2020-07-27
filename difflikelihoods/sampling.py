"""
sampling.py

We sample Metropolis-Hastings:
    * Random walk proposals
    * Langevin proposals
    * Langevin proposals with preconditioning
    * Hamiltonian MC
    * Hamiltonian MC with preconditioning

NOTE:
    The functionality of this module is restricted to log-densities,
    i.e. densities of the form p(s) = exp(-E(s)). We work with E(s) only.
    The reason is that in Bayesian inference, evaluations of exp(-E(s))
    are too instable in a numerical sense. 
"""

import collections
from abc import ABC, abstractmethod
import numpy as np
from difflikelihoods import logdensity




def metropolishastings_rw(logpdf, nsamps, initstate, pwidth, ninits):
    """
    Convenience function for Metropolis-Hastings sampling with
    random walk proposal kernel.
    """
    logdens = logdensity.LogDensity(logpdf)
    rwmh = RandomWalkMH(logdens)
    return rwmh.sample_nd(nsamps, initstate, pwidth, ninits)



def metropolishastings_lang(logpdf, loggrad, nsamps, initstate, pwidth, ninits):
    """
    Convenience function for Metropolis-Hastings sampling with
    Langevin dynamics proposal kernel.
    """
    logdens = logdensity.LogDensity(logpdf, loggrad)
    langmh = LangevinMH(logdens)
    return langmh.sample_nd(nsamps, initstate, pwidth, ninits)




def metropolishastings_plang(logpdf, loggrad, loghess, nsamps, initstate, pwidth, ninits):
    """
    Convenience function for Metropolis-Hastings sampling with
    Riemannian (preconditioned) Langevin dynamics proposal kernel.
    """
    logdens = logdensity.LogDensity(logpdf, loggrad, loghess)
    plangmh = PrecondLangevinMH(logdens)
    return plangmh.sample_nd(nsamps, initstate, pwidth, ninits)





def metropolishastings_ham(logpdf, loggrad, nsamps, initstate, stepsize, nsteps, ninits):
    """
    Convenience function for Hamiltonian MCMC.
    """
    logdens = logdensity.LogDensity(logpdf, loggrad)
    hmc = HamiltonianMC(logdens, nsteps)
    return hmc.sample_nd(nsamps, initstate, stepsize, ninits)





def metropolishastings_pham(logpdf, loggrad, loghess, nsamps, initstate, stepsize, nsteps, ninits):
    """
    Convenience function for preconditioned Hamiltonian MCMC.
    """
    logdens = logdensity.LogDensity(logpdf, loggrad, loghess)
    phmc = PrecondHamiltonianMC(logdens, nsteps)
    return phmc.sample_nd(nsamps, initstate, stepsize, ninits)




# Convenience data structure.
MCMCState = collections.namedtuple('MCMCState',
                                   'state logdens loggrad loghess')


class MetropolisHastings(ABC):
    """
    Abstract Metropolis-Hastings class. Contains everything but the
    proposal kernels.
    """
    def __init__(self, logdens):
        """
        Initialise MH sampler with a log-density function.

        Args:
            logdens:    LogDensity object, evaluations of a negative log-
                        density and derivatives
        """
        self.logdens = logdens


    def sample_nd(self, nsamps, init_state, pwidth, ninits=None, *optional):
        """
        """
        assert init_state_is_array(init_state), \
            "Please enter a (d,) dimensional initial state"
        states, logprobs = np.zeros((nsamps, len(init_state))), np.zeros(nsamps) 
        accepted = 0
        if ninits is None:
            ninits = 0
        currstate = self.evaluate_logdens(init_state)
        states[0], logprobs[0] = currstate.state, currstate.logdens
        for idx in range(1, nsamps):
            if idx < ninits:
                proposal, corrfact = self.generate_proposal(currstate, pwidth)
            else:
                proposal, corrfact = self.generate_proposal(currstate, 0.2*pwidth)
            currstate, is_accept = self.accept_or_reject(currstate, proposal, corrfact, idx, ninits)
            states[idx], logprobs[idx] = currstate.state.copy(), currstate.logdens.copy()
            if idx >= ninits:
                accepted = accepted + int(is_accept)
        ratio = accepted/nsamps
        return states, logprobs, ratio


    def evaluate_logdens(self, loc):
        """
        """
        logdenseval = self.logdens.eval(loc)
        if self.logdens.has_gradient:
            gradeval = self.logdens.gradeval(loc)
        else:
            gradeval = 0
        if self.logdens.has_hessian:
            hesseval = self.logdens.hesseval(loc)
        else:
            hesseval = 0
        return MCMCState(state=loc, logdens=logdenseval,
                         loggrad=gradeval, loghess=hesseval)


    def accept_or_reject(self, currstate, proposal, corrfact, idx, ninits):
        """
        """
        logaccprob = self.get_logaccprob(currstate, proposal, corrfact, idx, ninits)
        if logaccprob < 0 or logaccprob < -np.log(np.random.rand()):
            state = proposal
            is_accept = True
        else:
            state = currstate
            is_accept = False
        return state, is_accept


    def get_logaccprob(self, currstate, proposal, corrfact, idx, ninits):
        """
        Returns NEGATIVE log acceptance probability, i.e.
            corrected proposal - corrected currstate
        """
        if idx < ninits:
            corrfact = -corrfact
        return (corrfact) + (proposal.logdens - currstate.logdens)


    @abstractmethod
    def generate_proposal(self, *args):
        """
        """
        pass



def init_state_is_array(init_state):
    """
    Checks whether init_state is compliant with an Nd algorithm.
    That is, whether init_state is an (d,) np.ndarray.
    """
    assert(isinstance(init_state, np.ndarray)), \
        "Please enter init_state of shape (d,)"
    assert(len(init_state.shape) == 1), \
        "Please enter init_state of shape (d,)"
    return True



class RandomWalkMH(MetropolisHastings):
    """
    """
    def __init__(self, logdens):
        """
        """
        MetropolisHastings.__init__(self, logdens)


    def generate_proposal(self, currstate, pwidth):
        """
        """
        newloc = self.sample_randomwalk(currstate.state, pwidth)
        proposal = self.evaluate_logdens(newloc)
        corrfact = 0
        return proposal, corrfact


    def sample_randomwalk(self, mean, var):
        """
        """
        return mean + np.sqrt(var) * np.random.randn(len(mean))




class LangevinMH(MetropolisHastings):
    """
    """
    def __init__(self, logdens):
        """
        """
        MetropolisHastings.__init__(self, logdens)


    def generate_proposal(self, currstate, pwidth):
        """
        """
        newloc = self.sample_langevin(currstate, pwidth)
        proposal = self.evaluate_logdens(newloc)
        corrfact = self.compute_corrfact_langevin(currstate, proposal, pwidth)
        return proposal, corrfact


    def sample_langevin(self, currstate, pwidth):
        """
        """
        noise = np.random.randn(len(currstate.state))
        return currstate.state - pwidth*currstate.loggrad + np.sqrt(2*pwidth)*noise


    def compute_corrfact_langevin(self, currstate, proposal, pwidth):
        """
        """
        lognomin = self.kernel_langevin(currstate, proposal, pwidth)
        logdenom = self.kernel_langevin(proposal, currstate, pwidth)
        return lognomin - logdenom


    def kernel_langevin(self, state1, state2, pwidth):
        """
        """
        state2_dyn = state2.state - pwidth*state2.loggrad
        dist = np.linalg.norm(state1.state - state2_dyn)**2
        return 0.5*dist/(2*pwidth)



class PrecondLangevinMH(MetropolisHastings):
    """
    Preconditioning with (inverse) Hessian.
    """
    def __init__(self, logdens):
        """
        precondeval returns M (and not M^{-1}) as used in Cald&Gir
        """
        MetropolisHastings.__init__(self, logdens)


    def generate_proposal(self, currstate, pwidth):
        """
        """
        newloc = self.sample_langevin(currstate, pwidth)
        proposal = self.evaluate_logdens(newloc)
        corrfact = self.compute_corrfact_langevin(currstate, proposal, pwidth)
        return proposal, corrfact


    def sample_langevin(self, currstate, pwidth):
        """
        """
        noise = np.random.multivariate_normal(np.zeros(len(currstate.loghess)), np.linalg.inv(currstate.loghess))
        prec_dyn = np.linalg.solve(currstate.loghess, currstate.loggrad)
        return currstate.state - pwidth*prec_dyn + np.sqrt(2*pwidth)*noise


    def compute_corrfact_langevin(self, currstate, proposal, pwidth):
        """
        """
        lognomin = self.kernel_langevin(currstate, proposal, pwidth)
        logdenom = self.kernel_langevin(proposal, currstate, pwidth)
        return lognomin - logdenom


    def kernel_langevin(self, state1, state2, pwidth):
        """
        """
        prec_dyn = np.linalg.solve(state2.loghess, state2.loggrad)
        state2_dyn = state2.state - pwidth*prec_dyn
        difference = state1.state - state2_dyn
        return 0.5 * difference.dot(np.dot(state2.loghess, difference))/(2*pwidth)




class HamiltonianMC(MetropolisHastings):
    """
    """
    def __init__(self, logdens, nsteps):
        """
        """
        MetropolisHastings.__init__(self, logdens)
        self.nsteps = nsteps


    def generate_proposal(self, currstate, pwidth):
        """
        pwidth is used as stepsize for self.nsteps leapfrog steps.

        The correction factor is the quotient of the hamiltonian terms.
        """
        momentum = np.random.multivariate_normal(np.zeros(len(currstate.state)),
                                                 np.eye(len(currstate.state)))
        # hamilt = self.evaluate_hamiltonian(momentum, currstate)
        momentum_new, proposal = self.leapfrog_dynamics(momentum, currstate, pwidth)
        # prop_hamilt = self.evaluate_hamiltonian(momentum_new, proposal)
        corrfact = self.get_corrfact(momentum, momentum_new)
        return proposal, corrfact


    def leapfrog_dynamics(self, momentum, currstate, pwidth):
        """
        """
        proposal = currstate
        for idx in range(self.nsteps):
            momentum, proposal = self.compute_next_lfstep(momentum, proposal, pwidth)
        return momentum, proposal


    def compute_next_lfstep(self, momentum, proposal, pwidth):
        """
        """
        momentum = momentum - 0.5*pwidth*proposal.loggrad
        pstate = proposal.state + pwidth * momentum
        proposal = self.evaluate_logdens(pstate)
        momentum = momentum - 0.5*pwidth*proposal.loggrad
        return momentum, proposal

    def get_corrfact(self, mom_new, mom):
        """
        """
        return 0.5*(mom_new.T @ mom_new - mom.T @ mom)






class PrecondHamiltonianMC(MetropolisHastings):
    """
    In fact, the true name would be either
        * Riemannian-Gaussian HMC: if the preconditioner depends on the state
        * Euclidean-Gaussian HMC: if the preconditioner is constant
    [Girolami and Calderhead, 2011; Betancourt, 2018]
    """
    def __init__(self, logdens, nsteps):
        """
        evalprecond returns M (and not M^{-1}) as used in Cald&Gir.
        M is the Hessian
        """
        MetropolisHastings.__init__(self, logdens)
        self.nsteps = nsteps


    def generate_proposal(self, currstate, pwidth):
        """
        pwidth is used as stepsize for self.nsteps leapfrog steps.

        The correction factor is the quotient of the hamiltonian terms.
        """
        momentum = np.random.multivariate_normal(np.zeros(len(currstate.state)),
                                                 currstate.loghess)
        momentum_new, proposal = self.leapfrog_dynamics(momentum, currstate, pwidth)
        corrfact = self.get_corrfact(momentum, momentum_new, currstate, proposal)
        return proposal, corrfact


    def leapfrog_dynamics(self, momentum, currstate, pwidth):
        """
        """
        proposal = currstate
        for idx in range(self.nsteps):
            momentum, proposal = self.compute_next_lfstep(momentum, proposal, pwidth)
        return momentum, proposal


    def compute_next_lfstep(self, momentum, proposal, pwidth):
        """
        """
        momentum = momentum - 0.5*pwidth*proposal.loggrad
        pstate = proposal.state + pwidth * np.linalg.solve(proposal.loghess, momentum)
        proposal = self.evaluate_logdens(pstate)
        momentum = momentum - 0.5*pwidth*proposal.loggrad
        return momentum, proposal


    def get_corrfact(self, mom, mom_new, currstate, proposal):
        """
        """
        return 0.5*(mom_new.T @  np.linalg.solve(proposal.loghess, mom_new)\
            + np.log(np.linalg.det(proposal.loghess))\
            - mom.T @  np.linalg.solve(currstate.loghess, mom)
            - np.log(np.linalg.det(currstate.loghess)))


