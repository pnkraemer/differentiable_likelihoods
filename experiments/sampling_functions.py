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
from odefilters.sampling import metropolishastings_pham, metropolishastings_plang, metropolishastings_rw


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
    means = odesolver.get_trajectory_ddim(m, 1, 0)
    evalpts = 2.0 * np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert(np.prod(np.in1d(evalpts, tsteps))==1), print(evalpts[np.in1d(evalpts, tsteps)==False])
    evalidcs = [list(tsteps).index(evalpt) for evalpt in evalpts]
    data = np.array([means[evalidx] + np.sqrt(ivpvar)*np.random.randn()
                     for evalidx in evalidcs])
    ipdata = ip.InvProblemData(evalpts, data, ivpvar)
    return ipdata


if __name__ == "__main__":

    np.random.seed(1)

    # Set Model Parameters
    initial_value = np.array([20, 20])
    initial_time, end_time = 0., 5.
    ivpvar = 1e-2
    thetatrue = np.array([1.0, 0.1, 0.1, 1.0])
    ivp = linode.LotkaVolterra(initial_time, end_time, params=thetatrue, initval=initial_value)

    # Set Method Parameters
    h_for_data = (end_time - initial_time)/10000
    h = (end_time - initial_time)/200
    solver = linsolver.LinearisedODESolver(statespace.IBM(q=1, dim=2))
    ipdata = create_data(solver, ivp, thetatrue, h_for_data, ivpvar)
    iplklhd = ip.InvProblemLklhd(ipdata, ivp, solver, h, with_jacob=True)

    # Sample from posteriors
    niter = 50
    init_theta = np.array([.8, .2, .05, 1.1])
    samples_ham, probs_ham = hamiltonian(niter, iplklhd, init_theta, stepsize=.2, nsteps=6)
    samples_lang, probs_lang = langevin(niter, iplklhd, init_theta, stepsize=1.2)
    # samples_rw, probs_rw = randomwalk(niter, iplklhd, init_theta, stepsize=0.15)


    # Plot results
    plt.title("Likelihood Values")
    plt.style.use("seaborn-whitegrid")
    plt.rc('font', size=30) # controls default text sizes
    # plt.semilogy(probs_rw, ':', label="RW")
    plt.semilogy(probs_lang, linewidth=4, label="PMALA")
    plt.semilogy(probs_ham, linewidth=4, label="PHMC")
    # plt.semilogy(probs_rw, linewidth=4, label="RW")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.show()