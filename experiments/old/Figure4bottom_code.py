# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from odefilters import odesolver
from odefilters import linearised_odesolver as linsolver
from odefilters import linearised_ode as linode
from odefilters import statespace
from odefilters import inverseproblem as ip
from odesolvers.sampling import metropolishastings_pham, metropolishastings_plang, metropolishastings_rw

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
    means = odesolver.get_trajectory_ddim(m, 2, 0)
    evalpts = np.array([1., 2., 3., 4., 5.])
    evalpts = np.arange(.5, 5., 5/10)
    assert(np.prod(np.in1d(evalpts, tsteps))==1), print(evalpts[np.in1d(evalpts, tsteps)==False])
    noise = np.sqrt(ivpvar)*np.random.randn(len(evalpts)*2).reshape((len(evalpts), 2))
    evalidcs = [list(tsteps).index(evalpt) for evalpt in evalpts]
    data = means[evalidcs] + noise # this is 'wrong' noise
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
    h = (end_time - initial_time)/400
    solver = linsolver.LinearisedODESolver(statespace.IBM(q=1, dim=2))
    ipdata = create_data(solver, ivp, thetatrue, h_for_data, ivpvar)
    iplklhd = ip.InvProblemLklhd(ipdata, ivp, solver, h, with_jacob=True)

    # Sample from posteriors
    niter = 250
    init_theta = np.array([1, .2, .01, 1.1])
    np.random.seed(1)
    _ninits = 50
    samples_lang, probs_lang = langevin(niter, iplklhd, init_theta, stepsize=1.15, ninits=45)
    np.random.seed(1)
    samples_ham, probs_ham = hamiltonian(niter, iplklhd, init_theta, stepsize=0.35, nsteps=3, ninits=45)
    np.random.seed(1)
    samples_rw, probs_rw = randomwalk(niter, iplklhd, init_theta, stepsize=0.0005, ninits=45)



    # Plot results
    plt.style.use("./icmlstyle.mplstyle")


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Neg. log-likelihood")
    # ax1.set_ylim((1e-310, 1e10))
    # plt.semilogy(probs_rw, ':', label="RW")
    ax1.semilogy((probs_lang), ls='None', marker="^", label="PLMC", alpha=0.4, markevery=2)
    ax1.semilogy((probs_ham), ls='None', marker="d", label="PHMC", alpha=0.4, markevery=2)
    ax1.semilogy((probs_rw), ls='None', marker="s", label="RWM", alpha=0.4, markevery=2)
    # ax1.set_yticks([1e-300, 1e-200, 1e-100, 1e-0])
    # ax1.legend()

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Rel. Error")
    # plt.semilogy(probs_rw, ':', label="RW")
    ax2.semilogy(np.abs((samples_lang - thetatrue[np.newaxis, :])/thetatrue[np.newaxis, :]).mean(axis=1),  ls='None', marker="^", label="PLMC", alpha=0.4, markevery=2)
    ax2.semilogy(np.abs((samples_ham - thetatrue[np.newaxis, :])/thetatrue[np.newaxis, :]).mean(axis=1), ls='None', marker="d", label="PHMC", alpha=0.4, markevery=2)
    ax2.semilogy(np.abs((samples_rw - thetatrue[np.newaxis, :])/thetatrue[np.newaxis, :]).mean(axis=1), ls='None', marker="s", label="RWM", alpha=0.4, markevery=2)

    # ax1.spines['bottom'].set_position(('outward', 5))
    # ax2.spines['bottom'].set_position(('outward', 5))
    ax1.set_title("c", loc="left", fontweight='bold', ha='right')
    ax2.set_title("d", loc="left", fontweight='bold', ha='right')

    ax2.legend()

    ax1.minorticks_off()
    ax2.minorticks_off()
    plt.tight_layout()
    plt.savefig("./figures/figure4_sampling_right")

    plt.show()