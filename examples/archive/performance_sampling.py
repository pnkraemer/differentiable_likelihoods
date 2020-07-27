"""
performance_sampling.py

Compare performance gain in terms of
computational time of rewriting sampling module.

What we do:
    * We define 'fast' and 'slow' probability density functions,
    then we compute a number of samples with
    the old and with the new method, comparing the
    average computational time for each sample.
    * The study shows that we cut the time by 30%
    for fast evaluations and by 50% for slow evaluations.
"""
import time
import functools as ft
import numpy as np
from odefilters.sampling import metropolishastings_nd as mh_new
from odefilters.archive.old_sampling import metropolishastings_nd as mh_old


def statpdf(x, sleep):
    """
    Evaluate Gaussian density and
    sleep for given amount of time.
    """
    time.sleep(sleep)
    assert isinstance(x, np.ndarray) is True
    return np.exp(-0.5*np.linalg.norm(x)**2/2)/np.sqrt(2*np.pi)


def grad(x, sleep):
    """
    Evaluate gradient of Gaussian density
    and sleep for given amount of time.
    """
    time.sleep(sleep)
    assert isinstance(x, np.ndarray) is True
    return np.linalg.norm(x)


def compute_time_per_sample(samplerfct, _nsamps, _statpdf, _grad):
    """
    Computes Langevin MCMC samples and divides
    computational time by number of samples.
    """
    init_state = np.array([0.0])
    time_before = time.time()
    samplerfct(_nsamps, _statpdf, init_state, sampler="lang", grad=_grad)
    walltime = time.time() - time_before
    return walltime/nsamps


def print_description(_nsamps, _sleeptime):
    """
    Print description of study.
    """
    print("\nAverage computational (wall) time")
    print("per Langevin MCMC sample,")
    print("averaged over N=%u samples" % _nsamps)
    print("\nCheap densities are 1d")
    print("Std-Gaussians and expensive densities")
    print("sleep for t=%.0f mikrosec" % (sec_to_microsec(_sleeptime)))


def sec_to_microsec(sec):
    """Transforms seconds into microseconds."""
    return 1e06*sec


if __name__ == "__main__":

    # Method parameters
    nsamps = 250
    sleeptime = 1e-5
    print_description(nsamps, sleeptime)

    # Define fast and slow evaluation functions
    statpdf_fast = ft.partial(statpdf, sleep=0.0)
    grad_fast = ft.partial(grad, sleep=0.0)
    statpdf_slow = ft.partial(statpdf, sleep=sleeptime)
    grad_slow = ft.partial(grad, sleep=sleeptime)


    # Compute wall times
    t_old_fast = compute_time_per_sample(mh_old, nsamps, statpdf_fast, grad_fast)
    t_new_fast = compute_time_per_sample(mh_new, nsamps, statpdf_fast, grad_fast)
    t_old_slow = compute_time_per_sample(mh_old, nsamps, statpdf_slow, grad_slow)
    t_new_slow = compute_time_per_sample(mh_new, nsamps, statpdf_slow, grad_slow)
    ratio_old = t_new_fast/t_old_fast
    ratio_new = t_new_slow/t_old_slow

    # Print results
    print("\nCheap densities:")
    print("\tOld:\t%.0f mikrosec" % (sec_to_microsec(t_old_fast)))
    print("\tNew:\t%.0f mikrosec" % (sec_to_microsec(t_new_fast)))
    print("\tRatio:\t%.2f " % ratio_old)
    print("\nExpensive densities:")
    print("\tOld:\t%.0f mikrosec" % (sec_to_microsec(t_old_slow)))
    print("\tNew:\t%.0f mikrosec" % (sec_to_microsec(t_new_slow)))
    print("\tRatio:\t%.2f" % ratio_new)
    print()
