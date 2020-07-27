# coding=utf-8
"""
filter_car_trajectory.py

Checking the compliance of statespace.py and kalmanfilter.py
by tracking the movement of a car. If the output looks 'proper', the test passes.

Reference:
Example 4.3 in https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from difflikelihoods import statespace as stsp
from difflikelihoods import filters


def init_ssm(twidth):
    """
    Set up matrices in Example 4.3
    and initialise time-invariant state space model
    """
    A = np.eye(4) + np.diag(twidth * np.ones(2), 2)
    Q = _get_tcovar(twidth)
    H = np.eye(2, 4)
    R = np.diag(np.array([0.01**2, 0.01**2]))
    m0 = np.zeros(4)
    c0 = 0.0001 * np.eye(4)
    ssm = stsp.InvariantSSM(transmat=A, transcovar=Q, measmat=H, meascovar=R, init_mean=m0, init_covar=c0)
    return ssm


def _get_tcovar(twidth):
    """
    Auxiliary function which returns the transition covariance matrix of car model
    """
    diagonal = np.array([twidth**3/3., twidth**3/3., twidth, twidth])
    offdiagonal = np.array([twidth**2/2., twidth**2/2.])
    tcovar = np.diag(diagonal) + np.diag(offdiagonal, 2) +\
        np.diag(offdiagonal, -2)
    return tcovar


# Initialise parameters
twidth = 0.001
ntsteps = 1000

# Create state space model
ssm = init_ssm(twidth)
dataset = ssm.sample_trajectory(ntsteps)

# Create Kalmanfilter and filter out dataset
kfilt = filters.KalmanFilter(ssm)
means, covars = kfilt.filter(dataset)

# Visualise results
plt.title("Tracking the movement of a car")
plt.plot(dataset[:,0], dataset[:,1], '.', alpha=0.5, label="Dataset")
plt.plot(means[:,0], means[:,1], '-', linewidth=2, label="Filtered Trajectory")
plt.plot(means[0,0], means[0,1], 'o', label="Starting point at t=t0")
plt.plot(means[-1,0], means[-1,1], '^', label="Endpoint at t=tmax")
plt.legend()
plt.show()

# END OF FILE
