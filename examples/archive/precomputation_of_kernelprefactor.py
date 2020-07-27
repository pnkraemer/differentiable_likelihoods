# coding=utf-8
"""
sanitycheck_linearisation_of_filter_mvdata_multidim.py

This is a version of 
sanitycheck_linearisation_of_filter_mvdata_multidim.py 
with the difference that this version has multidim
parameter vector. 

We assert that the filter mean m(t_*) at time t_*
can be obtained via the function 
compute_filtering_mean_by_IBM_GP_regression.
"""
import time
import sys
import random

import numpy as np

from odefilters import linearised_odesolver as linsolver
from odefilters import linearised_ode as linode
from odefilters import statespace
from odefilters import linearisation as lin

def get_data_points(tsteps, N_data):
	"""
	computes array of N_data evenly spaced evaluation points
	along the time axis given by tsteps
	"""
	dist_index = np.floor(len(tsteps) / N_data).astype(int)
	return np.array([tsteps[(i+1)*dist_index] for i in range(N_data-1)] + [tsteps[-1]])

def valid_rhs_parts(data, rhs_parts, odeparams, N_sim, d):
	"""
	tests that data is correct by checking that it is
	(N_sim, params_dim, d)-shape and 
	rhs_parts_reproduces_data is true
	"""
	return (rhs_parts.shape == (N_sim, len(odeparams), d) and rhs_parts_reproduces_data(data, rhs_parts, odeparams))

def get_rhs_parts(data, diag_matrix):
	"""
	computes (N_sim, n_params, d)-shape array of rhs_parts (without parameters)
	"""
	# get dimensionality of arameter vector
	n_params = int(diag_matrix.shape[0] / 2)

	# divide the parameters out of the data
	data_wo_parameters = data @ np.linalg.inv(diag_matrix)

	# some storage for data_prats
	rhs_parts = np.zeros([N_sim, n_params, d])

	# reshape data_wo_parameters into rhs_parts
	for i_param in range(n_params):
		rhs_parts[:, i_param, 2 * i_param : 2 * (i_param + 1)] = data_wo_parameters[:, 2*i_param : 2*(i_param+1)]

	return rhs_parts

def rhs_parts_reproduces_data(data, rhs_parts, odeparams):
	"""
	test if rhs_parts multiplied with odeparams 
	indeed gives back data
	"""
	n_timestps = rhs_parts.shape[0]
	dim_ivp = rhs_parts.shape[2]

	data_test = np.zeros([n_timestps, dim_ivp])
	for i in range(n_timestps):
		for j in range(dim_ivp):
			data_test[i, j] = np.dot(rhs_parts[i, :, j], odeparams)

	return (np.linalg.norm(data_test - data) < 1e-13)

# generate a random seed and print it
# seed = random.randrange(sys.maxsize)
# rng = random.Random(seed)
# print("Seed was:", seed)

# set amount of data points
# N_data = np.random.poisson() + 2
N_data = 2

# set amount of model parameters
# params_dim = np.random.poisson() + 2 # dimensionality of ODE trajectory
params_dim = 2

# set dimensionality of ODE trajectory
d = 2 * params_dim

# define parameters of initial value problem
y0 = (2*np.random.rand(d)).reshape([d,])
y0_unc = np.zeros([d,d])
t0 = 0.1 + np.random.rand()
tmax = 1.1 + np.random.rand()

# set filter parameters
q = 1
h = np.random.choice([0.1, 0.01, 0.001])
R = 0.

# set up ODE solvers
ibm = statespace.IBM(q=q, dim=d)
solver = linsolver.LinearisedODESolver(ibm)

# define time steps for each forward simulation
tsteps = np.arange(t0, tmax + h, h)
tsteps_t0 = tsteps - tsteps[0]    # moving the time axis to tmin = 0

# define number of time steps for each forward simulation
N_sim = len(tsteps) 

# define (N_data,)-shape array of evaluation points 
prdct_tsteps = get_data_points(tsteps, N_data)

data_indices = lin.get_prdct_idcs(prdct_tsteps, tsteps)

kernel_prefactor = lin.compute_kernel_prefactor(solver.filt.ssm, R, tsteps, prdct_tsteps)

# draw ODE parameters
odeparams = np.array(2 * np.random.rand(params_dim))

# define matrix for linear ODE and compute its inverse
diag_matrix = np.diag(np.repeat(odeparams, 2))

# define initial value problem
ivp = linode.MatrixLinearODE(t0, tmax, diag_matrix, y0, y0_unc)
# ivp = ode.get_exp_multidim(t0, tmax, y0, y0_unc, diag_matrix, dim=d)

# solve ODE
tsteps, means, __, data, __ = solver.solve(ivp, stepsize=h)

# extract filtering mean estimate
mean = odesolver.get_trajectory_ddim(means, d, 0)

# compute mean at data_indices
mean_at_data = mean[data_indices]

# define rhs_parts, i.e. the data without parameters
rhs_parts = get_rhs_parts(data, diag_matrix)

# test that data is correct
assert(valid_rhs_parts(data, rhs_parts, odeparams, N_sim, d))

jacobian = lin.compute_jacobian(prdct_tsteps, tsteps, kernel_prefactor, rhs_parts)

postmean_lin = np.tile(y0, N_data) + jacobian @ odeparams
error_postmean = np.linalg.norm(postmean_lin - mean_at_data.flatten()) / np.linalg.norm(mean_at_data)

print('The error on the posterior mean is: %.e.' %error_postmean)

# final test over all rounds
assert(error_postmean < 1e-13)
print('The test was successful!')