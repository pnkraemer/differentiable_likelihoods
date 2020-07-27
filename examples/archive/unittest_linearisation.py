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
sys.path.append("../module/")

import numpy as np

from odefilters import odesolver
from odefilters import ode
from odefilters import statespace
from odefilters import linearisation as lin

def get_data_points(tsteps, N_data):
	"""
	computes array of N_data evenly spaced evaluation points
	along the time axis given by tsteps
	"""
	# define after how many indices there will be a new data point
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

	return (np.linalg.norm(data_test - data) < 1e-12)

def prdct_tsteps_ge_zero(prdct_tsteps):
	"""
	test that all data points in prdct_tsteps are greater than zero
	"""
	return all(datapt > 0 for datapt in prdct_tsteps)

# generate a random seed and print it
seed = random.randrange(sys.maxsize)
rng = random.Random(seed)
print("Seed was:", seed)

# set amount of data points
N_data = np.random.poisson() + 2

# set amount of model parameters
params_dim = np.random.poisson() + 2 # dimensionality of ODE trajectory

# set dimensionality of ODE trajectory
d = 2 * params_dim

# draw ODE parameters
odeparams = np.array(2 * np.random.rand(params_dim))

# define matrix for linear ODE and compute its inverse
diag_matrix = np.diag(np.repeat(odeparams, 2))

# define parameters of initial value problem
y0 = (2*np.random.rand(d)).reshape([d,])
y0_unc = np.zeros([d,d])
t0 = 0.1 + np.random.rand()
tmax = 1.1 + np.random.rand()

# define initial value problem
ivp = ode.MatrixLinearODE(t0, tmax, diag_matrix, y0, y0_unc)

# set filter parameters
q = 1
h = np.random.choice([0.1, 0.01, 0.001])
R = 0.

# set up ODE solvers
ibm = statespace.IBM(q=q, dim=d)
solver = odesolver.ODESolverWithData(ibm)

# solve ODE
tsteps, means, __, data, ___ = solver.solve(ivp, stepsize=h)

# define number of time steps
N_sim = len(tsteps)    # amount of time steps

# extract filtering mean estimate
mean = odesolver.get_trajectory_ddim(means, d, 0)

# define (N_data,)-shape array of evaluation points 
prdct_tsteps = get_data_points(tsteps, N_data)
assert(prdct_tsteps_ge_zero(prdct_tsteps))   # to make sure that the initial time is not contained in prdct_tsteps

# get indices of data points
data_indices = lin.get_prdct_idcs(prdct_tsteps, tsteps)

# compute mean at data_indices
mean_at_data = mean[data_indices]
assert((tsteps[data_indices] == prdct_tsteps).all())

# define rhs_parts, i.e. the data without parameters
rhs_parts = get_rhs_parts(data, diag_matrix)

# test that data is correct
assert(valid_rhs_parts(data, rhs_parts, odeparams, N_sim, d))

# define quantity that contains all data quantities for linearisation
derivative_data = (tsteps, rhs_parts, 0)

print('d is %i'%(d))
print('n is %i'%(params_dim))

# compute quantities for first-order Taylor expansion
(constant_term,  jacobian) = lin.compute_linearisation(solver.filt.ssm, y0, derivative_data, prdct_tsteps)

postmean_lin = constant_term + jacobian @ odeparams
error_postmean = np.linalg.norm(postmean_lin - mean_at_data.flatten()) / np.linalg.norm(mean_at_data)

print('The error on the posterior mean is: %.e.' %error_postmean)

assert(error_postmean < 1e-13)
print('The sanitycheck was successful!')