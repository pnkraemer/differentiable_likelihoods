# coding=utf-8
"""
linearisation.py

This module computes the linearisation of the map

(p_1,...,p_n) -> m(T_l) 
as
m(T_l) ~ m0 + J p,

where p_i are the parameters of the ODE and
T_l is a subset of the time points returned 
by the filter and m is the posterior mean at
time points T_l.

Note:
    We employ the assumption
    f() = p_1 * f_1() + ... + p_n * f_n().

Example:
    See example_linearisation.py
"""

import numpy as np
import scipy.sparse.linalg as spspl
from difflikelihoods import covariance as cov


def compute_linearisation(ssm, initial_value, derivative_data, prdct_tsteps):
    """
    Compute linearisation, i.e. a first-order Taylor expansion consisting of a
    constant term and Jacobian matrix, of the map 

    (p_1,...,p_n) -> m(T_l),

    where p_i are the parameters of the ODE and T_l is a subset of the time 
    points returned by the filter and m is the posterior mean at time points 
    T_l. Note that we employ the assumption

    f() = p_1 * f_1() + ... + p_n * f_n().

    By this assumption, the computation of the Jacobian is split up into a 
    kernel prefactor and a data factor.

    Args:
        ssm:            instance of statespace.SSM,
                        typically of statespace.IBM with q=1
        initial_value:  (ndim,) shaped array consisting of the ODE initial value
        derivative_data:tuple with three elements; namely 
                        * (ntsteps,) array of ntsteps time points covered by the filter, 
                        * (ntsteps, npar, ndim) array consisting of the ntsteps evaluations of 
                          (f_1,...,f_n) in d dimensions, 
                        * scalar, constant measurement variance of ODE filter.
                          Array input not supported as of yet.
        prdct_tsteps:   (ndata,) shaped array consting of prediction time points
    Returns:
        constant_term:  the constant term of the linearisation, shape (ndata*ndim,)
        jacobian:       the Jacobian matrix, shape (ndata*ndim, npar)
    """
    if np.isscalar(derivative_data[2]) is not True:
        raise TypeError("derivative_data[2] needs to be scalar.")
    tsteps, rhs_parts, measvar = derivative_data
    assert np.prod(np.in1d(prdct_tsteps, tsteps)) == 1
    kernel_prefactor = compute_kernel_prefactor(ssm, measvar, tsteps, prdct_tsteps)
    constant_term = np.tile(initial_value, len(prdct_tsteps))
    jacobian = compute_jacobian(prdct_tsteps, tsteps, kernel_prefactor, rhs_parts)
    return constant_term, jacobian


def compute_kernel_prefactor(ssm, measvar, tsteps, prdct_tsteps):
    """
    Computes kernel prefactor. The kernel prefactor times the
    data factor is the Jacobian. It only depends on the prior (as      
    well as on the observation variance measvar) and is independent of the evaluations 
    of the right-hand side of the ODE Filter. Therefore, in the context of 
    inverse problems, it can be precomputed for all samples of the ODE parameter 
    vector.  

    Args:
        ssm:            instance of statespace.SSM, typically statespace.IBM
        measvar:        (scalar) constant measurement variance of ODE filter
        tsteps:         (ntsteps,) shaped array of ntsteps time points used by the filter
        prdct_tsteps:   (ndata,) shaped array consting of prediction time points
    Returns:
        prefactor:      the kernel prefactor, shape (ndata*ndim, ntsteps-1)
    """
    assert np.prod(np.in1d(prdct_tsteps, tsteps)) == 1
    prdct_idcs = get_prdct_idcs(prdct_tsteps, tsteps)
    prefactor = np.zeros([len(prdct_idcs) * ssm.dim, len(tsteps) - 1])
    (kds, dkds_plus_mvar) = kernel_matrices(ssm, tsteps, measvar)
    ###########################################################################
    # compute kernel prefactor for all data indices
    #
    # (nb: they are the same for all dimensions
    #   if the kernel is the same for all dimensions
    #   as currently implemented)
    #
    ###########################################################################
    for l, prdct_idx in enumerate(prdct_idcs):
        content = compute_kernel_prefactor_at_idx(kds, dkds_plus_mvar, prdct_idx)
        prefactor[l * ssm.dim : (l + 1) * ssm.dim, :prdct_idx] = content
    #     print(l, prdct_idx, content)
    # sys.exit()
    return prefactor


def get_prdct_idcs(prdct_tsteps, tsteps):
    """
    Get the indices of prdct_tsteps in the time array tsteps.
    Args:
        prdct_tsteps:   (ndata,) shaped array consting of prediction time points
        tsteps:         (ntsteps,) shaped array of ntsteps time points used by the filter
    Raises:
        AssertionError: if not tsteps[result] == prdct_tsteps
    Returns:
        idcs:           indices of prdct_tsteps in the array tsteps, shape (ndata,)
    """
    idcs = np.zeros(len(prdct_tsteps))

    for (idx, time) in enumerate(prdct_tsteps):

        # find index of prdct_tsteps in tsteps and save it to idcs
        idcs[idx] = tsteps.tolist().index(time)

    idcs = idcs.astype(int)
    assert (tsteps[idcs] == prdct_tsteps).all()

    return idcs


def kernel_matrices(ssm, tsteps, measvar):
    """
    Compute kernel matrices needed for the kernel prefactor.
    Args:
        ssm:        instance of statespace.SSM, typically of statespace.IBM with q=1
        tsteps:     (ntsteps,) shaped array of ntsteps time points used by the filter
        measvar:    (scalar) constant measurement variance of ODE Filter
    Returns:
        kds:        the covariance matrix of kernel cov.ibm_covd on tsteps, shape 
                    (ntsteps-1, ntsteps-1)
        dkds_plus_mvar:
                    the covariance matrix of kernel cov.ibm_dcovd on tsteps 
                    plus measvar on the diagonal, shape (ntsteps-1, ntsteps-1)
    """
    # move the time axis to tmin = 0
    tsteps_from_t0 = tsteps - tsteps[0]

    # nb: indices [1:] so that the first time point is not used because it has variance 0
    kds = get_kds(ssm, tsteps_from_t0)
    dkds_plus_mvar = get_dkds_plus_mvar(ssm, tsteps_from_t0, measvar)

    return (kds, dkds_plus_mvar)


def get_kds(ssm, tsteps_t0):
    """
    Returns covariance matrix of left partial derivative of Integatred Brownian 
    motion kernel. 
    Args:
        ssm:        instance of statespace.SSM, typically of statespace.IBM with q=1
        tsteps_t0:  (ntsteps-1,) shaped array of ntsteps-1 time points starting from t=0
    Returns:
        covariance matrix of kernel cov.ibm_covd on tsteps_t0, shape (ntsteps-1, ntsteps-1) 
    """
    return cov.ibm_covd(tsteps_t0[1:], tsteps_t0[1:], ssm.diffconst, ssm.q)


def get_dkds_plus_mvar(ssm, tsteps_t0, measvar):
    """
    Returns covariance matrix of Integrated Brownian Motino differentiated
    from both sides
    Args:
        ssm:        instance of statespace.SSM, typically of statespace.IBM with q=1
        tsteps_t0:  (ntsteps-1,) shaped array of ntsteps-1 time points starting from t=0
        measvar:    (scalar) constant measurement variance of ODE filter
    Returns:
        covariance matrix of kernel cov.ibm_dcovd on tsteps plus measvar on 
        the diagonal, shape (ntsteps-1, ntsteps-1)
    """
    return get_dkds(ssm, tsteps_t0) + measvar * np.eye(len(tsteps_t0) - 1)


def get_dkds(ssm, tsteps_t0):
    """
    Returns covariance matrix of Integrated Brownian Motino differentiated
    from both sides.
    Args:
        ssm:        instance of statespace.SSM, typically of statespace.IBM with q=1
        tsteps_t0:  (ntsteps-1,) shaped array of ntsteps-1 time points starting from t=0
    Returns:
        covariance matrix of kernel cov.ibm_dcovd on tsteps, shape (ntsteps-1, ntsteps-1)
    """
    return cov.ibm_dcovd(tsteps_t0[1:], tsteps_t0[1:], ssm.diffconst, ssm.q)


def compute_kernel_prefactor_at_idx(kds, dkds_plus_mvar, prdct_idx):
    """
    Compute kernel factor from kernel matrices (kds, dkd_plus_measvar)
    at a specific data index.
    Args:
        kds:        (ntsteps-1, ntsteps-1) shaped array of the covariance matrix of kernel 
                    cov.ibm_covd on tsteps
        dkds_plus_mvar:
                    (ntsteps-1, ntsteps-1) shaped array of the covariance matrix of 
                    kernel cov.ibm_dcovd on tsteps plus measvar on the 
                    diagonal
        prdct_idx:  index of the prdct_tsteps in the array tsteps, integer
    Returns:
        the kernel prefactor for prdct_idx, shape (1, prdct_idx)
    """
    assert prdct_idx > 0, "Data point seems to at t0"
    kd_mat = kds[prdct_idx - 1, :prdct_idx].reshape([prdct_idx, 1])
    dkd_plus_measvar_mat = dkds_plus_mvar[:prdct_idx, :prdct_idx]
    return np.linalg.solve(dkd_plus_measvar_mat, kd_mat).T


def compute_jacobian(prdct_tsteps, tsteps, kernel_prefactor, rhs_parts):
    """
    Computes Jacobian matrix of the map 

    (p_1,...,p_n) -> m(T_l),

    where p_i are the parameters of the ODE and T_l is a subset of the time 
    points returned by the filter and m is the posterior mean at time points 
    T_l. 
    Note that we employ the assumption

    f() = p_1 * f_1() + ... + p_n * f_n().

    This function receives the kernel prefactor as an input. To compute
    the Jacobian from scratch, without access to the kernel prefactor, use the
    compute_linearisation function in this module.

    Args:
        prdct_tsteps:       (ndata,) shaped array consting of prediction time points
        tsteps:             (ntsteps,) shaped array of ntsteps time points used by the filter
        kernel_prefactor:   (ndata*ndim, ntsteps-1) shaped array
        rhs_parts:          (ntsteps, npar, ndim) shaped array consisting of the ntsteps evaluations of 
                            (f_1,...,f_n) in d dimensions 
    Returns: 
        jacobian:           Jacobian matrix, shape (ndata*ndim, npar)
    """

    data_factor = rhs_parts[1:] - rhs_parts[0]
    jacobian_wo_prior = compute_jacobian_wo_prior(kernel_prefactor, data_factor)
    jacobian = compute_jacobian_with_prior(
        jacobian_wo_prior, prdct_tsteps, tsteps, rhs_parts
    )
    return jacobian


def compute_jacobian_wo_prior(kernel_prefactor, data_factor):
    """
    Computes the Jacobian without the influence from the prior. In other words, 
    this function computes the Jacobian for a GP prior with zero-mean on the 
    derivative state.

    Args:
        kernel_prefactor:   (ndata*ndim, ntsteps-1) shaped array
        data_factor:        (ntsteps-1, npar, ndim) shaped array consisting of the ntsteps-1 evaluations 
                            (without the initial one) of (f_1,...,f_n) in d dimensions, 
                            from which the prior derivative has been subtracted
    Returns:
        jacobian_wo_prior:  Jacobian matrix without the influence of the prior, 
                            i.e. with zero-mean prior on the derivative, shape 
                            (ndata*ndim, npar) 
    """
    jacobian_wo_prior = np.zeros([kernel_prefactor.shape[0], data_factor.shape[1]])
    ode_dim = data_factor.shape[2]

    # compute Jacobian (one dimension at a time)
    for j in range(ode_dim):
        jacobian_wo_prior[j::ode_dim] = (
            kernel_prefactor[j::ode_dim] @ data_factor[:, :, j]
        )

    return jacobian_wo_prior


def compute_jacobian_with_prior(jacobian_wo_prior, prdct_tsteps, tsteps, rhs_parts):
    """
    This function computes the final Jacobian from the Jacobian without prior, 
    which ignores the influence from a (in general non-zero) prior derivative
    state.

    Args:
        jacobian_wo_prior:  (ndata*ndim, npar) shaped array with the Jacobian matrix without
                            the influence of the prior, i.e. with zero-mean prior
                            on the derivative 
        prdct_tsteps:       (ndata,) shaped array consting of prediction time points
        tsteps:             (ntsteps,) shaped array of ntsteps time points used by the filter
        rhs_parts:          (ntsteps, npar, ndim) shaped array consisting of the ntsteps evaluations of 
                            (f_1,...,f_n) in d dimensions 
    Returns:
        jacobian:           the Jacobian matrix, shape (ndata*ndim, npar)
    """
    ode_dim = rhs_parts.shape[2]
    data_dists = (
        prdct_tsteps - tsteps[0]
    )  # compute distance of prdct_tsteps to initial time
    jacobian = np.zeros(jacobian_wo_prior.shape)
    for (idx, data_dist) in enumerate(data_dists):
        jacobian[idx * ode_dim : (idx + 1) * ode_dim] = (
            jacobian_wo_prior[idx * ode_dim : (idx + 1) * ode_dim]
            + data_dist * rhs_parts[0].T
        )

    return jacobian
