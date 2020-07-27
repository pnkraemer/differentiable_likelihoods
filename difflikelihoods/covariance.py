# coding=utf-8
"""
covariance.py

Some common covariance functions K(x,y).
Evaluations are vectorised to ensure fast runtime.

Contents:
    * Matern family: Gaussian, Matern
      and exponential covariance kernel
    * Brownian motion family: Brownian motion,
      integrated Brownian motion and derivatives

Example:
    >>> import numpy as np
    >>> import covariance as cov
    >>> ptset = np.random.rand(15, 2)
    >>> covmat = cov.gausscov(ptset, ptset, corr_length=1.0)
"""
import numpy as np
import scipy.spatial as scs
import scipy.special as scspec


def gausscov(ptset1, ptset2, corr_length=1.0):
    """
    Gaussian covariance kernel: see Rasmussen & Williams (p.83, 2006).

    Args:
        ptset1: an (N, d) shaped array consisting of N points in d dimensions
        ptset2: an (N, d) shaped array consisting of N points in d dimensions
        corr_length: correlation length of Gaussian covariance (default: 1.0)
    Raises:
        AssertionError: if ptset1 or ptset2 are not in (N,d) shape
    Returns:
        covmat: The covariance matrix on ptset1 and ptset2
    """
    assert check_shape_of_ptsets(ptset1, ptset2) is True
    distmtrx = scs.distance_matrix(ptset1, ptset2)
    covmat = np.exp(-(distmtrx ** 2) / (2.0 * corr_length ** 2))
    return covmat


def expcov(ptset1, ptset2, corr_length=1.0):
    """
    Exponential covariance kernel: see Rasmussen & Williams (p.85, 2006).

    Args:
        ptset1: an (N, d) shaped array consisting of N points in d dimensions
        ptset2: an (N, d) shaped array consisting of N points in d dimensions
        corr_length: correlation length of Exponential cov. (default: 1.0)
    Raises:
        AssertionError: if ptset1 or ptset2 are not in (N,d) shape
    Returns:
        covmat: The covariance matrix on ptset1 and ptset2
    """
    assert check_shape_of_ptsets(ptset1, ptset2) is True
    distmtrx = scs.distance_matrix(ptset1, ptset2)
    covmat = np.exp(-distmtrx / (1.0 * corr_length))
    return covmat


def materncov(ptset1, ptset2, reg=1.5, corr_length=1.0):
    """
    Matérn covariance kernel: see Rasmussen & Williams (p.84, 2006).

    Args:
        ptset1: an (N, d) shaped array consisting of N points in d dimensions
        ptset2: an (N, d) shaped array consisting of N points in d dimensions
        reg: regularity of Matérn covariance (default: 1.5)
        corr_length: correlation length of Matern covariance (default: 1.0)
    Raises:
        AssertionError: if ptset1 or ptset2 are not in (N,d) shape
    Returns:
        covmat: The covariance matrix on ptset1 and ptset2
    """
    assert check_shape_of_ptsets(ptset1, ptset2) is True
    distmtrx = scs.distance_matrix(ptset1, ptset2)
    scaled_distmtrx = (np.sqrt(2.0 * reg) * distmtrx) / corr_length
    scaled_distmtrx[np.where(scaled_distmtrx > 0.0)] = (
        2.0 ** (1.0 - reg)
        / scspec.gamma(reg)
        * scaled_distmtrx[np.where(scaled_distmtrx > 0.0)] ** (reg)
        * scspec.kv(reg, scaled_distmtrx[np.where(scaled_distmtrx > 0.0)])
    )
    scaled_distmtrx[np.where(scaled_distmtrx <= 0.0)] = 1.0
    covmat = scaled_distmtrx
    return covmat


def check_shape_of_ptsets(ptset1, ptset2):
    """
    Asserts that ptset1 and ptset2 BOTH have shape (N, d).
    If not, an AssertionError is raised.
    """
    assert (
        np.isscalar(ptset1) is False and np.isscalar(ptset2) is False
    ), "Please enter arrays with shape (k, m) and (l, m)"
    assert (
        len(ptset1.shape) == 2 and len(ptset2.shape) == 2
    ), "Please enter arrays with shape (k, m) and (l, m)"
    assert (
        ptset1.shape[1] == ptset2.shape[1]
    ), "Please enter arrays with shape (k, m) and (l, m)"
    return True


def ibm_dcovd(ptset1, ptset2, diffconst=1.0, q=1):
    """
    Second derivative of q-times integrated
    Brownian motion covariance kernel,
    see Schober, Duvenaud & Hennig (Appendix B, 2014)
    for the recursion formula.

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
        diffconst:  diffusion constant of Brownian motion (sigma)
        q:          order of integration
    Returns:
        covmat:     covariance matrix on ptset1 and ptset2, shape (N, M)
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    assert q_in_valid_range(q) is True
    if q == 1:
        covmat = bmcov(ptset1, ptset2, diffconst)
    else:
        covmat = ibmcov(ptset1, ptset2, diffconst, q=q - 1)
    return covmat


def bmcov(ptset1, ptset2, diffconst=1.0):
    """
    Brownian motion covariance kernel,
    see Schober, Duvenaud & Hennig (Equation (17), 2014)

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
        diffconst:  diffusion constant of Brownian motion (sigma)
    Returns:
        covmat:     covariance matrix on ptset1 and ptset2, shape (N, M)
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    ptset1 = ptset1.reshape((len(ptset1), 1))
    ptset2 = ptset2.reshape((len(ptset2), 1))
    covmat = np.minimum(ptset1, ptset2.T)
    return diffconst ** 2 * covmat


def ibmcov(ptset1, ptset2, diffconst=1.0, q=1):
    """
    q-times integrated Brownian motion covariance kernel,
    see Schober, Duvenaud & Hennig (Appendix B, 2014)

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
        diffconst:  diffusion constant of Brownian motion (sigma)
        q:          order of integration
    Returns:
        covmat:     covariance matrix on ptset1 and ptset2, shape (N, M)
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    assert q_in_valid_range(q) is True
    if q == 1:
        covmat = ibmcov_q1(ptset1, ptset2, diffconst)
    elif q == 2:
        covmat = ibmcov_q2(ptset1, ptset2, diffconst)
    elif q == 3:
        covmat = ibmcov_q3(ptset1, ptset2, diffconst)
    return covmat


def ibmcov_q1(ptset1, ptset2, diffconst=1.0):
    """
    Once integrated Brownian motion covariance kernel,
    see Schober, Duvenaud & Hennig (Equation (18), 2014)

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
        diffconst:  diffusion constant of Brownian motion (sigma)
    Returns:
        covmat:     covariance matrix on ptset1 and ptset2, shape (N, M)
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    ptset1 = ptset1.reshape((len(ptset1), 1))
    ptset2 = ptset2.reshape((len(ptset2), 1))
    distmtrx = scs.distance_matrix(ptset1, ptset2)
    minmtrx = np.minimum(ptset1, ptset2.T)
    covmat = minmtrx ** 3 / 3.0 + distmtrx * minmtrx ** 2 / 2.0
    return diffconst ** 2 * covmat


def ibmcov_q2(ptset1, ptset2, diffconst=1.0):
    """
    Twice integrated Brownian motion covariance kernel,
    see Schober, Duvenaud & Hennig (Equation (21), 2014)

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
        diffconst:  diffusion constant of Brownian motion (sigma)
    Returns:
        covmat:     covariance matrix on ptset1 and ptset2, shape (N, M)
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    ptset1 = ptset1.reshape((len(ptset1), 1))
    ptset2 = ptset2.reshape((len(ptset2), 1))
    distmtrx = scs.distance_matrix(ptset1, ptset2)
    minmtrx = np.minimum(ptset1, ptset2.T)
    covmat = minmtrx ** 5 / 20.0 + (distmtrx / 12.0) * (
        (ptset1 + ptset2.T) * minmtrx ** 3 - minmtrx ** 4 / 2.0
    )
    return diffconst ** 2 * covmat


def ibmcov_q3(ptset1, ptset2, diffconst=1.0):
    """
    3-times integrated Brownian motion covariance kernel,
    see Schober, Duvenaud & Hennig (Equation (24), 2014)

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
        diffconst:  diffusion constant of Brownian motion (sigma)
    Returns:
        covmat:     covariance matrix on ptset1 and ptset2, shape (N, M)
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    ptset1 = ptset1.reshape((len(ptset1), 1))
    ptset2 = ptset2.reshape((len(ptset2), 1))
    distmtrx = scs.distance_matrix(ptset1, ptset2)
    minmtrx = np.minimum(ptset1, ptset2.T)
    maxmtrx = np.maximum(ptset1, ptset2.T)
    covmat = minmtrx ** 7 / 252.0 + distmtrx * minmtrx ** 4 / 720.0 * (
        5 * maxmtrx ** 2 + 2 * ptset1 * ptset2.T + 3 * minmtrx ** 2
    )
    return diffconst ** 2 * covmat


def ibm_covd(ptset1, ptset2, diffconst=1.0, q=1):
    """
    First derivative of q-times integrated
    Brownian motion covariance kernel,
    see Schober, Duvenaud & Hennig (Appendix B, 2014)

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
        diffconst:  diffusion constant of Brownian motion (sigma)
        q:          order of integration
    Returns:
        covmat:     covariance matrix on ptset1 and ptset2, shape (N, M)
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    assert q_in_valid_range(q) is True
    if q == 1:
        covmat = ibm_covd_q1(ptset1, ptset2, diffconst)
    elif q == 2:
        covmat = ibm_covd_q2(ptset1, ptset2, diffconst)
    elif q == 3:
        covmat = ibm_covd_q3(ptset1, ptset2, diffconst)
    return covmat


def q_in_valid_range(q):
    """
    Asserts that q is either 1, 2 or 3.
    For other values, the formulas have not been implemented.
    """
    assert 1 <= q <= 3, "Please enter q in {1,2,3}"
    return True


def ibm_covd_q1(ptset1, ptset2, diffconst=1.0):
    """
    First derivative of (once) integrated
    Brownian motion covariance kernel,
    see Schober, Duvenaud & Hennig (Equation (19), 2014)

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
        diffconst:  diffusion constant of Brownian motion (sigma)
    Returns:
        covmat:     covariance matrix on ptset1 and ptset2, shape (N, M)
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    pt1cols, pt2rows = create_aligned_copies(ptset1, ptset2)
    cond_less = lambda a, b: a ** 2 / 2.0
    cond_more = lambda a, b: a * b - b ** 2 / 2.0
    covmat = np.where(
        pt1cols <= pt2rows, cond_less(pt1cols, pt2rows), cond_more(pt1cols, pt2rows)
    )
    return diffconst ** 2 * covmat


def ibm_covd_q2(ptset1, ptset2, diffconst=1.0):
    """
    First derivative of twice integrated
    Brownian motion covariance kernel,
    see Schober, Duvenaud & Hennig (Equation (22), 2014)

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
        diffconst:  diffusion constant of Brownian motion (sigma)
    Returns:
        covmat:     covariance matrix on ptset1 and ptset2, shape (N, M)
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    pt1cols, pt2rows = create_aligned_copies(ptset1, ptset2)
    cond_less = lambda a, b: -1 * a ** 4 / 24.0 + b * a ** 3 / 6.0
    cond_more = lambda a, b: 1.0 / 24.0 * b ** 2 * (b ** 2 - 4 * a * b + 6 * a ** 2)
    covmat = np.where(
        pt1cols <= pt2rows, cond_less(pt1cols, pt2rows), cond_more(pt1cols, pt2rows)
    )
    return diffconst ** 2 * covmat


def ibm_covd_q3(ptset1, ptset2, diffconst=1.0):
    """
    First derivative of 3-times integrated
    Brownian motion covariance kernel,
    see Schober, Duvenaud & Hennig (Equation (25), 2014)

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
        diffconst:  diffusion constant of Brownian motion (sigma)
    Returns:
        covmat:     covariance matrix on ptset1 and ptset2, shape (N, M)
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    pt1cols, pt2rows = create_aligned_copies(ptset1, ptset2)
    cond_less = lambda a, b: a ** 4 / 720 * (15 * b ** 2 - 6 * a * b + a ** 2)
    cond_more = (
        lambda a, b: b ** 3
        / 720
        * (20 * a ** 3 - 15 * a ** 2 * b + 6 * a * b ** 2 - b ** 3)
    )
    covmat = np.where(
        pt1cols <= pt2rows, cond_less(pt1cols, pt2rows), cond_more(pt1cols, pt2rows)
    )
    return diffconst ** 2 * covmat


def create_aligned_copies(ptset1, ptset2):
    """
    Creates M horizontal copies of ptset1 and N vertical copies of ptset2.
    Used for element-wise comparisons.

    Args:
        ptset1:     shape (N,) array of N points in 1 dimension
        ptset2:     shape (M,) array of M points in 1 dimension
    Returns:
        pt1cols:    shape (N, M) array of M horizontal copies of ptset1
        pt2rows:    shape (N, M) array of N vertical copies of ptset2
    """
    assert ptset_is_timeseries(ptset1) is True
    assert ptset_is_timeseries(ptset2) is True
    pt1cols = np.outer(ptset1, np.ones(ptset2.shape[0]))
    pt2rows = np.outer(np.ones(ptset1.shape[0]), ptset2)
    return pt1cols, pt2rows


def ptset_is_timeseries(ptset):
    """
    Asserts that ptset has shape (N,)
    and is positive (required by BM)
    """
    assert isinstance(ptset, np.ndarray) is True, "Please enter a ptset of shape (N,)"
    assert len(ptset.shape) == 1, "Please enter a ptset of shape (N,)"
    assert (ptset > 0).all(), "BM requires t>0"
    return True


# def ptset_is_timeseries_matrix(ptset):
#     """
#     Asserts that ptset has shape (N, 1)
#     and is positive (required by BM)
#     """
#     assert(isinstance(ptset, np.ndarray) is True), \
#         "Please enter a ptset of shape (N, 1)"
#     assert(len(ptset.shape) == 2), \
#         "Please enter a ptset of shape (N, 1)"
#     assert(ptset.shape[1] == 1), \
#         "Please enter a ptset of shape (N, 1)"
#     assert((ptset > 0).all()), "BM requires t>0"
#     return True
