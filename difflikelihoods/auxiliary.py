# coding=utf-8
"""
auxiliary.py

Auxiliary functions, sporadically needed throughout different modules.

Todo:
    * Rename this module from auxiliary into gaussianpdf.py or sth
"""
import numpy as np


def gaussian_pdf(ptset, mean, covar):
    """
    Evaluates Gaussian density function in a vectorised way
    without relying on scipy.stats. Self defined is much quicker
    and subsequently speeds up tasks like integration massively.
    Used e.g. in bayesianquadrature.py.

    Args:
        ptset: either shape (N,) or shape (N,d)
        mean: either scalar or shape (d,)
        covar: either scalar or shape (d,d)
    Returns:
        result: evaluation of Gaussian PDF, shape (N,)
    """
    if np.isscalar(mean) is True:
        result = univariate_gaussian(ptset, mean, covar)
    else:
        result = multivariate_gaussian(ptset, mean, covar)
    return result


def univariate_gaussian(ptset, mean, covar):
    """
    Vectorised evaluation of univariate Gaussian pdf.

    Args:
        ptset: shape (N,)
        mean: scalar
        covar: scalar
    Raises:
        AssertionError: if ptset, mean and covar have not consistent shape
    Returns:
        result: evaluation of Gaussian PDF, shape (N,)
    """
    assert scalar_shape_point_mean_covar(ptset, mean, covar) is True
    pointreshape = ptset.reshape((len(ptset), 1))
    meanreshape = mean * np.ones(1)
    covarreshape = covar * np.ones((1, 1))
    result = multivariate_gaussian(pointreshape, meanreshape, covarreshape)
    return result


def scalar_shape_point_mean_covar(ptset, mean, covar):
    """Tests whether point, mean and covar are all scalars."""
    assert isinstance(ptset, np.ndarray) is True, "Please enter shape (n,) ptset"
    assert len(ptset.shape) == 1, "Please enter a shape (n,) ptset"
    assert np.isscalar(mean) is True, "Please enter a scalar mean value"
    assert np.isscalar(covar) is True, "Please enter a scalar covar value"
    return True


def multivariate_gaussian(ptset, meanvec, covmat):
    """
    Vectorised evaluation of multivariate Gaussian pdf.

    Args:
        ptset: shape (N, d)
        mean: shape (d,)
        covar: shape (d,)
    Raises:
        AssertionError: if ptset, mean and covar have not consistent shape
    Returns:
        result: evaluation of Gaussian PDF, shape (N,)
    """
    assert array_shape_point_mean_covar(ptset, meanvec, covmat) is True
    dim = meanvec.shape[0]
    ptset = ptset - meanvec
    scaling = 1.0 / (np.sqrt((2.0 * np.pi) ** dim * np.linalg.det(covmat)))
    solution = np.linalg.solve(covmat, ptset.T).T
    matvecprod = np.einsum("ij,ij->i", ptset, solution)
    if matvecprod > 100:
        gaussian = np.zeros(matvecprod.shape)
    else:
        gaussian = np.exp(-1 * matvecprod / 2)
    return scaling * gaussian


def array_shape_point_mean_covar(ptset, mean, covar):
    """Tests whether ptset, mean and covar have matching shapes."""
    assert isinstance(ptset, np.ndarray) is True, "Please enter shape (N,d) ptset"
    assert isinstance(mean, np.ndarray) is True, "Please enter shape (d,) mean"
    assert isinstance(covar, np.ndarray) is True, "Please enter shape (d,d) covar"
    assert len(ptset.shape) == 2, "Please enter an (N,d) shaped pointset"
    assert len(mean.shape) == 1, "Please enter an (d,) shaped mean"
    assert (
        len(covar.shape) == 2 and covar.shape[0] == covar.shape[1]
    ), "Please enter a covar with shape (d,d)"
    assert (
        ptset.shape[1] == mean.shape[0] == covar.shape[0]
    ), "Please enter consistent point, mean and covar"
    return True


# END OF FILE
