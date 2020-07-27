"""
Todo:
    * Make gradeval etc. independent of lkldeval (i.e. precomputations)
    * bring everything into the l(z) = exp(-E(z)) format and stick to it
    * Think about priors
    * UNITTEST!!!
    * Clean up the auxiliary functions --- are they all needed??
"""


import sys
from dataclasses import dataclass
import numpy as np
from difflikelihoods import linearisation
from difflikelihoods import odesolver as solver
from difflikelihoods import linearised_odesolver as linsolver
from difflikelihoods import linearised_ode as linode
from difflikelihoods import auxiliary as aux


@dataclass
class InvProblemData:
    """
    """

    evalpts: np.ndarray
    data: np.ndarray
    var: float


class InvProblemLklhd:
    """
    """

    def __init__(self, ipdata, ivp, linsolver, stepsize, with_jacob=False):
        """
        Args:
            ipdata:     InvProblemData object
            ivp:        LinearisedODE object
            linsolver:  LinearisedODESolver object
            stepsize:   scalar
            with_jacob: boolean, indicates if gradients can be evaluated

        If with_jacob is True, a kernel prefactor matrix is computed:
            kprefact:   ndarray of shape (ndata*ndim, ntsteps-1).
                        Vertical stack of all kprefactors at timesteps.
        """
        assert ivp.is_linearised is True, "Please enter a LinearisedODE object"
        self.ipdata = ipdata
        self.ivp = ivp
        self.linsolver = linsolver
        self.stepsize = stepsize
        self.tsteps, self.evalidcs = self.compute_tsteps()
        self.with_jacob = with_jacob
        if with_jacob is True:
            self.kprefact = self.compute_kernel_prefactor()

    def compute_tsteps(self):
        """
        Computes tsteps and indices of evalpts in tsteps.
        """
        tsteps, __, __, __, __ = self.linsolver.solve(self.ivp, self.stepsize)
        evalpts = self.ipdata.evalpts
        assert np.in1d(evalpts, tsteps).prod() == 1
        evalidcs = [list(tsteps).index(evalpt) for evalpt in evalpts]
        return tsteps, evalidcs

    def compute_kernel_prefactor(self):
        """
        """
        return linearisation.compute_kernel_prefactor(
            self.linsolver.filt.ssm, self.ipdata.var, self.tsteps, self.ipdata.evalpts
        )

    # def lklhdeval(self, par):
    #     """
    #     Evaluates l(s) ~ exp(-E(s)),
    #         E(s) = (data - m_s(t))^T S^{-1} (data-m_s(t))
    #     by regarding it as a Gaussian pdf evaluated at data with mean
    #     m_s(t) and covariance S.

    #     Args:
    #         par:        parameter to evaluate at, shape (npar,)
    #     Computes:
    #         self.mean:  Estimate for mean-values at evalpts given par.
    #                     vertical stack, shape (ndata*ndim,).
    #                         [(data1, dim1), (data1, dim2), ..., (dataM, dimD)]
    #         self.ipvar: Estimate for the variance, shape (ndata*ndim, ndata*ndim)
    #                     Should be block-diagonal:
    #                         ndata blocks of ndim x ndim matrices
    #         self.jacob: Estimate of Jacobian of filter-output w.r.t. par.
    #                     Product of kprefact and rhs_parts.
    #                     Shape (ndata*ndim, npar).
    #                     Same alignment as meanguess ("outer loop is ndata")
    #                     Not needed here, precomputed for self.gradeval().
    #     Returns:
    #                     Gaussian PDF with mean self.mean and covariance
    #                     self.ipvar evaluated at data; scalar
    #     """
    #     # print("lkld", par)
    #     # self.mean, self.ipvar, self.jacob = self.forwardsolve(par)
    #     # # print(self.ipdata.data.shape, self.mean.shape, self.ipvar.shape)
    #     # returnval = custom_gaussian_pdf(self.ipdata.data, self.mean, self.ipvar)
    #     return np.exp(-self.potenteval(par))

    def potenteval(self, par):
        """
        """
        self.mean, self.ipvar, self.jacob = self.forwardsolve(par)
        # print(self.ipdata.data.shape, self.mean.shape, self.ipvar.shape)
        returnval = custom_potential(self.ipdata.data, self.mean, self.ipvar)
        # print("potential", returnval, "at", par, "with lkld", np.exp(-returnval))
        return returnval

    def gradeval(self, par):
        """
        For a likelihood of the form l(s) ~ exp(-E(s)),
            E(s) = (data - m_s(t))^T S^{-1} (data - m_s(t))
        where s is par, this function evalutes
            \nabla_s E(s) = -J^T S^{-1} (data - m_s(t))
        In other words, it returns the gradient of the negative
        log-likelihood of the map par -> l(par).

        Args:
            par:    parameter, shape (npar,)
        Uses:
            flat_data:  shape (ndata*ndim,); flattened version
                        of self.ipdata.data
            self.mean:  Estimate for mean-values at evalpts given par.
                        vertical stack, shape (ndata*ndim,).
                            [(data1, dim1), (data1, dim2), ..., (dataM, dimD)]
            self.ipvar: Estimate for the variance, shape (ndata*ndim, ndata*ndim)
                        Should be block-diagonal:
                            ndata blocks of ndim x ndim matrices
        Returns:
            Gradient of the negative log-likelihood of the map
            par -> par.

        NOTE:
            Requires that lklhdeval(par) was evaluated at the same par
            beforehand (as self.mean and self.ipvar are being used).
        """
        # self.mean, self.ipvar, self.jacob = self.forwardsolve(par)      # necessary for HMC
        # print("grad", par)
        flat_data = self.ipdata.data.flatten()
        grad = np.dot(-self.jacob.T, np.linalg.solve(self.ipvar, flat_data - self.mean))
        # print("Gradnorm:", np.linalg.norm(grad))
        # print("Grad:", (grad))
        return grad

    def hesseval(self, par):
        """
        For a likelihood of the form l(s) ~ exp(-E(s)),
            E(s) = (data - m_s(t))^T S^{-1} (data - m_s(t))
        where s is par, this function evalutes
            \nabla_s^2 E(s) = J^T S^{-1} J^T
        In other words, it returns the Hessian of the negative
        log-likelihood of the map par -> l(par).

        Args:
            par:    parameter, shape (npar,)
        Uses:
            self.ipvar: Estimate for the variance, shape (ndata*ndim, ndata*ndim)
                        Should be block-diagonal:
                            ndata blocks of ndim x ndim matrices
        Returns:
            Gradient of the negative log-likelihood of the map
            par -> par.

        NOTE:
            Requires that lklhdeval(par) was evaluated at the same par
            beforehand (as self.mean and self.ipvar are being used).
        """
        # self.mean, self.ipvar, self.jacob = self.forwardsolve(par)      # necessary for HMC
        solved = np.linalg.solve(self.ipvar, self.jacob)
        hess = np.dot(self.jacob.T, solved)
        return hess

    # def precond_gradeval(self, par):
    #     """
    #     """
    #     # self.mean, self.ipvar, self.jacob = self.forwardsolve(par)      # necessary for HMC
    #     hess = self.hesseval(par)
    #     grad = self.gradeval(par)
    #     return np.linalg.solve(hess, grad)

    def forwardsolve(self, par):
        """
        Solves ODE at parameter par and returns mean, stdev and Jacobian
        in the right shape for usage in self.lklhdeval() and self.gradeval().

        Args:
            par: parameter to evaluate the likelihood at. Shape (npar,)

        Returns:
            meanguess:  Estimate for mean-values at evalpts given par.
                        vertical stack, shape (ndata*ndim,).
                            [(data1, dim1), (data1, dim2), ..., (dataM, dimD)]
            ipvar:      Estimate for the variance, shape (ndata*ndim, ndata*ndim)
                        Should be block-diagonal:
                            ndata blocks of ndim x ndim matrices
            jacob:      Estimate of Jacobian of filter-output w.r.t. par.
                        Product of kprefact and rhs_parts.
                        Shape (ndata*ndim, npar).
                        Same alignment as meanguess ("outer loop is ndata")
        """
        self.ivp.params = par
        __, mean, stdev, rhs_parts, __ = self.linsolver.solve(self.ivp, self.stepsize)
        meanguess, stdevguess = self.extract_measurement_guesses(mean, stdev)
        flat_meanguess = meanguess.flatten()
        ipvar = np.diag((self.ipdata.var + stdevguess ** 2).flatten())
        # print(ipvar)
        if self.with_jacob is True:
            jacob = linearisation.compute_jacobian(
                self.ipdata.evalpts, self.tsteps, self.kprefact, rhs_parts
            )
        else:
            jacob = None
        return flat_meanguess, ipvar, jacob

    def extract_measurement_guesses(self, mean, stdev):
        """
        Args:
            mean:   mean as returned by LinearisedODESolver.solve,
                    shape (ntsteps, ndim, q+1)
            stdev:  stdev as returned by LinearisedODESolver.solve
                    shape (ntsteps, ndim, q+1)
        Returns:
            mean_guess:     evaluations of mean approximation at datapts,
                            shape (ndata, ndim)
            stdev_guess:    evaluates of stdev approximation at datapts
                            shape (ndsta, ndim)
        """
        if np.isscalar(self.ivp.initval):
            odedim = 1
        else:
            odedim = len(self.ivp.initval)
        mean_trajectory = solver.get_trajectory_ddim(mean, odedim, 0)
        mean_guess = mean_trajectory[self.evalidcs]
        stdev_trajectory = solver.get_trajectory_ddim(stdev, odedim, 0)
        stdev_guess = stdev_trajectory[self.evalidcs]
        return mean_guess, stdev_guess


class InvProblemLklhdClassic(InvProblemLklhd):
    """
    """

    def forwardsolve(self, par):
        """
        Solves ODE at parameter par and returns mean, stdev and Jacobian
        in the right shape for usage in self.lklhdeval() and self.gradeval().

        Args:
            par: parameter to evaluate the likelihood at. Shape (npar,)

        Returns:
            meanguess:  Estimate for mean-values at evalpts given par.
                        vertical stack, shape (ndata*ndim,).
                            [(data1, dim1), (data1, dim2), ..., (dataM, dimD)]
            ipvar:      Estimate for the variance, shape (ndata*ndim, ndata*ndim)
                        Should be block-diagonal:
                            ndata blocks of ndim x ndim matrices
            jacob:      Estimate of Jacobian of filter-output w.r.t. par.
                        Product of kprefact and rhs_parts.
                        Shape (ndata*ndim, npar).
                        Same alignment as meanguess ("outer loop is ndata")
        """
        self.ivp.params = par
        __, mean, stdev, rhs_parts, __ = self.linsolver.solve(self.ivp, self.stepsize)
        meanguess, stdevguess = self.extract_measurement_guesses(mean, stdev)
        flat_meanguess = meanguess.flatten()
        ipvar = np.diag((self.ipdata.var * np.ones(stdevguess.shape)).flatten())
        # print(ipvar)
        if self.with_jacob is True:
            jacob = linearisation.compute_jacobian(
                self.ipdata.evalpts, self.tsteps, self.kprefact, rhs_parts
            )
        else:
            jacob = None
        return flat_meanguess, ipvar, jacob


def custom_gaussian_pdf(loc, mean, covar):
    """
    Evaluates multidimensional Gaussian PDF with mean mmat and
    variance vmat at point locmat.

    Args:
        loc:    shape (ndata, ndim)
                flattened into shape (ndata*ndim,)

    """
    reshaped_loc = loc.flatten().reshape((1, len(loc.flatten())))
    return aux.multivariate_gaussian(reshaped_loc, mean, covar)[0]


def custom_potential(loc, mean, covar):
    """
    Evaluates multidimensional Gaussian PDF with mean mmat and
    variance vmat at point locmat.

    Args:
        loc:    shape (ndata, ndim)
                flattened into shape (ndata*ndim,)

    """
    reshaped_loc = loc.flatten().reshape((1, len(loc.flatten())))
    return multivariate_potential(reshaped_loc, mean, covar)[0]


def multivariate_potential(ptset2, meanvec, covmat):
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
    # assert array_shape_point_mean_covar(ptset, meanvec, covmat) is True
    dim = meanvec.shape[0]
    ptset = ptset2 - meanvec
    # scaling = 1.0 / (np.sqrt((2.0 * np.pi)**dim * np.linalg.det(covmat)))
    # print(np.linalg.det(covmat))
    solution = np.linalg.solve(covmat, ptset.T).T
    # print(ptset, solution)
    matvecprod = np.einsum("ij,ij->i", ptset, solution)
    # if matvecprod > 100:
    #     gaussian = np.zeros(matvecprod.shape)
    # else:
    #     gaussian = matvecprod/2.
    # return scaling * gaussian
    return matvecprod / 2
