# # coding=utf-8
# """
# test_sampling.py
#
# Test the sampling.py module by
#     1) Making metropolishastings_nd() complaing about wrong inputs
#     2) Computing a Kolmogorov-Smirnov test to see
#        whether using a Gaussian distribution as input
#        yields samples which are (roughly) Gaussian distributed.
#
# Suggestions for further unittests are welcome.
# """
# import unittest
# import numpy as np
# from odefilters import sampling as samp
# import scipy.stats as sps
#
#
# def is_distribution_gaussian_1d(samples, mean, cov, siglevel=0.5*1e-4):
#     """
#     Performs Kolmogorov Smirnov test in 1Dto check
#     whether samples follow a Gaussian distribution.
#
#     Procedure:
#         Compute empirical CDF at sorted samples
#         and compute error w.r.t. true Gaussian CDF.
#         Reject hypothesis with given significance level
#
#     Reference:
#         https://en.wikipedia.org/wiki/Kolmogor
#         ov%E2%80%93Smirnov_test
#
#     Disclaimer:
#         We compute the RMSE between empirical and
#         true CDF instead of L^inf error, because
#         the yields more stable test results.
#
#     Args:
#         samples: states to be checked, shape (N,)
#         mean: expected mean, scalar
#         cov: expected covariance, scalar
#         siglevel: significance level, default is 0.0005
#     Returns:
#         is_accepted: boolean
#     """
#     sorted_samples = np.sort(samples)
#     empcdf = np.arange(1, len(samples) + 1) / len(samples)
#     truecdf = sps.norm.cdf(sorted_samples, loc=mean, scale=np.sqrt(cov))
#     rmse = np.linalg.norm(empcdf - truecdf)/np.sqrt(len(samples))
#     is_accepted = accept_or_reject(len(samples), siglevel, rmse)
#     return is_accepted
#
#
# def accept_or_reject(nsamps, siglevel, discrep):
#     """
#     Compares computed discrepancy to
#     allowed discrepancy according to test
#     statistics of Kolmogorov-Smirnov tests.
#     """
#     allowed_discrep = np.sqrt(-0.5*np.log(siglevel))/np.sqrt(nsamps)
#     if discrep < allowed_discrep:
#         is_accepted = True
#     else:
#         is_accepted = False
#     return is_accepted
#
#
# class TestKolmogorovSmirnov(unittest.TestCase):
#     """
#     Lets make sure the KS test above works.
#     """
#     def test_exponential_not_gaussian(self):
#         """
#         KS-test for Gaussian distribution should
#         reject samples from exponential distribution.
#         """
#         samples = np.random.standard_exponential(1000)
#         sample_mean = np.average(samples)
#         sample_cov = np.average(samples**2) - sample_mean**2
#         is_acc = is_distribution_gaussian_1d(samples, mean=sample_mean, cov=sample_cov)
#         self.assertEqual(is_acc, False)
#
#     def test_cauchy_not_gaussian(self):
#         """
#         KS-test for Gaussian distribution should
#         reject samples from Cauchy distribution.
#         """
#         samples = np.random.standard_cauchy(1000)
#         sample_mean = np.average(samples)
#         sample_cov = np.average(samples**2) - sample_mean**2
#         is_acc = is_distribution_gaussian_1d(samples, mean=sample_mean, cov=sample_cov)
#         self.assertEqual(is_acc, False)
#
#     def test_gaussian_wrong_mean(self):
#         """
#         KS-test for a different mean than the
#         one which generated the functions should reject.
#         """
#         samples = 2 + 1.0*np.random.randn(100)
#         is_acc = is_distribution_gaussian_1d(samples, mean=3, cov=1.0)
#         self.assertEqual(is_acc, False)
#
# class TestMetropolisHastings():
#     """
#     Set up test parameters for Metropolis Hastings.
#     """
#     def setUp(self):
#         """
#         Set up parameters for small MH problem.
#         """
#         self.mean = 1.0
#         self.nsamps = 350       # smaller than 350 is unreliable
#         self.init_state = np.array([self.mean])
#         self.pwidth = 0.75
#         self.dens = lambda x: np.exp(-(x-self.mean).dot((x-self.mean).T)/2.)/np.sqrt(2*np.pi)
#         self.grad = lambda x: (x-self.mean)/np.sqrt(2*np.pi)
#
#
# class TestIsGaussian(TestMetropolisHastings, unittest.TestCase):
#     """
#     Tests whether both MH versions give Gaussian samples
#     when statpdf is Gaussian.
#     """
#     def test_random_walk(self):
#         """Compute RW samples and perform KS test."""
#         states, likelihoods, __ = samp.metropolishastings_nd(self.nsamps, self.dens, self.init_state,
#                                                 self.pwidth, sampler="rw")
#         is_accepted = is_distribution_gaussian_1d(states[:, 0], self.mean, cov=1.0)
#         self.assertEqual(is_accepted, True)
#
#     def test_langevin(self):
#         """Compute LANGEVIN samples and perform KS test."""
#         states, likelihoods, __ = samp.metropolishastings_lang(
#             self.dens, self.grad, self.nsamps, self.init_state, self.pwidth, 0)
#
#         is_accepted = is_distribution_gaussian_1d(states[:, 0], self.mean, cov=1.0)
#         self.assertEqual(is_accepted, True)
#
#     def test_controlvariate(self):
#         """Compute LANG samples which should fail KS test with wrong mean."""
#         states, likelihoods, __ = samp.metropolishastings_nd(self.nsamps, self.dens, self.init_state,
#                                                 self.pwidth, sampler="lang", grad=self.grad)
#         is_accepted = is_distribution_gaussian_1d(states[:, 0], self.mean*2, cov=1.0)
#         self.assertEqual(is_accepted, False)
#
#
# class TestWrongInputs(TestMetropolisHastings, unittest.TestCase):
#     """
#     Tests whether metropolishastings_nd() complains about
#     inputs with unexpected shape or content.
#     """
#     def test_scalar_init_mean(self):
#         """
#         MH function should raise AssertionErrors
#         if init_state is a scalar.
#         """
#         with self.assertRaises(AssertionError):
#             samp.metropolishastings_nd(self.nsamps, self.dens, self.init_state[0],
#                                        self.pwidth, sampler="rw")
#
#     def test_wrong_sampler_keyword(self):
#         """
#         MH function should raise AssertionErrors if
#         sampler keyword is neither 'rw' nor 'lang'.
#         """
#         with self.assertRaises(AssertionError):
#             samp.metropolishastings_nd(self.nsamps, self.dens, self.init_state,
#                                        self.pwidth, sampler="rubbish")
#
#     def test_lang_without_gradient(self):
#         """
#         MH function should raise AssertionErrors if
#         sampler keyword is 'lang' but no gradient is provided.
#         """
#         with self.assertRaises(AssertionError):
#             samp.metropolishastings_nd(self.nsamps, self.dens, self.init_state,
#                                        self.pwidth, sampler="lang")
#
