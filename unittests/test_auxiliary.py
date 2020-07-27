# # coding=utf-8
# """
# test_auxiliary.py
#
# Test auxiliary functions.
# """
# import unittest
# import numpy as np
# import scipy.stats as sps
# from odefilters import auxiliary as aux
#
# np.random.seed(568)
#
#
#
# class TestGaussianPDF(unittest.TestCase):
#     """
#     Tests gaussian_pdf() function.
#     Checks whether it complains about
#     wrong inputs returns vectorised evaluations
#     and compares output to scipy.stats.
#     """
#     def setUp(self):
#         """
#         Set up pointsets, means and covariances
#         that 'would work'
#         """
#         self.ptset_univar = np.random.rand(10)
#         self.mean_univar = np.random.rand()
#         self.cov_univar = np.random.rand()
#
#         self.ptset_multivar = np.random.rand(10, 2)
#         self.mean_multivar = np.random.rand(2)
#         self.cov_multivar = np.random.rand() * np.eye(2)
#
#
#     def test_compare_with_scipystats_univar(self):
#         """Check whether the output is the same as for scipy.stats"""
#         scs_output = sps.norm.pdf(self.ptset_univar[0],
#                                   self.mean_univar, np.sqrt(self.cov_univar))
#         aux_output = aux.gaussian_pdf(self.ptset_univar,
#                                       self.mean_univar, self.cov_univar)[0]
#         self.assertAlmostEqual(aux_output, scs_output, places=15)
#
#
#     def test_inconsistent_univar(self):
#         """
#         Make gaussian_pdf() complain about
#         wrongly shaped inputs in UNIVARIATE setting.
#         """
#         self.check_proper_input_univar()
#         self.check_only_ptset_proper_univar()
#         self.check_only_mean_proper_univar()
#         self.check_only_covar_proper_univar()
#
#     def check_proper_input_univar(self):
#         """
#         Everything has the right shape---this should pass.
#         """
#         aux.gaussian_pdf(self.ptset_univar, self.mean_univar, self.cov_univar)
#
#     def check_only_ptset_proper_univar(self):
#         """
#         Only the pointset has the right shape.
#         Recycle 'proper' multivariate inputs as
#         wrong univariate inputs.
#         """
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_univar, self.mean_multivar, self.cov_univar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_univar, self.mean_univar, self.cov_multivar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_univar, self.mean_multivar, self.cov_multivar)
#
#     def check_only_mean_proper_univar(self):
#         """
#         Only the mean has the right shape.
#         Recycle 'proper' multivariate inputs as
#         wrong univariate inputs.
#         """
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_multivar, self.mean_univar, self.cov_univar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_univar, self.mean_univar, self.cov_multivar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_multivar, self.mean_univar, self.cov_multivar)
#
#     def check_only_covar_proper_univar(self):
#         """
#         Only the covariance has the right shape.
#         Recycle 'proper' multivariate inputs as
#         wrong univariate inputs.
#         """
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_multivar, self.mean_univar, self.cov_univar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_univar, self.mean_multivar, self.cov_univar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_multivar, self.mean_univar, self.cov_univar)
#
#
#     def test_compare_with_scipystats_multivar(self):
#         """
#         Check whether the output is the same as for scipy.stats.
#         """
#         scs_output = sps.multivariate_normal.pdf(self.ptset_multivar[0],
#                                                  self.mean_multivar, self.cov_multivar)
#         aux_output = aux.gaussian_pdf(self.ptset_multivar,
#                                       self.mean_multivar, self.cov_multivar)[0]
#         self.assertAlmostEqual(aux_output, scs_output, places=15)
#
#
#     def test_inconsistent_multivar(self):
#         """
#         Make gaussian_pdf() complain about
#         wrongly shaped inputs in MULTIVARIATE setting.
#         """
#         self.check_proper_input_multivar()
#         self.check_only_ptset_proper_multivar()
#         self.check_only_mean_proper_multivar()
#         self.check_only_covar_proper_multivar()
#
#     def check_proper_input_multivar(self):
#         """
#         Everything has the right shape---this should pass.
#         """
#         aux.gaussian_pdf(self.ptset_multivar, self.mean_multivar, self.cov_multivar)
#
#     def check_only_ptset_proper_multivar(self):
#         """
#         Only the pointset has the right shape.
#         Recycle 'proper' univariate inputs as
#         wrong multivariate inputs.
#         """
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_multivar, self.mean_multivar, self.cov_univar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_multivar, self.mean_univar, self.cov_multivar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_multivar, self.mean_univar, self.cov_univar)
#
#
#     def check_only_mean_proper_multivar(self):
#         """
#         Only the mean has the right shape.
#         Recycle 'proper' univariate inputs as
#         wrong multivariate inputs.
#         """
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_multivar, self.mean_multivar, self.cov_univar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_univar, self.mean_multivar, self.cov_multivar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_univar, self.mean_multivar, self.cov_univar)
#
#     def check_only_covar_proper_multivar(self):
#         """
#         Only the covar has the right shape.
#         Recycle 'proper' univariate inputs as
#         wrong multivariate inputs.
#         """
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_multivar, self.mean_univar, self.cov_multivar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_univar, self.mean_multivar, self.cov_multivar)
#         with self.assertRaises(AssertionError):
#             aux.gaussian_pdf(self.ptset_univar, self.mean_univar, self.cov_multivar)
