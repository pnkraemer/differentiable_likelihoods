# coding=utf-8
"""
test_statespace.py

[BOOK]: https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf

Tests include:
    * Successfully sampling a trajectory
    * Comparing A(h), Q(h) and H_1 to
      expected values for q=1 and q=2
    * Making sure that statespaces only accept np.ndarrays
"""
import unittest
import numpy as np
from odefilters import statespace as stsp


class TestInvariantSSM(unittest.TestCase):
    """
    Initialises a time-invariant state space model.
    See Example 4.3 in [BOOK] with q_1 = q_2 = 1.0 (for simplicity)
    """
    def setUp(self):
        twidth = 0.1 * np.random.rand()
        A = np.eye(4) + np.diag(twidth * np.ones(2), 2)
        Q = self._get_tcovar(twidth)
        H = np.eye(2, 4)
        R = np.diag(np.random.rand(2))
        m0 = np.random.rand(4)
        c0 = np.diag(np.random.rand(4))
        self.ssm = stsp.InvariantSSM(transmat=A, transcovar=Q, measmat=H, meascovar=R,
                                     init_mean=m0, init_covar=c0)

    def test_sample_trajectory(self):
        """
        Set up matrices in Example 4.3, initialise time-invariant state space model
        and sample a trajectory.
        Test passes if trajectory has correct shape.
        """
        traj = self.ssm.sample_trajectory(25)
        self.assertEqual(traj.shape[0], 25)
        self.assertEqual(traj.shape[1], len(self.ssm.measmat))


    def _get_tcovar(self, twidth):
        """
        Auxiliary function which returns the transition covariance matrix
        for Example 4.3 (which is fairly messy)
        """
        diagonal = np.array([twidth**3/3., twidth**3/3., twidth, twidth])
        offdiagonal = np.array([twidth**2/2., twidth**2/2.])
        tcovar = np.diag(diagonal) + np.diag(offdiagonal, 2) +\
            np.diag(offdiagonal, -2)
        return tcovar

        

class TestIBM():
    """
    Unittest for IBM class in 'statespace.py'.
    Test whether A(h) and Q(h) are computed correctly for different orders of q.
    """
    def test_transmat_as_expected(self):
        """
        Compares transition matrix to hard coded counterpart
        """
        transmat = self.ssm.get_transmat()
        expctd_transmat = self._expected_transmat
        error = np.linalg.norm(transmat - expctd_transmat)
        self.assertLess(error, 1e-15)

    def test_transcovar_as_expected(self):
        """
        Compares transition covariance to hard coded counterpart
        """
        transcovar = self.ssm.get_transcovar()
        expctd_transcovar = self._expected_transcovar
        error = np.linalg.norm(transcovar - expctd_transcovar)
        self.assertLess(error, 1e-15)

    def test_measmat_as_expected(self):
        """
        Compares measurement matrix to hard coded counterpart
        """
        measmat = self.ssm.get_measmat()
        expctd_measmat = self._expected_measmat
        error = np.linalg.norm(measmat - expctd_measmat)
        self.assertLess(error, 1e-15)


class TestQ1(TestIBM, TestInvariantSSM):

    def setUp(self):
        """
        Set up IBM(1) state space model and precompute hard coded A, Q, H for q=1
        """
        self._setup_ibm_q1()
        stepsize = 0.1 * np.random.rand()
        self.ssm.discretise(stepsize)
        self._expected_transmat = self._get_hardcoded_transmat_q1(stepsize)
        self._expected_transcovar = self._get_hardcoded_transcovar_q1(stepsize)
        self._expected_measmat = self._get_hardcoded_measmat_q1()

    def _setup_ibm_q1(self):
        """
        Set up IBM(1) state space model, i.e. q=1.
        """
        q = 1
        bm_amp = 1.0 + np.random.rand()
        dim = 1
        self.ssm = stsp.IBM(q, dim, bm_amp)

    def _get_hardcoded_transmat_q1(self, stepsize):
        """
        Return expected form for A(h) given q=1.
        """
        expctd_transmat = np.array([[1., stepsize], [0., 1.]])
        return expctd_transmat

    def _get_hardcoded_transcovar_q1(self, stepsize):
        """
        Return expected form for Q(h) given q=1.
        """
        expctd_transcovar = np.array([[stepsize**3/3., stepsize**2/2.],
                                      [stepsize**2/2., stepsize]])
        return self.ssm.diffconst**2 * expctd_transcovar

    def _get_hardcoded_measmat_q1(self):
        """
        Return expected form for H_1 given q=1.
        """
        expctd_measmat = np.array([[0., 1.]])
        return expctd_measmat


class TestQ2(TestIBM, TestInvariantSSM):

    def setUp(self):
        """
        Set up IBM(2) state space model and precompute hard coded A, Q, H.
        """
        self._setup_ibm_q2()
        stepsize = 0.1 * np.random.rand()
        self.ssm.discretise(stepsize)
        self._expected_transmat = self._get_hardcoded_transmat_q2(stepsize)
        self._expected_transcovar = self._get_hardcoded_transcovar_q2(stepsize)
        self._expected_measmat = self._get_hardcoded_measmat_q2()

    def _setup_ibm_q2(self):
        """
        Set up IBM(2) state space model.
        """
        q = 2
        bm_amp = 1.0 + np.random.rand()
        dim = 1
        self.ssm = stsp.IBM(q, dim, bm_amp)

    def _get_hardcoded_transmat_q2(self, stepsize):
        """
        Return expected form for A(h) given q=2.
        """
        expctd_transmat = np.array([[1., stepsize, stepsize**2/2.],
                                    [0., 1., stepsize],
                                    [0., 0., 1.]])
        return expctd_transmat

    def _get_hardcoded_transcovar_q2(self, stepsize):
        """
        Return expected form for Q(h) given q=2.
        """
        expctd_transcovar = np.array([[stepsize**5/20., stepsize**4/8., stepsize**3/6.],
                                      [stepsize**4/8., stepsize**3/3., stepsize**2/2.],
                                      [stepsize**3/6., stepsize**2/2., stepsize]])
        return self.ssm.diffconst**2 * expctd_transcovar

    def _get_hardcoded_measmat_q2(self):
        """
        Return expected form for H_1 given q=2.
        """
        expctd_measmat = np.array([[0., 1., 0.]])
        return expctd_measmat


class TestBadInputs(unittest.TestCase):
    """
    Tests whether inputs to InvariantSSM are not matrices,
    respectively arrays for init_mean.
    """
    def setUp(self):
        """Set up working input matrices"""
        self.good_transmat = np.eye(2)
        self.good_transcovar = np.eye(2)
        self.good_measmat = np.eye(2)
        self.good_meascovar = np.eye(2)
        self.good_init_mean = np.zeros(2)
        self.good_init_covar = np.eye(2)
    
    def test_assertion_error_for_floats(self):
        """
        Swap each working matrix for a nonworking matrix
        one by one and test whether AssertionErrors are raised.
        """
        self.check_wrong_transmat()
        self.check_wrong_transcovar()
        self.check_wrong_measmat()
        self.check_wrong_meascovar()
        self.check_wrong_init_mean()
        self.check_wrong_init_covar()

    def check_wrong_transmat(self):
        """
        Test passes if putting a float in
        for transmat raises an AssertionError.
        """
        bad_transmat = 0.1
        with self.assertRaises(AssertionError):
            self.ssm = stsp.InvariantSSM(transmat=bad_transmat,
                                         transcovar=self.good_transcovar,
                                         measmat=self.good_measmat,
                                         meascovar=self.good_meascovar,
                                         init_mean=self.good_init_mean,
                                         init_covar=self.good_init_covar)

    def check_wrong_transcovar(self):
        """
        Test passes if putting a float in
        for transcovar raises an AssertionError.
        """
        bad_transcovar = 0.1
        with self.assertRaises(AssertionError):
            self.ssm = stsp.InvariantSSM(transmat=self.good_transmat,
                                         transcovar=bad_transcovar,
                                         measmat=self.good_measmat,
                                         meascovar=self.good_meascovar,
                                         init_mean=self.good_init_mean,
                                         init_covar=self.good_init_covar)

    def check_wrong_measmat(self):
        """
        Test passes if putting a float in
        for measmat raises an AssertionError.
        """
        bad_measmat = 0.1
        with self.assertRaises(AssertionError):
            self.ssm = stsp.InvariantSSM(transmat=self.good_transmat,
                                         transcovar=self.good_transcovar,
                                         measmat=bad_measmat,
                                         meascovar=self.good_meascovar,
                                         init_mean=self.good_init_mean,
                                         init_covar=self.good_init_covar)

    def check_wrong_meascovar(self):
        """
        Test passes if putting a float in
        for meascovar raises an AssertionError.
        """
        bad_meascovar = 0.1
        with self.assertRaises(AssertionError):
            self.ssm = stsp.InvariantSSM(transmat=self.good_transmat,
                                         transcovar=self.good_transcovar,
                                         measmat=self.good_measmat,
                                         meascovar=bad_meascovar,
                                         init_mean=self.good_init_mean,
                                         init_covar=self.good_init_covar)

    def check_wrong_init_mean(self):
        """
        Test passes if putting a float in
        for init_mean raises an AssertionError.
        """
        bad_init_mean = 0.1
        with self.assertRaises(AssertionError):
            self.ssm = stsp.InvariantSSM(transmat=self.good_transmat,
                                         transcovar=self.good_transcovar,
                                         measmat=self.good_measmat,
                                         meascovar=self.good_meascovar,
                                         init_mean=bad_init_mean,
                                         init_covar=self.good_init_covar)

    def check_wrong_init_covar(self):
        """
        Test passes if putting a float in
        for init_covar raises an AssertionError.
        """
        bad_init_covar = 0.1
        with self.assertRaises(AssertionError):
            self.ssm = stsp.InvariantSSM(transmat=self.good_transmat,
                                         transcovar=self.good_transcovar,
                                         measmat=self.good_measmat,
                                         meascovar=self.good_meascovar,
                                         init_mean=self.good_init_mean,
                                         init_covar=bad_init_covar)

