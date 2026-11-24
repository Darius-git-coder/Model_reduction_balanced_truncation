"""Numerical properties of balanced realizations and their truncations."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from helpers import balancing, fdm_2d_poisson, lyapunov_gramians


class BalancingTests(unittest.TestCase):
    def test_balance_and_preserve_transfer_function(self):
        A = np.array([[-1., 2., 0.], [0., -3., 1.], [0., 0., -5.]])
        B, C = np.eye(3), np.eye(3)
        Ab, Bb, Cb, sigma = balancing(A, B, C, 1e-14)
        # With B=C=I, Bb and Cb are the two coordinate maps themselves.
        assert_allclose(Bb @ Cb, np.eye(3), atol=1e-13)
        P, Q = lyapunov_gramians(Ab, Bb, Cb)
        assert_allclose(P, np.diag(sigma), atol=1e-12)
        assert_allclose(Q, np.diag(sigma), atol=1e-12)
        for omega in (0., 1., 10.):
            original = C @ np.linalg.solve(1j*omega*np.eye(3)-A, B)
            balanced = Cb @ np.linalg.solve(1j*omega*np.eye(3)-Ab, Bb)
            assert_allclose(balanced, original, atol=1e-12)
        self.assertTrue(np.all(np.diff(sigma) <= 0))

    def test_remove_unreachable_state(self):
        A = np.diag([-1., -2., -3.])
        B = np.array([[1.], [0.], [0.]])
        C = np.array([[1., 0., 1.]])
        Ab, Bb, Cb, sigma = balancing(A, B, C, 1e-14)
        self.assertEqual(Ab.shape, (1, 1))
        assert_allclose(sigma, [0.5])
        assert_allclose(Cb @ np.linalg.solve(-Ab, Bb), [[1.]])

    def test_zero_output(self):
        Ab, Bb, Cb, sigma = balancing(-np.eye(2), np.ones((2, 1)), np.zeros((1, 2)), 1e-14)
        self.assertEqual((Ab.shape, Bb.shape, Cb.shape), ((0, 0), (0, 1), (1, 0)))
        self.assertEqual(sigma.size, 0)

    def test_heat_truncation_is_stable_and_obeys_sampled_error_bound(self):
        A, B, C1, *_ = fdm_2d_poisson(5)
        for C in (C1, np.eye(A.shape[0])):
            Ab, Bb, Cb, sigma = balancing(A, B, C, 1e-14)
            r = min(3, len(sigma)-1)
            Ar, Br, Cr = Ab[:r, :r], Bb[:r], Cb[:, :r]
            self.assertLess(np.max(np.linalg.eigvals(Ar).real), 0)
            bound = 2*np.sum(sigma[r:])
            for omega in np.r_[0., np.logspace(-3, 3, 20)]:
                G = C @ np.linalg.solve(1j*omega*np.eye(A.shape[0])-A.toarray(), B)
                Gr = Cr @ np.linalg.solve(1j*omega*np.eye(r)-Ar, Br)
                self.assertLessEqual(np.linalg.norm(G-Gr, 2), bound*1.001 + 1e-10)


if __name__ == "__main__":
    unittest.main()
