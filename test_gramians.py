"""Check Lyapunov signs, transpose placement and both heat-system outputs."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse as sp

from helpers import fdm_2d_poisson, lyapunov_gramians


class GramianTests(unittest.TestCase):
    def check_gramians(self, A, B, C):
        P, Q = lyapunov_gramians(A, B, C)
        A, B, C = [v.toarray() if sp.issparse(v) else v for v in (A, B, C)]
        assert_allclose(A @ P + P @ A.T, -B @ B.T, atol=1e-11)
        assert_allclose(A.T @ Q + Q @ A, -C.T @ C, atol=1e-11)
        for gramian in (P, Q):
            assert_allclose(gramian, gramian.T, atol=1e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(gramian).min(), -1e-12)
        return P, Q

    def test_scalar_closed_form(self):
        P, Q = self.check_gramians(np.array([[-2.0]]), np.array([[3.0]]), np.array([[4.0]]))
        assert_allclose(P, [[9 / 4]])
        assert_allclose(Q, [[4]])

    def test_nonsymmetric_system(self):
        # A symmetric Laplacian alone would not catch a missing transpose.
        A = np.array([[-1.0, 2.0, 0.0], [0.0, -2.0, 1.0], [0.0, 0.0, -3.0]])
        B = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        C = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
        self.check_gramians(A, B, C)

    def test_heat_system_both_outputs_and_sparse_inputs(self):
        A, B, C, *_ = fdm_2d_poisson(5)
        P1, _ = self.check_gramians(A, sp.csr_matrix(B), sp.csr_matrix(C))
        P2, _ = self.check_gramians(A, B, np.eye(A.shape[0]))
        # Controllability depends on A,B; changing the observation changes Q.
        assert_allclose(P1, P2)


if __name__ == "__main__":
    unittest.main()
