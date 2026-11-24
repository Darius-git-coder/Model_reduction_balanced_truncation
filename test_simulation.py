"""Numerical checks for the time integrator and the supplied spatial model."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from helpers import fdm_2d_poisson, implicit_euler
from visualize_system import simulate, estimate_error, build_interactive


class SimulationTests(unittest.TestCase):
    def test_dense_scalar_forced_solution_converges(self):
        # x'=-2x+3, x(0)=1 has x(t)=1.5-0.5*exp(-2t).
        exact = 1.5 - 0.5 * np.exp(-2)
        errors = []
        for steps in (50, 100):
            x = implicit_euler(np.array([[-2.0]]), np.array([[3.0]]), 1, lambda t: 1, [[1]], steps)
            self.assertEqual(x.shape, (1, steps + 1))
            self.assertEqual(x[0, 0], 1)
            errors.append(abs(x[0, -1] - exact))
        self.assertLess(errors[1], 0.55 * errors[0])

    def test_sparse_heat_eigenmode(self):
        # The first discrete sine mode decays with a known eigenvalue.
        d = 5
        A, B, C, X, Y = fdm_2d_poisson(d)
        initial = (np.sin(np.pi * X) * np.sin(np.pi * Y)).ravel()
        eigenvalue = -8 * (d + 1)**2 * np.sin(np.pi / (2 * (d + 1)))**2
        steps, tf = 100, 0.2
        x = implicit_euler(A, B, tf, lambda t: 0, initial, steps)
        expected = initial * (1 - tf / steps * eigenvalue)**(-steps)
        assert_allclose(x[:, -1], expected, atol=1e-13)
        self.assertTrue(np.all(np.diff(np.linalg.norm(x, axis=0)) < 0))
        assert_allclose(C @ initial, np.sum(initial) / (d + 1)**2)

    def test_right_endpoint_and_multiple_inputs(self):
        # For A=0 and u=(t,2t), backward Euler samples at t[k+1].
        x = implicit_euler(np.zeros((2, 2)), np.eye(2), 1, lambda t: [t, 2*t], np.zeros((2, 1)), 4)
        assert_allclose(x[:, -1], [0.625, 1.25])

    def test_error_map_uses_coincident_nodes_and_synchronized_frames(self):
        d, tf, steps = 3, 0.2, 20
        data = simulate(d, tf, steps)
        error = estimate_error(data, tf, steps, "sine")
        reference = simulate(2*d+1, tf, steps)
        assert_allclose(data[3], reference[3][1::2, 1::2])
        assert_allclose(data[4], reference[4][1::2, 1::2])
        expected = np.abs(data[6] - reference[6].reshape(7, 7, steps+1)[1::2, 1::2].reshape(d*d, steps+1))
        assert_allclose(error["absolute"], expected)
        assert_allclose(error["absolute"][:, 0], 0)
        fig = build_interactive(data, error)
        self.assertTrue(all(step.method == "skip" for step in fig.layout.sliders[0].steps))
        for frame in fig.frames:
            k = int(frame.name)
            self.assertEqual(list(frame.traces), [1, 2, 6, 7, 8])
            assert_allclose(np.array(frame.data[1].z)[1:-1, 1:-1].ravel(), expected[:, k])
            self.assertEqual(frame.data[-1].x[0], data[5][k])
            self.assertEqual(frame.data[-1].y[0], error["maximum"][k])

    def test_error_estimate_decreases_under_grid_refinement(self):
        maxima = []
        for d in (7, 15):
            data = simulate(d, 0.3, 300)
            maxima.append(np.max(estimate_error(data, 0.3, 300, "sine")["absolute"]))
        # A second-order spatial scheme should reduce the difference by ~4.
        self.assertLess(maxima[1], maxima[0] / 3)


if __name__ == "__main__":
    unittest.main()
