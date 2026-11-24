"""Check common-node error definitions and synchronized visual updates."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from compare_methods import compare, build_comparison, snapshot_fields


class ComparisonTests(unittest.TestCase):
    def test_common_grid_and_errors(self):
        result = compare(5, 2, 2, steps=40)
        assert_allclose(result["fine"][3][1::2,1::2], result["coarse"][3])
        assert_allclose(result["fine"][4][1::2,1::2], result["coarse"][4])
        reference = result["fine"][6].reshape(5,5,41)[1::2,1::2].reshape(4,41)
        assert_allclose(result["reference_common"], reference)
        assert_allclose(result["error_fdm"], np.abs(result["coarse"][6]-reference))
        assert_allclose(result["error_bt"], np.abs(result["bt_common"]-reference))
        for key in ("error_fdm", "error_bt", "error_difference"):
            assert_allclose(result[key][:,0], 0)
        assert_allclose(result["error_difference"], result["error_fdm"]-result["error_bt"])
        for field in snapshot_fields(result, 20):
            assert_allclose(field[[0,-1]], 0)
            assert_allclose(field[:,[0,-1]], 0)

    def test_comparison_frames_update_both_maps_and_markers(self):
        result = compare(5, 2, 2, steps=10)
        fig = build_comparison(result)
        self.assertEqual(list(fig.layout.meta["error_traces"]), [3,4])
        for frame in fig.frames:
            k = int(frame.name)
            self.assertEqual(frame.layout.meta["time"], result["fine"][5][k])
            assert_allclose(np.array(frame.data[3].z)[1:-1,1:-1].ravel(), result["error_fdm"][:,k])
            assert_allclose(np.array(frame.data[4].z)[1:-1,1:-1].ravel(), result["error_bt"][:,k])
            self.assertEqual(frame.data[-2].y[0], result["max_fdm"][k])
            self.assertEqual(frame.data[-1].y[0], result["max_bt"][k])

    def test_non_nested_grids_rejected(self):
        with self.assertRaisesRegex(ValueError, "Gitter"):
            compare(6, 2)


if __name__ == "__main__":
    unittest.main()
