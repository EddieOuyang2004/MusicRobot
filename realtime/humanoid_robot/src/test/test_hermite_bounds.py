"""Differential checks against continuous polynomial extrema."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_bridges import HermiteBridge
from hermite_bounds import classify


class BoundsTests(unittest.TestCase):
    def test_random_and_near_extrema(self):
        rng = np.random.default_rng(913)
        for _ in range(120):
            curve = HermiteBridge(float(rng.uniform(.1, 10)), rng.normal(size=(6, 8)))
            for derivative in range(4):
                lo, hi = curve.extrema(derivative)
                for margin in (-2e-6, 0., 2e-6):
                    lower, upper = lo-margin, hi+margin
                    expected = bool(np.all(lo >= lower-1e-7) and np.all(hi <= upper+1e-7))
                    self.assertEqual(curve.within_limits(derivative, lower, upper), expected)

    def test_uncertain_uses_original_roots(self):
        curve = HermiteBridge(1., np.array([[0.], [1.], [0.], [0.], [0.], [0.]]))
        with patch('motion_bridges.classify', return_value=np.array([0], dtype=np.int8)):
            self.assertTrue(curve.within_limits(0, np.array([0.]), np.array([1.])))
            self.assertFalse(curve.within_limits(0, np.array([0.]), np.array([.9])))

    def test_python_and_native_match(self):
        rng = np.random.default_rng(42)
        python = getattr(classify, 'py_func', classify)
        for degree in range(1, 6):
            c = rng.normal(size=(degree+1, 29))
            lo, hi = np.full(29, -1.), np.full(29, 1.)
            np.testing.assert_array_equal(classify(c, lo, hi), python(c, lo, hi))

    def test_constant_and_unlimited(self):
        c = np.zeros((6, 3))
        c[0] = [0., 1., -1.]
        np.testing.assert_array_equal(classify(c, np.full(3, -np.inf), np.full(3, np.inf)), [1, 1, 1])
        self.assertTrue(HermiteBridge(1., c).within_limits(0, np.full(3, -1.), np.full(3, 1.)))


if __name__ == '__main__':
    unittest.main()
