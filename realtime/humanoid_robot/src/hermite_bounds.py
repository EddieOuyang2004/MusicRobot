"""Bounded Bernstein checks for degree <= 5, with optional native compilation.

Returns -1 (violation), 0 (uncertain), 1 (within limits). Uncertain cases
must use continuous extrema. No fast-math: limit decisions need float64.
"""
import math
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        return lambda function: function


_POWER_TO_BERNSTEIN = np.zeros((6, 6, 6))
for n in range(6):
    for i in range(n + 1):
        for k in range(i + 1):
            _POWER_TO_BERNSTEIN[n, i, k] = math.comb(i, k) / math.comb(n, k)


@njit(cache=True, nogil=True)
def classify(coefficients, lower, upper):
    n = coefficients.shape[0] - 1
    result = np.zeros(coefficients.shape[1], dtype=np.int8)
    for joint in range(coefficients.shape[1]):
        if lower[joint] == -np.inf and upper[joint] == np.inf:
            result[joint] = 1
            continue
        scale = 1.0
        for k in range(n + 1):
            scale += abs(coefficients[k, joint])
        if not np.isfinite(scale):
            continue
        # Deliberately leave a rounding margin around the existing tolerance.
        guard = 256.0 * np.finfo(np.float64).eps * scale
        lo, hi = lower[joint] - 1e-7, upper[joint] + 1e-7
        violated = False
        for sample in range(9):
            u = sample / 8.0
            value = coefficients[n, joint]
            for k in range(n - 1, -1, -1):
                value = value * u + coefficients[k, joint]
            if value < lo - guard or value > hi + guard:
                violated = True
                break
        if violated:
            result[joint] = -1
            continue
        # Depth-first subdivision requires at most depth+1 pending intervals.
        stack = np.zeros((5, 6))
        depths = np.zeros(5, dtype=np.int64)
        for i in range(n + 1):
            for k in range(i + 1):
                stack[0, i] += _POWER_TO_BERNSTEIN[n, i, k] * coefficients[k, joint]
        count = 1
        uncertain = False
        while count:
            count -= 1
            points = stack[count].copy()
            depth = depths[count]
            minimum, maximum = points[0], points[0]
            for i in range(1, n + 1):
                minimum = min(minimum, points[i])
                maximum = max(maximum, points[i])
            if minimum >= lo + guard and maximum <= hi - guard:
                continue
            if points[0] < lo - guard or points[0] > hi + guard or points[n] < lo - guard or points[n] > hi + guard:
                violated = True
                break
            if depth == 3:
                uncertain = True
                continue
            left, right = np.zeros(6), np.zeros(6)
            left[0], right[n] = points[0], points[n]
            for level in range(1, n + 1):
                for i in range(n - level + 1):
                    points[i] = (points[i] + points[i + 1]) * 0.5
                left[level], right[n - level] = points[0], points[n - level]
            stack[count], stack[count + 1] = left, right
            depths[count], depths[count + 1] = depth + 1, depth + 1
            count += 2
        result[joint] = -1 if violated else (0 if uncertain else 1)
    return result


def warmup():
    """Compile/load the kernel before the control loop, not at the first join."""
    classify(np.zeros((6, 1)), np.array([-1.]), np.array([1.]))
