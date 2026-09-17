"""The value-loss machinery works at any action width, not just two.

`action_curvature` and `fit_value_law` were written for ForageWorld's 2-D
velocity command and hardcoded that 2 everywhere, so the intersection -- whose
signal commands a green time and whose car commands an acceleration, both
one-dimensional -- could not call them at all. That is why `grow` runs with
`labels=None` and its fitted-law source is dark.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.valuesplit import action_curvature, fit_value_law, local_optimum


def _sample(rng, n, K, A, u_star, m):
    """K actions per state with returns from a known concave quadratic."""
    a = rng.uniform(-3.0, 3.0, size=(n, K, A))
    r = a - u_star[:, None, :]
    G = -np.einsum("nkq,qm,nkm->nk", r, m, r)
    return a, G


def test_one_dimensional_curvature_and_peak_are_recovered():
    rng = np.random.default_rng(0)
    n, m = 200, np.array([[2.5]])
    u_star = rng.uniform(-1.0, 1.0, size=(n, 1))
    a, G = _sample(rng, n, 6, 1, u_star, m)
    M = action_curvature(a, G)
    assert M.shape == (n, 1, 1)
    # M is -Hessian of a quadratic with Hessian -2m
    assert np.allclose(M[:, 0, 0], 2.0 * m[0, 0], atol=1e-4)
    u, _ = local_optimum(a, G, lo=np.array([-3.0]), hi=np.array([3.0]))
    assert u.shape == (n, 1)
    assert np.allclose(u, u_star, atol=1e-4)


def test_two_dimensional_case_is_unchanged():
    rng = np.random.default_rng(1)
    n = 150
    m = np.array([[3.0, 0.5], [0.5, 1.5]])
    u_star = rng.uniform(-1.0, 1.0, size=(n, 2))
    a, G = _sample(rng, n, 12, 2, u_star, m)
    M = action_curvature(a, G)
    assert M.shape == (n, 2, 2)
    assert np.allclose(M, 2.0 * m, atol=1e-4)
    u, _ = local_optimum(a, G, lo=np.full(2, -3.0), hi=np.full(2, 3.0))
    assert np.allclose(u, u_star, atol=1e-4)


def test_a_convex_sample_falls_back_to_the_best_action_tried():
    """A sample with no peak in it says nothing about where one is, so the
    honest answer is the best action actually rolled out rather than an
    extrapolated optimum. Return rising away from the centre is that case."""
    rng = np.random.default_rng(2)
    n, K = 60, 5
    a = rng.uniform(-3.0, 3.0, size=(n, K, 1))
    G = (a[:, :, 0] ** 2)                           # convex: no interior peak
    u, M = local_optimum(a, G, lo=np.array([-3.0]), hi=np.array([3.0]))
    best = a[np.arange(n), np.argmax(G, axis=1)]
    assert np.allclose(u, best)
    assert (np.linalg.eigvalsh(M) >= -1e-9).all()   # still PSD


def test_noise_never_escapes_the_action_range():
    """A spuriously concave fit on noise may put its peak anywhere; it is
    clipped, and a peak that lands outside the range falls back instead."""
    rng = np.random.default_rng(3)
    n, K = 400, 5
    a = rng.uniform(-3.0, 3.0, size=(n, K, 1))
    G = rng.normal(size=(n, K))
    lo, hi = np.array([-3.0]), np.array([3.0])
    u, M = local_optimum(a, G, lo=lo, hi=hi)
    assert (u >= lo - 1e-9).all() and (u <= hi + 1e-9).all()
    assert (np.linalg.eigvalsh(M) >= -1e-9).all()


def test_the_fitted_law_recovers_a_known_linear_optimum():
    """u*(s) = 0.7 x - 0.2 should come back out of the M-weighted fit."""
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=(n, 1))
    X = np.hstack([x, np.ones((n, 1))])
    U = 0.7 * x - 0.2
    M = np.full((n, 1, 1), 2.0)
    th = fit_value_law(X, U, M)
    assert th.shape == (2, 1)
    assert np.allclose(th[:, 0], [0.7, -0.2], atol=1e-6)
