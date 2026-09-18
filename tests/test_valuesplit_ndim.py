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

import btind.valuesplit as VS
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


# ------------------------------------------------- standardised law fitting
def _prob(n=500, d=6, nu=2, seed=0, lab_frac=1.0):
    rng = np.random.default_rng(seed)
    scale = np.array([1.0, 50.0, 0.01, 200.0, 1.0, 3.0])[:d]
    X = np.hstack([rng.normal(0, 1, (n, d)) * scale, np.ones((n, 1))])
    U = rng.normal(0, 1, (n, nu))
    R = rng.normal(0, 1, (n, nu, nu))
    M = np.einsum("nij,nkj->nik", R, R) + np.eye(nu)
    if lab_frac < 1.0:
        off = rng.random(n) >= lab_frac
        U, M = U.copy(), M.copy()
        U[off], M[off] = 0.0, 0.0
    return X, U, M


def _raw_fit(X, U, M, ridge=1e-12, k=None):
    """`fit_value_law` as it was: solve in the caller's own column units."""
    d, nu = X.shape[1], np.shape(M)[1]

    def solve(Xs):
        Ai, bi = VS._Ab(Xs, U, M)
        A = Ai.sum(0) + ridge * np.eye(nu * Xs.shape[1])
        return np.linalg.solve(A, bi.sum(0)).reshape(Xs.shape[1], nu)

    th = solve(X)
    if k is None or k >= d - 1:
        return th
    lean = np.abs(th[:-1]).max(axis=1) * X[:, :-1].std(axis=0)
    cols = np.sort(np.argsort(-lean)[:k])
    idx = np.concatenate([cols, [d - 1]]).astype(int)
    out = np.zeros((d, nu))
    out[idx] = solve(X[:, idx])
    return out


def test_standardising_is_the_same_minimiser():
    """At lam 0 it is a reparameterisation, so the law must not move."""
    X, U, M = _prob()
    a = VS.fit_value_law(X, U, M, ridge=1e-12)
    b = _raw_fit(X, U, M)
    assert np.abs(a - b).max() / np.abs(b).max() < 1e-8


def test_sparse_selection_picks_the_same_columns():
    X, U, M = _prob()
    for k in (2, 3, 4):
        a = VS.fit_value_law(X, U, M, ridge=1e-12, k=k)
        b = _raw_fit(X, U, M, k=k)
        keep = lambda t: set(np.flatnonzero(np.abs(t[:-1]).max(1) > 1e-12))
        assert keep(a) == keep(b)
        assert np.abs(a - b).max() < 1e-9


def test_it_is_better_conditioned():
    """Columns spanning 0.01 to 200 make the raw normal matrix unusable."""
    X, U, M = _prob()
    C, S = VS._standardise(X)
    cond = lambda Q: np.linalg.cond(Q)
    raw = cond(VS._Ab(X, U, M)[0].sum(0))
    std = cond(VS._Ab((X - C) / S, U, M)[0].sum(0))
    assert std < raw / 1e5


def test_unlabelled_rows_are_dropped_not_fitted():
    """A zero-curvature row contributes nothing, and must cost nothing."""
    X, U, M = _prob(n=600, lab_frac=0.1)
    lab = np.abs(M.reshape(len(M), -1)).sum(1) > 1e-12
    full = VS.fit_value_law(X, U, M)
    C, S = VS._standardise(X)                    # scale still from every row
    Ai, bi = VS._Ab(((X - C) / S)[lab], U[lab], M[lab])
    nu = M.shape[1]
    th_s = np.linalg.solve(Ai.sum(0) + VS.RIDGE * np.eye(nu * X.shape[1]),
                           bi.sum(0)).reshape(X.shape[1], nu)
    hand = np.zeros_like(full)
    for c in range(X.shape[1]):
        hand[c] += th_s[c] / S[c]
        hand[-1] -= th_s[c] * C[c] / S[c]
    assert np.abs(full - hand).max() == 0.0


def test_shrinkage_shrinks_the_sweep_and_is_off_by_default():
    """What `lam` does, and that asking for none of it changes nothing.

    `lam` charges (slope / sweep)^2, so it pulls the command a law asks for
    back toward the range the world will actually execute. It is OFF by
    default because it was measured on the car and DOES NOT HELP: across
    lam 0.03 to 10 the share of commands on a clip fell from 83% to 0.8% and
    the return fell with it, -21261 at the most saturated end down to -34060,
    against an unshrunk incumbent at -16380. Saturation was not what was
    costing that car anything.
    """
    X, U, M = _prob(n=600, lab_frac=0.05)
    half = 3.55
    spread = X[:, :-1].std(axis=0)
    sweep = lambda th: float((np.abs(th[:-1, 0]) * spread).sum())
    base = sweep(VS.fit_value_law(X, U, M))
    prev = base
    for lam in (0.1, 1.0, 10.0):
        cur = sweep(VS.fit_value_law(X, U, M, sweep=half, lam=lam))
        assert cur < prev
        prev = cur
    assert prev < base / 2
    assert np.array_equal(VS.fit_value_law(X, U, M),
                          VS.fit_value_law(X, U, M, sweep=half, lam=0.0))
