"""Numba kernels for ForageWorld.

Profiling put 93% of search time inside env.step (14.0s of 15.1s), almost all of
it numpy temporaries: a 9-column copy of a 1.3M-row state array per step, plus
fancy-indexed column slices, norms and clips each allocating again. The fix is
not a faster numpy expression, it is to stop materialising intermediates at all.

Two kernels:

  step_kernel      one step, in place, for the closed-loop evaluator where the
                   action has to come back to Python each step for the policy.
  rollout_kernel   the whole H-step horizon fused, for search. Each row is an
                   independent trajectory, so this is embarrassingly parallel
                   (prange) and needs no state array beyond scalars in registers.
                   It also breaks out of the loop the moment a row dies, which
                   matters because most rows die early.

Parameters travel as a flat float64 array rather than a jitclass so the kernels
stay simple and cache-compilable; PARAM_NAMES documents the layout.
"""
import numpy as np
from numba import njit, prange

PARAM_NAMES = ("agent_speed", "threat_speed", "catch_r", "food_r", "e_decay",
               "eat_r", "caught_r", "starve_r", "step_cost", "gamma")

_EPS = 1e-9


@njit(cache=True, inline="always")
def _clip01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@njit(cache=True, inline="always")
def _clip11(x):
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


@njit(cache=True, inline="always")
def _advance(px, py, fx, fy, tx, ty, en, tt, ax, ay, p):
    """One step of the dynamics on scalars. Returns the new state + reward + done.

    Mirrors ForageWorld.step exactly, including the order in which the threat
    moves (after the agent) and the fact that reward on the terminating step is
    still collected.
    """
    px = _clip01(px + p[0] * _clip11(ax))
    py = _clip01(py + p[0] * _clip11(ay))

    dx = px - tx
    dy = py - ty
    n = np.sqrt(dx * dx + dy * dy)
    if n < _EPS:
        n = _EPS
    tx = _clip01(tx + p[1] * dx / n)
    ty = _clip01(ty + p[1] * dy / n)

    en -= p[4]
    tt += 1.0
    r = -p[8]

    dfx = fx - px
    dfy = fy - py
    ate = np.sqrt(dfx * dfx + dfy * dfy) < p[3]
    if ate:
        r += p[5]
        en = 1.0
        fx = np.random.random()
        fy = np.random.random()

    dtx = tx - px
    dty = ty - py
    caught = np.sqrt(dtx * dtx + dty * dty) < p[2]
    starved = en <= 0.0
    if caught:
        r += p[6]
    elif starved:
        r += p[7]

    return px, py, fx, fy, tx, ty, en, tt, r, (caught or starved)


@njit(cache=True, parallel=True)
def step_kernel(s, a, p, r, done):
    """In-place single step over a (n, 9) state array."""
    for i in prange(s.shape[0]):
        (s[i, 0], s[i, 1], s[i, 2], s[i, 3], s[i, 4], s[i, 5], s[i, 6],
         s[i, 7], ri, di) = _advance(
            s[i, 0], s[i, 1], s[i, 2], s[i, 3], s[i, 4], s[i, 5],
            s[i, 6], s[i, 7], a[i, 0], a[i, 1], p)
        r[i] = ri
        done[i] = di


@njit(cache=True, parallel=True)
def rollout_kernel(states, segs, H, seg_len, p, G):
    """Fused horizon. states (n, 9); segs (n, n_seg, 2); writes returns into G.

    No per-step state array is ever materialised -- the whole trajectory lives
    in registers -- and a dead row exits immediately instead of being masked.
    """
    n_seg = segs.shape[1]
    for i in prange(states.shape[0]):
        px = states[i, 0]
        py = states[i, 1]
        fx = states[i, 2]
        fy = states[i, 3]
        tx = states[i, 4]
        ty = states[i, 5]
        en = states[i, 6]
        tt = states[i, 7]
        g = 0.0
        disc = 1.0
        for h in range(H):
            k = h // seg_len
            if k >= n_seg:
                k = n_seg - 1
            px, py, fx, fy, tx, ty, en, tt, r, d = _advance(
                px, py, fx, fy, tx, ty, en, tt, segs[i, k, 0], segs[i, k, 1], p)
            g += disc * r
            disc *= p[9]
            if d:
                break
        G[i] = g


@njit(cache=True)
def seed_kernel(k):
    np.random.seed(k)


def params_array(env):
    return np.array([getattr(env, n) for n in PARAM_NAMES], dtype=np.float64)
