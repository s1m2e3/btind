"""Numba kernels for NestWorld: ForageWorld plus carry-and-deposit.

ONE CHANGE TO THE TASK, chosen to be the smallest edit that makes a behaviour
tree's TEMPORAL primitives necessary rather than decorative.

    ForageWorld   touch food -> +5, immediately
    NestWorld     touch food -> you are CARRYING. +7 arrives only when you
                  reach the nest, and only then does food respawn.

That single delay creates the thing the flat controller has no way to express:
a phase. "Seek" and "return" want opposite actions at the same position, so the
optimal policy is not a function of position alone, and the agent must run one
phase to a TERMINATION CONDITION -- reach food, reach nest -- rather than
choosing afresh each tick. In options language, the piece we never had to model
on ForageWorld was beta(s); here the task supplies two of them.

WHAT IS DELIBERATELY NOT CHANGED. `carrying` is in the observation, so the
problem stays Markov and a memoryless controller can still be optimal in
principle. That is the point: this variant isolates TEMPORAL EXTENT (does a
Sequence chaining two phases help?) from MEMORY (is a blackboard needed?). The
second question needs partial observability and is a separate variant, so that
whichever way the experiment lands we know which property caused it.

Energy still refills on eating, not on depositing: otherwise carrying and
starving would be the same decision, and the two subgoals would collapse back
into one.
"""
import numpy as np
from numba import njit, prange

PARAM_NAMES = ("agent_speed", "threat_speed", "catch_r", "food_r", "e_decay",
               "eat_r", "caught_r", "starve_r", "step_cost", "gamma",
               "nest_r", "deposit_r", "carry_decay", "day_len",
               "night_threat_speed", "food_persistent")

_EPS = 1e-9


def params_array(w):
    return np.array([getattr(w, n) for n in PARAM_NAMES], dtype=np.float64)


@njit(cache=True, inline="always")
def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@njit(cache=True, inline="always")
def _clip11(x):
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)


@njit(cache=True, inline="always")
def is_night(tt, day_len):
    """Cycle phase. Nights are as long as days, so the split is at the halfway
    point of a 2*day_len period; `tt` is the same clock the agent observes."""
    if day_len <= 0.0:
        return False
    return (tt % (2.0 * day_len)) >= day_len


@njit(cache=True, inline="always")
def _advance(px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz, ax, ay, p):
    """One step on scalars. State carries the nest position and the carry flag.

    Order matches ForageWorld: agent moves, then the threat pursues, then
    contacts are resolved -- so `d_threat` in the observation is the POST-move
    distance and a guard on it is testing what the agent can still react to.
    """
    px = _clip01(px + p[0] * _clip11(ax))
    py = _clip01(py + p[0] * _clip11(ay))

    # THE THREAT SLEEPS AT NIGHT. Speed, not presence: a sleeping threat still
    # catches an agent that walks into it, so night is an opportunity rather
    # than an exemption, and the danger is a place instead of a chase.
    spd = p[14] if is_night(tt, p[13]) else p[1]
    dx = px - tx
    dy = py - ty
    n = np.sqrt(dx * dx + dy * dy)
    if n < _EPS:
        n = _EPS
    tx = _clip01(tx + spd * dx / n)
    ty = _clip01(ty + spd * dy / n)

    en -= p[4] + (p[12] if cr > 0.5 else 0.0)
    tt += 1.0
    r = -p[8]

    if cr < 0.5:
        dfx = fx - px
        dfy = fy - py
        if np.sqrt(dfx * dfx + dfy * dfy) < p[3]:
            r += p[5]                      # picking up still pays a little:
            en = 1.0                       # without it the seek phase has no
            cr = 1.0                       # gradient until the first deposit
    else:
        dnx = nx - px
        dny = ny - py
        if np.sqrt(dnx * dnx + dny * dny) < p[10]:
            r += p[11]
            cr = 0.0
            if p[15] < 0.5:                # TELEPORTING food rewards searching
                fx = np.random.random()    # again; PERSISTENT food rewards
                fy = np.random.random()    # remembering where the site was,
                                           # which is the whole point of the
                                           # partial-observability variant

    dtx = tx - px
    dty = ty - py
    caught = np.sqrt(dtx * dtx + dty * dty) < p[2]
    starved = en <= 0.0
    if caught:
        r += p[6]
    elif starved:
        r += p[7]
    return (px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz, r,
            (caught or starved))


@njit(cache=True, parallel=True)
def step_kernel(s, a, p, r, done):
    """In-place single step over a (n, 12) state array."""
    for i in prange(s.shape[0]):
        (s[i, 0], s[i, 1], s[i, 2], s[i, 3], s[i, 4], s[i, 5], s[i, 6],
         s[i, 7], s[i, 8], s[i, 9], s[i, 10], s[i, 11], ri, di) = _advance(
            s[i, 0], s[i, 1], s[i, 2], s[i, 3], s[i, 4], s[i, 5], s[i, 6],
            s[i, 7], s[i, 8], s[i, 9], s[i, 10], s[i, 11], a[i, 0], a[i, 1], p)
        r[i] = ri
        done[i] = di


@njit(cache=True, parallel=True)
def rollout_kernel(states, segs, H, seg_len, p, G):
    """Fused horizon for the planner. No per-step state array is materialised."""
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
        cr = states[i, 8]
        nx = states[i, 9]
        ny = states[i, 10]
        nz = states[i, 11]
        g = 0.0
        disc = 1.0
        for h in range(H):
            k = h // seg_len
            if k >= n_seg:
                k = n_seg - 1
            (px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz, r, d) = _advance(
                px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz,
                segs[i, k, 0], segs[i, k, 1], p)
            g += disc * r
            disc *= p[9]
            if d:
                break
        G[i] = g


@njit(cache=True)
def seed_kernel(k):
    np.random.seed(k)
