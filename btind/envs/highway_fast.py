"""The highway model and the tree, fused into one compiled loop.

`highway_batch` advances every episode in lockstep over array columns, which
took 140 ms an episode down to 4.1. That is still ~270x NestWorld, because the
work per substep is small and numpy pays for an array pass on every one of them:
40 policy steps x 5 substeps x a handful of (n, V, V) reductions.

THE FUSION IS THE SAME MOVE `nest_fast` MAKES. Put the episode loop INSIDE the
row loop and compile it, so the per-episode state -- fifteen vehicles, the
latch, the blackboard -- lives in registers and locals rather than in arrays that
have to be re-read every step. The policy goes in with it: guards, terminations,
the write rule and the laws are flattened to arrays once and evaluated inline,
so a rollout never returns to Python.

WHAT IT CANNOT RUN, and it says so rather than guessing: a guard that reads
`V_hat` or `leverage` (there is no xgboost predict in a kernel), which is what
`uses_vq` reports, exactly as on NestWorld.

THE HEAD IS AN ARGMAX. `laws` is (arm, feature, action) and the kernel emits the
index of the best-scoring manoeuvre, which is what highway-env's
DiscreteMetaAction takes. The vector head is not implemented here because this
world does not have one.

CORRECTNESS IS A MEASUREMENT, NOT A CLAIM. `test_equivalence` in
`experiments/e25` runs the same banks through this and through `highway_batch`
and requires the returns to agree exactly -- the same discipline that held
MemBank to LandscapeBank at 0.0 and the NestWorld kernel to its Python path at
0.0. A fused kernel that silently disagrees with the model it was derived from
is worse than no kernel, because it is fast enough to be trusted.
"""
import numpy as np
from numba import njit, prange

from ..tick import (_fires, _tick, flatten, no_dev,  # noqa: F401  (re-exported)
                    no_trace)
from .highway_batch import (ACC_COMF, ACC_MAX, D0, DELTA, FAR, KP_A,
                            KP_HEADING, KP_LATERAL, LANE_CHANGE_DELAY, LANE_W,
                            LENGTH, MAX_SPEED, MAX_STEER, MIN_GAIN, MIN_SPEED,
                            R_COLLISION, R_RIGHT, R_SPEED, SPEED_RANGE,
                            TARGET_SPEEDS, TAU_W, WIDTH)


def params(env):
    """Every scalar the kernel needs, in one float array."""
    return np.array([env.n_lanes, env.n_veh, env.n_obs_veh, env.duration,
                     env.n_sub, env.dt, env.gamma], np.float64)


def uses_vq(bank, n_obs):
    """Only a GUARD can force the Python path, never a law's coefficients.

    The kernel sets V_hat and leverage to zero exactly as MemBank does with no
    critic attached, so a law coefficient on either multiplies a zero and the
    two paths agree. A guard COMPARING against them is the real
    incompatibility.

    Asking `_reads_vq` instead -- which is true as soon as any law touches those
    columns -- silently refuses every bank CEM has ever perturbed, because CEM
    makes theta dense. That cost NestWorld 63x once already; here it turned a
    10-second smoke test into a 500-second one before it was noticed.
    """
    from ..memory import _guards_read_vq
    return _guards_read_vq(bank, n_obs)


@njit(cache=True, inline="always")
def _nz(x):
    if x > 1e-2:
        return x
    if x < -1e-2:
        return x
    return 1e-2 if x >= 0.0 else -1e-2


@njit(cache=True, inline="always")
def _idm(v, target, gap, fv):
    """IDM acceleration -- the same two terms as the numpy model."""
    free = ACC_COMF * (1.0 - (max(v, 0.0) / abs(_nz(target))) ** DELTA)
    if gap >= FAR:
        return free
    dv = v - fv
    dstar = D0 + v * TAU_W + v * dv / (2.0 * np.sqrt(ACC_MAX * ACC_COMF))
    if dstar < 0.0:
        dstar = 0.0
    return free - ACC_COMF * (dstar / _nz(gap)) ** 2


@njit(cache=True, inline="always")
def _steer(y, h, v, tl):
    lat_cmd = -KP_LATERAL * (y - LANE_W * tl)
    a = lat_cmd / _nz(v)
    if a > 1.0:
        a = 1.0
    elif a < -1.0:
        a = -1.0
    hc = np.arcsin(a)
    if hc > np.pi / 4:
        hc = np.pi / 4
    elif hc < -np.pi / 4:
        hc = -np.pi / 4
    rate = KP_HEADING * (hc - h)
    b = LENGTH / 2.0 / _nz(v) * rate
    if b > 1.0:
        b = 1.0
    elif b < -1.0:
        b = -1.0
    st = np.arctan(2.0 * np.tan(np.arcsin(b)))
    if st > MAX_STEER:
        st = MAX_STEER
    elif st < -MAX_STEER:
        st = -MAX_STEER
    return st


@njit(cache=True, parallel=True)
def rollout(states, p, laws, mem_cols, w_col, w_thr, w_neg, n_obs, T, G, trace,
            dev,
            lit_col, lit_thr, lit_neg, cl_start, cl_len,
            b_col, b_thr, b_neg, b_start, b_len,
            f_col, f_thr, f_neg, f_start, f_len,
            a_col, a_thr, a_neg, a_start, a_len,
            law_start, n_steps, sticky):
    """Physics and tree together; call as
    `rollout(s, p, *world_args(f), T, G, trace, dev, *tick_args(f))` with
    `f = flatten(...)`. The arbitration is `tick._tick`, shared with the
    NestWorld kernel.

    `trace` is (n, T, d + 5) to record, per episode and tick, the full `z`
    followed by (law index, action, 0, latch, step); pass `tick.no_trace()` to
    skip it. A rollout that can be replayed tick by tick is what makes a
    disagreement with the numpy model localisable rather than merely detectable.

    `dev` is (n, 4): per episode (t0, k, a, unused). For k > 0 the action at
    ticks t0 <= t < t0 + k is `a` REGARDLESS of the tree, which then resumes.
    The tree's own latch and step keep evolving underneath, exactly as they
    would if the world had taken that action. Pass `tick.no_dev()` for none.
    Paired with the undeviated rollout on the same start this is an exact
    counterfactual: the return of doing `a` here instead of what the tree does.
    """
    n = states.shape[0]
    d = laws.shape[1]
    countdown = n_obs >= 1048576          # tick.COUNTDOWN_FLAG
    if countdown:
        n_obs = n_obs - 1048576
    tracing = trace.shape[0] == n
    deviating = dev.shape[0] == n
    nA = laws.shape[2]
    nmem = mem_cols.shape[0]
    n_lanes = int(p[0])
    V = int(p[1])
    OV = int(p[2])
    duration = int(p[3])
    n_sub = int(p[4])
    dt = p[5]
    gamma = p[6]

    for i in prange(n):
        x = np.empty(V); y = np.empty(V); h = np.empty(V); v = np.empty(V)
        tl = np.empty(V); tm = np.empty(V)
        for q in range(V):
            x[q] = states[i, 4 * q + 0]
            y[q] = states[i, 4 * q + 1]
            h[q] = states[i, 4 * q + 2]
            v[q] = states[i, 4 * q + 3]
            tl[q] = states[i, 4 * V + q]
            tm[q] = states[i, 5 * V + q]
        si = states[i, 6 * V + 0]
        crashed = states[i, 6 * V + 1] > 0.5
        z = np.zeros(d)
        slots = np.zeros(max(nmem, 1))
        have = False
        age = 0
        latch = -1
        step = 0
        g = 0.0
        disc = 1.0
        gap = np.empty(V)
        fv = np.empty(V)
        used = np.zeros(V, np.bool_)

        for t in range(duration if T <= 0 else min(T, duration)):
            # ---- observation: ego, then the OV-1 nearest vehicles ----------
            z[0] = x[0]
            z[1] = y[0]
            z[2] = v[0] * np.cos(h[0])
            z[3] = v[0] * np.sin(h[0])
            for q in range(V):
                used[q] = False
            for k in range(OV - 1):
                best = -1
                bd = 1e18
                for q in range(1, V):
                    if used[q]:
                        continue
                    dx = x[q] - x[0]
                    dy = y[q] - y[0]
                    dd = dx * dx + dy * dy
                    if dd < bd:
                        bd = dd
                        best = q
                b = 4 + 5 * k
                if best < 0:
                    z[b] = 0.0; z[b + 1] = FAR; z[b + 2] = FAR
                    z[b + 3] = 0.0; z[b + 4] = 0.0
                else:
                    used[best] = True
                    z[b] = 1.0
                    z[b + 1] = x[best] - x[0]
                    z[b + 2] = y[best] - y[0]
                    z[b + 3] = v[best] * np.cos(h[best]) - z[2]
                    z[b + 4] = v[best] * np.sin(h[best]) - z[3]
            base = 4 + 5 * (OV - 1)
            z[base] = t / duration                 # t_norm    (planted)
            z[base + 1] = states[i, 6 * V + 4]     # noise     (planted)
            z[n_obs] = 0.0
            z[n_obs + 1] = 0.0
            if nmem > 0:
                if have:
                    age += 1
                if w_col.shape[0] > 0 and _fires(z, w_col, w_thr, w_neg, 0,
                                                 w_col.shape[0]):
                    for q in range(nmem):
                        slots[q] = z[mem_cols[q]]
                    have = True
                    age = 0
                for q in range(nmem):
                    z[n_obs + 2 + q] = slots[q]
                z[n_obs + 2 + nmem] = 1.0 if have else 0.0
                z[n_obs + 3 + nmem] = age
                if countdown:
                    for q in range(nmem):
                        z[n_obs + 4 + nmem + q] = slots[q] - age
            z[d - 1] = 1.0

            # ---- the tree ---------------------------------------------------
            law, latch, step = _tick(z, latch, step,
                                     lit_col, lit_thr, lit_neg, cl_start, cl_len,
                                     b_col, b_thr, b_neg, b_start, b_len,
                                     f_col, f_thr, f_neg, f_start, f_len,
                                     a_col, a_thr, a_neg, a_start, a_len,
                                     law_start, n_steps, sticky)

            best_a = 0
            best_s = -1e18
            for k in range(nA):
                sc = 0.0
                for q in range(d):
                    sc += z[q] * laws[law, q, k]
                if sc > best_s:
                    best_s = sc
                    best_a = k
            if deviating and dev[i, 1] > 0.0 and t >= dev[i, 0] \
                    and t < dev[i, 0] + dev[i, 1]:
                best_a = int(dev[i, 2])
            if tracing and t < trace.shape[1]:
                for q in range(d):
                    trace[i, t, q] = z[q]
                trace[i, t, d] = law
                trace[i, t, d + 1] = best_a
                trace[i, t, d + 2] = 0.0
                trace[i, t, d + 3] = latch
                trace[i, t, d + 4] = step

            # ---- the meta-action --------------------------------------------
            if best_a == 0:
                tl[0] = tl[0] - 1.0
            elif best_a == 2:
                tl[0] = tl[0] + 1.0
            if tl[0] < 0.0:
                tl[0] = 0.0
            elif tl[0] > n_lanes - 1:
                tl[0] = n_lanes - 1.0
            if best_a == 3:
                si += 1.0
            elif best_a == 4:
                si -= 1.0
            if si < 0.0:
                si = 0.0
            elif si > len(TARGET_SPEEDS) - 1:
                si = len(TARGET_SPEEDS) - 1.0
            ego_target = TARGET_SPEEDS[int(si)]

            # ---- MOBIL, once per policy step --------------------------------
            for q in range(V):
                tm[q] += n_sub * dt
            for q in range(1, V):
                if tm[q] < LANE_CHANGE_DELAY:
                    continue
                g0 = FAR
                f0 = 0.0
                for r2 in range(V):
                    if r2 == q:
                        continue
                    if abs(tl[r2] - tl[q]) < 0.5 and x[r2] > x[q]:
                        dd = x[r2] - x[q]
                        if dd < g0:
                            g0 = dd
                            f0 = v[r2]
                a0 = _idm(v[q], 30.0, g0, f0)
                for dlane in (-1.0, 1.0):
                    cand = tl[q] + dlane
                    if cand < 0.0 or cand > n_lanes - 1:
                        continue
                    g1 = FAR
                    f1 = 0.0
                    for r2 in range(V):
                        if r2 == q:
                            continue
                        if abs(tl[r2] - cand) < 0.5 and x[r2] > x[q]:
                            dd = x[r2] - x[q]
                            if dd < g1:
                                g1 = dd
                                f1 = v[r2]
                    if _idm(v[q], 30.0, g1, f1) - a0 > MIN_GAIN:
                        tl[q] = cand
                        tm[q] = 0.0
                        break

            # ---- physics ----------------------------------------------------
            for _ in range(n_sub):
                for q in range(V):
                    ln_q = round(y[q] / LANE_W)
                    if ln_q < 0.0:
                        ln_q = 0.0
                    elif ln_q > n_lanes - 1:
                        ln_q = n_lanes - 1.0
                    g0 = FAR
                    f0 = 0.0
                    for r2 in range(V):
                        if r2 == q:
                            continue
                        ln_r = round(y[r2] / LANE_W)
                        if ln_r < 0.0:
                            ln_r = 0.0
                        elif ln_r > n_lanes - 1:
                            ln_r = n_lanes - 1.0
                        if abs(ln_r - ln_q) < 0.5 and x[r2] > x[q]:
                            dd = x[r2] - x[q]
                            if dd < g0:
                                g0 = dd
                                f0 = v[r2]
                    gap[q] = g0
                    fv[q] = f0
                for q in range(V):
                    if q == 0:
                        acc = KP_A * (ego_target - v[0])
                    else:
                        acc = _idm(v[q], 30.0, gap[q], fv[q])
                    if acc > ACC_MAX:
                        acc = ACC_MAX
                    elif acc < -ACC_MAX:
                        acc = -ACC_MAX
                    st = _steer(y[q], h[q], v[q], tl[q])
                    beta = np.arctan(0.5 * np.tan(st))
                    x[q] += v[q] * np.cos(h[q] + beta) * dt
                    y[q] += v[q] * np.sin(h[q] + beta) * dt
                    h[q] += v[q] * np.sin(beta) / (LENGTH / 2.0) * dt
                    v[q] += acc * dt
                    if v[q] > MAX_SPEED:
                        v[q] = MAX_SPEED
                    elif v[q] < MIN_SPEED:
                        v[q] = MIN_SPEED
                for q in range(1, V):
                    if abs(x[q] - x[0]) < LENGTH and abs(y[q] - y[0]) < WIDTH:
                        crashed = True

            # ---- reward -------------------------------------------------------
            on_road = (y[0] > -LANE_W * 0.5) and (y[0] < LANE_W * (n_lanes - 0.5))
            lane0 = round(y[0] / LANE_W)
            if lane0 < 0.0:
                lane0 = 0.0
            elif lane0 > n_lanes - 1:
                lane0 = n_lanes - 1.0
            fwd = v[0] * np.cos(h[0])
            frac = (fwd - SPEED_RANGE[0]) / (SPEED_RANGE[1] - SPEED_RANGE[0])
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            denom = max(n_lanes - 1, 1)
            raw = R_SPEED * frac + R_RIGHT * lane0 / denom
            if crashed:
                raw += R_COLLISION
            r = (raw - R_COLLISION) / ((R_SPEED + R_RIGHT) - R_COLLISION)
            if not on_road:
                r = 0.0
            g += disc * r
            disc *= gamma
            if crashed or (not on_road) or (t + 1 >= duration):
                break
        G[i] = g
