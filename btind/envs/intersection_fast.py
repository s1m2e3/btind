"""The intersection model and both trees, fused into one compiled loop.

The same move `nest_fast` and `highway_fast` make: the episode loop goes
inside the row loop, so forty vehicles, their latches, steps and blackboards,
and the signal's, live in locals; the vehicle tree and the signal tree are
flattened once and evaluated inline; a rollout never returns to Python.

TWO TREES, ONE TICK FUNCTION. Both trees are ordinary flattened banks and both
call `tick._tick`. The vehicle tree runs once per ACTIVE slot with that slot's
latch and step; the signal tree once per episode. A slot that is not active is
not ticked at all, and starts with an empty latch when it spawns -- which is
what the reference path does by resetting the row at spawn.

WHAT IS MIRRORED, in this order, because the order is semantics:
observe (signal state BEFORE this tick's update) -> both trees act -> signal
update -> spawn -> kinematics -> red-light running -> delay -> collisions
(rear-end, then crossing) -> exits -> reward. `tests/test_intersection_equiv.py`
holds this to `IntersectionBatch.python_rollout` to the bit.

DEVIATION AND TRACE name ONE AGENT per episode: `dev[i, 3]` is a slot index, or
-1 for the signal. The trace records that agent's z, law, action, latch and
step per tick, and the deviation overrides that agent's action at ticks
t0 <= t < t0 + k. That is how `explore.deviations` measures what one vehicle,
or the signal, would have gained by acting differently.
"""
import numpy as np
from numba import njit, prange

from ..kernlaw import kern_args, law_out
from ..tick import _fires, _tick, flatten, no_dev, no_trace, tick_args, world_args
from .intersection import (A_MAX, ACCELS, B_MAX, FAR, FIXED_GREEN, HALF_CONF,
                           QUEUE_GAP, R_GREEN, R_LEFT, R_RED_CAR, R_STOP, STOP_ZONE,
                           W_CAR_DELAY, W_COMFORT, W_DELAY, W_QUEUE, W_STUCK_CAR,
                           W_SPEED, W_STUCK,
                           L_VEH, N_PHASES, NEAR_INT, QUEUE_D, QUEUE_V, R_COLL,
                           R_EXIT, R_RED, SENSE_R, SIG_EMPTY, SIG_HIDDEN, SPAWN_GAP,
                           T_AR, T_V_MIN,
                           T_MAX, T_MIN,
                           V0)


def params(env, vehicle_bank=None, signal_bank=None, reward=None):
    vh = (vehicle_bank or {}).get("head", env.veh_head)
    sh = (signal_bank or {}).get("head", env.sig_head)
    reward = reward or env.reward_mode
    assert reward in ("team", "car")
    return np.array([env.N, env.M, env.dt, env.duration, env.gamma,
                     1.0 if env.event else 0.0,
                     1.0 if vh == "scalar" else 0.0,
                     1.0 if sh == "duration" else 0.0,
                     env.occlude_lo, env.occlude_hi,
                     1.0 if reward == "car" else 0.0], np.float64)


@njit(cache=True, inline="always")
def _live(kind, bound, t0, dt):
    """Ticks a live phase lasts from elapsed t0 (see the model's `_t_window`)."""
    k = np.ceil((bound - t0) / dt - 1e-9)
    if kind == 0:
        return (k if k > 0.0 else 0.0) + 1.0
    return k if k > 1.0 else 1.0


@njit(cache=True, inline="always")
def _t_window(m, phase, t_phase, in_ar, ar_t, plan, kind, green, fixed_green, dt):
    """(earliest, latest) ticks until movement m's light is observed to change,
    under controller `kind` (0 fixed, 1 duration, 2 argmax). Mirrors the numpy
    model's `_t_window` exactly."""
    if kind == 0:
        cur_min = _live(0, fixed_green[phase], t_phase, dt)
        cur_max = cur_min
    elif kind == 1 and plan > 0.0:
        cur_min = _live(1, plan, t_phase, dt)
        cur_max = cur_min
    else:
        cur_min = _live(2, T_MIN, t_phase, dt)
        cur_max = _live(2, T_MAX, t_phase, dt)
    ar_full = np.ceil(T_AR / dt - 1e-9)
    if (not in_ar) and green[phase, m]:
        return cur_min, cur_max
    if in_ar:
        ar_now = np.ceil((T_AR - ar_t) / dt - 1e-9)
        if ar_now < 1.0:
            ar_now = 1.0
        tmin = ar_now
        tmax = ar_now
    else:
        tmin = cur_min + ar_full
        tmax = cur_max + ar_full
    q = (phase + 1) % N_PHASES
    for _ in range(N_PHASES):
        if green[q, m]:
            break
        if kind == 0:
            f = _live(0, fixed_green[q], 0.0, dt)
            tmin += f + ar_full
            tmax += f + ar_full
        else:
            tmin += _live(2, T_MIN, 0.0, dt) + ar_full
            tmax += _live(2, T_MAX, 0.0, dt) + ar_full
        q = (q + 1) % N_PHASES
    return tmin, tmax


def geometry(env):
    g = env.geom
    M = env.M
    # per movement: sorted conflict-point arc-lengths, padded with FAR
    K = max(1, max(len(c) for c in env.cps))
    cps = np.full((M, K), 1e9)
    for m in range(M):
        cps[m, :len(env.cps[m])] = env.cps[m]
    return (np.ascontiguousarray(g["path_len"], np.float64),
            np.ascontiguousarray(g["s_stop"], np.float64),
            np.ascontiguousarray(g["s_junc"], np.float64),
            np.ascontiguousarray(env.s_spawn, np.float64),
            np.ascontiguousarray(env.s_exit, np.float64),
            np.ascontiguousarray(g["s_cp"], np.float64),
            np.ascontiguousarray(g["conf"], np.bool_),
            np.ascontiguousarray(g["dirs"], np.int64),
            np.ascontiguousarray(env.lane_group, np.int64),
            np.ascontiguousarray(g["to_edge"], np.int64),
            np.ascontiguousarray(g["green"], np.bool_),
            np.ascontiguousarray(FIXED_GREEN, np.float64),
            np.ascontiguousarray(cps, np.float64),
            np.ascontiguousarray(g["appr"], np.int64),
            np.ascontiguousarray(g["cum_f"], np.float64),
            np.ascontiguousarray(g["pts_f"], np.float64),
            np.ascontiguousarray(g["n_vert"], np.int64))


def run(env, vehicle_bank, signal_bank, s, T, trace=None, dev=None, reward=None):
    """Roll both trees from the states `s`. `signal_bank` None = fixed plan.
    `reward` "team" or "car" overrides the return the world would score.

    POPULATIONS. Either bank may be a LIST -- the partners a controller is
    trained against. The episodes are split into consecutive blocks, one per
    (vehicle, signal) pair, deterministically, so a paired test over two
    candidates still compares each episode against the same partner.
    """
    if isinstance(vehicle_bank, list) or isinstance(signal_bank, list):
        vbs = vehicle_bank if isinstance(vehicle_bank, list) else [vehicle_bank]
        sbs = signal_bank if isinstance(signal_bank, list) else [signal_bank]
        pairs = [(v, g) for v in vbs for g in sbs]
        G = np.empty(len(s))
        edges = np.linspace(0, len(s), len(pairs) + 1).round().astype(int)
        for (v, g), a, b in zip(pairs, edges[:-1], edges[1:]):
            if b <= a:
                continue
            tr = None if trace is None or trace.shape[0] != len(s) else trace[a:b]
            dv = None if dev is None or dev.shape[0] != len(s) else dev[a:b]
            G[a:b] = run(env, v, g, s[a:b], T, tr, dv, reward)
        return G
    if vehicle_bank is None:
        vehicle_bank = env.default_vehicle_bank()
    fv = flatten(vehicle_bank, len(env.veh_names))
    if signal_bank is None:
        fs = flatten(dict(clauses=[], laws=[], default=np.zeros((len(env.sig_names) + 3, 2)),
                          laws_on_z=True), len(env.sig_names))
        use_sig = False
    else:
        fs = flatten(signal_bank, len(env.sig_names))
        use_sig = True
    G = np.empty(len(s))
    rollout(np.ascontiguousarray(s), params(env, vehicle_bank, signal_bank, reward),
            *geometry(env), T, G,
            no_trace() if trace is None else trace,
            no_dev() if dev is None else dev,
            *world_args(fv), *world_args(fs), use_sig,
            tick_args(fv), tick_args(fs), kern_args(fv), kern_args(fs))
    return G


@njit(cache=True, inline="always")
def _argmax_out(u, nA):
    """The first action with the highest preference, as `_argmax_law` picks."""
    best_a = 0
    best_s = -1e18
    for k in range(nA):
        if u[k] > best_s:
            best_s = u[k]
            best_a = k
    return best_a


@njit(cache=True, inline="always")
def _argmax_law(z, laws, law, nA):
    best_a = 0
    best_s = -1e18
    for k in range(nA):
        sc = 0.0
        for q in range(z.shape[0]):
            sc += z[q] * laws[law, q, k]
        if sc > best_s:
            best_s = sc
            best_a = k
    return best_a


@njit(cache=True, inline="always")
def _write_mem(z, n_obs_flagged, slots, have, age, mem_cols, w_col, w_thr, w_neg):
    """The blackboard write, then the memory columns into z: slots, have, age,
    and `left_` columns when the flag bit is set on n_obs (tick.COUNTDOWN_FLAG).
    Returns (have, age); age counts ticks since the write, 0 on it."""
    nmem = mem_cols.shape[0]
    countdown = n_obs_flagged >= 1048576
    n_obs = n_obs_flagged - 1048576 if countdown else n_obs_flagged
    if nmem > 0:
        if have:
            age += 1
        if w_col.shape[0] > 0 and _fires(z, w_col, w_thr, w_neg, 0, w_col.shape[0]):
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
    return have, age


@njit(cache=True, parallel=True)
def rollout(states, p, path_len, s_stop, s_junc, s_spawn, s_exit, s_cp, conf,
            dirs, lane_group, to_edge, green, fixed_green, cps, appr, cum, pts, n_vert,
            T, G,
            trace, dev,
            vlaws, vmem_cols, vw_col, vw_thr, vw_neg, vn_obs,
            slaws, smem_cols, sw_col, sw_thr, sw_neg, sn_obs, use_sig, vt, st,
            vk, sk):
    n = states.shape[0]
    N = int(p[0])
    dt = p[2]
    duration = int(p[3])
    gamma = p[4]
    event = p[5] > 0.5
    vscalar = p[6] > 0.5
    sduration = p[7] > 0.5
    occ_lo = p[8]
    occ_hi = p[9]
    car_r = p[10] > 0.5
    # the controller the V2I message describes: 0 fixed plan, 1 green-time
    # tree, 2 per-tick switcher
    sig_kind = 0 if not use_sig else (1 if sduration else 2)
    dv = vlaws.shape[1]
    ds = slaws.shape[1]
    nAv = vlaws.shape[2]
    nAs = slaws.shape[2]
    nvmem = vmem_cols.shape[0]
    nsmem = smem_cols.shape[0]
    tracing = trace.shape[0] == n
    deviating = dev.shape[0] == n
    T_run = duration if T <= 0 else min(T, duration)
    # the largest anchor count of any law, for the kernel scratch buffers
    kmax_v = 1
    for l in range(vk[0].shape[0] - 1):
        if vk[0][l + 1] - vk[0][l] > kmax_v:
            kmax_v = vk[0][l + 1] - vk[0][l]
    kmax_s = 1
    for l in range(sk[0].shape[0] - 1):
        if sk[0][l + 1] - sk[0][l] > kmax_s:
            kmax_s = sk[0][l + 1] - sk[0][l]

    for i in prange(n):
        mv = np.empty(N, np.int64)
        s = np.empty(N)
        v = np.empty(N)
        status = np.empty(N)
        dep = np.empty(N)
        for q in range(N):
            mv[q] = int(states[i, q])
            s[q] = states[i, N + q]
            v[q] = states[i, 2 * N + q]
            status[q] = states[i, 3 * N + q]
            dep[q] = states[i, 4 * N + q]
        b = 5 * N
        told = np.empty(N)
        paid = np.empty(N)
        for q in range(N):
            told[q] = states[i, b + 12 + q]
            paid[q] = states[i, b + 12 + N + q]
        plan = states[i, b + 10]
        fg = np.empty(N_PHASES)                  # this episode's fixed plan
        for p4 in range(N_PHASES):
            fg[p4] = states[i, b + 12 + 2 * N + p4]
        phase = int(states[i, b + 0])
        t_phase = states[i, b + 1]
        in_ar = states[i, b + 2] > 0.5
        ar_t = states[i, b + 3]
        noise = states[i, b + 5]
        clock_phase = states[i, b + 11]         # hidden: see sample_starts

        latch = np.full(N, -1, np.int64)
        step = np.zeros(N, np.int64)
        vslots = np.zeros((N, max(nvmem, 1)))
        vhave = np.zeros(N, np.bool_)
        vage = np.zeros(N, np.int64)
        s_latch = -1
        s_step = 0
        sslots = np.zeros(max(nsmem, 1))
        shave = False
        sage = 0
        zv = np.zeros(dv)
        zs = np.zeros(ds)
        px = np.zeros(N)
        py = np.zeros(N)
        pvx = np.zeros(N)
        pvy = np.zeros(N)
        uv = np.zeros(nAv)
        us = np.zeros(nAs)
        kvv = np.zeros(kmax_v)
        kvs = np.zeros(kmax_s)
        act_v = np.zeros(N, np.int64)
        acc_v = np.zeros(N)
        has_lead = np.zeros(N, np.bool_)
        lead_gap = np.empty(N)
        lead_j = np.zeros(N, np.int64)
        spawned_grp = np.zeros(2 * 4 + 2, np.bool_)
        fresh = np.zeros(N, np.bool_)
        sp_sum = np.zeros(4)
        sp_cnt = np.zeros(4)
        q_wait = np.zeros(4)
        ts = int(dev[i, 3]) if deviating else -1
        g = 0.0
        disc = 1.0

        for t in range(T_run):
            time_now = t * dt
            # ---- observe + act: vehicles -------------------------------------
            green_now = not in_ar
            # positions and velocities in the plane, for the radius sensor
            for q in range(N):
                if status[q] != 1.0:
                    continue
                m = mv[q]
                kk = -1
                for vv in range(cum.shape[1]):
                    if s[q] >= cum[m, vv]:
                        kk += 1
                if kk > n_vert[m] - 2:
                    kk = n_vert[m] - 2
                if kk < 0:
                    kk = 0
                c0 = cum[m, kk]
                seg = cum[m, kk + 1] - c0
                f = (s[q] - c0) / seg
                px[q] = pts[m, kk, 0] + f * (pts[m, kk + 1, 0] - pts[m, kk, 0])
                py[q] = pts[m, kk, 1] + f * (pts[m, kk + 1, 1] - pts[m, kk, 1])
                pvx[q] = v[q] * ((pts[m, kk + 1, 0] - pts[m, kk, 0]) / seg)
                pvy[q] = v[q] * ((pts[m, kk + 1, 1] - pts[m, kk, 1]) / seg)
            for q in range(N):
                if status[q] != 1.0:
                    continue
                m = mv[q]
                # leader on my lane (before the box) or my exit edge (past it)
                past_q = s[q] > s_junc[m] + 2.0
                xq = path_len[m] - s[q]
                best = 1e18
                bj = 0
                for o in range(N):
                    if o == q or status[o] != 1.0:
                        continue
                    mr = mv[o]
                    past_r = s[o] > s_junc[mr] + 2.0
                    if (not past_q) and (not past_r) and lane_group[mr] == lane_group[m]:
                        if s[o] > s[q]:
                            dd = s[o] - s[q]
                            if dd < best:
                                best = dd
                                bj = o
                    if past_q and past_r and to_edge[mr] == to_edge[m]:
                        xr = path_len[mr] - s[o]
                        if xr < xq:
                            dd = xq - xr
                            if dd < best:
                                best = dd
                                bj = o
                has = best < 1e17
                gap = best - L_VEH if has else FAR
                d_here = s_stop[m] - s[q]
                if occ_hi > occ_lo and d_here >= occ_lo and d_here <= occ_hi:
                    has = False                  # the sensor is blind here
                    gap = FAR
                d_conf = FAR
                for k in range(cps.shape[1]):
                    dd = cps[m, k] - s[q]
                    if dd > 0.0 and dd < d_conf:
                        d_conf = dd
                if d_conf > FAR:
                    d_conf = FAR
                for k in range(dv):
                    zv[k] = 0.0
                zv[0] = v[q]
                d_stop = s_stop[m] - s[q]
                zv[1] = d_stop
                near = d_stop > 0.0 and d_stop <= NEAR_INT
                zv[2] = 1.0 if near else 0.0
                if event:
                    hear = near and told[q] < 0.5
                else:
                    hear = near or d_stop <= 0.0
                if hear:
                    zv[3] = 1.0 if (green_now and green[phase, m]) else 0.0
                    if event:
                        w_min, w_max = _t_window(m, phase, t_phase, in_ar, ar_t,
                                                 plan, sig_kind, green,
                                                 fg, dt)
                        zv[4] = w_min
                        zv[5] = w_max
                    else:
                        zv[4] = t_phase
                        zv[5] = SIG_HIDDEN
                    zv[6] = 1.0 if in_ar else 0.0
                else:
                    zv[3] = SIG_HIDDEN
                    zv[4] = SIG_HIDDEN
                    zv[5] = SIG_HIDDEN
                    zv[6] = SIG_HIDDEN
                if event and near:
                    told[q] = 1.0              # heard, at observation time
                zv[7] = gap
                zv[8] = (v[q] - v[bj]) if has else 0.0
                zv[9] = 1.0 if has else 0.0
                zv[10] = d_conf
                zv[11] = 1.0 if dirs[m] == 2 else 0.0
                zv[12] = 1.0 if dirs[m] == 0 else 0.0
                zv[13] = (t / duration + clock_phase) % 1.0
                zv[14] = noise
                # the radius sensor: every active vehicle within SENSE_R
                n_near = 0.0
                near_best = 1e18
                near_rate = 0.0
                riv_best = 1e18
                riv_d = FAR
                for o in range(N):
                    if o == q or status[o] != 1.0:
                        continue
                    ddx = px[o] - px[q]
                    ddy = py[o] - py[q]
                    dist = np.sqrt(ddx * ddx + ddy * ddy)
                    if dist > SENSE_R:
                        continue
                    n_near += 1.0
                    if dist < near_best:
                        near_best = dist
                        near_rate = ddx * (pvx[o] - pvx[q]) + ddy * (pvy[o] - pvy[q])
                    mo = mv[o]
                    if conf[m, mo]:
                        d_me = s_cp[m, mo] - s[q]
                        d_rv = s_cp[mo, m] - s[o]
                        if d_me > -HALF_CONF and d_rv > -HALF_CONF:
                            t_me = (d_me if d_me > 0.0 else 0.0) / (v[q] if v[q] > T_V_MIN else T_V_MIN)
                            t_rv = (d_rv if d_rv > 0.0 else 0.0) / (v[o] if v[o] > T_V_MIN else T_V_MIN)
                            gt = abs(t_rv - t_me)
                            if gt < riv_best:
                                riv_best = gt
                                riv_d = d_rv
                zv[15] = n_near
                if near_best < 1e17:
                    zv[16] = near_best
                    zv[17] = -(near_rate / near_best) if near_best > 1e-9 else 0.0
                else:
                    zv[16] = FAR
                    zv[17] = 0.0
                if riv_best < 1e17:
                    zv[18] = 1.0
                    zv[19] = riv_best
                    zv[20] = riv_d
                else:
                    zv[18] = 0.0
                    zv[19] = FAR
                    zv[20] = FAR
                vhave[q], vage[q] = _write_mem(zv, vn_obs, vslots[q], vhave[q],
                                               vage[q], vmem_cols, vw_col,
                                               vw_thr, vw_neg)
                zv[dv - 1] = 1.0
                law, latch[q], step[q] = _tick(zv, latch[q], step[q],
                                               vt[0], vt[1], vt[2], vt[3], vt[4], vt[5], vt[6], vt[7], vt[8], vt[9], vt[10], vt[11], vt[12], vt[13], vt[14], vt[15], vt[16], vt[17], vt[18], vt[19], vt[20], vt[21], vt[22])
                law_out(zv, law, vlaws, vk[0], vk[1], vk[2], vk[3], vk[4], vk[5],
                        vk[6], vk[7], vk[8], vk[9], uv, kvv)
                if vscalar:
                    u = uv[0]
                    if deviating and ts == q and dev[i, 1] > 0.0 and t >= dev[i, 0] \
                            and t < dev[i, 0] + dev[i, 1]:
                        u = dev[i, 2]
                    if u < -B_MAX:
                        u = -B_MAX
                    elif u > A_MAX:
                        u = A_MAX
                    acc_v[q] = u
                    a = 0
                else:
                    a = _argmax_out(uv, nAv)
                    if deviating and ts == q and dev[i, 1] > 0.0 and t >= dev[i, 0] \
                            and t < dev[i, 0] + dev[i, 1]:
                        a = int(dev[i, 2])
                    acc_v[q] = ACCELS[a]
                act_v[q] = a
                if tracing and ts == q and t < trace.shape[1]:
                    for k in range(dv):
                        trace[i, t, k] = zv[k]
                    trace[i, t, dv] = law
                    trace[i, t, dv + 1] = acc_v[q] if vscalar else a
                    trace[i, t, dv + 2] = 0.0
                    trace[i, t, dv + 3] = latch[q]
                    trace[i, t, dv + 4] = step[q]

            # ---- observe + act: signal ---------------------------------------
            for k in range(ds):
                zs[k] = 0.0
            for k in range(N_PHASES):
                zs[3 * N_PHASES + k] = FAR
            for q in range(N):
                if status[q] != 1.0:
                    continue
                m = mv[q]
                d_stop = s_stop[m] - s[q]
                if d_stop > 0.0 and d_stop < QUEUE_D:
                    for k in range(N_PHASES):
                        if green[k, m]:
                            zs[N_PHASES + k] += 1.0
                            zs[2 * N_PHASES + k] += v[q]
                            if d_stop < zs[3 * N_PHASES + k]:
                                zs[3 * N_PHASES + k] = d_stop
                            if v[q] < QUEUE_V:
                                zs[k] += 1.0
            for k in range(N_PHASES):
                if zs[N_PHASES + k] > 0.0:
                    zs[2 * N_PHASES + k] = zs[2 * N_PHASES + k] / zs[N_PHASES + k]
                else:
                    zs[2 * N_PHASES + k] = SIG_EMPTY
            zs[4 * N_PHASES + phase] = 1.0
            zs[5 * N_PHASES] = t_phase
            zs[5 * N_PHASES + 1] = 1.0 if in_ar else 0.0
            zs[5 * N_PHASES + 2] = (t / duration + clock_phase) % 1.0
            zs[5 * N_PHASES + 3] = noise
            s_law = -1
            dur = 0.0
            if use_sig:
                shave, sage = _write_mem(zs, sn_obs, sslots, shave, sage,
                                         smem_cols, sw_col, sw_thr, sw_neg)
                zs[ds - 1] = 1.0
                s_law, s_latch, s_step = _tick(zs, s_latch, s_step,
                                               st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], st[8], st[9], st[10], st[11], st[12], st[13], st[14], st[15], st[16], st[17], st[18], st[19], st[20], st[21], st[22])
                law_out(zs, s_law, slaws, sk[0], sk[1], sk[2], sk[3], sk[4], sk[5],
                        sk[6], sk[7], sk[8], sk[9], us, kvs)
                if sduration:
                    dur = us[0]
                    a_sig = 0
                else:
                    a_sig = _argmax_out(us, nAs)
            else:
                zs[ds - 1] = 1.0
                a_sig = 1 if t_phase >= fg[phase] else 0
            if deviating and ts == -1 and dev[i, 1] > 0.0 and t >= dev[i, 0] \
                    and t < dev[i, 0] + dev[i, 1]:
                if sduration and use_sig:
                    dur = dev[i, 2]
                else:
                    a_sig = int(dev[i, 2])
            if tracing and ts == -1 and t < trace.shape[1]:
                for k in range(ds):
                    trace[i, t, k] = zs[k]
                trace[i, t, ds] = s_law
                trace[i, t, ds + 1] = dur if (sduration and use_sig) else a_sig
                trace[i, t, ds + 2] = 0.0
                trace[i, t, ds + 3] = s_latch
                trace[i, t, ds + 4] = s_step

            # ---- signal update ------------------------------------------------
            r = 0.0
            was_ar = in_ar
            if was_ar:
                ar_t += dt
                if ar_t >= T_AR - 1e-9:
                    phase = (phase + 1) % N_PHASES
                    t_phase = 0.0
                    in_ar = False
                    ar_t = 0.0
            else:
                if sduration and use_sig:
                    if plan <= 0.0:
                        plan = dur
                        if plan < T_MIN:
                            plan = T_MIN
                        elif plan > T_MAX:
                            plan = T_MAX
                    t_phase += dt
                    if t_phase >= plan - 1e-9:
                        in_ar = True
                        ar_t = 0.0
                        plan = 0.0
                else:
                    t_phase += dt
                    if (a_sig == 1 and t_phase >= T_MIN) or t_phase >= T_MAX:
                        in_ar = True
                        ar_t = 0.0

            # ---- spawn ---------------------------------------------------------
            for k in range(spawned_grp.shape[0]):
                spawned_grp[k] = False
            for q in range(N):
                fresh[q] = False
            for q in range(N):
                if status[q] != 0.0 or dep[q] > time_now:
                    continue
                m = mv[q]
                grp = lane_group[m]
                if spawned_grp[grp]:
                    continue
                ok = True
                mn = 1e18
                for o in range(N):
                    if status[o] != 1.0 or lane_group[mv[o]] != grp:
                        continue
                    ahead = s[o] - s_spawn[m]
                    if ahead >= -1e-9 and ahead < mn:
                        mn = ahead
                if mn < 1e17 and mn < 2.0 * L_VEH + SPAWN_GAP:
                    ok = False
                if ok:
                    status[q] = 1.0
                    s[q] = s_spawn[m]
                    # v[q] already holds the slot's entry speed
                    latch[q] = -1
                    step[q] = 0
                    for k in range(vslots.shape[1]):
                        vslots[q, k] = 0.0
                    vhave[q] = False
                    vage[q] = 0
                    fresh[q] = True
                    spawned_grp[grp] = True

            # ---- kinematics, red running ----------------------------------------
            n_ran = 0
            green_sum = 0.0
            comfort_sum = 0.0
            # the traced car's OWN reward under the per-car reward, for the
            # trace (a per-car value function needs it; G is unchanged)
            r_me = 0.0
            for q in range(N):
                if status[q] != 1.0 or fresh[q]:
                    continue                     # spawned this tick: not driven yet
                m = mv[q]
                before = s[q]
                vold = v[q]
                vn = v[q] + acc_v[q] * dt
                if vn < 0.0:
                    vn = 0.0
                elif vn > V0:
                    vn = V0
                s[q] = s[q] + vn * dt
                v[q] = vn
                jerk = ((vn - vold) / dt / B_MAX) ** 2
                comfort_sum += jerk
                if q == ts:
                    r_me -= W_COMFORT * jerk * dt
                gm = (not in_ar) and green[phase, m]
                if before < s_stop[m] and s[q] >= s_stop[m]:
                    if gm:
                        green_sum += v[q] / V0
                        if q == ts:
                            r_me += R_GREEN * v[q] / V0
                    else:
                        n_ran += 1
                        if q == ts:
                            r_me -= R_RED_CAR
            r -= (R_RED_CAR if car_r else R_RED) * n_ran
            if car_r:
                r += R_GREEN * green_sum
                r -= W_COMFORT * comfort_sum * dt

            # ---- collisions -----------------------------------------------------
            n_rear = 0
            n_cross = 0
            for q in range(N):
                has_lead[q] = False
                lead_gap[q] = FAR
                if status[q] != 1.0:
                    continue
                m = mv[q]
                past_q = s[q] > s_junc[m] + 2.0
                xq = path_len[m] - s[q]
                best = 1e18
                bj = 0
                for rr in range(N):
                    if rr == q or status[rr] != 1.0:
                        continue
                    mr = mv[rr]
                    past_r = s[rr] > s_junc[mr] + 2.0
                    if (not past_q) and (not past_r) and lane_group[mr] == lane_group[m]:
                        if s[rr] > s[q]:
                            dd = s[rr] - s[q]
                            if dd < best:
                                best = dd
                                bj = rr
                    if past_q and past_r and to_edge[mr] == to_edge[m]:
                        xr = path_len[mr] - s[rr]
                        if xr < xq:
                            dd = xq - xr
                            if dd < best:
                                best = dd
                                bj = rr
                if best < 1e17:
                    has_lead[q] = True
                    lead_gap[q] = best - L_VEH
                    lead_j[q] = bj
            # ---- traffic performance, before collided cars are removed ----------
            for a4 in range(4):
                sp_sum[a4] = 0.0
                sp_cnt[a4] = 0.0
                q_wait[a4] = 0.0
            delay = 0.0
            n_stuck = 0
            n_stop = 0
            for q in range(N):
                if status[q] == 1.0:
                    m = mv[q]
                    a4 = appr[m]
                    vf = v[q] / V0
                    sp_sum[a4] += vf
                    sp_cnt[a4] += 1.0
                    delay += 1.0 - vf
                    if q == ts:
                        r_me -= W_CAR_DELAY * (1.0 - vf) * dt
                    if v[q] < QUEUE_V:
                        d_q = s_stop[m] - s[q]
                        red_q = in_ar or (not green[phase, m])
                        red_ok = red_q and (d_q <= NEAR_INT or not car_r)
                        legit = d_q > 0.0 and ((has_lead[q] and lead_gap[q] < QUEUE_GAP)
                                               or red_ok)
                        if not legit:
                            n_stuck += 1
                            if q == ts:
                                r_me -= W_STUCK_CAR * dt
                        if d_q > 0.0:
                            q_wait[a4] += 1.0
                        if car_r and red_q and d_q > 0.0 and d_q <= STOP_ZONE \
                                and paid[q] < 0.5:
                            paid[q] = 1.0
                            n_stop += 1
                            if q == ts:
                                r_me += R_STOP
                elif status[q] == 0.0 and dep[q] <= time_now:
                    a4 = appr[mv[q]]
                    sp_cnt[a4] += 1.0
                    delay += 1.0
                    q_wait[a4] += 1.0
                    if q == ts:
                        r_me -= W_CAR_DELAY * dt
            ms = 0.0
            na = 0.0
            for a4 in range(4):
                if sp_cnt[a4] > 0.0:
                    ms += sp_sum[a4] / sp_cnt[a4]
                    na += 1.0
            if na > 0.0 and not car_r:
                r += W_SPEED * (ms / na) * dt
            r -= (W_CAR_DELAY if car_r else W_DELAY) * delay * dt
            r -= (W_STUCK_CAR if car_r else W_STUCK) * n_stuck * dt
            if car_r:
                r += R_STOP * n_stop
            else:
                qsq = 0.0
                for a4 in range(4):
                    qsq += q_wait[a4] * q_wait[a4]
                r -= W_QUEUE * qsq * dt

            hit = np.zeros(N, np.bool_)
            fault = np.zeros(N, np.bool_)
            for q in range(N):
                if status[q] == 1.0 and has_lead[q] and lead_gap[q] < 0.0:
                    hit[q] = True
                    fault[q] = True                  # the striker, not the struck
                    hit[lead_j[q]] = True
                    n_rear += 1
            for q in range(N):
                if status[q] != 1.0:
                    continue
                m = mv[q]
                for rr in range(q + 1, N):
                    if status[rr] != 1.0:
                        continue
                    mr = mv[rr]
                    if not conf[m, mr]:
                        continue
                    if abs(s_cp[m, mr] - s[q]) < HALF_CONF and \
                            abs(s_cp[mr, m] - s[rr]) < HALF_CONF:
                        hit[q] = True
                        hit[rr] = True
                        fault[q] = True
                        fault[rr] = True
                        n_cross += 1
            n_fault = 0
            for q in range(N):
                if hit[q]:
                    status[q] = 2.0
                    v[q] = 0.0
                if fault[q]:
                    n_fault += 1
                    if q == ts:
                        r_me -= R_COLL
            r -= R_COLL * (n_fault if car_r else (n_rear + n_cross))

            # ---- exits ------------------------------------------------------------
            n_out = 0
            for q in range(N):
                if status[q] == 1.0 and s[q] >= s_exit[mv[q]]:
                    status[q] = 2.0
                    n_out += 1
                    if q == ts:
                        r_me += R_EXIT
            r += R_EXIT * n_out

            if t + 1 >= duration:
                n_left = 0
                for q in range(N):
                    if status[q] == 1.0 or (status[q] == 0.0 and dep[q] <= time_now):
                        n_left += 1
                        if q == ts:
                            r_me -= R_LEFT
                r -= R_LEFT * n_left

            if tracing and t < trace.shape[1]:
                # the tick's reward, in the slot the trace leaves free: the
                # traced car's own under the per-car reward, else the team's
                trace[i, t, (dv if ts >= 0 else ds) + 2] = (r_me if (car_r and ts >= 0)
                                                            else r)
            g += disc * r
            disc *= gamma
        G[i] = g
