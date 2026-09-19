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
                           QUEUE_GAP, R_GREEN, R_LEFT, R_RED_CAR, R_RED_FLOOR,
                           R_STOP, STOP_ZONE, T_SAFE, T_TTC, TTC_FLOOR,
                           W_CROSS_FAULT, W_TTC_LEAD, W_TTC_RIVAL,
                           PASS_GATE_D, W_FALSE_PASS,
                           W_CAR_DELAY, W_COMFORT, W_DELAY, W_OVER, COMMIT_D,
                           W_QUEUE, W_STUCK_CAR, W_STUCK_FREE, W_STUCK_QUEUE,
                           W_STUCK_EARLY, STOP_ZONE, NEAR_INT,
                           W_SPEED, W_STUCK,
                           L_VEH, N_PHASES, NEAR_INT, QUEUE_D, QUEUE_V, R_COLL,
                           NEAR_D, Q_OFF, QS_OFF, VA_OFF, N_OFF, V_OFF, NN_OFF,
                           D_OFF, PH_OFF, MISC_OFF,
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
                     1.0 if vh in ("scalar", "pass") else 0.0,
                     1.0 if sh == "duration" else 0.0,
                     env.occlude_lo, env.occlude_hi,
                     1.0 if reward == "car" else 0.0,
                     1.0 if vh == "pass" else 0.0], np.float64)


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
    vpass = p[11] > 0.5
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

    # the two approaches each phase serves, lower index first. They discharge
    # in parallel, so the queue that governs a phase is the longer of them.
    ph_a0 = np.zeros(N_PHASES, np.int64)
    ph_a1 = np.zeros(N_PHASES, np.int64)
    for k0 in range(N_PHASES):
        lo = 1 << 30
        hi = -1
        for m0 in range(green.shape[1]):
            if green[k0, m0]:
                a0 = appr[m0]
                if a0 < lo:
                    lo = a0
                if a0 > hi:
                    hi = a0
        ph_a0[k0] = lo
        ph_a1[k0] = hi

    # the band of each movement's path that holds any of its conflict points
    n_mv = conf.shape[0]
    cp_lo = np.empty(n_mv)
    cp_hi = np.empty(n_mv)
    for m0 in range(n_mv):
        lo = 1e18
        hi = -1e18
        for m1 in range(n_mv):
            if conf[m0, m1]:
                c = s_cp[m0, m1]
                if c < lo:
                    lo = c
                if c > hi:
                    hi = c
        cp_lo[m0] = lo
        cp_hi[m0] = hi

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
        # `rival_dt` kept per slot: `zv` is one reused scratch vector, and the
        # red-run charge below needs the gap the crossing car SAW this tick
        rdt = np.full(N, FAR)
        # the same trick for both time-to-collision columns: the charge below
        # reads the number the car SAW this tick, and `zv` is reused per slot
        lttc = np.full(N, FAR)
        rttc = np.full(N, FAR)
        pvx = np.zeros(N)
        pvy = np.zeros(N)
        uv = np.zeros(nAv)
        us = np.zeros(nAs)
        kvv = np.zeros(kmax_v)
        kvs = np.zeros(kmax_s)
        act_v = np.zeros(N, np.int64)
        acc_v = np.zeros(N)
        say_raw = np.zeros(N)               # this tick's raw declaration logit
        said = np.zeros(N)                  # last tick's, gated: what others hear
        has_lead = np.zeros(N, np.bool_)
        lead_gap = np.empty(N)
        lead_j = np.zeros(N, np.int64)
        act_idx = np.zeros(N, np.int64)     # active slots, rebuilt each tick
        va_n = np.zeros(2 * N_PHASES)       # cars per (phase, side), for va
        # A LEADER IS ALWAYS IN THE SAME LANE GROUP (or, past the junction, on
        # the same exit edge), so the scan never has to leave that bucket. Eight
        # groups and four edges here, which is where the 8x and 4x come from.
        n_grp = 0
        n_edg = 0
        for mm in range(len(lane_group)):
            if lane_group[mm] + 1 > n_grp:
                n_grp = lane_group[mm] + 1
            if to_edge[mm] + 1 > n_edg:
                n_edg = to_edge[mm] + 1
        grp_cnt = np.zeros(n_grp, np.int64)
        grp_slots = np.zeros((n_grp, N), np.int64)
        edg_cnt = np.zeros(n_edg, np.int64)
        edg_slots = np.zeros((n_edg, N), np.int64)
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
            n_act_s = 0
            for q in range(N):
                if status[q] == 1.0:
                    act_idx[n_act_s] = q
                    n_act_s += 1
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
                dvl = (v[q] - v[bj]) if has else 0.0
                zv[8] = dvl
                zv[9] = 1.0 if has else 0.0
                # TIME TO COLLISION WITH THE LEADER: only while closing, and
                # the gap floored at zero so an overlapping pair reads 0 and
                # not a negative time
                if has and dvl > 1e-6 and gap < 0.5 * FAR:
                    zv[24] = (gap if gap > 0.0 else 0.0) / dvl
                else:
                    zv[24] = FAR
                lttc[q] = zv[24]
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
                rttc_best = 1e18
                riv_d = FAR
                riv_o = -1
                # THE OTHER QUADRATIC LOOP, and the dominant one: this runs per
                # active car per tick to build its observation. Two changes, both
                # exact. The active list skips empty slots (the old scan tested
                # `status` on every one of N). And the range test is done on the
                # SQUARED distance, so the sqrt is paid only by the few pairs
                # actually within SENSE_R instead of by all N^2 -- d > R and
                # d^2 > R^2 agree for non-negative d, and `dist` is still the
                # true distance wherever it is kept.
                for oi in range(n_act_s):
                    o = act_idx[oi]
                    if o == q:
                        continue
                    ddx = px[o] - px[q]
                    ddy = py[o] - py[q]
                    d2 = ddx * ddx + ddy * ddy
                    if d2 > SENSE_R * SENSE_R:
                        continue
                    dist = np.sqrt(d2)
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
                                riv_o = o
                            # A COLLISION COURSE, NOT A NEAR MISS. Each of the
                            # two occupies its conflict point over an interval
                            # of half-width HALF_CONF; overlapping intervals
                            # are exactly the condition the crossing collision
                            # test checks, and contact begins when the later
                            # one arrives. Tracked on its OWN minimum: the
                            # rival nearest in arrival time is not always the
                            # one you are going to hit.
                            vq = v[q] if v[q] > T_V_MIN else T_V_MIN
                            vo = v[o] if v[o] > T_V_MIN else T_V_MIN
                            i_me = (d_me - HALF_CONF) / vq
                            if i_me < 0.0:
                                i_me = 0.0
                            o_me = (d_me + HALF_CONF) / vq
                            i_rv = (d_rv - HALF_CONF) / vo
                            if i_rv < 0.0:
                                i_rv = 0.0
                            o_rv = (d_rv + HALF_CONF) / vo
                            if i_me < o_rv and i_rv < o_me:
                                tc = i_me if i_me > i_rv else i_rv
                                if tc < rttc_best:
                                    rttc_best = tc
                zv[25] = rttc_best if rttc_best < 1e17 else FAR
                rttc[q] = zv[25]
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
                    # does the one that is coming have the right to?
                    zv[21] = 0.0 if (green_now and green[phase, mv[riv_o]])                         else 1.0
                    zv[22] = v[riv_o]
                    zv[23] = said[riv_o]            # what it said LAST tick
                else:
                    zv[18] = 0.0
                    zv[19] = FAR
                    zv[20] = FAR
                    zv[21] = 0.0
                    zv[22] = 0.0
                    zv[23] = 0.0
                rdt[q] = zv[19]
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
                    if vpass:
                        say_raw[q] = uv[1]
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
                zs[D_OFF + k] = FAR
                va_n[2 * k] = 0.0
                va_n[2 * k + 1] = 0.0
            for q in range(N):
                if status[q] != 1.0:
                    continue
                m = mv[q]
                d_stop = s_stop[m] - s[q]
                if d_stop > 0.0 and d_stop < QUEUE_D:
                    a_q = appr[m]
                    for k in range(N_PHASES):
                        if green[k, m]:
                            zs[N_OFF + k] += 1.0
                            zs[V_OFF + k] += v[q]
                            if d_stop < NEAR_D:
                                zs[NN_OFF + k] += 1.0
                            if d_stop < zs[D_OFF + k]:
                                zs[D_OFF + k] = d_stop
                            # same granularity as the queue: through and left
                            # are separable on one approach
                            sd = 0 if a_q == ph_a0[k] else 1
                            zs[VA_OFF + 2 * k + sd] += v[q]
                            va_n[2 * k + sd] += 1.0
                            if v[q] < QUEUE_V:
                                zs[Q_OFF + k] += 1.0
                                zs[QS_OFF + 2 * k + sd] += 1.0
            for k in range(N_PHASES):
                if zs[N_OFF + k] > 0.0:
                    zs[V_OFF + k] = zs[V_OFF + k] / zs[N_OFF + k]
                else:
                    zs[V_OFF + k] = SIG_EMPTY
                for sd in range(2):
                    if va_n[2 * k + sd] > 0.0:
                        zs[VA_OFF + 2 * k + sd] = zs[VA_OFF + 2 * k + sd] / va_n[2 * k + sd]
                    else:
                        zs[VA_OFF + 2 * k + sd] = SIG_EMPTY
            zs[PH_OFF + phase] = 1.0
            zs[MISC_OFF] = t_phase
            zs[MISC_OFF + 1] = 1.0 if in_ar else 0.0
            zs[MISC_OFF + 2] = (t / duration + clock_phase) % 1.0
            zs[MISC_OFF + 3] = noise
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
            red_sum = 0.0
            false_sum = 0.0
            green_sum = 0.0
            comfort_sum = 0.0
            ttc_sum = 0.0
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
                if vpass:
                    # GATED: only a car on red and still short of the line by
                    # less than PASS_GATE_D is saying anything at all. What it
                    # says is read by every OTHER car on the next tick.
                    d_pre = s_stop[m] - before
                    if (not gm) and d_pre > 0.0 and d_pre <= PASS_GATE_D                             and say_raw[q] > 0.0:
                        said[q] = 1.0
                    else:
                        said[q] = 0.0
                    if car_r and said[q] > 0.5 and v[q] < QUEUE_V:
                        false_sum += W_FALSE_PASS * dt
                        if q == ts:
                            r_me -= W_FALSE_PASS * dt
                if before < s_stop[m] and s[q] >= s_stop[m]:
                    if gm:
                        green_sum += v[q] / V0
                        if q == ts:
                            r_me += R_GREEN * v[q] / V0
                    else:
                        n_ran += 1
                        # PRICED BY THE CONFLICT, NOT BY THE FACT. `rdt[q]` is
                        # the car's own `rival_dt` this tick, so a car slipping
                        # through an empty junction pays the floor and one
                        # cutting across an arriving movement pays all of it.
                        rk = 1.0 - rdt[q] / T_SAFE
                        if rk < 0.0:
                            rk = 0.0
                        elif rk > 1.0:
                            rk = 1.0
                        chg = R_RED_FLOOR + (R_RED_CAR - R_RED_FLOOR) * rk
                        red_sum += chg
                        if q == ts:
                            r_me -= chg
                if car_r:
                    # TIME TO COLLISION, every tick, on both geometries. The
                    # excess of inverse TTC over the threshold, floored so one
                    # tick at a vanishing TTC cannot dominate an episode.
                    a = lttc[q]
                    if a < TTC_FLOOR:
                        a = TTC_FLOOR
                    e_l = 1.0 / a - 1.0 / T_TTC
                    if e_l < 0.0:
                        e_l = 0.0
                    b = rttc[q]
                    if b < TTC_FLOOR:
                        b = TTC_FLOOR
                    e_r = 1.0 / b - 1.0 / T_TTC
                    if e_r < 0.0:
                        e_r = 0.0
                    pen = W_TTC_LEAD * e_l + W_TTC_RIVAL * e_r
                    ttc_sum += pen
                    if q == ts:
                        r_me -= pen * dt
            r -= (red_sum if car_r else R_RED * n_ran)
            r -= false_sum
            if car_r:
                r += R_GREEN * green_sum
                r -= W_COMFORT * comfort_sum * dt
                r -= ttc_sum * dt

            # ---- collisions -----------------------------------------------------
            n_rear = 0
            n_cross = 0
            # A LEADER IS ALWAYS IN THE SAME LANE GROUP, or past the junction on
            # the same exit edge, so the scan never has to leave that bucket.
            # This was O(N^2) a tick -- at n_max=384 over 768 ticks, 113M pair
            # tests an episode against 3.8M at the old 112 slots. Buckets are
            # built in ascending slot order, so each presents the same
            # candidates in the same order the full scan did: ties break
            # identically and the result is bit-for-bit what it gave.
            for gg in range(n_grp):
                grp_cnt[gg] = 0
            for ee in range(n_edg):
                edg_cnt[ee] = 0
            for q in range(N):
                has_lead[q] = False
                lead_gap[q] = FAR
                if status[q] != 1.0:
                    continue
                m = mv[q]
                if s[q] > s_junc[m] + 2.0:
                    ee = to_edge[m]
                    edg_slots[ee, edg_cnt[ee]] = q
                    edg_cnt[ee] += 1
                else:
                    gg = lane_group[m]
                    grp_slots[gg, grp_cnt[gg]] = q
                    grp_cnt[gg] += 1
            for q in range(N):
                if status[q] != 1.0:
                    continue
                m = mv[q]
                past_q = s[q] > s_junc[m] + 2.0
                xq = path_len[m] - s[q]
                best = 1e18
                bj = 0
                if past_q:
                    ee = to_edge[m]
                    for ri in range(edg_cnt[ee]):
                        rr = edg_slots[ee, ri]
                        if rr == q:
                            continue
                        xr = path_len[mv[rr]] - s[rr]
                        if xr < xq:
                            dd = xq - xr
                            if dd < best:
                                best = dd
                                bj = rr
                else:
                    gg = lane_group[m]
                    for ri in range(grp_cnt[gg]):
                        rr = grp_slots[gg, ri]
                        if rr == q:
                            continue
                        if s[rr] > s[q]:
                            dd = s[rr] - s[q]
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
            stuck_w = 0.0
            n_stop = 0
            n_serv = 0
            for q in range(N):
                if status[q] == 1.0:
                    m = mv[q]
                    a4 = appr[m]
                    vf = v[q] / V0
                    sp_sum[a4] += vf
                    sp_cnt[a4] += 1.0
                    # relu: above free-flow is free, never a delay bonus
                    dl = 1.0 - vf
                    if dl < 0.0:
                        dl = 0.0
                    delay += dl
                    if q == ts:
                        r_me -= W_CAR_DELAY * dl * dt
                    d_now_q = s_stop[m] - s[q]
                    if green[phase, m] and d_now_q > 0.0 and d_now_q < COMMIT_D:
                        n_serv += 1
                    if v[q] < QUEUE_V:
                        d_q = d_now_q
                        red_q = in_ar or (not green[phase, m])
                        led_q = has_lead[q] and lead_gap[q] < QUEUE_GAP
                        # three cases, worst first: stopped on green with a free
                        # lane; stopped on green behind a queue that should be
                        # moving; stopped on red much too far short of the line
                        if d_q <= 0.0:
                            w_st = W_STUCK_FREE
                        elif not red_q and not led_q:
                            w_st = W_STUCK_FREE
                        elif not red_q:
                            w_st = W_STUCK_QUEUE
                        elif not led_q and d_q > NEAR_INT:
                            # a red cannot excuse a stop the car cannot see:
                            # beyond NEAR_INT `observe` hands it no signal at
                            # all, so standing there is a free-lane stop
                            w_st = W_STUCK_FREE
                        elif not led_q:
                            # by degree: the road left empty beyond a normal
                            # stopping zone, zero at the line
                            ex = d_q - STOP_ZONE
                            if ex < 0.0:
                                ex = 0.0
                            w_st = W_STUCK_EARLY * ex / NEAR_INT
                        else:
                            w_st = 0.0
                        if w_st > 0.0:
                            stuck_w += w_st
                            if q == ts:
                                r_me -= (W_STUCK_CAR / W_STUCK) * w_st * dt
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
            r -= ((W_STUCK_CAR / W_STUCK) if car_r else 1.0) * stuck_w * dt
            if car_r:
                r += R_STOP * n_stop
            else:
                qsq = 0.0
                for a4 in range(4):
                    qsq += q_wait[a4] * q_wait[a4]
                r -= W_QUEUE * qsq * dt
                # OVER-GREEN: green held on a phase with nobody within COMMIT_D
                # of its stop line. sumo_test's floor-of-zero case, charged
                # directly instead of waiting for someone else's delay.
                if n_serv == 0 and not in_ar:
                    r -= W_OVER * dt

            hit = np.zeros(N, np.bool_)
            fault = np.zeros(N, np.bool_)
            xfault = np.zeros(N, np.bool_)      # crossing, without the green
            for q in range(N):
                if status[q] == 1.0 and has_lead[q] and lead_gap[q] < 0.0:
                    hit[q] = True
                    fault[q] = True                  # the striker, not the struck
                    hit[lead_j[q]] = True
                    n_rear += 1
            # CROSSING COLLISIONS ONLY HAPPEN IN THE BOX. Every conflict point
            # of a movement lies in a narrow band of its path, so a car outside
            # [cp_lo, cp_hi] +- HALF_CONF cannot be in ANY conflict and its whole
            # inner loop is dead -- which is most cars most of the time, since
            # the approaches are long. With the active list this takes the pair
            # tests from N^2/2 down to (cars near the box)^2/2. Exact: the pairs
            # skipped are precisely those whose first condition was already false.
            for ai in range(n_act_s):
                q = act_idx[ai]
                m = mv[q]
                if s[q] < cp_lo[m] - HALF_CONF or s[q] > cp_hi[m] + HALF_CONF:
                    continue
                for ri in range(ai + 1, n_act_s):
                    rr = act_idx[ri]
                    mr = mv[rr]
                    if not conf[m, mr]:
                        continue
                    if abs(s_cp[m, mr] - s[q]) < HALF_CONF and \
                            abs(s_cp[mr, m] - s[rr]) < HALF_CONF:
                        hit[q] = True
                        hit[rr] = True
                        if not ((not in_ar) and green[phase, m]):
                            xfault[q] = True
                        if not ((not in_ar) and green[phase, mr]):
                            xfault[rr] = True
                        n_cross += 1
            n_fault = 0
            n_xfault = 0
            for q in range(N):
                if hit[q]:
                    status[q] = 2.0
                    v[q] = 0.0
                if fault[q]:
                    n_fault += 1
                    if q == ts:
                        r_me -= R_COLL
                if xfault[q]:
                    n_xfault += 1
                    if q == ts:
                        r_me -= W_CROSS_FAULT * R_COLL
            if car_r:
                r -= R_COLL * (n_fault + W_CROSS_FAULT * n_xfault)
            else:
                r -= R_COLL * (n_rear + n_cross)

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
