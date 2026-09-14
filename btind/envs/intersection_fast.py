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

from ..tick import _fires, _tick, flatten, no_dev, no_trace, tick_args, world_args
from .intersection import (A_MAX, ACCELS, B_MAX, DELAY_W, FAR, FIXED_GREEN, HALF_CONF,
                           L_VEH, N_PHASES, NEAR_INT, QUEUE_D, QUEUE_V, R_COLL,
                           R_EXIT, R_RED, SIG_HIDDEN, SPAWN_GAP, T_AR, T_MAX, T_MIN,
                           V0)


def params(env, vehicle_bank=None, signal_bank=None):
    vh = (vehicle_bank or {}).get("head", env.veh_head)
    sh = (signal_bank or {}).get("head", env.sig_head)
    return np.array([env.N, env.M, env.dt, env.duration, env.gamma,
                     1.0 if env.event else 0.0,
                     1.0 if vh == "scalar" else 0.0,
                     1.0 if sh == "duration" else 0.0,
                     env.occlude_lo, env.occlude_hi], np.float64)


@njit(cache=True, inline="always")
def _t_change(m, phase, t_phase, in_ar, ar_t, green, fixed_green, dt):
    """Ticks until movement m's light changes under the fixed plan (see the
    numpy model's `_t_change`)."""
    if (not in_ar) and green[phase, m]:
        t = fixed_green[phase] - t_phase
    else:
        t = (0.0 if in_ar else fixed_green[phase] - t_phase)
        t += (T_AR - ar_t) if in_ar else T_AR
        q = (phase + 1) % N_PHASES
        for _ in range(N_PHASES):
            if green[q, m]:
                break
            t += fixed_green[q] + T_AR
            q = (q + 1) % N_PHASES
    t = np.round(t / dt)
    return t if t > 0.0 else 0.0


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
            np.ascontiguousarray(cps, np.float64))


def run(env, vehicle_bank, signal_bank, s, T, trace=None, dev=None):
    """Roll both trees from the states `s`. `signal_bank` None = fixed plan."""
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
    rollout(np.ascontiguousarray(s), params(env, vehicle_bank, signal_bank),
            *geometry(env), T, G,
            no_trace() if trace is None else trace,
            no_dev() if dev is None else dev,
            *world_args(fv), *world_args(fs), use_sig,
            tick_args(fv), tick_args(fs))
    return G


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
            dirs, lane_group, to_edge, green, fixed_green, cps, T, G, trace, dev,
            vlaws, vmem_cols, vw_col, vw_thr, vw_neg, vn_obs,
            slaws, smem_cols, sw_col, sw_thr, sw_neg, sn_obs, use_sig, vt, st):
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
    dv = vlaws.shape[1]
    ds = slaws.shape[1]
    nAv = vlaws.shape[2]
    nAs = slaws.shape[2]
    nvmem = vmem_cols.shape[0]
    nsmem = smem_cols.shape[0]
    tracing = trace.shape[0] == n
    deviating = dev.shape[0] == n
    T_run = duration if T <= 0 else min(T, duration)

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
        for q in range(N):
            told[q] = states[i, b + 11 + q]
        plan = states[i, b + 10]
        phase = int(states[i, b + 0])
        t_phase = states[i, b + 1]
        in_ar = states[i, b + 2] > 0.5
        ar_t = states[i, b + 3]
        noise = states[i, b + 5]

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
        act_v = np.zeros(N, np.int64)
        acc_v = np.zeros(N)
        has_lead = np.zeros(N, np.bool_)
        lead_gap = np.empty(N)
        lead_j = np.zeros(N, np.int64)
        spawned_grp = np.zeros(2 * 4 + 2, np.bool_)
        fresh = np.zeros(N, np.bool_)
        ts = int(dev[i, 3]) if deviating else -1
        g = 0.0
        disc = 1.0

        for t in range(T_run):
            time_now = t * dt
            # ---- observe + act: vehicles -------------------------------------
            green_now = not in_ar
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
                    zv[4] = (_t_change(m, phase, t_phase, in_ar, ar_t, green,
                                       fixed_green, dt) if event else t_phase)
                    zv[5] = 1.0 if in_ar else 0.0
                else:
                    zv[3] = SIG_HIDDEN
                    zv[4] = SIG_HIDDEN
                    zv[5] = SIG_HIDDEN
                if event and near:
                    told[q] = 1.0              # heard, at observation time
                zv[6] = gap
                zv[7] = (v[q] - v[bj]) if has else 0.0
                zv[8] = 1.0 if has else 0.0
                zv[9] = d_conf
                zv[10] = 1.0 if dirs[m] == 2 else 0.0
                zv[11] = 1.0 if dirs[m] == 0 else 0.0
                zv[12] = (t / duration + (abs(noise) * 7.31) % 1.0) % 1.0
                zv[13] = noise
                vhave[q], vage[q] = _write_mem(zv, vn_obs, vslots[q], vhave[q],
                                               vage[q], vmem_cols, vw_col,
                                               vw_thr, vw_neg)
                zv[dv - 1] = 1.0
                law, latch[q], step[q] = _tick(zv, latch[q], step[q],
                                               vt[0], vt[1], vt[2], vt[3], vt[4], vt[5], vt[6], vt[7], vt[8], vt[9], vt[10], vt[11], vt[12], vt[13], vt[14], vt[15], vt[16], vt[17], vt[18], vt[19], vt[20], vt[21], vt[22])
                if vscalar:
                    u = 0.0
                    for k in range(dv):
                        u += zv[k] * vlaws[law, k, 0]
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
                    a = _argmax_law(zv, vlaws, law, nAv)
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
            for q in range(N):
                if status[q] != 1.0:
                    continue
                m = mv[q]
                d_stop = s_stop[m] - s[q]
                if d_stop > 0.0 and d_stop < QUEUE_D:
                    for k in range(N_PHASES):
                        if green[k, m]:
                            zs[N_PHASES + k] += 1.0
                            if v[q] < QUEUE_V:
                                zs[k] += 1.0
            zs[2 * N_PHASES + phase] = 1.0
            zs[3 * N_PHASES] = t_phase
            zs[3 * N_PHASES + 1] = 1.0 if in_ar else 0.0
            zs[3 * N_PHASES + 2] = (t / duration + (abs(noise) * 7.31) % 1.0) % 1.0
            zs[3 * N_PHASES + 3] = noise
            s_law = -1
            dur = 0.0
            if use_sig:
                shave, sage = _write_mem(zs, sn_obs, sslots, shave, sage,
                                         smem_cols, sw_col, sw_thr, sw_neg)
                zs[ds - 1] = 1.0
                s_law, s_latch, s_step = _tick(zs, s_latch, s_step,
                                               st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], st[8], st[9], st[10], st[11], st[12], st[13], st[14], st[15], st[16], st[17], st[18], st[19], st[20], st[21], st[22])
                if sduration:
                    for k in range(ds):
                        dur += zs[k] * slaws[s_law, k, 0]
                    a_sig = 0
                else:
                    a_sig = _argmax_law(zs, slaws, s_law, nAs)
            else:
                zs[ds - 1] = 1.0
                a_sig = 1 if t_phase >= fixed_green[phase] else 0
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
                    v[q] = V0
                    latch[q] = -1
                    step[q] = 0
                    for k in range(vslots.shape[1]):
                        vslots[q, k] = 0.0
                    vhave[q] = False
                    vage[q] = 0
                    fresh[q] = True
                    spawned_grp[grp] = True

            # ---- kinematics, red running, delay --------------------------------
            n_ran = 0
            delay = 0.0
            for q in range(N):
                if status[q] != 1.0 or fresh[q]:
                    continue                     # spawned this tick: not driven yet
                m = mv[q]
                before = s[q]
                vn = v[q] + acc_v[q] * dt
                if vn < 0.0:
                    vn = 0.0
                elif vn > V0:
                    vn = V0
                s[q] = s[q] + vn * dt
                v[q] = vn
                gm = (not in_ar) and green[phase, m]
                if before < s_stop[m] and s[q] >= s_stop[m] and not gm:
                    n_ran += 1
                delay += (V0 - v[q]) * dt
            for q in range(N):
                if status[q] == 0.0 and dep[q] <= time_now:
                    delay += V0 * dt             # due, blocked at the entrance
            r -= R_RED * n_ran
            r -= DELAY_W * delay

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
            hit = np.zeros(N, np.bool_)
            for q in range(N):
                if status[q] == 1.0 and has_lead[q] and lead_gap[q] < 0.0:
                    hit[q] = True
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
                        n_cross += 1
            for q in range(N):
                if hit[q]:
                    status[q] = 2.0
                    v[q] = 0.0
            r -= R_COLL * (n_rear + n_cross)

            # ---- exits ------------------------------------------------------------
            n_out = 0
            for q in range(N):
                if status[q] == 1.0 and s[q] >= s_exit[mv[q]]:
                    status[q] = 2.0
                    n_out += 1
            r += R_EXIT * n_out

            if tracing and t < trace.shape[1]:
                # the tick's reward, in the slot the trace leaves free
                trace[i, t, (dv if ts >= 0 else ds) + 2] = r
            g += disc * r
            disc *= gamma
        G[i] = g
