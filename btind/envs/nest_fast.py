"""A whole rollout inside one numba call: observe, decide, step, per row.

WHERE THE TIME WENT. A 400-episode, 400-step evaluation costs ~0.34s and almost
none of it is arithmetic: it is 400 Python ticks, each doing one `observe`, one
policy call and one kernel launch over a few hundred rows. The fixed cost per
tick is what dominates -- a 100-episode rollout costs 0.30s against 0.34s for
400, so three quarters of the work is overhead that does not care how much data
it is given. Stacking candidates into one batch (`fastroll`) recovers only 1.6x
because the per-tick Python remains.

The controller is threshold comparisons followed by an affine map. There is
nothing in it that needs Python, so the whole loop moves into the kernel and the
per-tick cost disappears entirely.

THE BANK, FLATTENED. Clauses become three parallel arrays plus an index, so a
guard is a slice rather than a list of lists:

    lit_col[i], lit_thr[i], lit_neg[i]      every literal of every clause
    cl_start[c], cl_len[c]                  where clause c's literals live
    laws[c]                                 (C+1, d, 2), the default last

EXACTNESS IS THE WHOLE POINT, so `check` compares this against the Python path
step for step on the same states. A fast evaluator that disagrees with the slow
one is not an optimisation, it is a second implementation of a different
controller -- and on a deterministic world agreement should be to the bit.

NOT SUPPORTED HERE: the latch and the blackboard. They need per-row state that
survives the tick, which the kernel can hold but which doubles the surface to
verify, so they stay on the Python path until this one is trusted. `MemBank`
remains the reference implementation either way.
"""
import numpy as np
from numba import njit, prange

from .nest_kernels import _advance, is_night

_EPS = 1e-9


def flatten(bank, n_obs):
    """Bank -> flat arrays the kernel can read. Laws must already be on z."""
    cols, thrs, negs, start, ln = [], [], [], [], []
    for cl in bank["clauses"]:
        start.append(len(cols))
        ln.append(len(cl))
        for j, t, n in cl:
            cols.append(int(j))
            thrs.append(float(t))
            negs.append(bool(n))
    laws = np.stack([np.asarray(t, float) for t in bank["laws"]]
                    + [np.asarray(bank["default"], float)]) \
        if bank["clauses"] else np.asarray(bank["default"], float)[None]
    return (np.array(cols, np.int64), np.array(thrs, np.float64),
            np.array(negs, np.bool_), np.array(start, np.int64),
            np.array(ln, np.int64), np.ascontiguousarray(laws))


@njit(cache=True, inline="always")
def _obs_row(px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz, p, o):
    """One row of NestWorld.observe, written into `o`. Mirrors nest.py."""
    dtx = tx - px
    dty = ty - py
    d_threat = np.sqrt(dtx * dtx + dty * dty)
    o[0] = d_threat
    o[1] = dtx / max(d_threat, _EPS)
    o[2] = dty / max(d_threat, _EPS)
    o[3] = en
    dfx = fx - px
    dfy = fy - py
    d_food = np.sqrt(dfx * dfx + dfy * dfy)
    o[4] = d_food
    o[5] = dfx / max(d_food, _EPS)
    o[6] = dfy / max(d_food, _EPS)
    o[7] = px
    o[8] = py
    o[9] = tt / p[16]                      # max_t
    o[10] = nz
    o[11] = cr
    dnx = nx - px
    dny = ny - py
    d_nest = np.sqrt(dnx * dnx + dny * dny)
    o[12] = d_nest
    o[13] = dnx / max(d_nest, _EPS)
    o[14] = dny / max(d_nest, _EPS)

    day_len = p[13]
    night = False
    if day_len > 0.0:
        ph = tt % (2.0 * day_len)
        night = ph >= day_len
        o[15] = 1.0 if night else 0.0
        o[16] = ((2.0 * day_len - ph) if night else (day_len - ph)) / day_len
    else:
        o[15] = 0.0
        o[16] = 1.0

    vis, nvis, tvis = p[17], p[18], p[19]
    scale = 1.0
    if vis > 0.0 and night:
        scale = nvis / max(vis, _EPS)
    if vis > 0.0:
        if d_food <= vis * scale:
            o[17] = 1.0
        else:
            o[17] = 0.0
            o[4] = 1.5
            o[5] = 0.0
            o[6] = 0.0
    else:
        o[17] = 1.0
    if tvis > 0.0:
        if d_threat <= tvis * scale:
            o[18] = 1.0
        else:
            o[18] = 0.0
            o[0] = 1.5
            o[1] = 0.0
            o[2] = 0.0
    else:
        o[18] = 1.0


@njit(cache=True, parallel=True)
def rollout_bank(states, p, lit_col, lit_thr, lit_neg, cl_start, cl_len, laws,
                 T, n_obs, G):
    """Advance every row under the bank for T steps, accumulating return."""
    n = states.shape[0]
    C = cl_start.shape[0]
    d = laws.shape[1]
    for i in prange(n):
        px = states[i, 0]; py = states[i, 1]
        fx = states[i, 2]; fy = states[i, 3]
        tx = states[i, 4]; ty = states[i, 5]
        en = states[i, 6]; tt = states[i, 7]
        cr = states[i, 8]; nx = states[i, 9]
        ny = states[i, 10]; nz = states[i, 11]
        z = np.zeros(d)                      # [obs, V_hat, leverage, 1]
        g = 0.0
        disc = 1.0
        for t in range(T):
            _obs_row(px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz, p, z)
            z[d - 1] = 1.0                   # intercept last, as design_matrix
            arm = C                          # the default law
            for c in range(C):
                ok = True
                for k in range(cl_start[c], cl_start[c] + cl_len[c]):
                    above = z[lit_col[k]] > lit_thr[k]
                    if lit_neg[k]:
                        above = not above
                    if not above:
                        ok = False
                        break
                if ok:
                    arm = c
                    break
            ux = 0.0
            uy = 0.0
            for q in range(d):
                ux += z[q] * laws[arm, q, 0]
                uy += z[q] * laws[arm, q, 1]
            nrm = np.sqrt(ux * ux + uy * uy)
            if nrm < 1e-9:
                nrm = 1e-9
            ux /= nrm
            uy /= nrm
            (px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz, r,
             done) = _advance(px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz,
                              ux, uy, p)
            g += disc * r
            disc *= p[9]
            if done:
                break
        G[i] = g


def params(env):
    """Kernel parameters plus the observation-only fields the rollout needs."""
    from .nest_kernels import params_array
    base = params_array(env)
    return np.concatenate([base, np.array(
        [env.max_t, env.vision_r, env.night_vision_r, env.threat_vision_r],
        float)])


def flatten_mem(bank, n_obs):
    """Bank with memory/latch -> flat arrays. Layout mirrors `mem_names`.

        z = [ obs , V_hat , leverage , slots... , have , 1 ]

    V_hat and leverage are zero here: the kernel cannot run an xgboost predict
    or a critic solve, so a bank whose guards mention them must stay on the
    Python path. `uses_vq` says which.
    """
    C = len(bank["clauses"])
    cols, thrs, negs, start, ln = [], [], [], [], []
    for cl in bank["clauses"]:
        start.append(len(cols)); ln.append(len(cl))
        for j, t, n in cl:
            cols.append(int(j)); thrs.append(float(t)); negs.append(bool(n))

    betas = bank.get("betas") or [None] * C
    bcol, bthr, bneg, bstart, blen = [], [], [], [], []
    for b in betas:
        bstart.append(len(bcol)); blen.append(0 if b is None else len(b))
        for j, t, n in (b or []):
            bcol.append(int(j)); bthr.append(float(t)); bneg.append(bool(n))

    m = bank.get("mem")
    mcols = np.array(m["cols"] if m else [], np.int64)
    wcol, wthr, wneg = [], [], []
    for j, t, n in ((m or {}).get("write") or []):
        wcol.append(int(j)); wthr.append(float(t)); wneg.append(bool(n))

    laws = np.stack([np.asarray(t, float) for t in bank["laws"]]
                    + [np.asarray(bank["default"], float)]) if C else \
        np.asarray(bank["default"], float)[None]
    st = np.array(bank.get("sticky") or [False] * C, np.bool_)
    return dict(
        lit_col=np.array(cols, np.int64), lit_thr=np.array(thrs, np.float64),
        lit_neg=np.array(negs, np.bool_), cl_start=np.array(start, np.int64),
        cl_len=np.array(ln, np.int64),
        b_col=np.array(bcol, np.int64), b_thr=np.array(bthr, np.float64),
        b_neg=np.array(bneg, np.bool_), b_start=np.array(bstart, np.int64),
        b_len=np.array(blen, np.int64), sticky=st, mem_cols=mcols,
        w_col=np.array(wcol, np.int64), w_thr=np.array(wthr, np.float64),
        w_neg=np.array(wneg, np.bool_),
        laws=np.ascontiguousarray(laws), n_obs=np.int64(n_obs))


def uses_vq(bank, n_obs):
    """True when some guard, beta or write rule reads V_hat or leverage."""
    from ..memory import _reads_vq
    return _reads_vq(bank, n_obs)


@njit(cache=True, inline="always")
def _fires(z, col, thr, neg, start, ln):
    for k in range(start, start + ln):
        above = z[col[k]] > thr[k]
        if neg[k]:
            above = not above
        if not above:
            return False
    return True


@njit(cache=True, parallel=True)
def rollout_mem(states, p, lit_col, lit_thr, lit_neg, cl_start, cl_len,
                b_col, b_thr, b_neg, b_start, b_len, sticky, mem_cols,
                w_col, w_thr, w_neg, laws, n_obs, T, G):
    """As `rollout_bank`, plus the latch and the blackboard.

    Both are per-row state that persists across ticks -- which in this kernel is
    simply a local variable, because the episode loop lives INSIDE the row loop.
    The Python implementation needs arrays and masks for the same thing.
    """
    n = states.shape[0]
    C = cl_start.shape[0]
    d = laws.shape[1]
    nmem = mem_cols.shape[0]
    for i in prange(n):
        px = states[i, 0]; py = states[i, 1]
        fx = states[i, 2]; fy = states[i, 3]
        tx = states[i, 4]; ty = states[i, 5]
        en = states[i, 6]; tt = states[i, 7]
        cr = states[i, 8]; nx = states[i, 9]
        ny = states[i, 10]; nz = states[i, 11]
        z = np.zeros(d)
        slots = np.zeros(max(nmem, 1))
        have = False
        latch = -1
        g = 0.0
        disc = 1.0
        for t in range(T):
            _obs_row(px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz, p, z)
            z[n_obs] = 0.0          # V_hat     (not available in the kernel)
            z[n_obs + 1] = 0.0      # leverage
            if nmem > 0:
                if w_col.shape[0] > 0 and _fires(z, w_col, w_thr, w_neg, 0,
                                                 w_col.shape[0]):
                    for q in range(nmem):
                        slots[q] = z[mem_cols[q]]
                    have = True
                for q in range(nmem):
                    z[n_obs + 2 + q] = slots[q]
                z[n_obs + 2 + nmem] = 1.0 if have else 0.0
            z[d - 1] = 1.0

            f = C
            for c in range(C):
                if _fires(z, lit_col, lit_thr, lit_neg, cl_start[c], cl_len[c]):
                    f = c
                    break
            arm = f
            if latch >= 0:
                preempt = f < latch
                if not preempt:
                    fired = False
                    if b_len[latch] > 0:
                        fired = _fires(z, b_col, b_thr, b_neg, b_start[latch],
                                       b_len[latch])
                    if not fired:
                        arm = latch
            latch = arm if (arm < C and sticky[arm]) else -1

            ux = 0.0
            uy = 0.0
            for q in range(d):
                ux += z[q] * laws[arm, q, 0]
                uy += z[q] * laws[arm, q, 1]
            nrm = np.sqrt(ux * ux + uy * uy)
            if nrm < 1e-9:
                nrm = 1e-9
            (px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz, r,
             done) = _advance(px, py, fx, fy, tx, ty, en, tt, cr, nx, ny, nz,
                              ux / nrm, uy / nrm, p)
            g += disc * r
            disc *= p[9]
            if done:
                break
        G[i] = g
