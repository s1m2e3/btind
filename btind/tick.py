"""One arbitration, compiled once, shared by every fused kernel.

The tree's per-tick decision -- which arm runs, which STEP of it, and what the
latch carries into the next tick -- existed in three places: `MemBank.arbitrate`
(numpy, over rows), `nest_fast.rollout_mem` and `highway_fast.rollout` (numba,
per row). Three copies of one rule is how the per-arm-list bug survived six
operators, and this file is where the rule grows next, so it is written once.

TWO IMPLEMENTATIONS REMAIN, ON PURPOSE. `MemBank` keeps its own vectorised
numpy arbitration and is the reference; both kernels call `_tick` here. The
equivalence tests compare the two, so a fault in either one is caught by the
other. Collapsing to a single implementation would make those tests test only
the physics.

WHAT AN ARM IS NOW. `laws[c]` is the law of step 0, as it always was. An arm may
carry further steps:

    steps[c]   None, or a list of (advance_clause, theta) for steps 1..K-1.
               While latched at step k < K-1, the arm moves to step k+1 on the
               tick `advance_clause` of step k+1 fires, and acts with that step's
               law on the same tick -- the same ordering the blackboard uses:
               the write and the action that reads it share a tick.
    betas[c]   the termination of the ARM, checked at every step -- the
               decorator wraps the whole Sequence. It reads as SUCCESS: the arm
               releases and the Fallback re-arbitrates. It is NOT the last
               step's exit only: measured, that version left an arm whose
               advance clause never fired latched forever, so "add a step" had
               no identity move and the richer class no longer contained the
               incumbent. With the termination global, a never-firing advance
               clause changes nothing, which is what a proposal starts from.
    fails[c]   None, or a clause. While latched, the tick it fires the arm
               releases AND is excluded from this tick's Fallback, so the arms
               BELOW it get the state. That is a child returning FAILURE to its
               parent Fallback, which is the status a flat guarded action never
               had. A one-step arm with `fails = None` is today's sticky arm.

A multi-step arm must be sticky: a step index only survives the tick inside the
latch. `check_arms` enforces it.

THE ORDER OF THE TICK, which is semantics rather than layout:

    1  fail        the latched arm's fail clause, evaluated first: a failing arm
                   cannot be "still running"
    2  match       first clause that fires, skipping the arm that just failed
    3  preempt     a higher arm that fires takes over (f < latch)
    4  success     `betas` releases the arm, at whatever step it is on
    5  advance     step k -> k+1 if step k+1's advance clause fires
    6  keep        otherwise the latched arm and step continue

With `steps`, `fails` and `betas` all None and `sticky` all False the tick is
`arm = f`, bit for bit the memoryless controller. Every richer bank contains
that one, which is what makes the acceptance test safe to run on it.
"""
import glob
import os

import numpy as np
from numba import njit


def _drop_stale_kernel_caches():
    """Delete a kernel's numba cache when this file is newer than the cache.

    `cache=True` is keyed on the SOURCE FILE OF THE FUNCTION BEING COMPILED.
    A kernel that inlines `_tick` from here is not recompiled when this file
    changes -- numba documents the limitation -- so after an edit to the
    arbitration the fused kernels kept running the previous rule while
    `MemBank` ran the new one. Measured: a multi-step highway bank disagreed by
    13 return units on one episode, and agreed to the bit as soon as the kernel
    file itself was touched. The same mechanism applies to constants imported
    from `highway_batch` and `nest_kernels`, which numba freezes at compile
    time, so those files are watched too.

    Cheap, runs once per process at import, and only deletes files numba will
    regenerate.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    envs = os.path.join(here, "envs")
    watched = [__file__, os.path.join(here, "kernlaw.py")] + [os.path.join(envs, n) for n in
                            ("highway_batch.py", "nest_kernels.py",
                             "intersection.py")]
    newest = max(os.path.getmtime(p) for p in watched if os.path.exists(p))
    for kern in ("highway_fast", "nest_fast", "intersection_fast"):
        for p in glob.glob(os.path.join(envs, "__pycache__", kern + ".*.nb[ic]")):
            try:
                if os.path.getmtime(p) < newest:
                    os.remove(p)
            except OSError:
                pass


_drop_stale_kernel_caches()


def n_steps_of(bank, c):
    st = bank.get("steps")
    return 1 + (len(st[c]) if st and st[c] else 0)


def law_of(bank, c, k):
    """The law arm `c` acts with at step `k`; `c < 0` is the default."""
    if c < 0:
        return bank["default"]
    return bank["laws"][c] if k == 0 else bank["steps"][c][k - 1][1]


def _lits(clauses):
    """Flatten a list of clauses (None allowed) into parallel literal arrays."""
    col, thr, neg, start, ln = [], [], [], [], []
    for cl in clauses:
        start.append(len(col))
        ln.append(0 if cl is None else len(cl))
        for j, t, n in (cl or []):
            col.append(int(j))
            thr.append(float(t))
            neg.append(bool(n))
    return (np.array(col, np.int64), np.array(thr, np.float64),
            np.array(neg, np.bool_), np.array(start, np.int64),
            np.array(ln, np.int64))


def flatten(bank, n_obs):
    """Bank -> flat arrays every kernel reads.

        z = [ obs , V_hat , leverage , slots... , have , 1 ]

    `laws` is (total_steps + 1, d, n_out) with the default LAST; `law_start[c]`
    is the flat index of arm c's step 0 and `law_start[C]` that of the default.
    `a_*` hold the advance clauses indexed by FLAT STEP: the clause at flat step
    s is the condition for entering s from the step before it, so step-0 entries
    are empty.
    """
    C = len(bank["clauses"])
    steps = bank.get("steps") or [None] * C
    betas = bank.get("betas") or [None] * C
    fails = bank.get("fails") or [None] * C
    lit_col, lit_thr, lit_neg, cl_start, cl_len = _lits(bank["clauses"])
    b_col, b_thr, b_neg, b_start, b_len = _lits(betas)
    f_col, f_thr, f_neg, f_start, f_len = _lits(fails)

    from . import kernlaw as KL
    laws, adv, law_start, n_steps, kerns = [], [], [], [], []
    for c in range(C):
        law_start.append(len(laws))
        laws.append(np.asarray(bank["laws"][c], float))
        kerns.append(KL.kern_of(bank, c, 0))
        adv.append(None)
        for k, (cl, th) in enumerate(steps[c] or []):
            adv.append(cl)
            laws.append(np.asarray(th, float))
            kerns.append(KL.kern_of(bank, c, k + 1))
        n_steps.append(1 + len(steps[c] or []))
    law_start.append(len(laws))
    laws.append(np.asarray(bank["default"], float))
    kerns.append(KL.kern_of(bank, -1, 0))
    if not adv:
        adv = [None]
    a_col, a_thr, a_neg, a_start, a_len = _lits(adv)

    m = bank.get("mem")
    w_col, w_thr, w_neg, _, _ = _lits([(m or {}).get("write") or []])
    kp = KL.pack(kerns, laws[0].shape[1], KL.bounds_of(bank))
    kp["has_kern"] = any(KL.n_points(k) for k in kerns)
    return dict(
        **kp,
        lit_col=lit_col, lit_thr=lit_thr, lit_neg=lit_neg, cl_start=cl_start,
        cl_len=cl_len,
        b_col=b_col, b_thr=b_thr, b_neg=b_neg, b_start=b_start, b_len=b_len,
        f_col=f_col, f_thr=f_thr, f_neg=f_neg, f_start=f_start, f_len=f_len,
        a_col=a_col, a_thr=a_thr, a_neg=a_neg, a_start=a_start, a_len=a_len,
        law_start=np.array(law_start, np.int64),
        n_steps=np.array(n_steps, np.int64),
        sticky=np.array(bank.get("sticky") or [False] * C, np.bool_),
        mem_cols=np.array((m or {}).get("cols", []), np.int64),
        w_col=w_col, w_thr=w_thr, w_neg=w_neg,
        laws=np.ascontiguousarray(np.stack(laws)),
        # n_obs carries the countdown flag in its sign-free high bit so the
        # kernels' signatures stay unchanged: n_obs + COUNTDOWN_FLAG when the
        # bank's blackboard exposes `left_<c>` columns
        n_obs=np.int64(n_obs + (COUNTDOWN_FLAG if (m or {}).get("countdown")
                                else 0)))


def no_trace():
    """Placeholder for a kernel's `trace` argument when no trace is wanted.

    ZERO episodes long, so it can never be mistaken for a real trace: the
    kernels test `trace.shape[0] == n`, and a placeholder of length 1 was a
    real trace for a one-episode batch -- whose column writes then ran past a
    (1, 1, 1) buffer, which compiled code does not check.
    """
    return np.zeros((0, 1, 1))


def no_dev():
    """Placeholder for a kernel's `dev` argument: no episode deviates."""
    return np.zeros((0, 4))


TRACE_EXTRA = 5     # per tick, after z: law index, action (or ux), uy, latch, step


def trace_array(n, T, d):
    """A trace buffer for `n` episodes of `T` ticks on a `d`-wide layout."""
    return np.zeros((n, T, d + TRACE_EXTRA))


COUNTDOWN_FLAG = 1 << 20


TICK_KEYS = ("lit_col", "lit_thr", "lit_neg", "cl_start", "cl_len",
             "b_col", "b_thr", "b_neg", "b_start", "b_len",
             "f_col", "f_thr", "f_neg", "f_start", "f_len",
             "a_col", "a_thr", "a_neg", "a_start", "a_len",
             "law_start", "n_steps", "sticky")
WORLD_KEYS = ("laws", "mem_cols", "w_col", "w_thr", "w_neg", "n_obs")


def tick_args(f):
    """The arrays `_tick` takes, in its argument order, for `rollout(..., *tick_args(f))`."""
    return tuple(f[k] for k in TICK_KEYS)


def world_args(f):
    """The arrays a kernel needs besides the tick: laws, blackboard, layout."""
    return tuple(f[k] for k in WORLD_KEYS)


@njit(cache=True, inline="always")
def _fires(z, col, thr, neg, start, ln):
    for k in range(start, start + ln):
        above = z[col[k]] > thr[k]
        if neg[k]:
            above = not above
        if not above:
            return False
    return True


@njit(cache=True, inline="always")
def _tick(z, latch, step,
          lit_col, lit_thr, lit_neg, cl_start, cl_len,
          b_col, b_thr, b_neg, b_start, b_len,
          f_col, f_thr, f_neg, f_start, f_len,
          a_col, a_thr, a_neg, a_start, a_len,
          law_start, n_steps, sticky):
    """One tick of the tree on one row. Returns (law index, latch, step).

    `latch` is the running arm or -1; `step` its step index. The law index is
    into the flat `laws` array, so the caller never re-derives (arm, step).
    """
    C = cl_start.shape[0]
    excl = -1
    if latch >= 0 and f_len[latch] > 0 and _fires(z, f_col, f_thr, f_neg,
                                                  f_start[latch], f_len[latch]):
        excl = latch
        latch = -1
        step = 0
    f = C
    for c in range(C):
        if c == excl:
            continue
        if _fires(z, lit_col, lit_thr, lit_neg, cl_start[c], cl_len[c]):
            f = c
            break
    arm = f
    k_new = 0
    if latch >= 0 and f >= latch:
        fired = False
        if b_len[latch] > 0:
            fired = _fires(z, b_col, b_thr, b_neg, b_start[latch], b_len[latch])
        if not fired:
            arm = latch
            k_new = step
            if step < n_steps[latch] - 1:
                nxt = law_start[latch] + step + 1
                if _fires(z, a_col, a_thr, a_neg, a_start[nxt], a_len[nxt]):
                    k_new = step + 1
    if arm < C and sticky[arm]:
        latch_out = arm
        step_out = k_new
    else:
        latch_out = -1
        step_out = 0
    law = law_start[arm] + k_new
    return law, latch_out, step_out
