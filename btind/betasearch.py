"""Terminations: which arms should persist, and what should stop them.

An arm today is re-chosen from scratch every tick. Making it STICKY means it
keeps running even after its own guard stops holding, until either a
higher-priority arm preempts it or its termination condition beta fires.

WHERE THE CANDIDATES COME FROM. Not from every arm equally -- from the ones the
controller keeps abandoning and re-entering. `churn` measures that directly: how
many times per episode each arm is left and taken up again, and how short its
dwell is. An arm with long uninterrupted dwell has nothing to gain from
stickiness; an arm that is picked up and dropped every few ticks is either
chattering on a boundary or being interrupted for good reason, and only a
rollout can tell those apart.

WHY beta = NOT(guard) IS THE NULL. Latching with that termination reproduces the
memoryless controller exactly, so it is the identity move, and every other
candidate is measured against a class that already contains the incumbent. That
is what makes the search safe rather than merely optimistic.

THE HYSTERESIS CASE, which is what this is really for. On ForageWorld the
learned tree was `flee <= 0.10`, a band, and `forage > 0.19` -- a Schmitt
trigger built in space because the language had no memory. One sticky arm with
`enter at 0.10, beta: d_threat > 0.18` says the same thing with one fewer branch
and two thresholds that state plainly what the band only implies.
"""
import numpy as np

from .landscape import _match_cols
from .structure import accept, score


def churn(env, bank, pol_fn, n_ep=200, T=400, seed=11):
    """Per-arm switch and dwell statistics under the current controller.

    Returns, for each arm, how often an episode leaves it and comes back, and
    the mean length of an uninterrupted run. Ranking by re-entries puts the
    arms that might want to persist at the front of the search.
    """
    A, AL, row_scale = _arms_on_kernel(env, bank, n_ep, T, seed)
    if A is not None:
        return _churn_stats(bank, A, AL, n_ep, row_scale)
    pol = pol_fn(bank)
    rng = np.random.default_rng(seed)
    env.seed_kernels(seed)
    s = (env.sample_starts(n_ep, rng) if hasattr(env, "sample_starts")
         else env.sample_states(n_ep, rng))
    if hasattr(pol, "reset"):
        pol.reset(n_ep)
    if hasattr(env, "_search_bank"):
        env._search_bank = bank
    alive = np.ones(n_ep, bool)
    hist, al = [], []
    for t in range(T):
        o = env.observe(s)
        Z = pol.z(o) if hasattr(pol, "z") else o
        a = (pol.arbitrate(Z) if hasattr(pol, "arbitrate")
             else np.zeros(len(o), int))
        hist.append(a.copy())
        # ONE ROW PER UNIT, NOT PER EPISODE. A world that runs the tree once
        # per vehicle observes k rows per episode; a row counts only while
        # its unit is active (`row_alive`), and every row of a finished
        # episode is dead.
        rows_alive = np.repeat(alive, len(o) // n_ep)
        if hasattr(env, "row_alive"):
            rows_alive = rows_alive & env.row_alive(s)
        al.append(rows_alive)
        s, _, done = env.step(s, pol.act(o))
        alive &= ~done
        if not alive.any():
            break
    A = np.array(hist).T
    AL = np.array(al).T
    return _churn_stats(bank, A, AL, n_ep)


def _arms_on_kernel(env, bank, n_ep, T, seed, max_slots=64):
    """(arm per row-tick, alive mask) from a fused-kernel trace, or (None, None).

    Churn needs two things a trace already records -- which arm each row took
    each tick, and whether the row was alive -- so stepping the world in Python
    to get them cost the 4500x the Python path runs at, every round. That is
    the stage a signal round appeared to hang in, and beta has never once been
    accepted for the light.
    """
    from .memory import mem_names
    from .structure import fast_rollout, kernel_for, starts
    from .tick import trace_array
    if not bank.get("clauses") or kernel_for(env, bank) is None:
        return None, None, 1.0
    d = len(mem_names(bank["names"], bank.get("mem"))) + 1
    s = starts(env, n_ep, seed)
    # which row the trace follows: the signal is one row an episode; a car is
    # one row per slot, so a sample of slots stands in for all of them
    who = [-1]
    n_all = 1
    if getattr(env, "agent", None) == "vehicle":
        dep = s[:, 4 * env.N:5 * env.N]
        who = [q for q in range(env.N) if np.isfinite(dep[:, q]).mean() > 0.3]
        n_all = len(who)
        if len(who) > max_slots:
            who = sorted(np.random.default_rng(seed).choice(who, max_slots,
                                                            replace=False))
    # EXACT FOR THE SIGNAL, SAMPLED FOR A CAR. The light is one row an episode,
    # so its trace is the whole thing; a car is one row per slot and tracing
    # every slot costs a rollout each, so a sample of them stands in. That makes
    # the car's dwell and share estimates (measured, 0.44 against 0.50 on a thin
    # 24-of-48 sample), which is what churn is for -- it RANKS which arms might
    # want hysteresis, and the paired rollout still decides what is kept.
    # RE-ENTRIES ARE A COUNT OVER ROWS, so a sampled subset of slots undercounts
    # them by exactly the sampling ratio -- 4.7x at 112 slots, 16x at 384 --
    # while dwell (a mean) and share (a ratio) are unbiased. The Python path
    # walked every row, so the sample has to be scaled back up to agree.
    row_scale = float(n_all) / max(len(who), 1) if who else 1.0
    A, AL = [], []
    for q in who:
        dev = np.zeros((len(s), 4))
        dev[:, 3] = q
        tr = trace_array(len(s), T, d)
        if fast_rollout(env, bank, s, T, trace=tr, dev=dev) is None:
            return None, None, 1.0
        A.append(tr[:, :, d].astype(int))
        AL.append(tr[:, :, d - 1] > 0.5)
    return np.concatenate(A, 0), np.concatenate(AL, 0), row_scale


def _churn_stats(bank, A, AL, n_ep, row_scale=1.0):
    n_rows = A.shape[0]
    out = {}
    C = len(bank["clauses"])
    for c in range(C):
        runs, reent, latchable = [], 0, 0
        for i in range(n_rows):
            row = A[i][AL[i]]
            if not len(row):
                continue
            inside = row == c
            d, seen = 0, 0
            for t in range(len(inside)):
                if inside[t]:
                    d += 1
                elif d:
                    runs.append(d)
                    d, seen = 0, seen + 1
            if d:
                runs.append(d)
                seen += 1
            reent += max(seen - 1, 0)
            # WHAT A LATCH CAN ACTUALLY ACT ON. `_tick` gives a running arm up
            # the moment a HIGHER-priority guard fires -- preemption beats
            # stickiness by construction -- so the only departure a beta can
            # prevent is one where this arm's own guard stopped holding and a
            # LOWER-priority arm (or the default) took over. Measured on a real
            # car tree: `is_left` churned 16.6 times an episode, every one of
            # them a preemption, and latching it was bit-identical over 100
            # episodes; `d_stop>85.104` churned 0.0 times and latching it
            # changed all 100. Ranking by re-entries sent the whole stage to
            # the one arm where stickiness was impossible.
            latchable += int(((row[:-1] == c) & (row[1:] > c)).sum())
        out[c] = dict(dwell=float(np.mean(runs)) if runs else 0.0,
                      reentries=reent * row_scale / max(n_ep, 1),
                      latchable=latchable * row_scale / max(n_ep, 1),
                      share=float((A[AL] == c).mean()) if AL.any() else 0.0)
    return out


def beta_candidates(Z, names, arm_clause, n_thr=6, cols=None):
    """Termination literals, drawn from the same vocabulary as any guard.

    The arm's OWN variables are offered first and at finer resolution, because
    the hysteresis case -- leave at a looser threshold than you entered at --
    lives entirely on them. Everything else in the alphabet follows.
    """
    own = [l[0] for l in arm_clause]
    cols = list(range(Z.shape[1])) if cols is None else list(cols)
    order = own + [c for c in cols if c not in own]
    out = []
    from .thresholds import literals
    for j in order:
        out += literals(Z[:, j], j, n_thr=(n_thr * 2 if j in own else n_thr),
                        lo=0.02, hi=0.98, name=names[j])
    return out


def search_beta(env, bank, names, Z, pol_fn, cur_G=None, arms=None, n_try=40,
                n_ep=600, T=400, seed=777, z=2.0, verbose=True):
    """Make one arm sticky, with the termination that wins its rollout.

    One arm per call and the most-abandoned first: stickiness changes which
    states the arms below ever see, so two simultaneous latches are priced
    against each other's stale distribution.
    """
    C = len(bank["clauses"])
    if not C:
        return bank, [], None
    ch = churn(env, bank, pol_fn)
    # BY WHAT A LATCH CAN CHANGE, not by how often the arm is left. See
    # `_churn_stats`: a departure to a higher-priority arm is preemption, which
    # stickiness cannot prevent, so counting it only points the search at arms
    # it cannot help.
    order = (arms if arms is not None else
             sorted(range(C), key=lambda c: -ch[c].get("latchable", 0.0)))
    cur = (score(env, bank, pol_fn, n_ep, T, seed) if cur_G is None else cur_G)
    log = []
    for c in order:
        if ch[c]["share"] < 0.02 or ch[c].get("latchable", 0.0) <= 0.0:
            continue        # nothing for a beta to hold through
        cands = beta_candidates(Z, names, bank["clauses"][c])
        rng = np.random.default_rng(0)
        if len(cands) > n_try:
            cands = [cands[i] for i in
                     rng.choice(len(cands), n_try, replace=False)]
        cands.append((None, "never (preemption only)"))
        best, best_d = None, 0.0
        for bcl, label in cands:
            betas = list(bank.get("betas") or [None] * C)
            st = list(bank.get("sticky") or [False] * C)
            betas[c], st[c] = bcl, True
            cand = dict(bank, betas=betas, sticky=st)
            ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed, z)
            log.append(dict(arm=c, beta=label, delta=d, accepted=bool(ok),
                            reentries=ch[c]["reentries"], dwell=ch[c]["dwell"]))
            if ok and d > best_d:
                best, best_d = cand, d
        if verbose:
            print("    arm %d (latchable %.1f/ep, re-entries %.1f/ep, "
                  "dwell %.1f): %s"
                  % (c, ch[c].get("latchable", 0.0), ch[c]["reentries"],
                     ch[c]["dwell"],
                     ("sticky, beta %s  %+.2f"
                      % (max((l for l in log if l["arm"] == c and l["accepted"]),
                             key=lambda l: l["delta"])["beta"], best_d))
                     if best else "nothing beat staying reactive"), flush=True)
        if best is not None:
            return best, log, ch
    return bank, log, ch
