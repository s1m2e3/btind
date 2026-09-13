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
    pol = pol_fn(bank)
    rng = np.random.default_rng(seed)
    env.seed_kernels(seed)
    s = (env.sample_starts(n_ep, rng) if hasattr(env, "sample_starts")
         else env.sample_states(n_ep, rng))
    if hasattr(pol, "reset"):
        pol.reset(n_ep)
    alive = np.ones(n_ep, bool)
    hist, al = [], []
    for t in range(T):
        o = env.observe(s)
        Z = pol.z(o) if hasattr(pol, "z") else o
        a = (pol.arbitrate(Z) if hasattr(pol, "arbitrate")
             else np.zeros(len(o), int))
        hist.append(a.copy())
        al.append(alive.copy())
        s, _, done = env.step(s, pol.act(o))
        alive &= ~done
        if not alive.any():
            break
    A = np.array(hist).T
    AL = np.array(al).T
    out = {}
    for c in range(len(bank["clauses"])):
        runs, reent = [], 0
        for i in range(n_ep):
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
        out[c] = dict(dwell=float(np.mean(runs)) if runs else 0.0,
                      reentries=reent / max(n_ep, 1),
                      share=float((A[AL] == c).mean()))
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
    for j in order:
        col = Z[:, j]
        qs = (np.linspace(0.05, 0.95, n_thr * 2) if j in own
              else np.linspace(0.15, 0.85, n_thr))
        for thr in np.unique(np.quantile(col, qs)):
            for neg in (False, True):
                frac = float((col <= thr).mean() if neg else (col > thr).mean())
                if 0.02 < frac < 0.98:
                    out.append(([[int(j), float(thr), bool(neg)]],
                                "%s%s%.3f" % (names[j], "<=" if neg else ">",
                                              thr)))
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
    order = (arms if arms is not None else
             sorted(range(C), key=lambda c: -ch[c]["reentries"]))
    cur = (score(env, bank, pol_fn, n_ep, T, seed) if cur_G is None else cur_G)
    log = []
    for c in order:
        if ch[c]["share"] < 0.02:
            continue
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
            print("    arm %d (re-entries %.1f/ep, dwell %.1f): %s"
                  % (c, ch[c]["reentries"], ch[c]["dwell"],
                     ("sticky, beta %s  %+.2f"
                      % (max((l for l in log if l["arm"] == c and l["accepted"]),
                             key=lambda l: l["delta"])["beta"], best_d))
                     if best else "nothing beat staying reactive"), flush=True)
        if best is not None:
            return best, log, ch
    return bank, log, ch
