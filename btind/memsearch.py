"""Discovering what to remember, and when to write it down.

THE SEARCH SPACE IS (what, when), AND THE SECOND HALF IS THE DANGEROUS ONE.
Measured on the masked NestWorld, with the SAME stored columns and the same
tree, only the write event differing:

    memory never written                      G  2.47
    write on `food_seen`                      G -0.07     worse than no memory
    write on `food_seen AND d_food <= 0.12`   G  9.68
    hand-written ceiling                      G 11.25

Latching at the moment of sighting stores a point up to `vision_r` = 0.30 away
from the thing being remembered, so the agent walks confidently back to the
wrong place. Fixing the event at "when you can see it" -- the obvious choice --
would have produced the conclusion that memory hurts.

NOTHING IS NAMED BY HAND. Every observation column is a candidate to store,
including the four planted distractors, and every write event is a literal drawn
from the same vocabulary the guards use. The task word "food" appears nowhere in
the search.

PROPOSE BY INFORMATION GAP, SELECT BY RETURN. The candidate space is thousands
of (columns, event) pairs and a rollout costs a third of a second, so it has to
be pruned by something cheap. The cheap thing is an information gap against a
PRIVILEGED reference: the planner sees the true state, so whatever it does that
an observation-only law cannot predict is exactly what the observation is
missing. Regress the planner's action on `z`, then on `z` plus a candidate's
latched columns; the rise in fit is what that candidate recovers.

ONE ROLLOUT SERVES EVERY CANDIDATE. The latched value of any (columns, event)
pair is a running "last value where the event fired", so a single recorded
trajectory answers the question for all of them offline. The expensive part is
labelling the recorded states with the planner, once.
"""
import time

import numpy as np

from .collect import design_matrix
from .landscape import _match_cols
from .structure import accept, score

_EPS = 1e-12


# --------------------------------------------------------------- recording
def record(env, policy, rng, n_ep=400, T=400):
    """Trajectories of states and observations under the current controller."""
    s = (env.sample_starts(n_ep, rng) if hasattr(env, "sample_starts")
         else env.sample_states(n_ep, rng))
    if hasattr(policy, "reset"):
        policy.reset(n_ep)
    d = env.observe(s).shape[1]
    OB = np.zeros((n_ep, T, d))
    ST = np.zeros((n_ep, T, s.shape[1]))
    AL = np.zeros((n_ep, T), bool)
    alive = np.ones(n_ep, bool)
    for t in range(T):
        AL[:, t] = alive
        ST[:, t] = s
        o = env.observe(s)
        OB[:, t] = o
        s, _, done = env.step(s, policy.act(o))
        alive &= ~done
        if not alive.any():
            break
    return OB, ST, AL


def latched(OB, cols, write, zb=None):
    """Replay a write rule over recorded observations, offline.

    Returns (values (n,T,k), have (n,T)) as they WOULD have been had the rule
    been in force -- which is not the same as having run the controller with it,
    since the trajectory would have differed. That is exactly why this ranks
    candidates rather than deciding them.
    """
    n, T, _ = OB.shape
    Z = OB if zb is None else zb
    fire = np.stack([_match_cols(write, Z[:, t]) for t in range(T)], 1)
    vals = OB[:, :, cols]
    out = np.zeros_like(vals)
    have = np.zeros((n, T), bool)
    cur = np.zeros((n, len(cols)))
    hv = np.zeros(n, bool)
    for t in range(T):
        f = fire[:, t]
        cur[f] = vals[f, t]
        hv |= f
        out[:, t] = cur
        have[:, t] = hv
    return out, have


# ------------------------------------------------------------------ ranking
def _r2(X, Y, tr, te):
    """HELD-OUT multi-output R^2: fit on `tr`, score on `te`.

    In-sample R^2 cannot rank these candidates, because every candidate ADDS
    columns and in-sample fit never falls. Measured: `noise` is constant within
    an episode on this world, so latching it is a pure duplicate carrying no
    information -- and it still took rank 1 of 3774 on in-sample gain, paired
    with a real column, purely as regression capacity. Held-out scoring prices
    capacity at zero, which is what the distractor control is for.
    """
    coef, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
    res = ((Y[te] - X[te] @ coef) ** 2).sum()
    tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum()
    return float(1.0 - res / max(tot, _EPS))


def rank(OB, AL, U_star, cands, w=None, max_rows=40000, rng=None):
    """Information gain of each (cols, write) candidate over the plain features.

    The baseline is the observation itself, so a candidate is credited only with
    what it adds -- a latched column that merely duplicates a live one scores
    zero, which is the behaviour that keeps the ranking honest.
    """
    rng = rng or np.random.default_rng(0)
    live = AL.reshape(-1)
    idx = np.flatnonzero(live)
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    O = OB.reshape(-1, OB.shape[2])[idx]
    Y = U_star[idx]
    base = design_matrix(O)
    half = rng.random(len(idx)) < 0.5
    tr, te = np.flatnonzero(half), np.flatnonzero(~half)
    r0 = _r2(base, Y, tr, te)
    out = []
    for cols, write, label in cands:
        v, h = latched(OB, list(cols), write)
        V = v.reshape(-1, len(cols))[idx]
        H = h.reshape(-1)[idx].astype(float)[:, None]
        gain = _r2(np.hstack([base, V, H]), Y, tr, te) - r0
        out.append(dict(cols=list(cols), write=write, label=label,
                        gain=float(gain), wrote=float(h.reshape(-1)[idx].mean())))
    return sorted(out, key=lambda d: -d["gain"]), r0


def candidates(names, OB, AL, n_thr=9, pair_adjacent=True, max_write=None):
    """Every column as a store; every literal as a write event.

    Pairs are formed only between ADJACENT columns. That is a structural prior
    about the observation layout, not about the task: a world that reports a
    vector reports its components side by side. Singletons cover the rest.
    """
    live = AL.reshape(-1)
    O = OB.reshape(-1, OB.shape[2])[live]
    d = O.shape[1]
    # THE GRID HAS TO REACH THE TAILS. Measured: the write event worth +9.7
    # return is `d_food <= 0.12`, which sits at quantile 0.13 of that column --
    # outside a 0.2-0.8 grid, so the winning candidate was not in the pool at
    # all. Masked columns make this worse: `d_food` is pinned at its 1.5
    # sentinel for 63% of steps, so the upper half of any quantile grid is one
    # repeated value and the informative range lives entirely in the tail.
    qs = np.linspace(0.05, 0.95, n_thr)
    writes = []
    for j in range(d):
        col = O[:, j]
        for thr in np.unique(np.quantile(col, qs)):
            for neg in (False, True):
                frac = float((col <= thr).mean() if neg else (col > thr).mean())
                if 0.03 < frac < 0.97:
                    writes.append(([[j, float(thr), bool(neg)]],
                                   "%s%s%.3f" % (names[j], "<=" if neg else ">",
                                                 thr)))
    if max_write:
        writes = writes[:max_write]
    stores = [((j,), names[j]) for j in range(d)]
    if pair_adjacent:
        stores += [((j, j + 1), "%s+%s" % (names[j], names[j + 1]))
                   for j in range(d - 1)]
    return [(c, w, "%s @ %s" % (cn, wn)) for c, cn in stores for w, wn in writes]


# ------------------------------------------------------------------- search
def search_memory(env, bank, names, pol_fn, n_obs, U_star, OB, AL,
                  n_try=24, n_ep=600, T=400, seed=777, z=2.0, verbose=True,
                  rng=None):
    """Rank offline, then rollout the best few. The rollout decides.

    A candidate is installed as a blackboard rule and NOTHING ELSE changes --
    no arm is added to use it. That is deliberate for the first pass: it asks
    whether merely HAVING the memory helps, which is a weaker question than the
    one we care about, so a null result here is not yet evidence against memory.
    """
    cands = candidates(names, OB, AL)
    ranked, r0 = rank(OB, AL, U_star, cands, rng=rng)
    if verbose:
        print("    baseline R2(planner action | obs) = %.3f, %d candidates"
              % (r0, len(cands)))
        for c in ranked[:5]:
            print("      %-44s gain %+.4f  (writes on %.0f%% of steps)"
                  % (c["label"], c["gain"], 100 * c["wrote"]))
    cur = score(env, bank, pol_fn, n_ep, T, seed)
    best, best_d, log = None, 0.0, []
    for c in ranked[:n_try]:
        cand = dict(bank, mem=dict(cols=c["cols"], write=c["write"],
                                   clear=None))
        ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed, z)
        log.append(dict(label=c["label"], gain=c["gain"], delta=d,
                        accepted=bool(ok)))
        if ok and d > best_d:
            best, best_d = cand, d
    if verbose:
        print("    rollout: %d/%d cleared the test%s"
              % (sum(l["accepted"] for l in log), len(log),
                 ("; best %s %+.2f" % (max(log, key=lambda l: l["delta"])["label"],
                                       best_d)) if best else ""))
    return (best or bank), log, ranked


# ------------------------------------------------- joint discovery (no proxy)
def discover(env, bank, names, pol_fn_for, n_obs, OB, AL, n_thr=7,
             screen_ep=120, confirm_ep=600, n_confirm=25, T=400, seed=777,
             z=2.0, verbose=True):
    """Search (what to store, when to write) jointly with an arm that uses it.

    TWO THINGS THIS LEARNED THE HARD WAY.

    A MEMORY RULE ALONE IS A NO-OP. Nothing reads the columns, so every rollout
    ties and nothing is ever accepted. The unit of search has to be the rule
    PLUS an arm guarded on `have_mem` whose law walks to the latched point.

    AND THE RANKING PROXY DOES NOT WORK. Scoring candidates by how much a
    latched column improves prediction of the privileged planner's action put
    the known-good store at rank 6 of 8214, tied with `noise`, with 5818
    candidates scoring positive -- no discrimination at all. A latched value
    correlated with hidden state predicts the planner better whether or not it
    helps the controller. What replaced it is not a better proxy but a cheaper
    version of the real thing: a short rollout screens the grid, a long one
    confirms the survivors.
    """
    from .lawsearch import memory_primitives
    from .memory import mem_names, widen

    zn0 = mem_names(names, None)
    cands = [c for c in candidates(names, OB, AL, n_thr=n_thr) if len(c[0]) == 2]
    if verbose:
        print("    memory grid: %d joint candidates" % len(cands), flush=True)

    def with_mem(mem):
        zn = mem_names(names, mem)
        b = widen(bank, zn0, zn)
        idx = {n: i for i, n in enumerate(zn)}
        prim = memory_primitives(zn, mem, names)
        b = dict(b, mem=mem,
                 clauses=[[l[:] for l in c] for c in bank["clauses"]]
                         + [[[idx["have_mem"], 0.5, False]]],
                 laws=list(b["laws"]) + [prim["to_mem"]])
        return b, zn

    base_cheap = score(env, bank, pol_fn_for(zn0), screen_ep, T, seed)
    t0, rows = time.time(), []
    for i, (cols, write, label) in enumerate(cands):
        mem = dict(cols=list(cols), write=write, clear=None)
        b, zn = with_mem(mem)
        g = score(env, b, pol_fn_for(zn), screen_ep, T, seed)
        rows.append((float((g - base_cheap).mean()), mem, label))
        if verbose and (i + 1) % 600 == 0:
            print("      screened %d/%d  best %+.2f  [%.0fs]"
                  % (i + 1, len(cands), max(r[0] for r in rows),
                     time.time() - t0), flush=True)
    rows.sort(key=lambda r: -r[0])

    base_full = score(env, bank, pol_fn_for(zn0), confirm_ep, T, seed)
    best, best_d, log = None, 0.0, []
    for d_screen, mem, label in rows[:n_confirm]:
        b, zn = with_mem(mem)
        ok, dl, _ = accept(env, b, pol_fn_for(zn), base_full, confirm_ep, T,
                           seed, z)
        log.append(dict(label=label, screen=d_screen, delta=dl,
                        accepted=bool(ok)))
        if ok and dl > best_d:
            best, best_d, best_label = b, dl, label
    if verbose:
        print("    confirmed %d/%d; %s"
              % (sum(l["accepted"] for l in log), len(log),
                 ("winner %s %+.2f" % (best_label, best_d)) if best
                  else "nothing cleared the test"), flush=True)
    return (best or bank), log
