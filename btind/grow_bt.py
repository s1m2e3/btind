"""Grow a tree from nothing, by rollout. No planner, no labels, no proxy.

WHY THIS REPLACES IMITATION ON A MASKED WORLD. `rsfi` buys guards by fitting to
the CEM planner's labels, and the planner is PRIVILEGED: it plans from the true
state, including the food the controller cannot see. Under masking no observable
guard can explain those labels, so the search grabs whatever correlates -- and
it does. Measured on the masked NestWorld, imitation emitted

    Sequence[ t_norm<=0.618 AND is_night<=0.000 AND carrying<=0.000 , Action ]

one arm, matching 100% of states, containing a planted distractor, scoring
-7.56. A partition that claims everything is not a partition, and no amount of
downstream refinement recovered it: six scheduled rounds got it to 1.71 against
a hand-written reactive 2.47. The starting structure dominates everything.

WHAT REPLACES IT. An empty bank, and arms added one at a time, each one a pair

    guard   a literal from the same vocabulary every other search uses
    law     an affine primitive from the library: toward, away, or tangent to
            any bearing pair the observation exposes

accepted only when the paired rollout says return rose. Nothing is fitted to
anything. The planner is never called, which on a partially observed world is
not a saving but a correctness argument: there is no teacher whose answers the
student could follow.

PROPOSE CHEAPLY, SELECT EXPENSIVELY, and here the cheap thing is the SAME
objective at lower resolution -- a 120-episode rollout screens the grid, a
600-episode paired test confirms the survivors. Six proxies have failed in this
project and the seventh failed last week; a short rollout is the one proposal
distribution that has never mis-ranked.

POSITION IS SEARCHED, NOT ASSUMED. A Fallback is priority-ordered, so where an
arm is inserted changes which rows it claims. Appending blindly is how the
memory search came to test 3132 candidates that were all behaviourally identical
to the incumbent: every one sat below an arm that already matched everything.
"""
import time

import numpy as np

from .lawsearch import library
from .structure import accept, score


def guard_pool(Z, names, cols=None, n_thr=7, lo=0.03, hi=0.97):
    """Every literal the alphabet admits, at quantiles that reach the tails.

    The tails matter: on a masked world a distance column is pinned at its
    sentinel for most steps, so the informative range is entirely in the tail
    and a 0.2-0.8 grid contains none of it.
    """
    out = []
    cols = range(Z.shape[1]) if cols is None else cols
    for j in cols:
        col = Z[:, j]
        for thr in np.unique(np.quantile(col, np.linspace(0.05, 0.95, n_thr))):
            for neg in (False, True):
                frac = float((col <= thr).mean() if neg else (col > thr).mean())
                if lo < frac < hi:
                    out.append(([[int(j), float(thr), bool(neg)]],
                                "%s%s%.3f" % (names[j], "<=" if neg else ">",
                                              thr)))
    return out


def _insert(bank, clause, law, pos):
    cl = [[l[:] for l in c] for c in bank["clauses"]]
    laws = list(bank["laws"])
    cl.insert(pos, [l[:] for l in clause])
    laws.insert(pos, law)
    return dict(bank, clauses=cl, laws=laws)


def best_default(env, bank, names, zn, pol_fn, n_ep, T, seed, verbose=True):
    """The law the tree falls back on, chosen by rollout before any arm exists."""
    best, best_g, label = bank["default"], None, "incumbent"
    for nm, th in library(names, bank["default"], zn=zn, mem=bank.get("mem")):
        cand = dict(bank, default=th)
        g = score(env, cand, pol_fn, n_ep, T, seed)
        if best_g is None or g.mean() > best_g.mean():
            best, best_g, label = th, g, nm
    if verbose:
        print("    default law: %-16s G %.2f" % (label, best_g.mean()),
              flush=True)
    return dict(bank, default=best), best_g


def grow(env, bank, names, zn, pol_fn, Z, max_arms=6, n_law=6, screen_ep=120,
         confirm_ep=600, n_confirm=20, T=400, seed=777, z=2.0, cols=None,
         verbose=True):
    """Add arms greedily while a rollout says they pay. Returns (bank, log)."""
    lib = library(names, None, zn=zn, mem=bank.get("mem"))
    log, t0 = [], time.time()

    # Which laws are worth pairing with a guard? Score each as the DEFAULT
    # first -- a law that is useless everywhere is unlikely to be the thing a
    # region wants, and this prunes the grid by more than half for 14 rollouts.
    ranked = []
    for nm, th in lib:
        g = score(env, dict(bank, default=th), pol_fn, screen_ep, T, seed)
        ranked.append((float(g.mean()), nm, th))
    ranked.sort(reverse=True)
    laws = [(nm, th) for _, nm, th in ranked[:n_law]]
    if verbose:
        print("    law shortlist: %s" % ", ".join(nm for nm, _ in laws),
              flush=True)

    guards = guard_pool(Z, names, cols=cols)
    for k in range(max_arms):
        cur_cheap = score(env, bank, pol_fn, screen_ep, T, seed)
        cur_full = score(env, bank, pol_fn, confirm_ep, T, seed)
        rows = []
        for gcl, glabel in guards:
            for lname, th in laws:
                cand = _insert(bank, gcl, th, 0)
                g = score(env, cand, pol_fn, screen_ep, T, seed)
                rows.append((float((g - cur_cheap).mean()), gcl, th, glabel,
                             lname))
        rows.sort(key=lambda r: -r[0])

        best, best_d, best_desc = None, 0.0, None
        for d, gcl, th, glabel, lname in rows[:n_confirm]:
            for pos in range(len(bank["clauses"]) + 1):
                cand = _insert(bank, gcl, th, pos)
                ok, dl, _ = accept(env, cand, pol_fn, cur_full, confirm_ep, T,
                                   seed, z)
                log.append(dict(arm=k, guard=glabel, law=lname, pos=pos,
                                screen=d, delta=dl, accepted=bool(ok)))
                if ok and dl > best_d:
                    best, best_d = cand, dl
                    best_desc = "%s -> %s @%d" % (glabel, lname, pos)
        if best is None:
            if verbose:
                print("    arm %d: nothing cleared the test -- stopping [%.0fs]"
                      % (k, time.time() - t0), flush=True)
            break
        bank = best
        if verbose:
            print("    arm %d: %-44s %+6.2f  [%.0fs]"
                  % (k, best_desc, best_d, time.time() - t0), flush=True)
    return bank, log
