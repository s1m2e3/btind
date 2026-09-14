"""Grow a tree arm by arm, by measured return. One grower for both worlds.

This is `rsfi`'s machinery, kept intact and freed of the two things that are
only valid when the expert can see what the controller sees.

WHAT IS KEPT, because every good result in this project rests on it:

    CONJUNCTIONS      candidate guards have arity 1..max_arity, not one literal.
                      "not carrying AND cannot see food" is a region; neither
                      half of it is.
    DRIFT             the accepted arm is re-proposed with a jittered threshold,
                      so a cut that is nearly right gets refined rather than
                      having to be re-found.
    SEED CLAUSES      informed proposals (an evolved bank, a previous run) enter
                      the same pool as the random ones and win or lose the same
                      rollout.
    min_n             a region must hold `min_n` rows to be considered at all.
                      Dropping this is what let `bear_food_x <= -0.986` -- about
                      1% of states, a sliver with no meaning -- be bought for
                      +0.16, three times over.
    claimed           a new arm is matched only against rows no earlier arm took,
                      so arms are proposed against the rows they would actually
                      receive.
    A LAW PER REGION  the question is "is this region worth having GIVEN THE BEST
                      LAW FOR IT", which is not the same as "given one of six
                      generic directions". Asking the weaker question makes real
                      regions look worthless and noise slivers look comparable.

WHAT CHANGES, and only this. `rsfi` fits that per-region law with
`fit_value_law` on the PLANNER's labels. On a masked world the planner is
privileged -- it plans from the true state, including food the controller cannot
see -- so no observable law can reproduce its labels and the fit is noise. Here
the law for a candidate region can come from any of three sources, and they are
all offered to the same rollout:

    library   an affine primitive: toward, away or tangent to a bearing pair
    fitted    `fit_value_law` on the region's rows        (needs labels)
    gradient  the parent's law plus a step along dQ/du    (needs a critic)

On a fully observed world all three are available and the rollout picks. On a
masked one the labels are dropped and the other two carry it, which is the
adaptation rather than an amputation.

AND AN EFFECT-SIZE FLOOR. A deterministic simulator gives the paired test a zero
noise floor, so an arm worth +0.11 return is STATISTICALLY CERTAIN and
practically nothing. Significance stopped being a filter the moment the noise
went away; `min_gain` is what replaces it.
"""
import time

import numpy as np

from .collect import design_matrix
from .evotm import Alphabet, _match, _rand_literal, dedupe_literals
from .lawsearch import library
from .structure import accept, score
from .valuesplit import fit_value_law


def _hot_rows(bank, obs, Z, qhat, frac=0.25):
    """Rows where the controller is leaving the most value on the table.

    `evotm` anchors newborn literals on the worst-fit quartile of a regression
    residual; with a critic the on-policy version of that is the size of
    dQ/du at the action the controller is taking -- how much return the current
    law is giving up right here. No planner, no labels.
    """
    if qhat is None:
        return None
    from .qwire import assign, _law
    Xd = design_matrix(Z if bank.get("laws_on_z") else obs)
    a = assign(bank["clauses"], Z)
    g = np.zeros((len(obs), 2))
    for c in np.unique(a):
        idx = np.flatnonzero(a == c)
        u = np.clip(Xd[idx] @ _law(bank, int(c)), -1, 1)
        g[idx] = qhat.grad_u(obs[idx], u)
    n = np.linalg.norm(g, axis=1)
    return np.flatnonzero(n >= np.quantile(n, 1 - frac))


def _clause_pool(rng, alpha, Z, hot, pool, max_arity, last, seeds):
    """Random conjunctions, drift of the last accepted arm, and any seeds."""
    out = []
    for _ in range(pool):
        k = 1 + int(rng.integers(max_arity))
        out.append(dedupe_literals([_rand_literal(rng, alpha, Z, hot)
                                    for _ in range(k)]))
    if last is not None:
        for _ in range(6):
            cl = [l[:] for l in last]
            i = int(rng.integers(len(cl)))
            j = cl[i][0]
            cl[i][1] = float(np.clip(
                cl[i][1] + rng.normal(0, 0.06 * (alpha.hi[j] - alpha.lo[j])),
                alpha.lo[j], alpha.hi[j]))
            out.append(dedupe_literals(cl))
    return out + [[l[:] for l in c] for c in (seeds or [])]


def _laws_for(bank, region, names, zn, obs, Z, labels, qhat, w, lib, arm_parent,
              etas=(0.05, 0.2)):
    """Every law worth trying on one candidate region, from every source.

    THE FITTED LAW IS THE PRIMARY SOURCE, as it has been since e18: solve the
    M-weighted value loss on exactly the rows the region holds. The library is a
    PRIOR -- a vocabulary of named directions I wrote -- and naming the answer is
    not discovering it, so it is off by default and reported separately when on.
    """
    out = list(lib)
    Xd = design_matrix(Z if bank.get("laws_on_z") else obs)
    if labels is not None:
        U, M = labels
        Mw = M[region] if w is None else (w[region, None, None] * M[region])
        try:
            out.append(("fitted", fit_value_law(Xd[region], U[region], Mw)))
        except np.linalg.LinAlgError:
            pass
    if qhat is not None:
        th = arm_parent
        u = np.clip(Xd[region] @ th, -1, 1)
        ww = np.ones(len(region)) if w is None else w[region]
        gr = (ww[:, None] * Xd[region]).T @ qhat.grad_u(obs[region], u)
        for e in etas:
            out.append(("grad%.2f" % e, th + e * gr / max(ww.sum(), 1e-9)))
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
        g = score(env, dict(bank, default=th), pol_fn, n_ep, T, seed)
        if best_g is None or g.mean() > best_g.mean():
            best, best_g, label = th, g, nm
    if verbose:
        print("    default law: %-22s G %.2f" % (label, best_g.mean()),
              flush=True)
    return dict(bank, default=best), best_g


def grow(env, bank, names, zn, pol_fn, obs, Z, max_arms=6, pool=60,
         max_arity=3, min_n=400, min_gain=0.3, n_law=8, labels=None, qhat=None,
         w=None, seed_clauses=None, screen_ep=120, confirm_ep=600,
         n_confirm=12, T=400, seed=777, z=2.0, rng=None, verbose=True,
         use_library=False):
    """Add arms while a rollout says they pay by more than `min_gain`."""
    rng = rng or np.random.default_rng(0)
    alpha = Alphabet(n_thresholds=9).fit(Z, list(range(Z.shape[1])))
    lib = (library(names, None, zn=zn, mem=bank.get("mem"))
           if use_library else [])

    # NO GLOBAL SHORTLIST. Ranking the library by how each law performs as the
    # DEFAULT -- applied everywhere -- prunes exactly the laws a REGION needs.
    # Measured: `to_food` scores badly globally (ignore the threat, never
    # deliver) and was cut from the shortlist, so the seeking arm the grower
    # correctly found, `carrying<=0` over 6182 rows, had no "go to the food"
    # available and settled for a gradient nudge off `from_threat`. A global
    # criterion deciding what a regional search may consider is the same
    # mistake as every proxy failure here, one level up.
    if verbose:
        print("    law sources: %s%s%s"
              % ("fitted " if labels is not None else "",
                 "gradient " if qhat is not None else "",
                 ("library(%d)" % len(lib)) if lib else ""), flush=True)

    claimed = np.zeros(len(Z), bool)
    last, log, t0 = None, [], time.time()
    for k in range(max_arms):
        cur_cheap = score(env, bank, pol_fn, screen_ep, T, seed)
        cur_full = score(env, bank, pol_fn, confirm_ep, T, seed)
        hot = _hot_rows(bank, obs, Z, qhat)
        if hot is not None:
            hot = hot[~claimed[hot]]
        cands = _clause_pool(rng, alpha, Z, hot, pool, max_arity, last,
                             seed_clauses if k == 0 else None)

        rows = []
        for cl in cands:
            m = _match(cl, Z) & ~claimed
            region = np.flatnonzero(m)
            if len(region) < min_n:
                continue
            parent = bank["default"]
            for lname, th in _laws_for(bank, region, names, zn, obs, Z, labels,
                                       qhat, w, lib, parent):
                g = score(env, _insert(bank, cl, th, 0), pol_fn, screen_ep, T,
                          seed)
                rows.append((float((g - cur_cheap).mean()), cl, th, lname,
                             len(region)))
        if not rows:
            if verbose:
                print("    arm %d: no candidate region held %d rows -- stopping"
                      % (k, min_n), flush=True)
            break
        rows.sort(key=lambda r: -r[0])

        best, best_d, desc = None, min_gain, None
        for d, cl, th, lname, nrow in rows[:n_confirm]:
            for pos in range(len(bank["clauses"]) + 1):
                cand = _insert(bank, cl, th, pos)
                ok, dl, _ = accept(env, cand, pol_fn, cur_full, confirm_ep, T,
                                   seed, z)
                log.append(dict(arm=k, law=lname, pos=pos, screen=d, delta=dl,
                                rows=nrow, accepted=bool(ok and dl > min_gain)))
                if ok and dl > best_d:
                    best, best_d, best_cl = cand, dl, cl
                    desc = "%s -> %s @%d (%d rows)" % (
                        " AND ".join("%s%s%.3f" % (names[j] if j < len(names)
                                                   else zn[j],
                                                   "<=" if n else ">", t)
                                     for j, t, n in cl), lname, pos, nrow)
        if best is None:
            if verbose:
                print("    arm %d: nothing cleared %+.2f -- stopping [%.0fs]"
                      % (k, min_gain, time.time() - t0), flush=True)
            break
        bank, last = best, best_cl
        claimed |= _match(best_cl, Z)
        if verbose:
            print("    arm %d: %-58s %+6.2f  [%.0fs]"
                  % (k, desc, best_d, time.time() - t0), flush=True)
    return bank, log
