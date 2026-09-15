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
from .lawsearch import library, structural_primitives
from .structure import accept, score
from .valuesplit import fit_value_law


def failure_states(env, bank, pol_fn, n_ep=400, T=400, seed=11, lookback=8):
    """Observations from the steps just before the controller dies.

    `evotm` anchored newborn literals on the worst-fit quartile of a regression
    residual, and `_hot_rows` does the on-policy version with a critic. Neither
    is available in pure-RL mode -- but a rollout reports something better than
    a residual: WHERE IT WENT WRONG. The states in the few steps before a catch
    or a starvation are exactly the region a new arm should be carved around.

    This is what the quantile alphabet cannot reach on its own. Coverage states
    put `d_threat` uniformly in [0.08, 0.80], so its 5% quantile is 0.11 and no
    threshold below that is ever proposed -- while the region worth +5.87 is
    `d_threat <= 0.09`. Thresholds taken from failure states sit where the
    trouble is, not where the sampling happens to be dense.
    """
    from .memory import MemBank
    pol = pol_fn(bank)
    rng = np.random.default_rng(seed)
    env.seed_kernels(seed)
    s = (env.sample_starts(n_ep, rng) if hasattr(env, "sample_starts")
         else env.sample_states(n_ep, rng))
    if hasattr(pol, "reset"):
        pol.reset(n_ep)
    alive = np.ones(n_ep, bool)
    hist = []
    out = []
    for t in range(T):
        o = env.observe(s)
        hist.append(o.copy())
        if len(hist) > lookback:
            hist.pop(0)
        s, r, done = env.step(s, pol.act(o))
        died = done & alive & (r < -1.0)          # caught or starved, not a step cost
        if died.any():
            for h in hist:
                out.append(h[died])
        alive &= ~done
        if not alive.any():
            break
    return np.concatenate(out, 0) if out else np.zeros((0, hist[0].shape[1]))


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


def _prune(cl, Z, lo=0.005, hi=0.90):
    """Drop literals that say nothing, and clauses that select nothing.

    `dedupe_literals` merges literals sharing a FEATURE AND DIRECTION, so it
    cannot see that `carrying<=1.000` is vacuous beside `carrying>0.000` -- the
    directions differ. A literal matching 90% of rows adds no region and only
    costs a reader something to hold.

    THE FLOOR IS DELIBERATELY TINY, and this was learned the hard way. A 10%
    floor plus `min_n=400` (6.7% of the coverage rows) excluded `d_threat<=0.09`
    -- 1.3% of states, and the single most valuable region in the task: adding
    that one arm by hand took the tree from 6.84 to 11.60, past the hand-written
    reference. Both filters were added the same day to stop junk slivers worth
    +0.16, and they threw out the region worth +5.87 with them.

    SIZE DOES NOT DISTINGUISH A SLIVER FROM A CRISIS. `bear_food_x<=-0.986` and
    `d_threat<=0.09` are the same size and differ only in value, which the
    rollout already measures and `min_gain` already filters. Cutting on size was
    filtering the one thing we can measure with the one thing we cannot.
    """
    keep = []
    for j, t, n in cl:
        f = float((Z[:, j] <= t).mean() if n else (Z[:, j] > t).mean())
        if lo < f < hi:
            keep.append([int(j), float(t), bool(n)])
    if not keep:
        return None
    m = np.ones(len(Z), bool)
    for j, t, n in keep:
        above = Z[:, j] > t
        m &= (~above if n else above)
    f = m.mean()
    return keep if lo < f < hi else None


def _clause_pool(rng, alpha, Z, hot, pool, max_arity, last, seeds,
                 weights=None, cols=None):
    """Random conjunctions, drift of the last accepted arm, and any seeds.

    Columns are drawn from LEARNED weights when they are supplied: a column that
    has produced accepted arms before is proposed more often, an untried column
    outranks one that has failed repeatedly, and nothing is ever excluded.
    """
    from .proposal import rand_literal
    out = []
    for _ in range(pool):
        # ARITY BIASED LOW. A uniform draw over 1..max_arity spends most of the
        # pool on conjunctions, and a conjunction wins its rollout for one of
        # its literals while the others ride along. Simple guards first.
        k = 1 if rng.random() < 0.6 else 1 + int(rng.integers(max_arity))
        out.append(dedupe_literals([rand_literal(rng, alpha, Z, hot, weights,
                                                 cols) for _ in range(k)]))
    if last is not None:
        for _ in range(6):
            cl = [l[:] for l in last]
            i = int(rng.integers(len(cl)))
            j = cl[i][0]
            cl[i][1] = float(np.clip(
                cl[i][1] + rng.normal(0, 0.06 * (alpha.hi[j] - alpha.lo[j])),
                alpha.lo[j], alpha.hi[j]))
            out.append(dedupe_literals(cl))
    out = out + [[l[:] for l in c] for c in (seeds or [])]
    pruned = [_prune(c, Z) for c in out]
    return [c for c in pruned if c is not None]


def _laws_for(bank, region, names, zn, obs, Z, labels, qhat, w, lib, arm_parent,
              etas=(0.05, 0.2), n_perturb=8, sigma=0.4, rng=None,
              structural=True, n_sample=60):
    """Every law worth trying on one candidate region, from every source.

    THE FITTED LAW IS THE PRIMARY SOURCE, as it has been since e18: solve the
    M-weighted value loss on exactly the rows the region holds. The library is a
    PRIOR -- a vocabulary of named directions I wrote -- and naming the answer is
    not discovering it, so it is off by default and reported separately when on.
    """
    out = list(lib)
    if structural:
        d_law, n_out = np.shape(arm_parent)
        head = bank.get("head", "vector" if n_out == 2 else "argmax")
        if head in ("scalar", "duration"):
            # A CONTINUOUS LEAF: constants across the world's range and
            # proportional terms, from `u_range` the world declares.
            from .lawsearch import scalar_primitives
            lo, hi = bank.get("u_range", (-1.0, 1.0))
            out += list(scalar_primitives(zn, d_law, lo, hi).items())
        elif head == "vector":
            out += list(structural_primitives(zn, d_law).items())
        else:
            # AN ARGMAX HEAD HAS NO GEOMETRY, so the paired "toward/away"
            # vocabulary means nothing on it. `discrete_primitives` offers the
            # constant preferences and single-column scores instead, which name
            # only the action set the environment defines.
            from .lawsearch import discrete_primitives
            out += list(discrete_primitives(zn, n_out, d_law, rng=rng,
                                            n_sample=n_sample).items())
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
    # PERTURBATIONS OF THE PARENT LAW, which need nothing at all. Without a
    # planner, a critic or a library there is no other source, and an arm whose
    # law equals its parent's changes no behaviour, so every rollout ties and no
    # region is ever bought. These are the seeds a per-arm CEM then refines.
    if n_perturb:
        rng = rng or np.random.default_rng(0)
        for i in range(n_perturb):
            out.append(("rand%d" % i,
                        arm_parent + sigma * rng.standard_normal(
                            np.shape(arm_parent))))
    if bank.get("prior") == "const":
        # a constant prior: every candidate is its intercept, and duplicates go
        from .kernlaw import constrain
        seen, kept = set(), []
        for nm, th in out:
            th = constrain(bank, th)
            key = np.round(th, 9).tobytes()
            if key not in seen:
                seen.add(key)
                kept.append((nm, th))
        out = kept
    return out


def _insert(bank, clause, law, pos):
    from .kernlaw import constrain
    from .memory import insert_arm
    return insert_arm(bank, clause, constrain(bank, law), pos)


def _with_prefix(prefix, cands):
    """Every candidate guard conjoined with the parent's literals, verbatim.

    Verbatim matters: the nested tree is recovered by factoring literals the
    children share EXACTLY, so a prefix literal that got merged into a tighter
    one would silently detach the child from its parent. A candidate literal
    identical to a prefix literal is dropped; a candidate that reduces to the
    prefix alone is dropped too, since it would be the parent over again.
    """
    keys = {(int(j), float(t), bool(n)) for j, t, n in prefix}
    out, seen = [], set()
    for cl in cands:
        extra = [l for l in cl if (int(l[0]), float(l[1]), bool(l[2])) not in keys]
        if not extra:
            continue
        full = [list(l) for l in prefix] + [list(l) for l in extra]
        sig = tuple(sorted((int(j), float(t), bool(n)) for j, t, n in full))
        if sig not in seen:
            seen.add(sig)
            out.append(full)
    return out


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
         max_arity=3, min_n=60, min_gain=0.3, n_law=8, labels=None, qhat=None,
         w=None, seed_clauses=None, screen_ep=120, confirm_ep=600,
         n_confirm=12, T=400, seed=777, z=2.0, rng=None, verbose=True,
         use_library=False, cem_region=True, cem_top=10, cem_iter=3,
         cem_K=24, cem_sigma=0.4, cols=None, structural=True, weights=None,
         prefix=None, parent_law=None, where=None, law_sample=60,
         n_perturb=8):
    """Add arms while a rollout says they pay by more than `min_gain`.

    REGION HOOKS, for growing INSIDE an existing child (`subtree.py`):
        prefix      literals conjoined onto every candidate guard -- the parent
                    child's own guard, kept verbatim so the new arm factors
                    into a nested subtree under it
        parent_law  the law new arms start from, instead of the root default
        where       callable(bank) -> (screen_pos, confirm_positions), so
                    candidates are screened and placed inside the parent's
                    block rather than at the top of the root Fallback
    With all three None the grower is exactly what it was.
    """
    rng = rng or np.random.default_rng(0)
    alpha = Alphabet(n_thresholds=9).fit(
        Z, list(range(Z.shape[1])) if cols is None else list(cols))
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
        print("    law sources: %s%s%s%s"
              % ("structural " if structural else "", "", "", "") + "%s%s%sparent-perturbations"
              % ("fitted " if labels is not None else "",
                 "gradient " if qhat is not None else "",
                 ("library(%d) " % len(lib)) if lib else ""), flush=True)

    claimed = np.zeros(len(Z), bool)
    last, log, t0 = None, [], time.time()
    for k in range(max_arms):
        cur_cheap = score(env, bank, pol_fn, screen_ep, T, seed)
        # CONFIRM ON FRESH EPISODES: the best of the screen carries the luck of
        # the screening episodes, so it is priced on a different seed
        seed_c = seed + 7919
        cur_full = score(env, bank, pol_fn, confirm_ep, T, seed_c)
        hot = _hot_rows(bank, obs, Z, qhat)
        if hot is not None:
            hot = hot[~claimed[hot]]
        cands = _clause_pool(rng, alpha, Z, hot, pool, max_arity, last,
                             seed_clauses if k == 0 else None, weights, cols)
        if prefix:
            cands = _with_prefix(prefix, cands)
        screen_pos, positions = (where(bank) if where is not None else
                                 (0, list(range(len(bank["clauses"]) + 1))))

        rows = []
        for cl in cands:
            m = _match(cl, Z) & ~claimed
            region = np.flatnonzero(m)
            if len(region) < min_n:
                continue
            parent = bank["default"] if parent_law is None else parent_law
            for lname, th in _laws_for(bank, region, names, zn, obs, Z, labels,
                                       qhat, w, lib, parent, rng=rng,
                                       structural=structural,
                                       n_sample=law_sample,
                                       n_perturb=n_perturb):
                g = score(env, _insert(bank, cl, th, screen_pos), pol_fn,
                          screen_ep, T, seed)
                rows.append((float((g - cur_cheap).mean()), cl, th, lname,
                             len(region)))
        if not rows:
            if verbose:
                print("    arm %d: no candidate region held %d rows -- stopping"
                      % (k, min_n), flush=True)
            break
        rows.sort(key=lambda r: -r[0])
        # NEGATIVE EVIDENCE COMES FROM THE SCREEN, not the shortlist. Recording
        # only the candidates that reach the confirm stage teaches the proposal
        # weights about winners and nothing about losers -- measured, 525
        # recorded tries left every top-weighted column at "accepted 0 / 0",
        # because the columns that keep failing were never written down.
        for d, cl, th, lname, nrow in rows[n_confirm:]:
            log.append(dict(arm=k, law=lname, screen=d, rows=nrow,
                            clause=[l[:] for l in cl], stage="screen",
                            accepted=False))

        # THE BEST LAW FOR THE REGION, found the way everything else here is
        # found. `rsfi` answered "is this region worth having, GIVEN the best
        # law for it" with a least-squares fit to planner labels. Without a
        # planner the same question is answered by cross-entropy search on the
        # law parameters, scored by rollout -- 72 rollouts per region, which at
        # 1.2ms is under a tenth of a second. A random perturbation of the
        # parent law cannot answer it: in 44 dimensions it almost never lands on
        # the direction a region actually wants, so real regions score like
        # noise and the grower buys neither.
        if cem_region:
            from .lawcem import cem_law
            tuned = []
            for d, cl, th, lname, nrow in rows[:cem_top]:
                probe = _insert(bank, cl, th, screen_pos)
                th2, _ = cem_law(env, probe, screen_pos, pol_fn, n_iter=cem_iter,
                                 K=cem_K, sigma0=cem_sigma, n_ep=screen_ep,
                                 T=T, seed=seed, rng=rng)
                g = score(env, _insert(bank, cl, th2, screen_pos), pol_fn,
                          screen_ep, T, seed)
                tuned.append((float((g - cur_cheap).mean()), cl, th2,
                              lname + "+cem", nrow))
            rows = sorted(tuned + rows, key=lambda r: -r[0])

        best, best_d, desc = None, min_gain, None
        for d, cl, th, lname, nrow in rows[:n_confirm]:
            for pos in positions:
                cand = _insert(bank, cl, th, pos)
                ok, dl, _ = accept(env, cand, pol_fn, cur_full, confirm_ep, T,
                                   seed_c, z)
                log.append(dict(arm=k, law=lname, pos=pos, screen=d, delta=dl,
                                rows=nrow, clause=[l[:] for l in cl],
                                accepted=bool(ok and dl > min_gain)))
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
