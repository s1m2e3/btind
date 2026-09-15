r"""Learning where a leaf's inducing points go, and what each one says -- by rollout.

A kernel law (`kernlaw.py`) is a few rules "in state x_i do y_i" on top of the
affine law the leaf already had. This module finds them. No expert and no
gradient; three moves, each priced by the paired test like every other move:

    ADD      a point is proposed where exploration says the leaf is wrong, and
             kept if the tree with it clears `min_gain`
    TUNE     CEM over every point's location, target and the lengthscales
    PRUNE    a point whose removal costs less than `margin` goes

WHERE A POINT IS PROPOSED. `explore.deviations` pairs an episode with the same
episode in which one agent took a different action for k ticks. On this world
that difference is an exact advantage, so a deviation that paid is a triple the
law can be anchored on directly:

    location  x = z[cols] at the deviation tick
    target    discrete: the law's own preferences there, with the action that
              paid lifted above the rest; continuous: the command that paid

A learned critic (`intersection_critic.py`) proposes the same triples from far
more states than a batch of deviations touches; both go through the same test.

WHERE THE TREE FAILED is the third source (`anchors`): the states a few ticks
before a red-light run, a crash or a stuck spell (`coverage_rows`), each offered
as a point with the command at a few levels -- full braking, half, hold, full
acceleration -- or, on a discrete leaf, each action in turn. Measured why: in
4-16 veh/h traffic only 10 of 154 deviations paid in a round, none of them near a
red light, so the point that pays there (+12.3, z 3.5) was never proposed.

CONFIRMATION IS ON FRESH EPISODES. Proposals are screened on one seed and the
best few confirmed on ANOTHER (`seed_c`), with the incumbent rescored there.
Screening and confirming on the same episodes was a winner's curse: the best of
300 candidates on those episodes carries their luck, so accepted gains were
inflated and vanished on the next round's seed -- the accept-then-prune flip
seen at rung 0. The tuner and the prune use the confirmation seed as well.

A JOINT SET ACROSS LAWS (`joint`). The trees are SHARED: the car that stops at a
red and the car that runs into it run the same policy, so a red stop and the
following that makes it safe live in different laws of one tree and pay only
together. Measured at 16-50 veh/h on a tree with neither: a red stop alone cost
-7.9 with crashes 0.02 -> 0.52 an episode. So after the per-law passes, the best
critic-set of each of the busiest laws is applied to the tree AT ONCE and that
tree is screened and confirmed as one candidate.

A KERNEL CAN TAKE ON A COLUMN LATER (`widen_kernel`). Its first point fixes its
columns, and a first point is often a proxy: measured, a red-brake point on
(speed, light) at rung 0 locked the default's kernel to those two, and the red
stop the next rung needed -- on distance to the line -- could not be proposed.
A law with points is therefore also offered proposals on its columns plus one
more from the pool, up to `max_cols`. The lengthscale is shared by every point
of a column, so the existing points cannot simply ignore the new one: each is
REPLICATED at the new column's few values (a light: -1, 0, 1) or at three of
its quantiles, which reproduces the old law almost exactly, and the new point
can then sit at one value of it. Prune trims the replicas the tree does not
need.

WHICH COLUMNS A KERNEL READS. The arm's own guard columns first -- the region is
already defined on them -- then the columns that best separate the deviations
that paid from the ones that did not, on the rows this law owns. A hint can be
given (`hint_cols`, what we are allowed to tell the model); it is offered first
and still has to earn its place through the rollouts. The planted columns are
never excluded, so the audit that catches them in guards catches them here.

LENGTHSCALES start from the columns' own structure over the rows the law owns
(`init_ls`): half the spread of a continuous column, and a quarter of the gap
between neighbouring values of a column that takes only a few (the light is
-1 / 0 / 1). Measured why: at the spread (0.8 for the light) a braking point at
red kept 29% of its weight on green, cars slowed on green and in the box, and
the point lost 38; at a quarter of the gap it gained +10.1.
"""
import time

import numpy as np

from . import explore as EX
from . import kernlaw as KL
from .collect import design_matrix
from .structure import accept, score
from .tick import n_steps_of


def _own_reward(env):
    """Measure inside this block on the agent's own reward, not the objective
    the round is accepting on."""
    from contextlib import nullcontext
    return env.reward_as(None) if hasattr(env, "reward_as") else nullcontext()


def flat_laws(bank):
    """(arm, step) per flat law index, in `tick.flatten` order; default last."""
    out = []
    for c in range(len(bank["clauses"])):
        out += [(c, k) for k in range(n_steps_of(bank, c))]
    return out + [(-1, 0)]


def _theta(bank, c, k):
    from .tick import law_of
    return np.asarray(law_of(bank, c, k), float)


def _guard_cols(bank, c):
    if c < 0:
        return []
    cols = [l[0] for l in bank["clauses"][c]]
    for adv, _ in (bank.get("steps") or [None] * (c + 1))[c] or []:
        cols += [l[0] for l in adv]
    return list(dict.fromkeys(cols))


def init_ls(Zown, cols, few=5):
    """Initial lengthscales: categorical columns sharp, continuous ones broad."""
    out = []
    for j in cols:
        vals = np.unique(Zown[:, j]) if len(Zown) else np.zeros(1)
        if len(vals) <= few:
            gaps = np.diff(vals)
            out.append(0.25 * float(gaps.min()) if len(gaps) else 1.0)
        else:
            out.append(max(0.5 * float(Zown[:, j].std()), 1e-3))
    return np.array(out)


def widen_kernel(kern, col, Zown, few=5):
    """The same kernel reading one more column, its points replicated across
    that column's values so the law is (nearly) unchanged until a point is
    added at one of them."""
    vals = np.unique(Zown[:, col]) if len(Zown) else np.zeros(1)
    if len(vals) > few:
        vals = np.quantile(Zown[:, col], (0.1, 0.5, 0.9))
    ls_new = init_ls(Zown, [col])[0] if len(Zown) else 1.0
    M = len(kern["X"])
    X = np.vstack([np.hstack([kern["X"], np.full((M, 1), v)]) for v in vals])
    Y = np.vstack([kern["Y"]] * len(vals))
    return KL.make(list(kern["cols"]) + [int(col)], X, Y,
                   np.concatenate([kern["ls"], [ls_new]]))


def choose_cols(Z, paid, n_cols, prefer=(), hint=(), n_obs=None):
    """Columns for a new kernel: hints, then guard columns, then separation."""
    n_obs = Z.shape[1] if n_obs is None else n_obs
    sd = Z[:, :n_obs].std(0)
    live = [j for j in range(n_obs) if sd[j] > 1e-9]
    out = [j for j in list(hint) + list(prefer) if j in live]
    out = list(dict.fromkeys(out))[:n_cols]
    if len(out) < n_cols and paid.any() and (~paid).any():
        sep = np.abs(Z[paid][:, live].mean(0) - Z[:, live].mean(0)) / sd[live]
        for i in np.argsort(-sep):
            if len(out) >= n_cols:
                break
            if live[i] not in out:
                out.append(live[i])
    return out


def candidates(ex, bank, c, k, kern, head, n_prop, u_range=None):
    """Point proposals (X_row, Y_row, adv) for law (c, k) from paid deviations."""
    L = flat_laws(bank).index((c, k))
    own = ex["law"].astype(int) == L
    paid = own & (ex["adv"] > 0)
    if not paid.any():
        return []
    idx = np.flatnonzero(paid)
    idx = idx[np.argsort(-ex["adv"][idx])]
    th = _theta(bank, c, k)
    out = []
    for i in idx:
        z = ex["z0"][i:i + 1]
        x = z[0, kern["cols"]]
        # a point that sits on top of one already there says nothing new
        if len(kern["X"]) and (np.abs(kern["X"] - x) / kern["ls"]).sum(1).min() < 0.5:
            continue
        near = [o for o in out
                if (np.abs(o[0] - x) / kern["ls"]).sum() < 0.5 and o[2] != float(ex["adv"][i])]
        if near:
            continue
        u0 = KL.evaluate(th, kern if KL.n_points(kern) else None, design_matrix(z),
                         bounds=KL.bounds_of(bank))[0]
        if head == "argmax":
            a = int(ex["a"][i, 0])
            y = u0.copy()
            y[a] = u0.max() + 1.0 + 0.25 * (u0.max() - u0.min())
            ys = [y]
        else:
            # THE COMMAND THAT PAID IS A DIRECTION, NOT A DOSE. A deviation is
            # drawn from five levels across the range, so the one that paid is
            # often too much: measured, braking at -4.5 on a red near the line
            # lost 3.3 while -2.7 gained 15.5. Points along the way from the
            # law's current command to the paying one are all proposed.
            a = float(ex["a"][i, 0])
            ys = [np.array([u0[0] + f * (a - u0[0])]) for f in (1.0, 0.75, 0.5, 0.25)]
        for y in ys:
            out.append((x, y, float(ex["adv"][i])))
        if len(out) >= n_prop:
            break
    return [(kern, x[None], y[None], a, "dev") for x, y, a in out]


def bound_seeds(ex, bank, c, k, kern, bounds, Zown):
    """For a continuous leaf with no points yet: the leaf's two extreme
    responses as a PAIR of points -- the minimum command where braking hardest
    paid most, the maximum where accelerating hardest did -- and, whatever the
    deviations say, the pair across the spread of the leaf's own columns in
    both orientations. The rollout decides which, if any, the leaf keeps."""
    lo, hi = bounds
    L = flat_laws(bank).index((c, k))
    own = ex["law"].astype(int) == L
    out = []
    lvl = ex["a"][:, 0]
    best = {}
    for name, target in (("lo", lo), ("hi", hi)):
        m = own & (ex["adv"] > 0) & (np.abs(lvl - target) < 1e-6)
        if m.any():
            i = int(np.flatnonzero(m)[np.argmax(ex["adv"][m])])
            best[name] = (ex["z0"][i, kern["cols"]], float(ex["adv"][i]))
    if "lo" in best and "hi" in best:
        out.append((kern, np.vstack([best["lo"][0], best["hi"][0]]),
                    np.array([[lo], [hi]]), best["lo"][1] + best["hi"][1], "bound"))
    if len(Zown):
        q10 = np.quantile(Zown[:, kern["cols"]], 0.1, axis=0)
        q90 = np.quantile(Zown[:, kern["cols"]], 0.9, axis=0)
        for a, b in ((lo, hi), (hi, lo)):
            out.append((kern, np.vstack([q10, q90]), np.array([[a], [b]]), 0.0, "bound"))
    return out


def anchor_candidates(Zanc, arms, bank, c, k, kern, head, n_rows):
    """Points at failure states this law owns (step 0), a few commands each."""
    if Zanc is None or k != 0:
        return []
    rows = np.flatnonzero(arms == c)
    if not len(rows):
        return []
    th = _theta(bank, c, k)
    bounds = KL.bounds_of(bank)
    kept = []
    for i in rows:
        x = Zanc[i, kern["cols"]]
        if len(kern["X"]) and (np.abs(kern["X"] - x) / kern["ls"]).sum(1).min() < 0.5:
            continue
        if any((np.abs(Zanc[j, kern["cols"]] - x) / kern["ls"]).sum() < 0.5 for j in kept):
            continue
        kept.append(i)
        if len(kept) >= n_rows:
            break
    out = []
    for i in kept:
        x = Zanc[i, kern["cols"]]
        u0 = KL.evaluate(th, kern if KL.n_points(kern) else None, design_matrix(Zanc[i:i + 1]),
                         bounds=bounds)[0]
        if head == "argmax":
            for a in range(len(u0)):
                y = u0.copy()
                y[a] = u0.max() + 1.0 + 0.25 * (u0.max() - u0.min())
                out.append((kern, x[None], y[None], 0.0, "anchor"))
        else:
            lo, hi = bounds if bounds is not None else (u0[0] - 1.0, u0[0] + 1.0)
            for yv in (lo, 0.5 * lo, min(max(0.0, lo), hi), hi):
                out.append((kern, x[None], np.array([[yv]]), 0.0, "anchor"))
    return out


def _add(kern, x, y):
    return KL.make(kern["cols"], np.vstack([kern["X"], np.atleast_2d(x)]),
                   np.vstack([kern["Y"], np.atleast_2d(y)]), kern["ls"])


def add_points(env, bank, c, k, cands, kern, pol_fn, cur, cfg, verbose=False,
               regen=None):
    """Greedy: screen every candidate, confirm the best few, keep the best.

    A candidate is (kernel it joins, x, y, adv). While the law has no points the
    candidates may sit on DIFFERENT column sets -- which columns a kernel reads
    is decided here, by rollout, not by the statistic that proposed them. After
    the first point is kept, `regen(kern)` re-proposes on the chosen columns.
    """
    log = []
    for _ in range(cfg["max_points"]):
        if not cands:
            break
        cur_cheap = score(env, bank, pol_fn, cfg["screen_ep"], cfg["T"], cfg["seed"])
        rows = []
        for j, (kb, x, y, adv, src) in enumerate(cands):
            b = KL.with_kern(bank, c, k, _add(kb, x, y))
            g = score(env, b, pol_fn, cfg["screen_ep"], cfg["T"], cfg["seed"])
            rows.append(((g - cur_cheap).mean(), j))
        rows.sort(key=lambda r: -r[0])
        best = None
        for d_screen, j in rows[:cfg["n_confirm"]]:
            kb, x, y, _, src = cands[j]
            kc = _add(kb, x, y)
            cand = KL.with_kern(bank, c, k, kc)
            ok, d, g = accept(env, cand, pol_fn, cur, cfg["n_ep"], cfg["T"],
                              cfg["seed_c"], cfg["z"])
            log.append(dict(op="add", arm=c, step=k, screen=float(d_screen), src=src,
                            delta=d, accepted=bool(ok and d > cfg["min_gain"])))
            if ok and d > cfg["min_gain"] and (best is None or d > best[0]):
                best = (d, j, kc, cand, g)
        if best is None:
            break
        d, j, kern, bank, cur = best
        src = cands[j][4]
        cands = (regen(kern) if regen is not None else
                 [cd for i, cd in enumerate(cands) if i != j])
        if verbose:
            print("      + point %d on %s: %s  %+.2f  [%s]"
                  % (KL.n_points(kern), _where(c, k), _pt(kern, -1, cfg), d, src),
                  flush=True)
    return bank, kern, cur, log


def cem_kernel(env, bank, c, k, kern, pol_fn, cur, cfg, rng, scale, y_sigma,
               verbose=False):
    """CEM over locations (in column-spread units), targets and log lengthscales."""
    M, D = kern["X"].shape
    nA = kern["Y"].shape[1]
    mu = np.concatenate([(kern["X"] / scale).ravel(), kern["Y"].ravel() / y_sigma,
                         np.log(kern["ls"])])
    sig = np.concatenate([np.full(M * D, 0.25), np.full(M * nA, 0.5), np.full(D, 0.3)])

    bounds = KL.bounds_of(bank)

    def unpack(p):
        X = p[:M * D].reshape(M, D) * scale
        Y = p[M * D:M * D + M * nA].reshape(M, nA) * y_sigma
        if bounds is not None:
            Y = np.clip(Y, bounds[0], bounds[1])
        return KL.make(kern["cols"], X, Y, np.exp(p[M * D + M * nA:]))
    n_el = max(3, cfg["cem_K"] // 4)
    for it in range(cfg["cem_iter"]):
        P = mu[None] + sig[None] * rng.standard_normal((cfg["cem_K"], len(mu)))
        P[0] = mu
        g = np.array([score(env, KL.with_kern(bank, c, k, unpack(p)), pol_fn,
                            cfg["screen_ep"], cfg["T"], cfg["seed"]).mean() for p in P])
        el = np.argsort(-g)[:n_el]
        mu = P[el].mean(0)
        sig = np.maximum(P[el].std(0), 0.03)
    kc = unpack(mu)
    cand = KL.with_kern(bank, c, k, kc)
    ok, d, g = accept(env, cand, pol_fn, cur, cfg["n_ep"], cfg["T"], cfg["seed_c"], cfg["z"])
    keep = bool(ok and d > cfg["min_gain"])
    if verbose:
        print("      tune %d points on %s: %+.2f %s"
              % (M, _where(c, k), d, "accepted" if keep else "rejected"), flush=True)
    rec = dict(op="cem", arm=c, step=k, delta=d, accepted=keep)
    return (cand, kc, g, rec) if keep else (bank, kern, cur, rec)


def prune_points(env, bank, c, k, kern, pol_fn, cur, cfg, verbose=False):
    """Drop every point the tree does not measurably need, one at a time."""
    log = []
    i = KL.n_points(kern) - 1
    while i >= 0 and KL.n_points(kern):
        keep = [j for j in range(KL.n_points(kern)) if j != i]
        kc = (KL.make(kern["cols"], kern["X"][keep], kern["Y"][keep], kern["ls"])
              if keep else None)
        cand = KL.with_kern(bank, c, k, kc)
        # A POINT IS PRUNED ONLY WHEN THE TEST IS CONFIDENT IT IS WORTH LESS THAN
        # THE MARGIN. The non-inferiority test used elsewhere prunes on
        # inconclusive evidence, and with a per-episode spread of ~50 against a
        # point worth +2 that flipped every round: measured, rung 0 accepted the
        # same red-brake point at +2.12 and pruned it at +0.76 on the next seed.
        g = score(env, cand, pol_fn, cfg["n_ep"], cfg["T"], cfg["seed_c"])
        dd = g - cur
        se = max(dd.std() / np.sqrt(len(dd)), 1e-12)
        d = float(dd.mean())
        ok = bool(d - cfg["z"] * se > -cfg["prune_margin"])
        log.append(dict(op="prune", arm=c, step=k, delta=d, accepted=ok))
        if ok:
            if verbose:
                print("      - point %d on %s pruned (%+.2f)" % (i, _where(c, k), d),
                      flush=True)
            bank, cur = cand, g
            kern = kc if kc is not None else dict(kern, X=kern["X"][:0], Y=kern["Y"][:0])
        i -= 1
    return bank, kern, cur, log


def _where(c, k):
    return "default" if c < 0 else ("arm %d" % c + (" step %d" % k if k else ""))


def _pt(kern, i, cfg):
    names = cfg.get("zn")
    x = ", ".join("%s=%.3g" % (names[j] if names else "z%d" % j, v)
                  for j, v in zip(kern["cols"], kern["X"][i]))
    y = kern["Y"][i]
    acts = cfg.get("actions")
    if len(y) == 1:
        return "(%s) -> %.3g" % (x, y[0])
    what = (acts[int(np.argmax(y))] if acts and len(y) > 1 else "%.3g" % y[0])
    return "(%s) -> %s" % (x, what)


# SCREENING ON THE TEST'S OWN EPISODES (screen_ep None = n_ep). A point worth
# +10 against a per-episode spread of ~50 is noise at 120 episodes, and among
# 300 proposals the top four screened were never the good one: measured, the
# rung-0 search accepted nothing at 120 and found its red slowdown at 500.
KDEFAULTS = dict(n_laws=2, max_points=3, n_cols=2, max_cols=3, col_pool=5, n_prop=24,
                 n_confirm=8,
                 dev_ep=3000, ks=(1, 3, 8), screen_ep=None, cem_iter=3, cem_K=24,
                 prune_margin=0.25, min_share=0.05, hint_cols=(), n_anchor=6)


def search_kernels(env, bank, names, zn, pol_fn, cur, T, seed, z=2.0, min_gain=0.3,
                   n_ep=300, rng=None, verbose=True, critic=None, anchors=None, **kw):
    """Grow, tune and prune inducing points on the laws that carry the most rows.

    `critic`, when given, is called as critic(bank, c, k, kern, head) and returns
    extra (x, y, adv) proposals for that law, ranked with the deviations'.
    """
    cfg = dict(KDEFAULTS, **kw)
    cfg.update(T=T, seed=seed, z=z, min_gain=min_gain, n_ep=n_ep, zn=zn,
               actions=bank.get("actions"))
    if cfg["screen_ep"] is None:
        cfg["screen_ep"] = n_ep
    cfg["seed_c"] = seed + 7919
    cur = score(env, bank, pol_fn, n_ep, T, cfg["seed_c"])     # the incumbent, fresh episodes
    rng = rng or np.random.default_rng(seed)
    head = bank.get("head")
    t0 = time.time()
    # THE PROPOSALS COME FROM THE AGENT'S OWN RETURN even when acceptance is on
    # another: an exact counterfactual is only informative if it is measured in
    # the units the agent is responsible for
    with _own_reward(env):
        ex = EX.deviations(env, bank, n_ep=cfg["dev_ep"], T=T, seed=seed + 5, rng=rng,
                           ks=cfg["ks"])
    if ex is None or not len(ex["adv"]):
        return bank, cur, []
    laws = flat_laws(bank)
    share = np.bincount(ex["law"].astype(int), minlength=len(laws)) / len(ex["law"])
    order = [int(i) for i in np.argsort(-share) if share[i] >= cfg["min_share"]]
    Zall = ex["z0"]
    joint = {}                     # law -> its critic-sets, for the joint candidate
    Zanc = arms_anc = None
    if anchors is not None and len(anchors):
        pa = pol_fn(bank)
        pa.reset(len(anchors))
        Zanc = pa.z(np.asarray(anchors, float), update=False)
        arms_anc = pa.arbitrate(Zanc)
    log = []
    for L in order[:cfg["n_laws"]]:
        c, k = laws[L]
        own = ex["law"].astype(int) == L
        paid = ex["adv"][own] > 0
        kern = KL.kern_of(bank, c, k)
        n_out = np.asarray(bank["default"]).shape[1]

        def empty(cols):
            return KL.make(cols, np.zeros((0, len(cols))), np.zeros((0, n_out)),
                           init_ls(Zall[own], cols))

        def propose(kb, n):
            cs = candidates(ex, bank, c, k, kb, head, n)
            if critic is not None:
                cs = cs + [(kb, np.atleast_2d(t[0]), np.atleast_2d(t[1]), t[2],
                            t[3] if len(t) > 3 else "critic")
                           for t in critic(bank, c, k, kb, head)]
            cs = sorted(cs, key=lambda t: -t[3])
            sets_ = [t for t in cs if t[4] == "critic-set"]
            if sets_:
                joint.setdefault((c, k), []).extend(sets_)
            return cs + anchor_candidates(Zanc, arms_anc, bank, c, k, kb, head,
                                          cfg["n_anchor"])
        if KL.n_points(kern):
            cands = propose(kern, cfg["n_prop"])
            sets = [kern["cols"]]
            if len(kern["cols"]) < cfg["max_cols"]:
                pool = choose_cols(Zall[own], paid, cfg["col_pool"], _guard_cols(bank, c),
                                   cfg["hint_cols"], n_obs=len(names))
                for col in pool:
                    if col in kern["cols"]:
                        continue
                    kw_ = widen_kernel(kern, col, Zall[own])
                    sets.append(kw_["cols"])
                    cands += propose(kw_, max(4, cfg["n_prop"] // 3))
        else:
            # WHICH COLUMNS: every set of `n_cols` from a short pool -- hints,
            # the arm's guard columns, then the columns that best separate the
            # deviations that paid -- each proposing its own points, all
            # screened together
            from itertools import combinations
            pool = choose_cols(Zall[own], paid, cfg["col_pool"], _guard_cols(bank, c),
                               cfg["hint_cols"], n_obs=len(names))
            sets = [list(s) for s in combinations(pool, min(cfg["n_cols"], len(pool)))]
            per = max(2 if head == "argmax" else 8, cfg["n_prop"] // max(1, len(sets)))
            cands = [cd for s in sets for cd in propose(empty(s), per)]
            bnd = KL.bounds_of(bank)
            if bnd is not None:
                cands = [cd for s in sets
                         for cd in bound_seeds(ex, bank, c, k, empty(s), bnd, Zall[own])] + cands
        if verbose:
            print("    kernel on %s (%.0f%% of deviations, %d paid): %d column sets "
                  "from {%s}, %d proposals"
                  % (_where(c, k), 100 * share[L], int(paid.sum()), len(sets),
                     ",".join(sorted({zn[j] for s in sets for j in s})), len(cands)),
                  flush=True)
        bank, kern, cur, alog = add_points(env, bank, c, k, cands, kern, pol_fn, cur,
                                           cfg, verbose,
                                           regen=lambda kb: propose(kb, cfg["n_prop"]))
        if verbose and KL.n_points(kern):
            print("      columns kept: %s" % ",".join(zn[j] for j in kern["cols"]),
                  flush=True)
        log += alog
        if KL.n_points(kern):
            scale = np.maximum(Zall[own][:, kern["cols"]].std(0), 1e-3)
            y_sig = (1.0 if head == "argmax" else
                     0.25 * float(np.subtract(*reversed(bank.get("u_range")
                                                        or env.u_range))))
            bank, kern, cur, rec = cem_kernel(env, bank, c, k, kern, pol_fn, cur, cfg,
                                              rng, scale, y_sig, verbose)
            log.append(rec)
            bank, kern, cur, plog = prune_points(env, bank, c, k, kern, pol_fn, cur,
                                                 cfg, verbose)
            log += plog
    if verbose:
        n_pts = sum(KL.n_points(KL.kern_of(bank, c, k)) for c, k in flat_laws(bank))
        print("    kernels: %d inducing points in the tree [%.0fs]; %s"
              % (n_pts, time.time() - t0, source_summary(log)), flush=True)
    if len(joint) >= 2:
        bank, cur, jlog = joint_sets(env, bank, joint, pol_fn, cur, cfg, verbose)
        log += jlog
    # back on the caller's seed, which the rest of the round compares against
    cur = score(env, bank, pol_fn, n_ep, T, seed)
    return bank, cur, log


def joint_sets(env, bank, joint, pol_fn, cur, cfg, verbose=False, top=2):
    """The best critic-sets of several laws applied together, tested as one."""
    from itertools import product
    laws = sorted(joint)[:3]
    per_law = [sorted(joint[L], key=lambda t: -t[3])[:top] for L in laws]
    cur_cheap = score(env, bank, pol_fn, cfg["screen_ep"], cfg["T"], cfg["seed"])
    rows = []
    for combo in product(*per_law):
        b = bank
        for (c, k), (kb, X, Y, adv, src) in zip(laws, combo):
            b = KL.with_kern(b, c, k, _add(kb, X, Y))
        g = score(env, b, pol_fn, cfg["screen_ep"], cfg["T"], cfg["seed"])
        rows.append((float((g - cur_cheap).mean()), b, combo))
    rows.sort(key=lambda r: -r[0])
    log = []
    for d_screen, b, combo in rows[:cfg["n_confirm"] // 2 or 1]:
        ok, d, g = accept(env, b, pol_fn, cur, cfg["n_ep"], cfg["T"], cfg["seed_c"], cfg["z"])
        keep = bool(ok and d > cfg["min_gain"])
        log.append(dict(op="add", arm=laws[0][0], step=0, screen=d_screen, delta=d,
                        src="joint", accepted=keep))
        if keep:
            if verbose:
                print("      + joint set on %s: %+.2f" % (" & ".join(_where(c, k) for c, k in laws), d),
                      flush=True)
            return b, g, log
    return bank, cur, log


def source_summary(log):
    """Per proposal source: how many reached confirmation, how many were kept --
    the audit of which loop (deviations, critic, failure anchors, bound seeds)
    is producing the points the tree keeps."""
    out = []
    for src in ("dev", "critic", "critic-set", "joint", "anchor", "bound"):
        rows = [e for e in log if e.get("op") == "add" and e.get("src") == src]
        if rows:
            out.append("%s %d confirmed/%d kept" % (src, len(rows),
                                                    sum(1 for e in rows if e["accepted"])))
    return "; ".join(out) if out else "no proposal reached confirmation"
