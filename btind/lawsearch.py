"""Control laws chosen by measured return, not by fitting a proxy.

e18 moved the GUARDS off a regression criterion and onto measured return, and
every result since has rested on that. The LAWS were never moved. They are still
whatever `fit_value_law` produces from the planner's labels, and on NestWorld
that costs more than the partition ever did:

    hand-coded affine laws      G 23.95     cos to the labels 0.800
    OLS on the same labels      G 13.43     cos to the labels 0.819
    M-weighted (the incumbent)  G 12.01     cos to the labels 0.817

The fitted laws match the labels BETTER and score ten return units WORSE, with
the same partition, the same labels and the same policy class. That is the sixth
time a proxy has improved in this project while return fell, and the first time
it happened inside the control law rather than the criterion that picks guards.

WHY REGRESSION LOSES TO A SELECTOR. Least squares spreads its error evenly over
the rows it is given, including over the direction that carries the behaviour. A
law that is exactly `bear_nest` is wrong on every noisy label and right about the
only thing that matters; a law fitted to those labels is slightly wrong about
everything, and slightly wrong about which way to walk is a controller that
arrives late.

THE LIBRARY IS THE TASK'S OWN VOCABULARY. Every candidate here is affine, so the
policy class does not change and nothing becomes less readable: a bearing pair
in the observation gives "toward", "away" and the two tangents, and pairs of
those give the blends. `bear_threat` tangents matter specifically -- fleeing
straight away from a pursuer that is faster in a corner is worse than orbiting
it, and no regression on planner labels reliably recovers that.

SELECTION IS THE SAME TEST AS EVERYWHERE ELSE. Paired episodes, one arm at a
time in priority order, accept only what clears z standard errors. On a
deterministic world the pairing is exact, so the only variance left is
heterogeneity of the effect across starting states -- which is what the standard
error is supposed to measure.
"""
import numpy as np

from .collect import design_matrix
from .structure import accept


def _theta(d, spec):
    """Affine law from a list of (column, row-of-u, coefficient) triples."""
    th = np.zeros((d, 2))
    for col, row, c in spec:
        th[col, row] += c
    return th


def structural_primitives(names, d=None):
    """Toward, away and both tangents for EVERY ADJACENT COLUMN PAIR.

    No column is named. The only assumption is that a world reporting a
    direction reports its two components side by side, which is the same
    structural prior the memory search uses to decide what may be stored
    together -- and it is a PRIOR, not a discovery: it asserts that sparse
    selectors on paired columns are the laws worth trying.

    It has to be said plainly because it is load-bearing. Measured with the
    right partition, selector laws scored 23.95 where laws fitted to planner
    labels scored 13.43, and cross-entropy search from a random law stalls at
    -11.6: in 44 dimensions it essentially never finds `u = bear_food`. The
    junk pairs this enumerates -- `(t_norm, noise)` and the like -- cost nothing
    but a rollout each, and the rollout rejects them.
    """
    d = (len(names) + 1) if d is None else d
    out = {}
    for j in range(len(names) - 1):
        a, b = j, j + 1
        tag = "%s|%s" % (names[a], names[b])
        out["to[" + tag + "]"] = _theta(d, [(a, 0, 1.0), (b, 1, 1.0)])
        out["from[" + tag + "]"] = _theta(d, [(a, 0, -1.0), (b, 1, -1.0)])
        out["cw[" + tag + "]"] = _theta(d, [(b, 0, -1.0), (a, 1, 1.0)])
        out["ccw[" + tag + "]"] = _theta(d, [(b, 0, 1.0), (a, 1, -1.0)])
    return out


def discrete_primitives(names, n_act, d, rng=None, n_sample=None):
    """The law vocabulary for an argmax head. It names NOTHING but the actions.

    Two families, and both are structural in the same sense
    `structural_primitives` is -- they assert a SHAPE for a useful law, not a
    strategy for the task.

        CONSTANT PREFERENCE    bias on action k, zero elsewhere: "always k".
                               These are the honest null. The only thing named
                               is the action set, which the environment defines
                               and hands us; choosing among them is the search's
                               job, and a one-arm bank carrying one of them is
                               exactly the constant-action baseline it has to
                               beat.

        SINGLE-COLUMN SCORE    theta[j, k] = +-1: "prefer action k in proportion
                               to column j". Every column against every action,
                               both signs, including the planted `t_norm` and
                               `noise` -- which is the point. A vocabulary that
                               excluded the distractors could not be caught
                               buying them.

    There is deliberately no pairing here. On a vector head, adjacent columns
    are a direction and `to[a|b]` means something; an argmax over preferences
    has no such geometry, so inventing combinations would be inventing task
    knowledge. Anything richer than a single column is left to CEM, which
    refines theta as a whole and needs no vocabulary at all.
    """
    rng = rng or np.random.default_rng(0)
    out = {}
    for k in range(n_act):
        th = np.zeros((d, n_act))
        th[-1, k] = 1.0
        out["always[%d]" % k] = th
    pool = []
    for j in range(len(names)):
        for k in range(n_act):
            for sgn in (1.0, -1.0):
                pool.append((j, k, sgn))
    if n_sample is not None and len(pool) > n_sample:
        idx = rng.choice(len(pool), n_sample, replace=False)
        pool = [pool[i] for i in idx]
    for j, k, sgn in pool:
        th = np.zeros((d, n_act))
        th[j, k] = sgn
        out["%s%s->%d" % ("+" if sgn > 0 else "-", names[j], k)] = th
    return out


def scalar_primitives(names, d, lo, hi, scales=(1.0, 0.1), Z=None):
    """The law vocabulary for a one-output continuous head. Names nothing.

    CONSTANT LEVELS across the world's range -- the honest null for a command
    that is clipped to [lo, hi] -- and single-column proportional terms at two
    scales, both signs: "command in proportion to column j".

    WHY THE TERMS ARE CENTRED AND SCALED when `Z` is given. `sgn * sc * column`
    only means anything if 0 lies inside [lo, hi]. It does for a car, whose
    range is [-B_MAX, A_MAX]. It does NOT for the signal, whose command is a
    green time in [5, 45]: every negative coefficient on a non-negative column
    lands below 5, and so does every 0.1 coefficient on a column smaller than
    50, so the clip turns them all into the same constant. Measured on a real
    tree, 38 of 50 laws in the step search's pool were the CONSTANT 5 after
    clipping, 50 laws gave 17 distinct commands, and 2300 screened candidates
    were 19 distinct controllers -- the search paid 119 s to rank copies.

    With `Z` a term is `mid + sgn * sc * (column - median) * half / spread`,
    so one column's 10th-to-90th-percentile sweep moves the command across
    half the range whatever its units, and both signs say something. Without
    `Z` the old uncentred form is returned unchanged, so a caller that does not
    pass it -- and a world whose range straddles 0 -- is unaffected.
    """
    out = {}
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        th = np.zeros((d, 1))
        th[-1, 0] = lo + f * (hi - lo)
        out["const[%.3g]" % th[-1, 0]] = th
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    for j in range(len(names)):
        gain, centre = 1.0, 0.0
        if Z is not None and j < np.shape(Z)[1]:
            col = np.asarray(Z[:, j], float)
            spread = float(np.percentile(col, 90) - np.percentile(col, 10))
            if spread <= 1e-9:
                # a column that does not move cannot carry a proportional term
                continue
            gain, centre = half / spread, float(np.median(col))
        for sc in scales:
            for sgn in (1.0, -1.0):
                th = np.zeros((d, 1))
                th[j, 0] = sgn * sc * gain
                th[-1, 0] = mid - sgn * sc * gain * centre if Z is not None else 0.0
                out["%s%.3g*%s" % ("+" if sgn > 0 else "-", sc, names[j])] = th
    return out


def primitives(names, extras=()):
    """The affine vocabulary a world's observation columns admit.

    For every `bear_<x>_x` / `bear_<x>_y` pair: toward, away, and both tangents.
    A tangent is the 90-degree rotation (-by, bx), which is still affine and is
    how a slower pursuer is evaded without being cornered.

    This version reads column NAMES, which is a stronger prior than
    `structural_primitives` -- it knows which pairs are directions. Kept for the
    named worlds; the structural one is what a new world gets.
    """
    d = len(names) + 1
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for n in names:
        if not n.startswith("bear_") or not n.endswith("_x"):
            continue
        tag = n[5:-2]
        bx, by = idx[n], idx["bear_%s_y" % tag]
        out["to_" + tag] = _theta(d, [(bx, 0, 1.0), (by, 1, 1.0)])
        out["from_" + tag] = _theta(d, [(bx, 0, -1.0), (by, 1, -1.0)])
        out["cw_" + tag] = _theta(d, [(by, 0, -1.0), (bx, 1, 1.0)])
        out["ccw_" + tag] = _theta(d, [(by, 0, 1.0), (bx, 1, -1.0)])
    for k, v in extras:
        out[k] = v
    return out


def memory_primitives(zn, mem, names):
    """Laws that steer by a remembered value: toward it, and away from it.

    `mem_<c> - <c>` is affine in the layout and the action is normalised, so
    "walk to the point I latched" is an ordinary law in the existing class --
    no new machinery, and it only exists for columns a search actually chose to
    store. Pairs only: a single stored scalar has no direction.
    """
    if not mem or len(mem["cols"]) != 2:
        return {}
    idx = {n: i for i, n in enumerate(zn)}
    d = len(zn) + 1
    c0, c1 = mem["cols"]
    m0, m1 = idx["mem_%s" % names[c0]], idx["mem_%s" % names[c1]]
    l0, l1 = idx[names[c0]], idx[names[c1]]
    return {"to_mem": _theta(d, [(m0, 0, 1.), (l0, 0, -1.),
                                 (m1, 1, 1.), (l1, 1, -1.)]),
            "from_mem": _theta(d, [(m0, 0, -1.), (l0, 0, 1.),
                                   (m1, 1, -1.), (l1, 1, 1.)])}


def library(names, base=None, blend_with=("from_threat", "cw_threat"),
            weights=(0.6,), zn=None, mem=None):
    """Primitives, a few blends, and the incumbent law, as (label, theta) pairs.

    Blends are restricted to threat-relative components because that is the only
    place two objectives genuinely compete at the same instant: heading for food
    while keeping a pursuer at arm's length. Everything else is a choice of one
    target, which the primitives already express.
    """
    prim = primitives(zn or names)
    prim.update(memory_primitives(zn, mem, names) if zn else {})
    out = [(k, v) for k, v in prim.items()]
    for a in ("to_food", "to_nest"):
        if a not in prim:
            continue
        for b in blend_with:
            if b not in prim:
                continue
            for w in weights:
                out.append(("%s+%g*%s" % (a, w, b), prim[a] + w * prim[b]))
    if base is not None:
        out.append(("fitted", base))
    return out


def search_arm(env, bank, arm, cands, pol_fn, cur_G, n_ep=600, T=400,
               seed=777, z=2.0, verbose=False):
    """Replace one arm's law with the best candidate that wins its rollout."""
    best, best_G, log = None, cur_G, []
    for label, th in cands:
        cand = dict(bank)
        if arm < 0:
            cand["default"] = th
        else:
            laws = list(bank["laws"])
            laws[arm] = th
            cand["laws"] = laws
        ok, dlt, g = accept(env, cand, pol_fn, cur_G, n_ep, T, seed, z)
        log.append(dict(arm=int(arm), law=label, delta=dlt, accepted=bool(ok)))
        if ok and dlt > (best_G.mean() - cur_G.mean()):
            best, best_G = (label, th), g
    if best is None:
        return bank, cur_G, log
    out = dict(bank)
    if arm < 0:
        out["default"] = best[1]
    else:
        laws = list(bank["laws"])
        laws[arm] = best[1]
        out["laws"] = laws
    out.setdefault("law_names", {})[arm] = best[0]
    return out, best_G, log


def search_laws(env, bank, names, pol_fn, cur_G=None, n_ep=600, T=400,
                seed=777, z=2.0, use_fitted=True, verbose=True, zn=None):
    """One pass over the arms, in priority order, then the default.

    Greedy and in order, because an arm's value depends on what the arms above
    it have already taken: changing the flee law changes which states ever reach
    the arm below, so a candidate scored against the old flee law is scored
    against a distribution that no longer exists.
    """
    # The reference has to be measured at THIS function's episode count. A
    # cheaper search is fine; comparing 600 episodes against a 1000-episode
    # reference is not, and numpy only catches it because the shapes differ.
    from .structure import score
    out, G, log = dict(bank), score(env, bank, pol_fn, n_ep, T, seed), []
    out["law_names"] = dict(bank.get("law_names", {}))
    order = list(range(len(bank["clauses"]))) + [-1]
    for arm in order:
        base = (bank["default"] if arm < 0 else bank["laws"][arm])
        cands = library(names, base if use_fitted else None, zn=zn,
                        mem=bank.get("mem"))
        out, G, lg = search_arm(env, out, arm, cands, pol_fn, G, n_ep, T, seed,
                                z)
        log += lg
        if verbose:
            acc = [l for l in lg if l["accepted"]]
            print("    arm %2d: %-22s %s" % (
                arm, out.get("law_names", {}).get(arm, "kept fitted"),
                ("+%.2f" % max(l["delta"] for l in acc)) if acc
                else "nothing beat the incumbent"), flush=True)
    return out, G, log
