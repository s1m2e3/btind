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


def primitives(names, extras=()):
    """The affine vocabulary a world's observation columns admit.

    For every `bear_<x>_x` / `bear_<x>_y` pair: toward, away, and both tangents.
    A tangent is the 90-degree rotation (-by, bx), which is still affine and is
    how a slower pursuer is evaded without being cornered.
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
