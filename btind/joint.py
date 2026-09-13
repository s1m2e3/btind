"""Conjunctive split criterion: the value slope AND the control law must break.

e08 measured the blind spot. Running the same instability test with the target
swapped from the action to the value gave completely different answers:

    target u* (action):  energy=288  d_threat=70  d_food=27  pos_x=17   null 11.2
    target V  (value):   pos_y=65    pos_x=61     energy=23  d_threat=21 null  6.7

`energy` dominates the action test and nearly vanishes from the value test,
because its boundary is an artifact of the planner's finite horizon -- beyond
H*e_decay of energy, starvation is invisible to the search, so u* flips
discontinuously while the achievable value does not. The cut moves with H
(0.402 / 0.508 / 0.591 at H = 40 / 80 / 120), which is proof it belongs to the
labelling procedure and not to ForageWorld. Meanwhile pos_x/pos_y top the value
test for a real reason -- being cornered against a wall genuinely lowers
achievable return -- and the action-only criterion threw that away.

So: fit TWO affine models per node, one for the action and one for the value,
and require a split to break both.

    z_u(t) = path_u(t) / crit_u        action-law fluctuation, in null units
    z_V(t) = path_V(t) / crit_V        value-model fluctuation, in null units
    score  = max_t min(z_u(t), z_V(t))

Normalising each path by its own permutation null makes the two comparable (the
score vectors have different dimensions, 2(d+1) against (d+1), so their nulls
differ). Taking the elementwise MIN is the conjunction: the statistic is only
large where both models break, and the argmax is a location where both break.
A split needs score > 1, i.e. both clear their own 95th percentile at that point.

Leaf laws are still fitted under the M-weighted value loss from valuesplit --
selection asks where to cut, the leaf law asks what to do once cut.
"""
import numpy as np

from .collect import design_matrix, leverage_weights
from .fluctuation import weighted_ols, scores, instability
from .valuesplit import fit_value_law


def joint_paths(X, U, V, w, Z, cand_vars, n_perm=200, rng=None):
    """Normalised fluctuation paths for both targets, plus the combined score.

    Returns dict with per-candidate score, cut index, and both raw statistics
    so a caller can report WHY a split was taken or refused.
    """
    rng = rng or np.random.default_rng(0)
    out = {}
    for key, Y in (("u", U), ("v", V.reshape(-1, 1))):
        _, resid = weighted_ols(X, Y, w)
        res = instability(scores(X, resid, w), Z[:, cand_vars],
                          n_perm=n_perm, rng=rng)
        out[key] = (res["paths"], float(np.quantile(res["null"], 0.95)),
                    res["stat"])

    (pu, cu, su), (pv, cv, sv) = out["u"], out["v"]
    n = X.shape[0]
    lo, hi = int(0.10 * n), int(0.90 * n)
    scores_, cuts = np.zeros(len(cand_vars)), np.zeros(len(cand_vars), int)
    for k in range(len(cand_vars)):
        comb = np.minimum(pu[k] / max(cu, 1e-9), pv[k] / max(cv, 1e-9))
        seg = comb[lo:hi]
        cuts[k] = lo + int(np.argmax(seg))
        scores_[k] = float(seg.max())
    return dict(score=scores_, cut=cuts, z_u=su / max(cu, 1e-9),
                z_v=sv / max(cv, 1e-9), crit_u=cu, crit_v=cv)


def grow_joint(obs, U, V, gap, M, split_vars, names, coh=None, min_samples=250,
               max_depth=5, n_perm=200, seed=0, _depth=0, _guard=None,
               _rng=None, verbose=True):
    from .grow import Node

    rng = _rng or np.random.default_rng(seed)
    guard = _guard or []
    X = design_matrix(obs)
    w = leverage_weights(gap, coh)
    node = Node(n=len(obs), depth=_depth, theta=fit_value_law(X, U, M),
                guard=list(guard))

    pad = "  " * _depth
    if len(obs) < 2 * min_samples or _depth >= max_depth:
        if verbose:
            print(f"{pad}leaf n={len(obs)} (size/depth)")
        return node

    r = joint_paths(X, U, V, w, obs, split_vars, n_perm=n_perm, rng=rng)
    k = int(np.argmax(r["score"]))
    node.stat, node.crit = float(r["score"][k]), 1.0

    if node.stat <= 1.0:
        if verbose:
            print(f"{pad}leaf n={len(obs)}  best={names[split_vars[k]]} "
                  f"score {node.stat:.2f} <= 1 (z_u {r['z_u'][k]:.1f}, "
                  f"z_v {r['z_v'][k]:.1f})")
        return node

    j = split_vars[k]
    at = float(np.sort(obs[:, j])[r["cut"][k]])
    m = obs[:, j] < at
    if min(m.sum(), (~m).sum()) < min_samples:
        if verbose:
            print(f"{pad}leaf n={len(obs)} (orphan)")
        return node

    node.split_var, node.split_at = j, at
    if verbose:
        print(f"{pad}split n={len(obs)} {names[j]} < {at:.3f}  "
              f"score {node.stat:.2f}  (z_u {r['z_u'][k]:.1f}, z_v {r['z_v'][k]:.1f})")

    kw = dict(split_vars=split_vars, names=names, min_samples=min_samples,
              max_depth=max_depth, n_perm=n_perm, _rng=rng, verbose=verbose)
    ch = (None, None) if coh is None else (coh[m], coh[~m])
    node.left = grow_joint(obs[m], U[m], V[m], gap[m], M[m], coh=ch[0],
                           _depth=_depth + 1, _guard=guard + [(j, at, True)], **kw)
    node.right = grow_joint(obs[~m], U[~m], V[~m], gap[~m], M[~m], coh=ch[1],
                            _depth=_depth + 1, _guard=guard + [(j, at, False)], **kw)
    return node


def grow_veto(obs, U, V, gap, M, split_vars, names, coh=None, ratio_max=4.0,
              min_samples=250, max_depth=4, n_perm=150, seed=0, _depth=0,
              _guard=None, _rng=None, verbose=True):
    """sup-LM selection, but variables that move the ACTION without moving the
    VALUE are vetoed as labelling artifacts.

    e08 showed the conjunctive min(z_u, z_V) criterion costs too much resolution
    -- it produced 5 leaves against sup-LM's 8 at H=40 and controlled far worse
    (-0.6 vs 4.8). The value signal is better used as a VETO than as a
    requirement: keep sup-LM's power to rank and locate, and merely refuse
    variables whose action-instability is not backed by any value-instability.

        ratio = z_u / z_V       energy 7.0 | d_threat 1.8 | d_food 0.8 | pos_x 0.2

    This derives the hand-banning of `energy` that has topped every comparison
    since e06, instead of being told it.
    """
    from .grow import Node
    from .collect import design_matrix as _dm

    rng = _rng or np.random.default_rng(seed)
    guard = _guard or []
    X, w = _dm(obs), leverage_weights(gap, coh)
    node = Node(n=len(obs), depth=_depth, theta=fit_value_law(X, U, M),
                guard=list(guard))
    pad = "  " * _depth
    if len(obs) < 2 * min_samples or _depth >= max_depth:
        return node

    r = joint_paths(X, U, V, w, obs, split_vars, n_perm=n_perm, rng=rng)
    ratio = r["z_u"] / np.maximum(r["z_v"], 1e-9)
    ok = (r["z_u"] > 1.0) & (ratio <= ratio_max)
    if not ok.any():
        if verbose:
            print(f"{pad}leaf n={len(obs)} (nothing survives the veto)")
        return node

    cand = np.where(ok)[0]
    k = int(cand[np.argmax(r["z_u"][cand])])
    j = split_vars[k]
    at = float(np.sort(obs[:, j])[r["cut"][k]])
    m = obs[:, j] < at
    if min(m.sum(), (~m).sum()) < min_samples:
        return node

    node.split_var, node.split_at, node.stat = j, at, float(r["z_u"][k])
    if verbose:
        vetoed = [names[split_vars[i]] for i in range(len(split_vars))
                  if r["z_u"][i] > 1.0 and ratio[i] > ratio_max]
        print(f"{pad}split n={len(obs)} {names[j]} < {at:.3f} "
              f"(z_u {r['z_u'][k]:.1f}, ratio {ratio[k]:.1f})"
              + (f"   vetoed: {vetoed}" if vetoed else ""))

    kw = dict(split_vars=split_vars, names=names, ratio_max=ratio_max,
              min_samples=min_samples, max_depth=max_depth, n_perm=n_perm,
              _rng=rng, verbose=verbose)
    ch = (None, None) if coh is None else (coh[m], coh[~m])
    node.left = grow_veto(obs[m], U[m], V[m], gap[m], M[m], coh=ch[0],
                          _depth=_depth + 1, _guard=guard + [(j, at, True)], **kw)
    node.right = grow_veto(obs[~m], U[~m], V[~m], gap[~m], M[~m], coh=ch[1],
                           _depth=_depth + 1, _guard=guard + [(j, at, False)], **kw)
    return node


def artifact_ratio(obs, U, V, gap, coh, split_vars, n_perm=120, rng=None):
    """z_u / z_V per candidate: action-instability not backed by value.

    A genuine mode moves the value surface and the control law together. A
    labelling artifact -- a discontinuity manufactured by the expert's finite
    horizon -- moves only the action.
    """
    X = design_matrix(obs)
    w = leverage_weights(gap, coh)
    r = joint_paths(X, U, V, w, obs, split_vars, n_perm=n_perm, rng=rng)
    return r["z_u"] / np.maximum(r["z_v"], 1e-9), r


def veto_vars(ratio, split_vars, factor=1.5):
    """Ban variables whose ratio is an OUTLIER among the candidates.

    An absolute cutoff does not survive DAgger: measured across rounds, energy's
    ratio fell 5.7 -> 3.5 -> 3.5 while every other variable sat at 0.2-2.0, so a
    fixed `ratio_max=4` banned it for two rounds and then silently stopped,
    and the controller detached from the hand-banned reference at exactly that
    round. Comparing each variable against the largest ratio among the OTHERS is
    scale-free and held for all 7 rounds.

    Note M (the curvature of Q in action space) must NOT be added to the
    denominator as extra "value support": energy has the highest M-instability
    of any variable (14.7 -> 33.2 across rounds vs d_threat's 1.2 -> 6.3),
    because horizon truncation reshapes the whole local Q surface. Using
    max(z_V, z_M, z_gap) as the denominator collapses the separation entirely
    (energy 1.7, d_threat 1.8) and excuses the artifact.
    """
    keep, banned = [], []
    for i, j in enumerate(split_vars):
        others = np.delete(ratio, i)
        (banned if ratio[i] > factor * others.max() else keep).append(j)
    return (keep or list(split_vars)), banned
