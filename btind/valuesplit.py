"""Split on control value instead of statistical significance.

e06 produced a clean dissociation: `energy` is the most detectable split we have
(sup-LM 254, chosen by 20/20 seeds, threshold reproducible to +-0.001, matching a
closed-form prediction) and REMOVING it improves closed-loop return by 1.4. The
sup-LM criterion answers "is there a parameter change here", which with n=5000 a
tiny but consistent change passes easily. It never asks what the change is worth.

This module asks that instead. Two ingredients.

1. ACTION CURVATURE. The CEM sample already contains a local sample of Q(s,.):
   K first-actions with their returns. Regress a quadratic through them per state
   and the negative Hessian M_i is the curvature of return in action space -- how
   fast value falls off as the action drifts from optimal. Where return is flat
   in u, action error is free; where it is sharply peaked, error is expensive.

2. VALUE LOSS AS THE OBJECTIVE. With u* a local optimum the first-order term
   vanishes, so the value cost of a control law is the M-weighted squared error

       L(R) = sum_i  r_i^T M_i r_i ,    r_i = u*_i - Theta^T x_i

   Fitting affine leaves under that loss has a closed form, and so does its
   minimum: L(R) = c(R) - b(R)^T A(R)^-1 b(R), where

       A[(p,q),(k,m)] = sum_i X[i,p] X[i,k] M[i,q,m]
       b[(p,q)]       = sum_i X[i,p] (M_i u*_i)[q]
       c              = sum_i u*_i^T M_i u*_i

   The c terms cancel across a split, leaving

       Gain = b_L^T A_L^-1 b_L  +  b_R^T A_R^-1 b_R  -  b^T A^-1 b

   which is exactly XGBoost's G^2/(H+lambda) with G -> b, H -> A, matrix-valued,
   and with affine leaves rather than constants. Per-sample contributions to A
   and b are accumulated by cumulative sum over each candidate ordering, so a
   whole variable is scanned for the price of one pass plus one small solve per
   candidate cut -- the same trick that makes boosted trees fast.
"""
import numpy as np

RIDGE = 1e-6


# --------------------------------------------------------------------- M_i
def action_curvature(elite_or_all_a, G, clip_q=99.0):
    """Per-state curvature of return w.r.t. the first action.

    a: (n, K, 2) candidate first actions, G: (n, K) their returns.
    Returns M (n, 2, 2), PSD, = -Hessian of the fitted quadratic.
    """
    a = elite_or_all_a
    ax, ay = a[:, :, 0], a[:, :, 1]
    Z = np.stack([np.ones_like(ax), ax, ay, ax * ax, ay * ay, ax * ay], axis=2)
    ZtZ = np.einsum("nkp,nkq->npq", Z, Z)
    Zty = np.einsum("nkp,nk->np", Z, G)
    ZtZ[:, np.arange(6), np.arange(6)] += 1e-8
    # numpy 2 treats a 2-D rhs as a matrix, not a stack of vectors
    beta = np.linalg.solve(ZtZ, Zty[..., None])[..., 0]    # (n, 6)

    H = np.empty((len(a), 2, 2))
    H[:, 0, 0] = 2.0 * beta[:, 3]
    H[:, 1, 1] = 2.0 * beta[:, 4]
    H[:, 0, 1] = H[:, 1, 0] = beta[:, 5]
    M = -H                                                 # return is maximised

    # project to PSD: a non-concave fit means the sample says nothing here
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    cap = np.percentile(w[w > 0], clip_q) if (w > 0).any() else 1.0
    w = np.minimum(w, cap)
    return np.einsum("nij,nj,nkj->nik", V, w, V)


# ------------------------------------------------------- fit / loss / gain
def _Ab(X, U, M):
    """Per-sample contributions to A ((d,2,d,2)) and b ((d,2)), flattened."""
    n, d = X.shape
    Ai = np.einsum("ip,ik,iqm->ipqkm", X, X, M).reshape(n, 2 * d, 2 * d)
    Mu = np.einsum("iqm,im->iq", M, U)
    bi = np.einsum("ip,iq->ipq", X, Mu).reshape(n, 2 * d)
    return Ai, bi


def _score(A, b, ridge=RIDGE):
    """b^T A^-1 b, the reducible part of the value loss."""
    A = A + ridge * np.eye(A.shape[-1])
    try:
        return float(b @ np.linalg.solve(A, b))
    except np.linalg.LinAlgError:
        return float(b @ np.linalg.lstsq(A, b, rcond=None)[0])


def fit_value_law(X, U, M, ridge=RIDGE):
    """Affine law minimising the M-weighted value loss. Returns theta (d, 2)."""
    d = X.shape[1]
    Ai, bi = _Ab(X, U, M)
    A = Ai.sum(0) + ridge * np.eye(2 * d)
    return np.linalg.solve(A, bi.sum(0)).reshape(d, 2)


def best_value_split(X, U, M, Z, cand_vars, n_cand=48, min_frac=0.10,
                     ridge=RIDGE):
    """Scan every candidate variable and cut; return the best value gain.

    Returns (var_index_into_cand_vars, threshold, gain, parent_score).
    """
    n, d = X.shape
    Ai, bi = _Ab(X, U, M)
    A_tot, b_tot = Ai.sum(0), bi.sum(0)
    parent = _score(A_tot, b_tot, ridge)

    lo, hi = int(min_frac * n), int((1 - min_frac) * n)
    if hi <= lo:
        return None, None, -np.inf, parent

    best = (None, None, -np.inf)
    for jj, j in enumerate(cand_vars):
        order = np.argsort(Z[:, j], kind="mergesort")
        zs = Z[order, j]
        Ac = np.cumsum(Ai[order], axis=0)
        bc = np.cumsum(bi[order], axis=0)
        for k in np.unique(np.linspace(lo, hi - 1, n_cand).astype(int)):
            g = (_score(Ac[k], bc[k], ridge)
                 + _score(A_tot - Ac[k], b_tot - bc[k], ridge) - parent)
            if g > best[2]:
                best = (jj, float(zs[k]), g)
    return best[0], best[1], best[2], parent


# ------------------------------------------------------------ best-first grow
def grow_value(obs, U, M, split_vars, names, null_var=None, max_leaves=12,
               min_samples=250, null_factor=2.0, n_cand=48, verbose=True):
    """Best-first (lossguide) growth on the value gain.

    Depth-limited greedy growth was the real problem behind the e06 `energy`
    dissociation: both criteria pick `energy` at the root, but splitting every
    node down to a fixed depth spends the budget evenly instead of spending it
    where value is. Best-first keeps a frontier and always splits the leaf with
    the largest gain, so a leaf budget is allocated globally.

    Stopping uses the planted null feature as a running calibrator: a split must
    beat `null_factor` x the best gain achievable on a variable that carries no
    structure by construction. (Without a planted null, permute the candidate
    column instead -- same idea, more expensive.)
    """
    from .grow import Node

    def mk(idx, depth, guard):
        Xs = np.hstack([obs[idx], np.ones((len(idx), 1))])
        return Node(n=len(idx), depth=depth,
                    theta=fit_value_law(Xs, U[idx], M[idx]), guard=list(guard))

    root_idx = np.arange(len(obs))
    root = mk(root_idx, 0, [])
    frontier = [(root, root_idx)]

    def propose(idx):
        if len(idx) < 2 * min_samples:
            return None
        Xs = np.hstack([obs[idx], np.ones((len(idx), 1))])
        jj, at, g, _ = best_value_split(Xs, U[idx], M[idx], obs[idx],
                                        split_vars, n_cand=n_cand)
        if jj is None:
            return None
        if null_var is not None:
            _, _, gn, _ = best_value_split(Xs, U[idx], M[idx], obs[idx],
                                           [null_var], n_cand=n_cand)
            if g <= null_factor * max(gn, 1e-9):
                return None
        return split_vars[jj], at, g

    props = {id(n): propose(i) for n, i in frontier}
    while len(root.leaves()) < max_leaves:
        live = [(n, i) for n, i in frontier if props.get(id(n)) is not None]
        if not live:
            break
        node, idx = max(live, key=lambda ni: props[id(ni[0])][2])
        j, at, g = props[id(node)]
        m = obs[idx, j] < at
        li, ri = idx[m], idx[~m]
        if min(len(li), len(ri)) < min_samples:
            props[id(node)] = None
            continue
        node.split_var, node.split_at, node.stat = j, at, g
        node.left = mk(li, node.depth + 1, node.guard + [(j, at, True)])
        node.right = mk(ri, node.depth + 1, node.guard + [(j, at, False)])
        if verbose:
            print(f"  split n={len(idx)} on {names[j]} < {at:.3f}  gain {g:.0f}")
        frontier = [(n, i) for n, i in frontier if n is not node]
        frontier += [(node.left, li), (node.right, ri)]
        props.pop(id(node), None)
        props[id(node.left)] = propose(li)
        props[id(node.right)] = propose(ri)
    return root
