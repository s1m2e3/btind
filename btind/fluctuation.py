"""Parameter-instability testing for model-based recursive partitioning.

The question at a node is not 'does the action differ across this split' but
'is ONE local law still adequate for the whole region'. We fit a local linear
(TSK) law, take the per-sample score -- the gradient of the loss w.r.t. the law's
parameters -- and ask whether that score is orthogonal to each candidate
splitting variable. Sum of scores is zero at the optimum by construction, so all
the information lives in how they *fluctuate* when ordered by a variable.

Null distribution is obtained by permutation rather than from tabulated
Brownian-bridge critical values: simpler to get right, and free at this scale.
"""
import numpy as np


def weighted_ols(X, Y, w):
    sw = np.sqrt(np.maximum(w, 0.0))[:, None]
    theta, *_ = np.linalg.lstsq(X * sw, Y * sw, rcond=None)
    return theta, Y - X @ theta


def scores(X, resid, w):
    """psi_i = w_i * vec(x_i outer eps_i), shape (n, d*m)."""
    n, d = X.shape
    m = resid.shape[1]
    return (w[:, None, None] * X[:, :, None] * resid[:, None, :]).reshape(n, d * m)


def _inv_sqrt(J, ridge=1e-8):
    vals, vecs = np.linalg.eigh(J)
    vals = np.maximum(vals, ridge * max(vals.max(), 1e-12))
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def sup_lm_path(psi_sorted, J_inv_sqrt, trim=0.10):
    """sup-LM statistic and the whole fluctuation path, for one ordering."""
    n = psi_sorted.shape[0]
    W = np.cumsum(psi_sorted, axis=0) / np.sqrt(n)
    path = ((W @ J_inv_sqrt) ** 2).sum(axis=1)
    lo, hi = int(trim * n), int((1 - trim) * n)
    seg = path[lo:hi]
    k = lo + int(np.argmax(seg))
    return float(seg.max()), k, path


def instability(psi, Z, n_perm=200, trim=0.10, rng=None):
    """sup-LM statistic + permutation p-value for every column of Z."""
    rng = rng or np.random.default_rng(0)
    n, p = Z.shape
    J_inv_sqrt = _inv_sqrt(psi.T @ psi / n)

    stats, kstars, paths = np.zeros(p), np.zeros(p, dtype=int), []
    for j in range(p):
        order = np.argsort(Z[:, j], kind="mergesort")
        stats[j], kstars[j], path = sup_lm_path(psi[order], J_inv_sqrt, trim)
        paths.append(path)

    # One permutation null shared across columns: under H0 the statistic is
    # order-invariant, so a random ordering is a draw from the same null.
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b], _, _ = sup_lm_path(psi[rng.permutation(n)], J_inv_sqrt, trim)

    pvals = np.array([(1.0 + (null >= s).sum()) / (n_perm + 1.0) for s in stats])
    return dict(stat=stats, pval=pvals, kstar=kstars, paths=paths, null=null)
