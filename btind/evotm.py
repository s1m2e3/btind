r"""Evolved clause bank: BT guards as Tsetlin-style conjunctions, thresholds
sampled and drifted rather than grown.

Ported from the EVOTM system (egt_tm.py / tabular_tm.py) with one structural
change forced by our problem.

WHAT CARRIES OVER
  * quantile alphabet as a STARTING set, not a fixed grid (Booleanizer): cuts
    begin at interior quantiles, deduped against ties;
  * thresholds then DRIFT -- mutation jitters thr by N(0, sig_thr), so the
    population searches threshold space continuously instead of being pinned to
    the grid. This is the part a tree grower cannot do: recursive partitioning
    picks a cut once and never revisits it.
  * specialize / generalize moves (append or delete a literal), bounded by
    max_arity, which is the readability constraint on a BT guard;
  * newborn literals anchored on the worst-fit quartile (`hot`), so proposals
    go where the current controller is failing;
  * crossover by pooling both parents' literals and taking k <= max_arity;
  * replicator-mutator selection: fitness-proportional parents, then
    cross / mutate / faithful-copy;
  * a similarity penalty so the bank keeps distinct species instead of
    collapsing onto one clause.

WHAT CHANGES
  EVOTM fits a ridge readout over clause firings, summing them. A behaviour
  tree cannot sum guards -- a Fallback takes the FIRST matching child. So the
  bank here is PRIORITY-ORDERED: a state is handled by the first clause that
  matches it, and each clause carries its own affine control law fitted under
  the M-weighted value loss from valuesplit. That is a Fallback of Sequences,
  emitted directly:

      Fallback
      |-- Sequence[ clause_1 , Action(u = K1 x + b1) ]
      |-- Sequence[ clause_2 , Action(u = K2 x + b2) ]
      \-- Action(default law)

  Overlapping clauses ordered by priority are a strictly better fit to BT
  semantics than the disjoint binary partition a tree grower produces -- and
  it is what the Fallback node was built to express in the first place.
"""
import numpy as np
from sklearn.linear_model import Lasso

from .collect import design_matrix
from .valuesplit import fit_value_law


# ----------------------------------------------------------------- alphabet
class Alphabet:
    """Per-feature interior quantiles, deduped. The seed set for thresholds."""

    def __init__(self, n_thresholds=8):
        self.k = n_thresholds

    def fit(self, X, feats):
        qs = np.linspace(0, 1, self.k + 2)[1:-1]
        self.feats = list(feats)
        self.thr = {j: np.unique(np.quantile(X[:, j], qs)) for j in feats}
        self.lo = {j: float(X[:, j].min()) for j in feats}
        self.hi = {j: float(X[:, j].max()) for j in feats}
        return self


def _rand_literal(rng, alpha, X, hot):
    """(feat, thr, neg). Anchored on a hot row when one is supplied."""
    j = int(rng.choice(alpha.feats))
    if hot is not None and len(hot) and rng.random() < 0.5:
        thr = float(X[hot[int(rng.integers(len(hot)))], j])
    else:
        t = alpha.thr[j]
        thr = float(t[int(rng.integers(len(t)))]) if len(t) else \
            float(rng.uniform(alpha.lo[j], alpha.hi[j]))
    return [j, thr, bool(rng.random() < 0.5)]


def dedupe_literals(cl):
    """Merge literals on the same (feature, direction): keep the tighter one.

    Without this the population fills with `pos_y>0.423 AND pos_y>0.423`, which
    spends guard budget saying one thing -- the same failure `_dedupe` guards
    against on the threshold alphabet in tabular_tm.py.
    """
    best = {}
    for j, thr, neg in cl:
        k = (j, neg)
        if k not in best:
            best[k] = thr
        else:
            best[k] = min(best[k], thr) if neg else max(best[k], thr)
    return [[j, thr, neg] for (j, neg), thr in best.items()]


def _match(clause, X):
    m = np.ones(X.shape[0], dtype=bool)
    for j, thr, neg in clause:
        t = X[:, j] > thr
        m &= (~t if neg else t)
    return m


def match_matrix(clauses, X):
    if not clauses:
        return np.zeros((X.shape[0], 0), dtype=bool)
    return np.stack([_match(c, X) for c in clauses], axis=1)


# ------------------------------------------------------------------ scoring
def l1_order(Mm, per_default, alpha=None, max_keep=32, standardize=True):
    """Select and RANK clauses by an L1 readout on the single-law value loss.

    EVOTM's `--readout l1` zeroes most clauses so the survivors are the model.
    Here the regression target is the per-sample value loss of ONE global affine
    law, and the readout is constrained positive: a clause earns weight only by
    marking states that a single law handles badly, which is exactly the states
    that deserve their own BT arm. Weight then orders the Fallback.

    This replaces ordering by specificity. Specificity is a proxy -- it assumes
    narrow means important -- and it let 24-parameter laws be fitted on ~150-row
    slivers, which drove in-sample loss down (100% -> 81.5%) while closed-loop
    return fell to -3.6. L1 prices a clause by what it explains instead.

    STANDARDIZE. A raw 0/1 indicator for a clause matching a fraction p of rows
    has std sqrt(p(1-p)), so under a fixed L1 penalty a narrow clause is shrunk
    ~1/sqrt(p) harder than a broad one for the same per-row effect. Since arity
    and coverage are inversely related, an unstandardized readout is a hidden
    prior against high-arity clauses: it decides the arity distribution before
    fitness ever sees it. Scaling the columns to unit variance makes the penalty
    scale-free in coverage, which is what `Lasso` assumes of its design anyway.

    SELECTION IS STANDARDIZED, ORDERING IS NOT. The returned weights are always
    in raw 0/1 units, where a coefficient is the mean loss ELEVATION of the rows
    a clause matches -- a per-row intensity. Ordering by that puts the most
    acute exception at the front of the Fallback, which is also what keeps a
    nested ladder in the right order: the superset of a narrow clause averages
    in its easier outer ring and prices lower.
    """
    X = Mm.astype(np.float64)
    if X.shape[1] == 0:
        return np.array([], dtype=int), np.zeros(0)
    if alpha is None:
        alpha = 0.01 * float(np.std(per_default))
    if standardize:
        p = X.mean(0)
        sd = np.sqrt(np.maximum(p * (1.0 - p), 1e-12))
        X = X / sd
    las = Lasso(alpha=max(alpha, 1e-9), positive=True, fit_intercept=True,
                max_iter=4000)
    las.fit(X, per_default)
    w = las.coef_ / sd if standardize else las.coef_   # back to raw 0/1 units
    keep = np.flatnonzero(w > 0)
    return keep[np.argsort(-w[keep])][:max_keep], w


def priority_order(Mm, max_cover=1.0):
    """Most SPECIFIC clause first; clauses covering most of the space dropped.

    Priority cannot be list position: selection reshuffles the population every
    generation, so a clause matching 90% of rows randomly lands at the front and
    starves every clause behind it -- measured, 3-6 of 40 clauses ever received
    a row and the loss did not improve across 60 generations. EVOTM never meets
    this because its ridge readout SUMS clause firings; a Fallback takes the
    first match, so the order has to be a deterministic function of the clause
    set. Ascending match count is both that and the convention a person writes
    by hand: specific exceptions above general defaults.
    """
    cnt = Mm.sum(0)
    # Broad clauses are KEPT, not dropped: specificity ordering already places
    # them late, where a Fallback wants its general arms. Capping cover at 0.5
    # deleted the mid-level fallbacks and dumped most of the state space on the
    # single default law, which is the -1.4 global-affine policy.
    live = np.flatnonzero((cnt > 0) & (cnt < max_cover * Mm.shape[0]))
    return live[np.argsort(cnt[live])]


def _assign(Mm, order):
    """First matching clause in priority order wins. -1 = default leaf."""
    out = np.full(Mm.shape[0], -1, dtype=int)
    for c in order[::-1]:
        out[Mm[:, c]] = c
    return out


def _laws_and_loss(assign, C, Xd, U, M, min_n):
    """Fit one affine law per region under the M-weighted value loss."""
    laws, per = {}, np.zeros(len(U))
    for c in list(range(C)) + [-1]:
        idx = np.flatnonzero(assign == c)
        if len(idx) < min_n:
            continue
        laws[c] = fit_value_law(Xd[idx], U[idx], M[idx])
    default = laws.get(-1)
    if default is None:
        default = fit_value_law(Xd, U, M)
        laws[-1] = default
    for c, th in laws.items():
        idx = np.flatnonzero(assign == c)
        if not len(idx):
            continue
        r = U[idx] - Xd[idx] @ th
        per[idx] = np.einsum("iq,iqm,im->i", r, M[idx], r)
    orphan = np.flatnonzero(~np.isin(assign, list(laws)))
    if len(orphan):
        r = U[orphan] - Xd[orphan] @ default
        per[orphan] = np.einsum("iq,iqm,im->i", r, M[orphan], r)
    return laws, per


def _clause_utility(assign, per, per_default, C):
    """How much better each clause's own law does than the default, on its rows.

    Cheap stand-in for leave-one-out: EVOTM reads utility off the ridge weights,
    which has no analogue once the readout is a priority chain rather than a sum.
    """
    u = np.zeros(C)
    for c in range(C):
        idx = np.flatnonzero(assign == c)
        u[c] = (per_default[idx] - per[idx]).sum() if len(idx) else 0.0
    return u


def _fitness(util, Mm, arity, arity_pen, sim_pen):
    f = util - arity_pen * np.asarray(arity, float)
    f = f - f.min() + 1e-9
    cnt = Mm.sum(0).astype(float)
    inter = (Mm.T.astype(np.float32) @ Mm.astype(np.float32))
    union = cnt[:, None] + cnt[None, :] - inter
    jac = inter / np.maximum(union, 1.0)
    np.fill_diagonal(jac, 0.0)
    f = f * np.exp(-sim_pen * jac.max(1))
    # Rank-sharpened: raw utility sums are dominated by a few clauses, which
    # flattens the selection probabilities and left the loss oscillating
    # (88.2 -> 89.7 -> 87.2 -> 85.8 -> 91.4) instead of descending.
    r = np.empty(len(f))
    r[np.argsort(f)] = np.arange(len(f))
    return (r + 1.0) ** 2, float(jac.max(1).mean())


# -------------------------------------------------------------------- evolve
def evolve_bank(obs, U, M, split_vars, names, C=40, generations=60,
                max_arity=3, n_thresholds=8, sig_thr_frac=0.06, p_cross=0.25,
                mu=0.55, arity_pen=0.0, sim_pen=1.5, min_n=120, max_cover=1.0,
                max_keep=32, seed=0, l1_standardize=True, Xd=None,
                verbose=True):
    rng = np.random.default_rng(seed)
    alpha = Alphabet(n_thresholds).fit(obs, split_vars)
    sig = {j: sig_thr_frac * (alpha.hi[j] - alpha.lo[j]) for j in split_vars}
    # `Xd` overrides the regressors so the GUARDS can be evolved over a feature
    # set the LAWS never see -- e15 gates on a learned V_hat column without
    # handing the affine law a 25th parameter, which would confound the two.
    Xd = design_matrix(obs) if Xd is None else Xd

    theta_d = fit_value_law(Xd, U, M)
    rd = U - Xd @ theta_d
    per_default = np.einsum("iq,iqm,im->i", rd, M, rd)
    base = per_default.sum()

    def rand_clause(hot=None):
        k = 1 + int(rng.integers(max_arity))
        return [_rand_literal(rng, alpha, obs, hot) for _ in range(k)]

    def mutate(cl, hot):
        cl = [lit[:] for lit in cl]
        r = rng.random()
        if r < 0.20 and len(cl) < max_arity:               # specialize
            cl.append(_rand_literal(rng, alpha, obs, hot))
            return cl
        if r < 0.35 and len(cl) > 1:                       # generalize
            del cl[int(rng.integers(len(cl)))]
            return cl
        i = int(rng.integers(len(cl)))
        if rng.random() < 0.15:                            # hot-anchored newborn
            cl[i] = _rand_literal(rng, alpha, obs, hot)
            return cl
        j = cl[i][0]                                       # THRESHOLD DRIFT
        cl[i][1] = float(np.clip(cl[i][1] + rng.normal(0, sig[j]),
                                 alpha.lo[j], alpha.hi[j]))
        if rng.random() < 0.10:
            cl[i][2] = not cl[i][2]
        return cl

    def crossover(a, b):
        pool = [l[:] for l in a] + [l[:] for l in b]
        rng.shuffle(pool)
        return pool[:min(int(rng.integers(1, max_arity + 1)), len(pool))]

    clauses = [rand_clause() for _ in range(C)]
    best, best_state, trace = np.inf, None, []

    for g in range(generations):
        clauses = [dedupe_literals(c) for c in clauses]
        Mm = match_matrix(clauses, obs)
        order, wl1 = l1_order(Mm, per_default, max_keep=max_keep,
                              standardize=l1_standardize)
        if len(order) == 0:
            order = priority_order(Mm, max_cover)
        assign = _assign(Mm, order)
        laws, per = _laws_and_loss(assign, C, Xd, U, M, min_n)
        tot = per.sum()
        trace.append(tot / base)
        if tot < best:
            best = tot
            best_state = ([[l[:] for l in c] for c in clauses], laws,
                          assign.copy(), order.copy())
        util = _clause_utility(assign, per, per_default, C)
        fit, sim = _fitness(util, Mm, [len(c) for c in clauses], arity_pen, sim_pen)

        if verbose and g % 15 == 0:
            print(f"  [gen {g:3d}] loss {tot:.0f} ({100*tot/base:.1f}% of one "
                  f"law)  used {len(laws)-1:2d}/{C}  sim {sim:.2f}  "
                  f"arity {np.mean([len(c) for c in clauses]):.1f}")
        if g == generations - 1:
            break

        hot = np.flatnonzero(per >= np.quantile(per, 0.75))
        p = fit / fit.sum()
        parents = rng.choice(C, size=C, p=p)
        nxt = []
        for pa in parents:
            r = rng.random()
            if r < p_cross:
                nxt.append(crossover(clauses[pa], clauses[int(rng.choice(C, p=p))]))
            elif r < p_cross + mu:
                nxt.append(mutate(clauses[pa], hot))
            else:
                nxt.append([l[:] for l in clauses[pa]])
        clauses = nxt

    cl, laws, assign, order = best_state
    keep = [c for c in order if c in laws and (assign == c).sum() >= min_n]

    # THE ARITY FUNNEL. Clause arity is gated three times, and reporting only
    # the survivors cannot say which gate binds:
    #   population -> what evolution actually proposed and kept alive;
    #   selected   -> what the L1 readout gave positive weight;
    #   surviving  -> what still held >= min_n rows after the arms above it in
    #                 the Fallback took theirs. High-arity clauses are narrow by
    #                 construction, so min_n is an arity ceiling in disguise.
    def _mar(ix):
        return float(np.mean([len(cl[c]) for c in ix])) if len(ix) else 0.0
    return dict(clauses=[cl[c] for c in keep], laws=[laws[c] for c in keep],
                default=laws[-1], loss=best, base=base, names=names,
                cover=[float((assign == c).mean()) for c in keep],
                arity_pop=float(np.mean([len(c) for c in cl])),
                arity_sel=_mar(order), arity_keep=_mar(keep),
                n_pop=len(cl), n_sel=len(order), n_keep=len(keep),
                trace=np.array(trace))


# --------------------------------------------------------------------- emit
class BankPolicy:
    def __init__(self, bank):
        self.b = bank

    def act(self, obs):
        Xd = design_matrix(obs)
        out = Xd @ self.b["default"]
        done = np.zeros(len(obs), dtype=bool)
        for cl, th in zip(self.b["clauses"], self.b["laws"]):
            m = _match(cl, obs) & ~done
            if m.any():
                out[m] = Xd[m] @ th
                done |= m
        n = np.linalg.norm(out, axis=1, keepdims=True)
        # A DIRECTION, NOT A VECTOR. Least squares shrinks the magnitude of
        # its prediction toward the mean, and a shrunk action is a slower
        # agent: measured on NestWorld, clipping to the disk instead of
        # normalising cost 5.0 return units on an identical bank.
        return out / np.maximum(n, 1e-9)


def clause_text(cl, names):
    return " AND ".join(
        f"{names[j]}{'<=' if neg else '>'}{thr:.3f}" for j, thr, neg in cl)


def bank_bt(bank):
    names = bank["names"]
    out = ["Fallback"]
    for cl in bank["clauses"]:
        out.append(f"|-- Sequence[ {clause_text(cl, names)} , Action ]")
    out.append("\\-- Action(default)")
    return "\n".join(out)
