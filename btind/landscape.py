"""Partitioning by the shape of the value mountain, not by the observations.

THE CONSTRAINT THAT SHAPES THIS FILE. A guard in an emitted behaviour tree can
only mention what the agent knows AT THE TICK. That splits every landscape
quantity into two classes and they cannot be used the same way:

    GUARD-USABLE   V_hat(obs)          how high on the mountain
                   leverage(obs)       lambda_max of M(obs) -- does the action
                                       matter here at all
                   both are pure functions of the current observation, so a
                   Fallback can test them.

    PROPOSAL-ONLY  drift_k(s_t)  = (V_hat(s_{t+k}) - V_hat(s_t)) / k
                   advantage(s_t) = r_t + gamma V_hat(s_{t+1}) - V_hat(s_t)
                   both look FORWARD. They are labels on training rows, not
                   features: a guard that mentions drift is not causal and
                   cannot be executed.

So the loop proposes in landscape space and emits in observation space:
`translate_cut` takes a subset defined by a forward-looking quantity and finds
the observation literal that best reproduces it. The artifact stays readable and
causal; the landscape only ever decides WHERE to look.

THE FOUR REGIMES, and which ones deserve an arm.

    steep            |drift| high          the side of the mountain
    good flat        |drift| low, V high   the summit -- already solved
    bad flat         |drift| low, V low    a basin

and crossed with leverage, which is what makes the difference operational:

                        leverage LOW              leverage HIGH
    steep          the slope carries you       an arm pays: the action
                   regardless                  decides how fast you climb
    good flat      DISREGARD: nothing to       rare; usually means the summit
                   gain and nothing to lose    is narrow and must be held
    bad flat       DISREGARD: doomed. Return   an arm pays: a RECOVERABLE
                   is flat in u because no     basin the current law is
                   action helps                mishandling

That cross is the point of the file. e15 gated on V_hat alone and lost (8.79
against 10.53) because a level cannot tell those two bad-flat cells apart, and
half of the states it gated on were the doomed kind, where a specialised law
buys exactly nothing and costs a region boundary.
"""
import numpy as np

from .collect import design_matrix

_EPS = 1e-12


# ------------------------------------------------------------------ rollout
def landscape_rollout(env, policy, vhat, rng, n_ep=1200, T=200, k=10):
    """Roll the controller and record the value profile along the way.

    Returns flat arrays over the live steps of every episode: states, obs,
    timestep, occupancy weight gamma^t, V_hat, forward drift, advantage.
    """
    s = env.sample_states(n_ep, rng)
    d = env.observe(s).shape[1]
    OB = np.zeros((n_ep, T, d))
    ST = np.zeros((n_ep, T, s.shape[1]))
    RW = np.zeros((n_ep, T))
    AL = np.zeros((n_ep, T), bool)
    alive = np.ones(n_ep, bool)
    for t in range(T):
        AL[:, t] = alive
        ST[:, t] = s
        ob = env.observe(s)
        OB[:, t] = ob
        s, r, done = env.step(s, policy.act(ob))
        RW[:, t] = r * alive
        alive &= ~done
        if not alive.any():
            break

    V = vhat.predict(OB.reshape(-1, d)).reshape(n_ep, T)
    ar = np.arange(T)
    ahead = np.minimum(ar + k, T - 1)
    nxt = np.minimum(ar + 1, T - 1)
    drift = (V[:, ahead] - V) / k
    adv = RW + env.gamma * V[:, nxt] - V
    # a step whose window runs past death has no forward value: mask it out
    ok = AL & AL[:, ahead] & (ar + k < T)[None, :]

    live = AL.reshape(-1)
    t_all = np.tile(ar, (n_ep, 1)).reshape(-1)
    return dict(state=ST.reshape(-1, s.shape[1])[live], obs=OB.reshape(-1, d)[live],
                t=t_all[live], w=env.gamma ** t_all[live],
                V=V.reshape(-1)[live], drift=drift.reshape(-1)[live],
                adv=adv.reshape(-1)[live], drift_ok=ok.reshape(-1)[live])


def subsample(L, n, rng):
    m = len(L["obs"])
    i = rng.choice(m, size=min(n, m), replace=False)
    return {k: (v[i] if isinstance(v, np.ndarray) and len(v) == m else v)
            for k, v in L.items()}


# ------------------------------------------------------------------ features
def leverage(qhat, obs):
    """lambda_max of M(obs): how sharply return falls off as the action drifts.

    The on-policy replacement for the planner's `gap`. `gap` asks how much the
    action mattered to a replanner at a teleported state; this asks how much it
    matters to THIS controller where it actually is.
    """
    _, M, _ = qhat.coef(obs)
    return np.linalg.eigvalsh(M)[:, -1]


def guard_features(obs, vhat, qhat, names):
    """[obs, V_hat, leverage] -- everything a guard is allowed to mention."""
    V = vhat.predict(obs)[:, None]
    lam = leverage(qhat, obs)[:, None]
    return np.hstack([obs, V, lam]), list(names) + ["V_hat", "leverage"]


# ------------------------------------------------------------------- regimes
def regimes(L, lam, q_flat=0.25, q_steep=0.75):
    """Label every row steep / good flat / bad flat, crossed with leverage."""
    ad = np.abs(L["drift"])
    ok = L["drift_ok"]
    lo, hi = np.quantile(ad[ok], [q_flat, q_steep])
    v_med = np.median(L["V"])
    lam_med = np.median(lam)
    flat, steep = ok & (ad <= lo), ok & (ad >= hi)
    return dict(steep=steep, flat=flat,
                good_flat=flat & (L["V"] >= v_med),
                bad_flat=flat & (L["V"] < v_med),
                high_lev=lam >= lam_med, lam_med=float(lam_med),
                v_med=float(v_med), flat_cut=float(lo), steep_cut=float(hi))


def regime_table(L, lam, g, R=None):
    """Occupancy mass and gradient mass per regime cell.

    `g` is the per-row residual DPG gradient. The last column is what the cell
    is WORTH: sum of w_i ||g_i||, the magnitude of the policy-gradient signal
    sitting in it. A cell with mass and no gradient is already solved; a cell
    with gradient and no mass is not worth an arm; only mass AND gradient pays.
    """
    R = regimes(L, lam) if R is None else R
    w, gn = L["w"], np.linalg.norm(g, axis=1)
    tot_w = w.sum()
    rows = []
    for name in ("steep", "good_flat", "bad_flat"):
        for lev, tag in ((R["high_lev"], "high"), (~R["high_lev"], "low")):
            m = R[name] & lev
            if not m.any():
                rows.append(dict(regime=name, lev=tag, mass=0.0, V=0.0,
                                 drift=0.0, lam=0.0, gnorm=0.0, signal=0.0))
                continue
            rows.append(dict(
                regime=name, lev=tag, mass=float(w[m].sum() / tot_w),
                V=float(np.average(L["V"][m], weights=w[m])),
                drift=float(np.average(L["drift"][m], weights=w[m])),
                lam=float(np.average(lam[m], weights=w[m])),
                gnorm=float(np.average(gn[m], weights=w[m])),
                signal=float((w[m] * gn[m]).sum() / tot_w)))
    return rows


def worth_an_arm(R, lam_min_frac=1.0):
    """Rows where an arm could pay: leverage above median, not a good flat.

    This is the DISREGARD rule, applied before any rollout is spent. A doomed
    basin (flat, low value, no leverage) and a solved summit (flat, high value,
    no leverage) are both states where every law scores the same, so a boundary
    drawn through them can only cost readability.
    """
    return R["high_lev"] & ~(R["good_flat"] & ~R["high_lev"])


# -------------------------------------------------------- gradient splitting
def grad_split(Z, g, w, cand_cols, n_cand=40, min_frac=0.08):
    """Separation of the MEAN gradient across a cut. Kept, but not used.

    This is the obvious criterion and it is wrong here, in a way worth keeping
    on file. It asks whether the two sides want different average actions -- and
    on ForageWorld the answer is always yes along the threat bearing, because
    inside the flee region the best direction rotates with `bear_threat`. It
    proposed exactly that cut on every iteration of a first run (gain ~4e4 on
    `bear_threat_y`), and every one lost its rollout by about a return unit.

    The reason is that an affine law ALREADY has `bear_threat_x/y` as regressors,
    so the disagreement it detects is disagreement the incumbent law can express
    without a boundary. A split criterion has to price what one affine law can
    do, not what one constant can. `grad_split_affine` does; this does not.
    """
    n = len(Z)
    S_tot, W_tot = (w[:, None] * g).sum(0), w.sum()
    parent = float(S_tot @ S_tot) / max(W_tot, _EPS)
    lo, hi = int(min_frac * n), int((1 - min_frac) * n)
    best = (None, None, -np.inf)
    if hi <= lo:
        return best
    for j in cand_cols:
        o = np.argsort(Z[:, j], kind="mergesort")
        zs = Z[o, j]
        Sc = np.cumsum((w[o, None] * g[o]), axis=0)
        Wc = np.cumsum(w[o])
        for kk in np.unique(np.linspace(lo, hi - 1, n_cand).astype(int)):
            SL, WL = Sc[kk], Wc[kk]
            SR, WR = S_tot - SL, W_tot - WL
            if WL <= _EPS or WR <= _EPS:
                continue
            gain = (SL @ SL) / WL + (SR @ SR) / WR - parent
            if gain > best[2]:
                best = (int(j), float(zs[kk]), float(gain))
    return best


def grad_split_affine(X, Z, g, w, cand_cols, n_cand=40, min_frac=0.08,
                      ridge=1e-6):
    """Best cut by gradient signal that TWO affine laws can explain and one cannot.

    Within a region the best affine correction to the laws is the weighted
    least-squares fit of the residual gradient field g on the regressors X, and
    the signal it captures is b^T A^-1 b with

        A = sum_i w_i x_i x_i^T        b = sum_i w_i x_i g_i^T

    so a cut is worth proposing exactly when splitting raises that:

        gain = b_L^T A_L^-1 b_L + b_R^T A_R^-1 b_R - b^T A^-1 b

    the same XGBoost-style accumulate-and-cumsum as `best_value_split`, with the
    gradient field in place of the label and occupancy in place of uniform
    weight. Because the parent term is itself an affine fit, disagreement the
    incumbent law can already absorb scores ZERO -- which is the whole
    correction over `grad_split`.
    """
    n, d = X.shape
    Ai = w[:, None, None] * np.einsum("ip,iq->ipq", X, X)
    Bi = w[:, None, None] * np.einsum("ip,iq->ipq", X, g)
    A_tot, B_tot = Ai.sum(0), Bi.sum(0)
    I = ridge * np.eye(d)

    def sc(A, B):
        try:
            return float((B * np.linalg.solve(A + I, B)).sum())
        except np.linalg.LinAlgError:
            return 0.0

    parent = sc(A_tot, B_tot)
    lo, hi = int(min_frac * n), int((1 - min_frac) * n)
    best = (None, None, -np.inf)
    if hi <= lo:
        return best
    for j in cand_cols:
        o = np.argsort(Z[:, j], kind="mergesort")
        zs = Z[o, j]
        Ac, Bc = np.cumsum(Ai[o], 0), np.cumsum(Bi[o], 0)
        for kk in np.unique(np.linspace(lo, hi - 1, n_cand).astype(int)):
            gain = (sc(Ac[kk], Bc[kk])
                    + sc(A_tot - Ac[kk], B_tot - Bc[kk]) - parent)
            if gain > best[2]:
                best = (int(j), float(zs[kk]), float(gain))
    return best


def translate_cut(obs, target, w, cand_cols, n_cand=40):
    """Closest observation literal to a subset defined any way at all.

    Proposals may be found on a forward-looking quantity; a guard cannot mention
    one. This finds the (feature, threshold, sense) whose half-space best
    reproduces `target` under occupancy weighting, scored by weighted balanced
    accuracy so a rare regime is not approximated by "everything".
    """
    pos, neg_m = target, ~target
    Wp, Wn = w[pos].sum(), w[neg_m].sum()
    if Wp <= _EPS or Wn <= _EPS:
        return None
    best = (None, None, None, -np.inf)
    for j in cand_cols:
        qs = np.quantile(obs[:, j], np.linspace(0.05, 0.95, n_cand))
        for thr in np.unique(qs):
            hit = obs[:, j] > thr
            tp = w[pos & hit].sum() / Wp
            fp = w[neg_m & hit].sum() / Wn
            for sense_neg, tpr, fpr in ((False, tp, fp), (True, 1 - tp, 1 - fp)):
                bal = 0.5 * (tpr + (1 - fpr))
                if bal > best[3]:
                    best = (int(j), float(thr), bool(sense_neg), float(bal))
    return best


# ------------------------------------------------------------- proposals
def propose(bank, assign_fn, Z, names, g, w, mask_ok, arms=None, per_arm=3,
            min_n=400, max_arity=3, n_cand=40, X=None):
    """Candidate clauses, one family per arm, ranked by gradient separation.

    Each proposal SPECIALISES the arm it came from -- the arm's own literals
    plus the new cut -- so it lands directly above that arm in the Fallback and
    inherits exactly the rows it was fitted to explain. A cut proposed against
    an arm and then inserted anywhere else is priced against a counterfactual
    that never happens.

    `mask_ok` is the disregard rule from `worth_an_arm`: rows where no law can
    pay are removed before the scan, not rejected after a rollout.
    """
    a = assign_fn(bank["clauses"], Z)
    out = []
    ids = range(-1, len(bank["clauses"])) if arms is None else arms
    cand_cols = list(range(Z.shape[1]))
    for c in ids:
        rows = np.flatnonzero((a == c) & mask_ok)
        if len(rows) < 2 * min_n:
            continue
        base = [] if c < 0 else [l[:] for l in bank["clauses"][c]]
        if len(base) >= max_arity:
            continue
        used = {l[0] for l in base}
        cols = [j for j in cand_cols if j not in used]
        for _ in range(per_arm):
            j, thr, gain = (
                grad_split(Z[rows], g[rows], w[rows], cols, n_cand=n_cand)
                if X is None else
                grad_split_affine(X[rows], Z[rows], g[rows], w[rows], cols,
                                  n_cand=n_cand))
            if j is None or gain <= 0:
                break
            cols = [cc for cc in cols if cc != j]
            for sense in (False, True):
                out.append(dict(arm=int(c), gain=float(gain),
                                clause=base + [[int(j), float(thr), sense]],
                                col=int(j)))
    return sorted(out, key=lambda d: -d["gain"])


def propose_from_regimes(R, Z, w, mask_ok, cand_cols, base_clauses=(),
                         targets=("bad_flat", "steep")):
    """Clauses aimed at a regime CELL, translated into a testable literal.

    The cell is the hypothesis -- "a recoverable basin the current law mishandles"
    is `bad_flat & high leverage` -- and `translate_cut` turns it into the single
    half-space that best reproduces it under occupancy weighting. What gets
    emitted is an ordinary literal on an ordinary feature; the regime only chose
    which subset to aim at.
    """
    out = []
    for name in targets:
        m = R[name] & R["high_lev"] & mask_ok
        if m.sum() < 50 or (~m).sum() < 50:
            continue
        t = translate_cut(Z, m, w, cand_cols)
        if t is None:
            continue
        j, thr, neg, bal = t
        out.append(dict(arm=-1, gain=float(bal), col=int(j), regime=name,
                        clause=[l[:] for l in base_clauses] + [[j, thr, neg]]))
    return out


def uses_landscape(clauses, n_obs):
    return any(l[0] >= n_obs for cl in clauses for l in cl)


class LandscapeBank:
    """Executable bank whose GUARDS may mention V_hat and leverage.

    Laws stay on `obs` alone, deliberately: e15 showed that letting the law see
    V_hat as a 25th regressor confounds "the partition can reference value" with
    "the control law got a strong nonlinear feature", and only the first is the
    claim being tested. The augmented columns are computed only when some clause
    actually mentions one -- a bank of plain observation literals pays nothing
    for the option.
    """

    def __init__(self, bank, vhat, qhat, n_obs):
        self.b, self.v, self.q, self.n_obs = bank, vhat, qhat, n_obs
        self.aug_needed = uses_landscape(bank["clauses"], n_obs)

    def _guard_mat(self, obs):
        if not self.aug_needed:
            return obs
        return np.hstack([obs, self.v.predict(obs)[:, None],
                          leverage(self.q, obs)[:, None]])

    def act(self, obs):
        Z = self._guard_mat(obs)
        Xd = design_matrix(obs)
        out = Xd @ self.b["default"]
        done = np.zeros(len(obs), bool)
        for cl, th in zip(self.b["clauses"], self.b["laws"]):
            m = _match_cols(cl, Z) & ~done
            if m.any():
                out[m] = Xd[m] @ th
                done |= m
        n = np.linalg.norm(out, axis=1, keepdims=True)
        # A DIRECTION, NOT A VECTOR. Least squares shrinks the magnitude of
        # its prediction toward the mean, and a shrunk action is a slower
        # agent: measured on NestWorld, clipping to the disk instead of
        # normalising cost 5.0 return units on an identical bank.
        return out / np.maximum(n, 1e-9)


def _match_cols(clause, Z):
    m = np.ones(Z.shape[0], bool)
    for j, thr, neg in clause:
        t = Z[:, j] > thr
        m &= (~t if neg else t)
    return m


def default_clause(clauses):
    """The region the Fallback reaches by omission, as an explicit conjunction.

    A Fallback needs an unconditional last child or it can return FAILURE, and a
    controller must act every tick -- so a default arm is structurally required.
    What is NOT required is leaving it anonymous, and on this task the anonymous
    arm is the interesting one: with `d_threat<=0.103` and `d_threat>0.241`
    above it, the default owns the band 0.103 < d_threat <= 0.241, which is the
    flee radius the whole controller is built around, named by subtraction.

    Two costs to that. A reader has to complement the arms above to see what
    triggers it, and -- worse for the loop -- the band has no literals of its
    own, so `boundary_grad` cannot move its edges independently: sliding the
    flee threshold moves where fleeing ends AND where the band starts, as one.

    Negating a disjunction of conjunctions gives a conjunction only when every
    clause is a single literal, which is the shape this pipeline emits; anything
    wider stays anonymous rather than being approximated, because an approximate
    guard would change what the tree does.
    """
    if not clauses or any(len(c) != 1 for c in clauses):
        return None
    return [[j, thr, not neg] for c in clauses for (j, thr, neg) in c]


def materialise_default(bank):
    """Give the band its own arm. Semantics unchanged: it matches exactly the
    rows nothing above it matched, and carries the same law it already had."""
    dc = default_clause(bank["clauses"])
    if dc is None:
        return bank, False
    from .memory import insert_arm
    return insert_arm(bank, dc, bank["default"], len(bank["clauses"])), True
