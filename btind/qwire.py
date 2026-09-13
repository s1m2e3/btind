"""Wiring the critic back into the two things a bank is made of.

The bank is a pair (guards, laws) and until now each half was fitted against a
quantity that is not the objective:

    laws     fit_value_law on (u*, M) from the PLANNER, at teleported states,
             every row weighted equally
    guards   evolved against the same loss, or -- in rollout_select -- proposed
             blind and accepted only by a rollout, with no per-arm signal at all

With Q^pi in hand both halves get the same, correct signal, and each is still a
PROPOSAL: nothing enters the bank without clearing the paired rollout test.

  LAWS. The region-restricted deterministic policy gradient is

      grad_{theta_c} J = E_{s ~ d_gamma^pi} [ 1{s in R_c} x(s) dQ^pi/du ]

  and with Q quadratic in u, dQ/du = M(x)(u*(x) - u), so a zero of the gradient
  is exactly the minimiser of the M-weighted value loss `fit_value_law` already
  solves -- with two changes. The rows are the states the controller occupies,
  weighted by gamma^t (occupancy), and (u*, M) come from Q^pi rather than Q*.
  The solve does not change; what it is solving for does.

  GUARDS. A threshold is not a regression parameter, so it has no residual. Its
  derivative is a BOUNDARY term: moving tau by dtau hands the states in a thin
  shell across the cut to a different arm, and what that is worth is the value
  difference between the two arms' laws ON THE SHELL,

      dJ/dtau = int_{x_j = tau} d_gamma^pi(s) [ Q(s, theta_new) - Q(s, theta_cur) ]

  which is estimable by simply evaluating both laws' actions under the critic at
  the states that lie in the shell. Nothing is differentiated through the sim.

WHICH ARM INHERITS. In a Fallback the answer is not "the default": it is the
first clause below `c` that matches, which is what `_owner` computes by rerunning
priority assignment with `c` removed. Pricing a clause against the global default
-- what `_clause_utility` does -- is the priority-blindness that made in-sample
utility and measured return disagree.
"""
import numpy as np

from .collect import design_matrix
from .evotm import _match
from .policies import evaluate
from .rollout_select import _Bank
from .valuesplit import fit_value_law


# --------------------------------------------------------------- occupancy
def onpolicy_states(env, policy, n_states, rng, n_ep=400, T=200, gamma=None):
    """Visited states WITH their timestep, so gamma^t is available as a weight.

    `dagger.collect_onpolicy_states` drops t: a state the controller reaches at
    t=3 and one it reaches at t=120 enter the law fit with the same weight,
    although the discounted objective prices the second at gamma^117 of the
    first. The weight is what the objective says it is -- and MEASURED, on
    ForageWorld, it does almost nothing: over four runs the same law step is
    worth +1.78 with gamma^t weights and +1.86 with uniform ones, a difference
    well inside the run-to-run spread. Median visit time is ~75 steps, so the
    weight mostly rescales rows that the affine law was fitting the same way
    regardless. Kept because it is correct, reported because it is not the
    reason anything improved.
    """
    gamma = env.gamma if gamma is None else gamma
    s = env.sample_states(n_ep, rng)
    alive = np.ones(n_ep, bool)
    pool, tt = [], []
    for t in range(T):
        if not alive.any():
            break
        pool.append(s[alive].copy())
        tt.append(np.full(int(alive.sum()), t))
        s, _, done = env.step(s, policy.act(env.observe(s)))
        alive &= ~done
    S = np.concatenate(pool, 0)
    t = np.concatenate(tt)
    take = min(n_states, len(S))
    i = rng.choice(len(S), size=take, replace=False)
    return S[i], t[i], gamma ** t[i]


# ------------------------------------------------------------- assignment
def regressors(bank, obs, Z):
    """The matrix a law multiplies.

    A law reads `z` once the bank has memory columns, and `obs` before that.
    Getting this wrong is silent: the shapes still conform for a while, and the
    law simply binds its coefficients to the wrong features.
    """
    return design_matrix(Z if bank.get("laws_on_z") and Z is not None else obs)


def assign(clauses, obs):
    """First matching clause in list order wins. -1 = the default arm."""
    out = np.full(len(obs), -1, int)
    for c in range(len(clauses) - 1, -1, -1):
        out[_match(clauses[c], obs)] = c
    return out


def _owner(clauses, obs, drop):
    """Who would own each row if clause `drop` were not in the Fallback."""
    out = np.full(len(obs), -1, int)
    for c in range(len(clauses) - 1, -1, -1):
        if c == drop:
            continue
        out[_match(clauses[c], obs)] = c
    return out


def _law(bank, c):
    return bank["default"] if c < 0 else bank["laws"][c]


def act_of(bank, Xd, arms):
    """The action each row would receive from the arm named in `arms`."""
    u = Xd @ bank["default"]
    for c in np.unique(arms[arms >= 0]):
        m = arms == c
        u[m] = Xd[m] @ bank["laws"][c]
    n = np.linalg.norm(u, axis=1, keepdims=True)
    return u / np.maximum(n, 1.0)


# ------------------------------------------------------------------- laws
def refit_laws(bank, obs, U, M, w=None, min_n=150, Xd_fn=design_matrix,
               Z=None):
    """Per-arm closed-form solve on the rows that arm owns, weighted by w.

    w enters as w_i * M_i because the loss is r' (w M) r -- the occupancy weight
    and the curvature multiply, they do not compete.
    """
    Xd = regressors(bank, obs, Z)
    a = assign(bank["clauses"], obs if Z is None else Z)
    Mw = M if w is None else (w[:, None, None] * M)
    laws, n_refit = [], 0
    for c in range(len(bank["clauses"])):
        idx = np.flatnonzero(a == c)
        if len(idx) >= min_n:
            laws.append(fit_value_law(Xd[idx], U[idx], Mw[idx]))
            n_refit += 1
        else:
            laws.append(bank["laws"][c])
    idx = np.flatnonzero(a == -1)
    default = (fit_value_law(Xd[idx], U[idx], Mw[idx]) if len(idx) >= min_n
               else bank["default"])
    out = dict(bank, laws=laws, default=default)
    out["n_refit"] = n_refit
    return out


def law_gradient_norm(bank, obs, qhat, w=None, Z=None):
    """||grad_theta_c J|| per arm, in the units of the DPG expression.

    Diagnostic only -- it says which arms the critic thinks are mis-fitted, and
    it is the quantity that should fall after a refit.
    """
    Xd = regressors(bank, obs, Z)
    a = assign(bank["clauses"], obs if Z is None else Z)
    w = np.ones(len(obs)) if w is None else w
    g = {}
    for c in list(range(len(bank["clauses"]))) + [-1]:
        idx = np.flatnonzero(a == c)
        if not len(idx):
            continue
        u = np.clip(Xd[idx] @ _law(bank, c), -1, 1)
        gr = qhat.grad_u(obs[idx], u)
        g[c] = float(np.linalg.norm(
            (w[idx, None] * Xd[idx]).T @ gr) / max(len(idx), 1))
    return g


# -------------------------------------------------------------- boundaries
def boundary_grad(bank, obs, w, qhat, eps_frac=0.05, lo_hi=None,
                  Z=None):
    """dJ/dtau for every literal in the bank. Positive => raise the threshold.

    For `x_j > tau`, raising tau EVICTS the shell just above tau from this
    clause, so the integrand is Q(inheritor) - Q(this clause). For `x_j <= tau`
    it ADMITS that same shell, so the sign flips and the inheritor is this
    clause. Rows are counted only where the clause's OTHER literals hold --
    a shell state that fails another literal never changes hands.
    """
    Xd = regressors(bank, obs, Z)
    Z = obs if Z is None else Z
    a = assign(bank["clauses"], Z)
    out = []
    for c, cl in enumerate(bank["clauses"]):
        alt = _owner(bank["clauses"], Z, c)
        for k, (j, thr, neg) in enumerate(cl):
            lo, hi = (lo_hi[j] if lo_hi else
                      (float(Z[:, j].min()), float(Z[:, j].max())))
            eps = max(eps_frac * (hi - lo), 1e-6)
            others = np.ones(len(Z), bool)
            for kk, lit in enumerate(cl):
                if kk != k:
                    t = Z[:, lit[0]] > lit[1]
                    others &= (~t if lit[2] else t)
            shell = others & (Z[:, j] > thr) & (Z[:, j] <= thr + eps)
            if neg:                       # x_j <= tau : raising tau ADMITS
                free = shell & (a != c) & ~_first_before(bank, Z, c)
                cur, new = a, np.full(len(obs), c)
                rows = np.flatnonzero(free)
            else:                         # x_j >  tau : raising tau EVICTS
                cur, new = a, alt
                rows = np.flatnonzero(shell & (a == c))
            if len(rows) < 8:
                out.append(dict(c=c, k=k, j=j, thr=float(thr), neg=bool(neg),
                                grad=0.0, n=int(len(rows)), eps=float(eps)))
                continue
            cur, new = cur[rows], new[rows]
            q_cur = qhat.q(obs[rows], act_of(bank, Xd[rows], cur))
            q_new = qhat.q(obs[rows], act_of(bank, Xd[rows], new))
            g = float((w[rows] * (q_new - q_cur)).sum() / (len(obs) * eps))
            out.append(dict(c=c, k=k, j=j, thr=float(thr), neg=bool(neg),
                            grad=g, n=int(len(rows)), eps=float(eps)))
    return sorted(out, key=lambda d: -abs(d["grad"]))


def _first_before(bank, obs, c):
    """True where some clause with priority above `c` already matches."""
    m = np.zeros(len(obs), bool)
    for cc in range(c):
        m |= _match(bank["clauses"][cc], obs)
    return m


def move_threshold(bank, c, k, new_thr):
    cl = [[l[:] for l in cc] for cc in bank["clauses"]]
    cl[c][k][1] = float(new_thr)
    return dict(bank, clauses=cl)


# ------------------------------------------------------------ the two steps
def _score(env, bank, n_ep, T, seed, pol_fn=None):
    """Per-episode returns, with the kernel re-seeded before every candidate.

    `evaluate`'s seed fixes the starting states, but the food respawn draw lives
    in the kernel's global numpy stream and numba keeps that stream PER THREAD
    inside a prange, so re-seeding does not make a rollout reproducible: two
    evaluations of the same bank still differ, by 0.57 return units on 300
    episodes and 0.28 on 1000. Pairing is therefore through the starting state
    only -- but the paired standard error measures exactly this residual (over
    six repeats of an identical pair, |mean d| / se = 0.85), so the z-test stays
    calibrated and the fix is episodes, not seeds. 1000 is the working point.
    """
    env.seed_kernels(seed)
    pol = (_Bank(bank["clauses"], bank["laws"], bank["default"], design_matrix)
           if pol_fn is None else pol_fn(bank))
    return evaluate(env, pol, n_ep=n_ep, T=T, seed=seed)["G"]


def paired_accept(env, cand, ref_G, n_ep, T, seed, z=2.0, pol_fn=None):
    """Accept only if the paired per-episode gain clears z standard errors."""
    g = _score(env, cand, n_ep, T, seed, pol_fn)
    d = g - ref_G
    se = d.std() / np.sqrt(len(d))
    return (d.mean() > z * max(se, 1e-12)), float(d.mean()), g


def blend(bank, other, alpha):
    """(1-alpha) * this bank's laws + alpha * the other's. alpha=1 is a full step."""
    return dict(bank,
                laws=[(1 - alpha) * t + alpha * o
                      for t, o in zip(bank["laws"], other["laws"])],
                default=(1 - alpha) * bank["default"] + alpha * other["default"])


def dpg_step(bank, obs, w, qhat, eta, min_n=150, grad_fn=None,
             Z=None):
    """theta_c <- theta_c + eta * sum_i w_i x_i (dQ/du)_i^T, per arm.

    The raw region-restricted DPG. It consumes only the DIRECTION field of the
    gradient source, never M^-1, and that is the whole difference between a step
    that works and one that does not. Averaged over four rebuilds of the e18
    incumbent, one step of this at eta 0.02 is worth +1.78 held-out return; the
    closed-form refit to the SAME critic argmax is worth -14.55. Inverting a
    curvature whose direction agrees with a re-probed reference field at median
    cos ~0.44 compounds the error in b and M together, and lands the controller
    somewhere no line search was consulted about. Using the direction only, with
    the distance chosen by measured return, cannot do that: the worst outcome of
    a bad direction is a rejected candidate.
    """
    Xd = regressors(bank, obs, Z)
    a = assign(bank["clauses"], obs if Z is None else Z)
    laws = list(bank["laws"])
    default = bank["default"]
    for c in list(range(len(bank["clauses"]))) + [-1]:
        idx = np.flatnonzero(a == c)
        if len(idx) < min_n:
            continue
        u = np.clip(Xd[idx] @ _law(bank, c), -1, 1)
        gr = (qhat.grad_u(obs[idx], u) if grad_fn is None
              else grad_fn(idx, u))
        g = (w[idx, None] * Xd[idx]).T @ gr / w[idx].sum()
        if c < 0:
            default = default + eta * g
        else:
            laws[c] = laws[c] + eta * g
    return dict(bank, laws=laws, default=default)


def law_step(env, bank, obs, w, qhat, cur_G, mode="newton",
             alphas=(0.25, 0.5, 1.0), etas=(0.05, 0.15, 0.4), n_ep=300, T=200,
             seed=777, z=2.0, min_n=150, pol_fn=None, Z=None):
    """Move the laws toward Q^pi; keep the best step that clears the paired test.

    Two step shapes, both proposals:
      newton  the closed-form refit to (u*, M) read off the critic, damped by
              alpha -- a full Gauss-Newton step when alpha=1
      grad    the raw DPG direction with a step size

    The line search is over MEASURED return, so a critic that is right in
    direction and wrong in scale still produces a usable move: it decides where,
    the rollout decides how far.
    """
    best, best_G, log = bank, cur_G, []
    if mode == "newton":
        U, M = qhat.targets(obs)
        full = refit_laws(bank, obs, U, M, w=w, min_n=min_n, Z=Z)
        cands = [(a, blend(bank, full, a)) for a in alphas]
        n_refit = full["n_refit"]
    else:
        cands = [(e, dpg_step(bank, obs, w, qhat, e, min_n, Z=Z))
                 for e in etas]
        n_refit = len(bank["clauses"]) + 1
    for step, cand in cands:
        ok, d, g = paired_accept(env, cand, cur_G, n_ep, T, seed, z, pol_fn)
        log.append(dict(kind="laws", mode=mode, step=float(step),
                        accepted=bool(ok), delta=d))
        if ok and d > (best_G.mean() - cur_G.mean()):
            best, best_G = cand, g
    return best, best_G, dict(mode=mode, n_refit=n_refit,
                              accepted=any(l["accepted"] for l in log),
                              best=max(l["delta"] for l in log), tries=log)


def guard_step(env, bank, obs, w, qhat, cur_G, n_try=4, steps=(0.5, 1.0, 2.0),
               n_ep=300, T=200, seed=777, z=2.0, eps_frac=0.05, lo_hi=None,
               pol_fn=None, Z=None):
    """Move the threshold the boundary term ranks highest. One move per call."""
    grads = boundary_grad(bank, obs, w, qhat, eps_frac, lo_hi, Z=Z)
    log = []
    for gd in grads[:n_try]:
        if gd["grad"] == 0.0:
            continue
        s = np.sign(gd["grad"])
        for mult in steps:
            cand = move_threshold(bank, gd["c"], gd["k"],
                                  gd["thr"] + s * mult * gd["eps"])
            ok, d, g = paired_accept(env, cand, cur_G, n_ep, T, seed, z,
                                     pol_fn)
            log.append(dict(kind="guard", c=gd["c"], k=gd["k"], j=gd["j"],
                            grad=gd["grad"], thr=gd["thr"],
                            new_thr=gd["thr"] + s * mult * gd["eps"],
                            accepted=bool(ok), delta=d))
            if ok:
                return cand, g, log
    return bank, cur_G, log


# ---------------------------------------------------------------- the loop
def polish(env, bank, vhat_fn, qhat_fn, rng, n_iter=3, n_states=20000,
           n_ep_sel=1000, T=200, sel_seed=777, z=2.0, mode="grad",
           etas=(0.02, 0.05, 0.1, 0.2), alphas=(0.15, 0.25, 0.5),
           n_guard=4, eps_frac=0.05, lo_hi=None, verbose=True):
    """Two timescales: laws every iteration, one guard move behind them.

    Fast and slow are not an aesthetic choice. A guard move changes which rows
    each law owns, so a threshold proposed against stale laws is priced against
    the wrong counterfactual; refitting the laws first means the boundary term
    compares two arms that are each already the best fit to what they hold.

    Every candidate -- law step and guard move alike -- is still accepted only
    by the paired rollout test. The critic decides DIRECTION, the rollout decides
    WHETHER, and the line search decides how far.
    """
    log = []
    for it in range(n_iter):
        pol = _Bank(bank["clauses"], bank["laws"], bank["default"],
                    design_matrix)
        vh = vhat_fn(pol)
        S, tt, w = onpolicy_states(env, pol, n_states, rng)
        obs = env.observe(S)
        qh = qhat_fn(pol, S, vh)
        cur = _score(env, bank, n_ep_sel, T, sel_seed)

        bank, cur, li = law_step(env, bank, obs, w, qh, cur, mode=mode,
                                 alphas=alphas, etas=etas, n_ep=n_ep_sel, T=T,
                                 seed=sel_seed, z=z)
        bank, cur, gi = guard_step(env, bank, obs, w, qh, cur, n_try=n_guard,
                                   n_ep=n_ep_sel, T=T, seed=sel_seed, z=z,
                                   eps_frac=eps_frac, lo_hi=lo_hi)
        rec = dict(it=it, G_sel=float(cur.mean()), law=li,
                   guard=[g for g in gi if g["accepted"]] or gi[:1],
                   n_states=len(S), w_mean=float(w.mean()))
        log.append(rec)
        if verbose:
            acc = [g for g in gi if g["accepted"]]
            print("  [it %d] laws %s (best %+.2f)  guard %s  G_sel %.2f"
                  % (it, "accepted" if li["accepted"] else "rejected",
                     li["best"],
                     ("%s -> %.3f" % (bank["names"][acc[0]["j"]],
                                      acc[0]["new_thr"])) if acc else "none",
                     cur.mean()), flush=True)
    return bank, log


class TabulatedGrad:
    """A gradient field supplied per row instead of by a model: M_i (u*_i - u).

    This is the planner used as a GRADIENT SOURCE rather than as a regression
    target, which is the only way to compare it with the critic under identical
    step machinery. Without it, "planner" and "critic" differ in two things at
    once -- where the direction comes from and whether the step is a full
    Gauss-Newton solve or a line-searched move -- and the ablation says nothing.
    """

    def __init__(self, M, u_star):
        self.M, self.u = M, u_star

    def __call__(self, idx, u):
        return np.einsum("iqm,im->iq", self.M[idx], self.u[idx] - u)
