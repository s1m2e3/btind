"""Q-hat: a critic of the CONTROLLER, quadratic in the action.

WHY A CRITIC AT ALL. Every label the laws are fitted against comes from the
planner: `u*` is argmax_a Q*(s,a) over 256 sequences and `M` is -Hessian_a of
that same Q*, both evaluated at teleported i.i.d. states. Two things are wrong
with that for a feedback loop:

  * it is Q*, not Q^pi. The planner replans from scratch every step; the bank
    does not. Fitting the bank's laws to the curvature of a policy that is not
    the bank is off-policy in the one place it cannot be corrected by more data.
  * it carries no occupancy. A state the controller visits at t=3 and a state it
    never reaches enter the fit with the same weight.

WHY QUADRATIC IN u, AND NOT A NETWORK. Three reasons, in order:

  * the law solve needs exactly two things -- M(x) and u*(x) -- and a quadratic
    critic gives both in closed form (M = -H_u, u* = M^-1 b). `fit_value_law`
    then stays what it already is: one linear solve, no inner optimisation.
  * the gradient is what we consume, not the value. A fit that is accurate in
    LEVEL and wrong in SLOPE is useless here, and a low-capacity model whose
    action-dependence is exactly the shape we need beats a flexible one whose
    gradient is unconstrained. (A linear-in-features V-hat is the degenerate
    case: its gradient is a constant vector.)
  * it stays comparable to the incumbent. `action_curvature` fits precisely this
    quadratic, per state, on the planner's cloud. Same functional form, three
    changes: the target is Q^pi not Q*, the coefficients are a smooth function
    of the state rather than an independent fit per state, and the states are
    the ones the controller actually occupies.

THE TARGET. Q^pi(s,u) = r(s,u) + sum_{k=1..h-1} gamma^k r_k + gamma^h V-hat(s_h)
where the first action is the probe u and every later action comes from the
policy. h-step + bootstrap for the same reason vhat.py uses it: rewards are +5 /
-20 and rare against a constant -0.02, so a short window is almost pure step
cost, and running the horizon out raises variance faster than it removes bias.

HOW GOOD IT IS, AND THE HONEST COMPARISON. Scored against a re-probed reference
gradient field (see `local_q_grad`, whose self-agreement caps the scale at cos
0.84), this critic agrees in direction at cos ~0.21 and median ~0.44, and the
planner it replaces scores ~0.24 / ~0.46. So on DIRECTION they are level: the
on-policy critic is NOT better at the thing the law step consumes. Where it does
win is SCALE -- median relative error 1.6 against the planner 2.1 -- which
matters only because a badly scaled gradient wastes line-search evaluations.
This is written here rather than in a commit message because the obvious reading
of this module (Q^pi beats Q*, that is why it exists) is not what the numbers
say; what earns the return is the step shape it makes possible.

THE DISCONTINUITY. Eating teleports the food (kernels.py: `fx = random()`), so
the realised transition genuinely jumps and no pathwise derivative exists there.
Q^pi is an EXPECTATION over that jump, so it stays differentiable in u wherever
the probability of the event is -- which is why the sensitivity is taken through
the critic instead of through the dynamics. `n_rep` averages independent
continuations so the respawn draw is integrated rather than sampled once.
"""
import numpy as np

from .collect import design_matrix

_EPS = 1e-9


# ------------------------------------------------------------------ features
class QFeatures:
    """obs -> Phi. Affine part, RBFs on the radial axes, and RADIAL x BEARING.

    The interaction block is the one that matters and the one a plain feature
    list cannot express. The best action here is always some blend of "toward
    food" and "away from threat" -- two unit vectors the observation already
    carries -- and the whole content of the policy is the GAIN on each blend
    component as a function of the two distances and the energy. A critic whose
    b(x) is linear in obs can represent the two directions but not the gains, so
    its argmax points the same way at d_threat 0.05 as at 0.5. Tensoring an RBF
    basis on each radial axis against each bearing pair gives exactly the
    state-dependent gain, and keeps the model linear in its parameters.

    Curvature stays on the radial block alone (`quad_idx`): M is what weights
    every row of the law solve, and a curvature that can also swing with bearing
    buys in-sample fit and pays for it in a noisier weight on every arm.
    """

    def __init__(self, radial=(0, 3, 4), bearing=((1, 2), (5, 6)), n_rbf=6,
                 width=1.25):
        self.radial, self.bearing = list(radial), [tuple(b) for b in bearing]
        self.n_rbf, self.width = n_rbf, width

    def fit(self, obs):
        qs = np.linspace(0.08, 0.92, self.n_rbf)
        self.centers = {j: np.quantile(obs[:, j], qs) for j in self.radial}
        self.sig = {}
        for j in self.radial:
            d = np.diff(self.centers[j])
            self.sig[j] = max(self.width * (d.mean() if len(d) else 1.0), 1e-3)
        self.p_lin = obs.shape[1] + 1
        self.n_rad = len(self.radial) * self.n_rbf
        return self

    def _rbf(self, obs, j):
        z = (obs[:, [j]] - self.centers[j][None, :]) / self.sig[j]
        return np.exp(-0.5 * z * z)

    def transform(self, obs):
        R = [self._rbf(obs, j) for j in self.radial]
        blocks = [design_matrix(obs)] + R
        for Rj in R:
            for (bx, by) in self.bearing:
                blocks.append(Rj * obs[:, [bx]])
                blocks.append(Rj * obs[:, [by]])
        return np.hstack(blocks)

    def quad_idx(self, p):
        """Columns allowed to modulate the CURVATURE: intercept + radial RBFs."""
        return np.array([self.p_lin - 1]
                        + list(range(self.p_lin, self.p_lin + self.n_rad)), int)


# ------------------------------------------------------------------- fitting
def _ridge(Z, y, lam):
    """Ridge on standardised columns; returns coefficients in raw units."""
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Zs = (Z - mu) / sd
    A = Zs.T @ Zs + lam * len(Z) * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Zs.T @ (y - y.mean()))
    return w / sd, float(y.mean() - (mu / sd) @ w)


class QHat:
    """Q(x,u) = a(x) + b(x).u - 0.5 u' M(x) u, coefficients linear in Phi."""

    def __init__(self, feats, w, qidx, m_floor=1e-3, zfn=None):
        self.f, self.w, self.qidx, self.m_floor = feats, w, qidx, m_floor
        self.zfn = zfn

    # -- design -------------------------------------------------------------
    def _design(self, Phi, u):
        Pq = Phi[:, self.qidx]
        ux, uy = u[:, [0]], u[:, [1]]
        return np.hstack([Phi, Phi * ux, Phi * uy,
                          Pq * ux * ux, Pq * uy * uy, Pq * ux * uy])

    def q(self, obs, u, Phi=None):
        Phi = self.f.transform(self._z(obs)) if Phi is None else Phi
        return self._design(Phi, u) @ self.w[0] + self.w[1]

    # -- coefficients -------------------------------------------------------
    def _z(self, obs):
        """Widen an observation to the layout the critic was fitted on."""
        return obs if self.zfn is None else self.zfn(obs)

    def coef(self, obs, Phi=None):
        """(b (n,2), M (n,2,2) PSD, a (n,)) of the local quadratic."""
        Phi = self.f.transform(self._z(obs)) if Phi is None else Phi
        p, q = Phi.shape[1], len(self.qidx)
        w = self.w[0]
        a = Phi @ w[:p] + self.w[1]
        b = np.stack([Phi @ w[p:2 * p], Phi @ w[2 * p:3 * p]], 1)
        Pq = Phi[:, self.qidx]
        cxx = Pq @ w[3 * p:3 * p + q]
        cyy = Pq @ w[3 * p + q:3 * p + 2 * q]
        cxy = Pq @ w[3 * p + 2 * q:3 * p + 3 * q]
        H = np.empty((len(Phi), 2, 2))
        H[:, 0, 0], H[:, 1, 1] = 2 * cxx, 2 * cyy
        H[:, 0, 1] = H[:, 1, 0] = cxy
        return b, psd(-H, floor=self.m_floor), a

    def grad_u(self, obs, u, Phi=None):
        """dQ/du = b(x) - M(x) u. The only thing the law solve consumes."""
        b, M, _ = self.coef(obs, Phi)
        return b - np.einsum("iqm,im->iq", M, u)

    def targets(self, obs, Phi=None, clip=1.0):
        """(u*, M) -- the pair `fit_value_law` expects, read off Q^pi."""
        b, M, _ = self.coef(obs, Phi)
        u = np.linalg.solve(M, b[..., None])[..., 0]
        n = np.linalg.norm(u, axis=1, keepdims=True)
        return u / np.maximum(n / clip, 1.0), M


def psd(M, floor=1e-3, clip_q=99.0):
    """Project to PSD with a floor, so M^-1 b stays finite.

    Same eigen-clip as `action_curvature`, plus a floor: a state where the
    critic says return is FLAT in the action has no argmax to solve for, and
    without the floor it contributes an arbitrary u* to the law fit.
    """
    w, V = np.linalg.eigh(M)
    pos = w[w > 0]
    cap = np.percentile(pos, clip_q) if pos.size else 1.0
    w = np.clip(w, floor * max(cap, 1e-9), cap)
    return np.einsum("nij,nj,nkj->nik", V, w, V)


# ------------------------------------------------------------------- probes
def _disk(rng, n, scale=1.0):
    ang = rng.uniform(0, 2 * np.pi, n)
    mag = scale * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([mag * np.cos(ang), mag * np.sin(ang)], 1)


def probe_actions(policy_u, rng, n_probe, sigma=0.5):
    """Probe cloud per state: the policy's own action, jitters, and uniforms.

    The cloud has to cover where the law MIGHT move, not only where it is: a
    critic fitted only near pi(s) has no information about the gradient that
    would carry the law away from pi(s), which is the one thing we ask it for.
    A third of the cloud is uniform on the disk for that reason.
    """
    n = len(policy_u)
    U = np.empty((n, n_probe, 2))
    U[:, 0] = policy_u
    n_unif = max(1, n_probe // 3)
    for k in range(1, n_probe - n_unif):
        U[:, k] = policy_u + sigma * rng.standard_normal((n, 2))
    for k in range(n_probe - n_unif, n_probe):
        U[:, k] = _disk(rng, n)
    nrm = np.linalg.norm(U, axis=2, keepdims=True)
    return U / np.maximum(nrm, 1.0)


def probe_returns(env, policy, states, U, vhat, h=10, n_rep=2, seed=0,
                  mem_state=None):
    """h-step return of (probe action, then the policy), bootstrapped by V-hat.

    Returns (n, n_probe) discounted returns. `n_rep` independent continuations
    are averaged: the food respawn is the only stochastic element in the sim,
    and it is exactly what a single continuation would bake in.
    """
    n, n_probe, _ = U.shape
    S0 = np.repeat(states, n_probe, axis=0)
    A0 = U.reshape(n * n_probe, 2)
    out = np.zeros(n * n_probe)
    for rep in range(n_rep):
        env.seed_kernels(seed + 7919 * rep)
        s = S0.copy()
        # A STATEFUL CONTROLLER HAS TO BE PROBED FROM ITS OWN STATE. The env
        # holds the world; the latch and the blackboard live in the policy, so
        # continuing from a recorded state means restoring both. Probing a
        # memory controller from a blank blackboard measures a different
        # controller -- one that has just forgotten everything.
        if mem_state is not None and hasattr(policy, "set_state"):
            lat, slots, have = mem_state
            policy.set_state(np.repeat(lat, n_probe),
                             np.repeat(slots, n_probe, axis=0),
                             np.repeat(have, n_probe))
        elif hasattr(policy, "reset"):
            policy.reset(len(s))
        G = np.zeros(len(s))
        alive = np.ones(len(s), bool)
        disc = 1.0
        s, r, done = env.step(s, A0)
        G += r
        alive &= ~done
        for k in range(1, h):
            disc *= env.gamma
            if not alive.any():
                break
            a = policy.act(env.observe(s))
            s, r, done = env.step(s, a)
            G += disc * r * alive
            alive &= ~done
        if vhat is not None and alive.any():
            G += disc * env.gamma * vhat.predict(env.observe(s)) * alive
        out += G
    return (out / n_rep).reshape(n, n_probe)


def fit_qhat(env, policy, states, vhat, rng, n_probe=16, h=5, n_rep=3,
             sigma=0.5, lam=1e-2, n_rbf=8, seed=0, feats=None, center=True,
             zfn=None, mem_state=None):
    """Probe the controller at `states`, fit Q^pi(x,u). Returns (QHat, info).

    CENTERED BY DEFAULT, and this is the single largest change to the fit. Every
    quantity the loop consumes is a WITHIN-STATE difference:

        dQ/du, u* = M^-1 b, M                 -- derivatives in u
        Q(s, theta_new) - Q(s, theta_cur)     -- the boundary integrand

    so the level a(x) is a nuisance that carries ~80% of the variance in y and
    none of the signal. Subtracting the per-state mean from both y and the design
    removes it exactly (the a-block is constant within a state, so it centers to
    zero and is dropped from the solve), leaving the ridge to spend its whole
    budget on the action dependence rather than on being a value function. The
    returned Q is then defined only up to a per-state constant -- which is all
    any consumer here needs.
    """
    obs = env.observe(states)
    U = probe_actions(policy.act(obs), rng, n_probe, sigma)
    Y = probe_returns(env, policy, states, U, vhat, h=h, n_rep=n_rep, seed=seed,
                      mem_state=mem_state)

    # THE CRITIC SEES WHAT THE CONTROLLER SEES. With a blackboard, Q^pi is a
    # function of (observation, memory) -- a critic fitted on the observation
    # alone is estimating the value of a different, amnesiac policy, and the law
    # gradient it hands back is the gradient of that.
    zx = obs if zfn is None else zfn(obs)
    feats = (feats or QFeatures(n_rbf=n_rbf)).fit(zx)
    Phi = feats.transform(zx)
    n, n_probe = Y.shape
    P = np.repeat(Phi, n_probe, axis=0)
    qh = QHat(feats, (None, 0.0), feats.quad_idx(Phi.shape[1]), zfn=zfn)
    Z = qh._design(P, U.reshape(-1, 2))
    y = Y.reshape(-1)

    p = Phi.shape[1]
    if center:
        Z = (Z.reshape(n, n_probe, -1)
             - Z.reshape(n, n_probe, -1).mean(1, keepdims=True)).reshape(len(y), -1)
        y = (Y - Y.mean(1, keepdims=True)).reshape(-1)
        w, b0 = _ridge(Z[:, p:], y, lam)
        w = np.concatenate([np.zeros(p), w])
    else:
        w, b0 = _ridge(Z, y, lam)
    qh.w, qh.centered = (w, b0), bool(center)

    pred = (Z[:, p:] @ w[p:] if center else Z @ w) + b0
    ss = ((y - y.mean()) ** 2).sum()
    info = dict(n_rows=len(y), n_par=int((w != 0).sum()),
                r2=float(1 - ((y - pred) ** 2).sum() / max(ss, _EPS)),
                r2_kind="within-state" if center else "total",
                y_std=float(y.std()))
    return qh, info


# ---------------------------------------------------------------- validation
def fd_grad(env, policy, states, u, vhat, rng, eps=0.08, h=10, n_rep=6, seed=0):
    """Central-difference dQ^pi/du at (states, u), by re-probing the sim.

    This is the ground truth the critic is scored against. It is expensive and
    noisy (the respawn draw again), which is the whole reason for fitting a
    critic instead of calling this inside a loop.
    """
    n = len(states)
    U = np.empty((n, 4, 2))
    for k, (d, sgn) in enumerate([(0, +1), (0, -1), (1, +1), (1, -1)]):
        U[:, k] = u
        U[:, k, d] += sgn * eps
    G = probe_returns(env, policy, states, U, vhat, h=h, n_rep=n_rep, seed=seed)
    return np.stack([(G[:, 0] - G[:, 1]) / (2 * eps),
                     (G[:, 2] - G[:, 3]) / (2 * eps)], 1)


def grad_score(g_hat, g_ref):
    """Cosine and relative error of a gradient field against the reference."""
    nh = np.linalg.norm(g_hat, axis=1)
    nr = np.linalg.norm(g_ref, axis=1)
    ok = (nh > _EPS) & (nr > _EPS)
    cos = (g_hat[ok] * g_ref[ok]).sum(1) / (nh[ok] * nr[ok])
    rel = np.linalg.norm(g_hat[ok] - g_ref[ok], axis=1) / nr[ok]
    return dict(cos=float(cos.mean()), cos_med=float(np.median(cos)),
                agree=float((cos > 0).mean()), rel=float(np.median(rel)),
                n=int(ok.sum()))


def local_q_grad(env, policy, states, u, vhat, rng, K=64, n_rep=8, h=5,
                 sigma=0.45, seed=0):
    """Reference gradient: a quadratic fitted per state on a big probe cloud.

    Four-point finite differences do NOT work here as a reference. Measured on
    ForageWorld, two independent FD estimates of the same gradient agree at
    cos 0.03 (eps 0.05) and 0.24 (eps 0.35) -- the estimator is noise, because
    a +-eps nudge either changes nothing or flips a catch/meal event worth
    +-5 to +-20. Averaging a QUADRATIC over K probes and n_rep continuations
    uses every probe to estimate the same two-dimensional slope instead of
    differencing two of them, which is the same reason `search` reads its label
    off an elite SET rather than off the single best sequence.
    """
    n = len(states)
    U = np.clip(u[:, None, :] + sigma * rng.standard_normal((n, K, 2)), -1, 1)
    U[:, 0] = u
    G = probe_returns(env, policy, states, U, vhat, h=h, n_rep=n_rep, seed=seed)
    ax, ay = U[:, :, 0], U[:, :, 1]
    Z = np.stack([np.ones_like(ax), ax, ay, ax * ax, ay * ay, ax * ay], 2)
    ZtZ = np.einsum("nkp,nkq->npq", Z, Z)
    ZtZ[:, np.arange(6), np.arange(6)] += 1e-6
    beta = np.linalg.solve(ZtZ, np.einsum("nkp,nk->np", Z, G)[..., None])[..., 0]
    gx = beta[:, 1] + 2 * beta[:, 3] * u[:, 0] + beta[:, 5] * u[:, 1]
    gy = beta[:, 2] + 2 * beta[:, 4] * u[:, 1] + beta[:, 5] * u[:, 0]
    return np.stack([gx, gy], 1)
