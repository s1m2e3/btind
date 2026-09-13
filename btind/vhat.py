"""A learned, continuous value estimate that a guard is allowed to mention.

e13 tried to partition by the planner's V and failed for a specific reason: V is
a TABLE, produced by the CEM search at the states we happened to label, with no
value at a new state. Approximating it and binning hard was worse than not
partitioning at all (103% of one global law), because a misassigned state gets
handed a law specialised for a regime it is not in.

This is the other way round. Fit a continuous V-hat(obs) by fitted value
iteration on the policy's own rollouts, and hand it to the guard alphabet as one
more feature whose threshold DRIFTS like any other. There is no assignment step
to get wrong: `V_hat <= 3.2` is a half-space in exactly the sense `d_threat <=
0.1` is, the evolution can move the cut, and a clause that does not earn its
place is dropped by the same L1 readout as everything else.

WHY THE TARGET IS BOOTSTRAPPED AND NOT MONTE-CARLO. Measured on ForageWorld
against a low-variance ground truth (the mean of 24 independent continuations
from each state, whose noise floor caps R2 at ~0.97):

    5-step return, no bootstrap            R2 -0.272
    20-step return, no bootstrap           R2  0.061
    20-step + bootstrap, 3 FVI sweeps      R2  0.200   (vs 0.189 for full MC)

Both halves are load-bearing: the horizon has to be long enough to see a meal
(rewards are +5 / -20 and rare, with a constant -0.02 otherwise, so a 5-step
window is almost pure step cost), and bootstrapping then recovers the rest of
the horizon at far lower variance than running it out.

WHAT IT ESTIMATES, AND WHY THAT IS THE POINT. V-hat is V^pi -- the value of the
controller we are actually running -- whereas the planner's V is V*-flavoured,
the value of replanning from scratch with 256 sequences. On the states the
policy visits, V-hat predicts realised return far better than the planner does
(R2 0.596 against 0.089, corr 0.822 against 0.544); on teleported states the
planner wins (0.368 against 0.262) because it genuinely solves each one. For a
partition that has to gate THIS controller's behaviour, V^pi on the distribution
the controller induces is the right quantity, and it is the one that is cheap.
"""
import numpy as np

from .collect import design_matrix
from .evotm import _match


class ValueHat:
    """obs -> scalar value estimate, plus the augmented feature matrix."""

    def __init__(self, model, name="V_hat"):
        self.m = model
        self.name = name

    def predict(self, obs):
        return np.asarray(self.m.predict(np.asarray(obs, np.float32)),
                          dtype=np.float64)

    def augment(self, obs):
        return np.hstack([obs, self.predict(obs)[:, None]])


def rollout_returns(env, policy, states, T, seed):
    """(obs, reward, alive) along real trajectories. No planner involved."""
    rng = np.random.default_rng(seed)
    s = states.copy()
    n = len(s)
    alive = np.ones(n, bool)
    d = env.observe(s).shape[1]
    OB = np.zeros((n, T, d), np.float32)
    RW = np.zeros((n, T), np.float32)
    AL = np.zeros((n, T), bool)
    for t in range(T):
        ob = env.observe(s)
        OB[:, t] = ob
        AL[:, t] = alive
        s, r, done = env.step(s, policy.act(ob), rng)
        RW[:, t] = r * alive
        alive &= ~done
    return OB, RW, AL


def fit_vhat(env, policy, n_ep=1500, T=200, n_step=20, sweeps=3, seed=21,
             n_estimators=300, max_depth=6, lr=0.06, extra_states=None):
    """Fitted value iteration on the policy's rollouts. Returns a ValueHat.

    Rollouts start from teleported states so the estimator sees both the uniform
    distribution the partition is defined over and the on-policy manifold the
    controller actually inhabits -- V-hat degrades off its training distribution
    (R2 0.596 on-policy against 0.262 teleported), so the training set has to
    span both or the guard gets a feature it cannot trust where it matters.
    """
    from xgboost import XGBRegressor           # imported after numba, always

    rng = np.random.default_rng(seed)
    S = env.sample_states(n_ep, rng)
    if extra_states is not None and len(extra_states):
        k = min(len(extra_states), n_ep // 2)
        S[:k] = extra_states[rng.choice(len(extra_states), k, replace=False)]
    OB, RW, AL = rollout_returns(env, policy, S, T, seed)
    G = env.gamma
    X = OB.reshape(-1, OB.shape[2])
    live = AL.reshape(-1)
    ar = np.arange(T)

    def target(vh):
        out = np.zeros((n_ep, T))
        for k in range(n_step):
            idx = np.minimum(ar + k, T - 1)
            out += (G ** k) * RW[:, idx] * (ar + k < T)
        if vh is not None:
            idx = np.minimum(ar + n_step, T - 1)
            boot = vh.predict(OB[:, idx].reshape(-1, OB.shape[2])
                              ).reshape(n_ep, T)
            out += (G ** n_step) * boot * (AL[:, idx] & (ar + n_step < T))
        return out.reshape(-1)

    vh = None
    for _ in range(sweeps):
        y = target(vh)
        m = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=lr, verbosity=0)
        m.fit(X[live], y[live])
        vh = m
    return ValueHat(vh)


class GatedBankPolicy:
    """A clause bank whose GUARDS may mention V-hat, while its LAWS may not.

    Keeping V-hat out of the design matrix is what makes the comparison mean
    something: any change in return then comes from the partition being allowed
    to reference value, and not from the control law gaining a 25th parameter
    that happens to be a strong nonlinear feature. `law_on_aug=True` runs the
    other arm deliberately -- laws regressed on [obs, V_hat] as well -- so the
    two effects are separated rather than conflated.
    """

    def __init__(self, bank, vhat, law_on_aug=False):
        self.b, self.v, self.law_on_aug = bank, vhat, law_on_aug

    def act(self, obs):
        aug = self.v.augment(obs)
        Xd = design_matrix(aug if self.law_on_aug else obs)
        out = Xd @ self.b["default"]
        done = np.zeros(len(obs), bool)
        for cl, th in zip(self.b["clauses"], self.b["laws"]):
            m = _match(cl, aug) & ~done
            if m.any():
                out[m] = Xd[m] @ th
                done |= m
        n = np.linalg.norm(out, axis=1, keepdims=True)
        # A DIRECTION, NOT A VECTOR. Least squares shrinks the magnitude of
        # its prediction toward the mean, and a shrunk action is a slower
        # agent: measured on NestWorld, clipping to the disk instead of
        # normalising cost 5.0 return units on an identical bank.
        return out / np.maximum(n, 1e-9)


# ------------------------------------------------ the loop version (e16)
class TrajBuffer:
    """Rollout chunks, newest last, oldest evicted.

    The buffer exists because V-hat is refitted every round against a policy
    that has just changed, and fitting only on the newest rollouts makes the
    critic chase the actor: V-hat describes the current policy, the partition
    uses V-hat to build a new policy, which has a different V-hat. Averaging
    over the last few policies is the standard damping, and it is why n_step
    should be SHORTER here than in the one-shot fit -- the older chunks were
    generated by policies that no longer act this way, so every one of the n
    steps inside a target carries off-policy bias, and bootstrapping is what
    lets the horizon be recovered without paying it.
    """

    def __init__(self, max_chunks=3):
        self.chunks = []
        self.max_chunks = max_chunks

    def add(self, OB, RW, AL):
        self.chunks.append((OB, RW, AL))
        self.chunks = self.chunks[-self.max_chunks:]
        return self

    def n_states(self):
        return int(sum(AL.sum() for _, _, AL in self.chunks))


def _chunk_target(OB, RW, AL, gamma, n_step, vh):
    n, T, _ = OB.shape
    ar = np.arange(T)
    out = np.zeros((n, T))
    for k in range(n_step):
        idx = np.minimum(ar + k, T - 1)
        out += (gamma ** k) * RW[:, idx] * (ar + k < T)
    if vh is not None:
        idx = np.minimum(ar + n_step, T - 1)
        boot = vh.predict(OB[:, idx].reshape(-1, OB.shape[2])).reshape(n, T)
        out += (gamma ** n_step) * boot * (AL[:, idx] & (ar + n_step < T))
    return OB.reshape(-1, OB.shape[2]), out.reshape(-1), AL.reshape(-1)


def fit_vhat_buffer(buf, gamma, n_step=10, sweeps=3, warm=None, max_rows=200000,
                    n_estimators=200, max_depth=6, lr=0.08, seed=0):
    """Refit V-hat on everything in the buffer, bootstrapping from `warm`.

    `warm` is the previous round's estimate: passing it means sweep 1 already
    bootstraps off a converged critic instead of off zero, which is what keeps
    the per-round cost at a few sweeps instead of a full re-solve.
    """
    from xgboost import XGBRegressor

    rng = np.random.default_rng(seed)
    vh = warm
    for _ in range(sweeps):
        Xs, ys = [], []
        for OB, RW, AL in buf.chunks:
            X, y, live = _chunk_target(OB, RW, AL, gamma, n_step, vh)
            Xs.append(X[live])
            ys.append(y[live])
        X, y = np.vstack(Xs), np.concatenate(ys)
        if len(X) > max_rows:
            idx = rng.choice(len(X), max_rows, replace=False)
            X, y = X[idx], y[idx]
        m = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=lr, verbosity=0)
        m.fit(X, y)
        vh = ValueHat(m)
    return vh


def mc_return_to_go(RW, AL, gamma):
    n, T = RW.shape
    out = np.zeros((n, T))
    acc = np.zeros(n)
    for t in range(T - 1, -1, -1):
        acc = RW[:, t] + gamma * acc
        out[:, t] = acc
    return out
