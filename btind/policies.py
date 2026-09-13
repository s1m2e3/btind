"""Executable policies + closed-loop evaluation. Nothing here was ever run before."""
import numpy as np
from .collect import design_matrix, leverage_weights
from .fluctuation import weighted_ols
from .search import search


def _clip_disk(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1.0)


class TreePolicy:
    def __init__(self, root):
        self.root = root

    def act(self, obs):
        X = design_matrix(obs)
        out = np.zeros((len(obs), 2))
        self._apply(self.root, X, obs, np.arange(len(obs)), out)
        return _clip_disk(out)

    def _apply(self, node, X, obs, idx, out):
        if len(idx) == 0:
            return
        if node.is_leaf:
            out[idx] = X[idx] @ node.theta
            return
        m = obs[idx, node.split_var] < node.split_at
        self._apply(node.left, X, obs, idx[m], out)
        self._apply(node.right, X, obs, idx[~m], out)


class GlobalAffine:
    """One affine law, no partition. THE baseline the tree has to beat."""
    def __init__(self, obs, u, gap, coh=None):
        self.theta, _ = weighted_ols(design_matrix(obs), u,
                                     leverage_weights(gap, coh))

    def act(self, obs):
        return _clip_disk(design_matrix(obs) @ self.theta)


class SklearnPolicy:
    def __init__(self, model, obs, u, gap, coh=None):
        self.m = model
        self.m.fit(obs, u, sample_weight=leverage_weights(gap, coh))

    def act(self, obs):
        return _clip_disk(np.asarray(self.m.predict(obs)).reshape(len(obs), 2))


class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def act(self, obs):
        ang = self.rng.uniform(0, 2 * np.pi, len(obs))
        mag = np.sqrt(self.rng.uniform(0, 1, len(obs)))
        return np.stack([mag * np.cos(ang), mag * np.sin(ang)], 1)


class CEMPolicy:
    """Replans from scratch every step. Upper bound, not a deployable policy."""
    def __init__(self, env, rng, K=96, H=25, n_seg=3, n_iter=2):
        self.env, self.rng = env, rng
        self.kw = dict(K=K, H=H, n_seg=n_seg, n_iter=n_iter)

    def act_from_state(self, s):
        return _clip_disk(search(self.env, s, self.rng, **self.kw)["u_star"])


def evaluate(env, policy, n_ep=3000, T=200, seed=0, from_state=False):
    rng = np.random.default_rng(seed)
    # An episode START may be a different distribution from a LABEL state: the
    # planner needs coverage, a rollout needs a situation the task begins in.
    s = (env.sample_starts(n_ep, rng) if hasattr(env, "sample_starts")
         else env.sample_states(n_ep, rng))
    # A STATEFUL POLICY GETS ITS STATE CLEARED AT THE EPISODE BOUNDARY. Without
    # this a latched mode or a remembered target leaks from one evaluation into
    # the next, which looks like memory working and is memory cheating.
    if hasattr(policy, "reset"):
        policy.reset(n_ep)
    alive = np.ones(n_ep, bool)
    G = np.zeros(n_ep)
    steps = np.zeros(n_ep)
    eaten = np.zeros(n_ep)
    curve = []
    disc = 1.0
    for _ in range(T):
        a = policy.act_from_state(s) if from_state else policy.act(env.observe(s))
        prev_e = s[:, 6].copy()
        s, r, done = env.step(s, a, rng)
        eaten += (s[:, 6] > prev_e) & alive
        G += disc * r * alive
        steps += alive
        alive &= ~done
        disc *= env.gamma
        curve.append(alive.mean())
        if not alive.any():
            break
    curve += [0.0] * (T - len(curve))
    return dict(G=G, steps=steps, eaten=eaten, survived=alive,
                curve=np.array(curve))
