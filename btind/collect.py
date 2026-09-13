"""Build the (state, action, value, leverage) dataset by stochastic search."""
import numpy as np
from .envs.forage import ForageWorld, OBS_NAMES
from .search import search
from .valuesplit import action_curvature


def collect(n_states=5000, seed=0, K=256, H=40, n_seg=4, n_iter=3, env=None,
            label='medoid'):
    rng = np.random.default_rng(seed)
    env = env or ForageWorld()
    s = env.sample_states(n_states, rng)
    o = env.observe(s)
    r = search(env, s, rng, K=K, H=H, n_seg=n_seg, n_iter=n_iter, label=label)
    return dict(obs=o, obs_names=OBS_NAMES, state=s, u_star=r["u_star"],
                u_mean=r["u_mean"], u_medoid=r["u_medoid"], u_best=r["u_best"],
                V=r["V"], gap=r["gap"], coherence=r["coherence"], seed=seed,
                M=action_curvature(r["a0_all"], r["G_all"]))


def design_matrix(obs):
    """[obs, 1] -- intercept last so feature indices stay aligned with OBS_NAMES."""
    return np.hstack([obs, np.ones((obs.shape[0], 1))])


def leverage_weights(gap, coherence=None, lo=1.0, hi=99.0, coh_floor=0.0):
    """gap as sample weights, clipped and normalised to mean 1.

    This is how value enters the split criterion: regions where the action
    choice does not matter get little say in where the tree branches.

    Passing `coherence` additionally downweights states where the elite set
    disagreed -- there the label itself is untrustworthy, not just unimportant.
    """
    w = np.clip(gap, *np.percentile(gap, [lo, hi]))
    w = np.maximum(w, 0.0)
    if coherence is not None:
        w = w * np.where(coherence >= coh_floor, coherence, 0.0)
    return w / max(w.mean(), 1e-12)
