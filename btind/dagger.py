"""DAgger: make the fitting distribution match the execution distribution.

The closed-loop failure in e04 was not a structure problem. Every fitted policy
was trained on i.i.d. teleported states and then executed on states drawn from
its own trajectories, with nothing feeding the second distribution back into the
first, so errors compounded with no correction and all of them died by step 50
while the replanning oracle stayed 95% alive.

The loop: fit -> roll out -> collect the states actually visited -> relabel those
with the CEM planner -> aggregate -> refit.

One detail that matters for THIS project specifically: the seed set of teleported
i.i.d. states is never dropped. Split locations depend on the state distribution,
and DAgger deliberately concentrates that distribution onto the policy's own
trajectory manifold -- which is exactly the coverage problem teleporting was
introduced to solve, since a competent policy stops visiting the states that
define its own boundaries. Aggregating rather than replacing keeps uniform
coverage for split-finding and adds on-trajectory states for label relevance.
"""
import numpy as np

from .search import search


def collect_onpolicy_states(env, policy, n_states, rng, n_ep=400, T=200):
    """Roll the policy out and subsample the internal states it actually visits."""
    s = env.sample_states(n_ep, rng)
    alive = np.ones(n_ep, dtype=bool)
    pool = []
    for _ in range(T):
        if not alive.any():
            break
        pool.append(s[alive].copy())
        s, _, done = env.step(s, policy.act(env.observe(s)), rng)
        alive &= ~done
    S = np.concatenate(pool, axis=0)
    take = min(n_states, len(S))
    return S[rng.choice(len(S), size=take, replace=False)]


def label_states(env, states, rng, label="medoid", **kw):
    """Ask the planner what to do at these states. This is the DAgger expert."""
    r = search(env, states, rng, label=label, **kw)
    return (env.observe(states), r["u_star"], r["gap"], r["coherence"])


def dagger(env, seed, fit_fn, rng, n_iter=3, n_new=2000, label="medoid",
           verbose=True):
    """seed = (obs, u, gap, coherence); fit_fn(obs, u, gap, coh) -> policy.

    Returns one policy per round (round 0 is the seed-only fit) plus the
    aggregated dataset, so the experiment can evaluate every round.
    """
    obs, u, gap, coh = seed
    policies, sizes = [], []
    for it in range(n_iter + 1):
        pol = fit_fn(obs, u, gap, coh)
        policies.append(pol)
        sizes.append(len(obs))
        if verbose:
            print(f"    round {it}: fit on n={len(obs)}")
        if it == n_iter:
            break
        S = collect_onpolicy_states(env, pol, n_new, rng)
        o2, u2, g2, c2 = label_states(env, S, rng, label=label)
        obs = np.vstack([obs, o2])
        u = np.vstack([u, u2])
        gap = np.concatenate([gap, g2])
        coh = np.concatenate([coh, c2])
    return policies, sizes, (obs, u, gap, coh)
