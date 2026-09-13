"""Stochastic search over action sequences. The data generator.

No policy is authored or imitated anywhere. For each sampled state we fire K
piecewise-constant action sequences through the sim, refine them with a few CEM
iterations, and read off:

    V(s)         value          mean elite return
    u*(s)        action         mean elite first-segment action
    gap(s)       leverage       elite mean - population mean: how much the
                                choice of action actually matters at s
    coherence(s) unimodality    ||mean a|| / mean||a|| over the elites of the
                                FIRST (unrefined) iteration, in [0,1]

gap and coherence are the two signals a cloned expert can never provide -- an
expert has already collapsed the ambiguity they measure. gap is what stops the
inducer spending tree capacity where no decision exists; low coherence marks
states where several very different actions are near-optimal, i.e. boundaries.

Two implementation choices that matter:

* Action sequences are piecewise-constant over n_seg segments, not i.i.d. per
  step. An i.i.d. random walk in action space has near-zero net displacement
  over a 40-step horizon, so it never reaches food or escapes a threat and the
  search sees no structure at all.
* CEM refines toward a single mode by construction, so coherence is read from
  iteration 0 only. Reading it from the converged population would measure how
  hard CEM collapsed, not how ambiguous the state is.
"""
import numpy as np


def _rollout(env, states, segs, rng, H):
    """segs (n, K, n_seg, 2) -> discounted returns (n, K), via the fused kernel."""
    n, K, n_seg, _ = segs.shape
    return env.rollout_returns(
        np.repeat(states, K, axis=0), segs.reshape(n * K, n_seg, 2), H
    ).reshape(n, K)


def _elites(G, a0, elite_frac):
    n, K = G.shape
    n_el = max(4, int(round(elite_frac * K)))
    idx = np.argsort(-G, axis=1)[:, :n_el]
    rows = np.arange(n)[:, None]
    return idx, rows, G[rows, idx], a0[rows, idx]


def search(env, states, rng, K=256, H=40, n_seg=4, elite_frac=0.10,
           n_iter=3, sigma0=0.7, sigma_floor=0.12, label='medoid'):
    n = states.shape[0]
    mu = np.zeros((n, n_seg, 2))
    sigma = np.full((n, n_seg, 2), sigma0)

    coherence = None
    for it in range(n_iter):
        segs = np.clip(
            mu[:, None] + sigma[:, None] * rng.standard_normal((n, K, n_seg, 2)),
            -1.0, 1.0,
        )
        G = _rollout(env, states, segs, rng, H)
        a0_all = segs[:, :, 0, :]
        idx, rows, elite_G, elite_a = _elites(G, a0_all, elite_frac)

        if it == 0:
            m = elite_a.mean(axis=1)
            coherence = np.linalg.norm(m, axis=1) / np.maximum(
                np.linalg.norm(elite_a, axis=2).mean(axis=1), 1e-9)

        elite_seq = segs[rows, idx]                      # (n, n_el, n_seg, 2)
        mu = elite_seq.mean(axis=1)
        sigma = np.maximum(elite_seq.std(axis=1), sigma_floor)

    V = elite_G.mean(axis=1)

    # Three candidate labels. The elite MEAN is the obvious choice and the wrong
    # one: where the elite set is multimodal it averages "go left" and "go right"
    # into "stand still", producing a target no controller should reproduce.
    # The MEDOID is the elite closest to all other elites -- always an actual
    # member of the set, so it can never be an average of opposing directions.
    u_mean = elite_a.mean(axis=1)
    u_best = elite_a[:, 0, :]                      # elites are sorted by -G
    d = np.linalg.norm(elite_a[:, :, None, :] - elite_a[:, None, :, :], axis=-1)
    u_medoid = elite_a[np.arange(n), np.argmin(d.sum(axis=2), axis=1)]

    return dict(V=V,
                u_star={"mean": u_mean, "medoid": u_medoid, "best": u_best}[label],
                u_mean=u_mean, u_medoid=u_medoid, u_best=u_best,
                gap=V - G.mean(axis=1),
                coherence=coherence,
                elite_a=elite_a,
                a0_all=a0_all,
                G_all=G)
