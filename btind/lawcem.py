"""Search the control law directly, by rollout. No expert, no critic.

WHY NOT THE CRITIC. On NestWorld the action-quadratic critic is at chance:
measured against a re-probed reference field whose own self-agreement is 0.82,
it scores cos 0.05 and 53% sign agreement unmasked, 0.03 and 50% masked. That is
not a bug in the fit -- it is the world. The critic estimates how return responds
to perturbing ONE action, and here the agent moves 2.5% of the map per step
against a 138-step discount half-life, so a single action barely moves the
return at all. On ForageWorld, where a step is 6% of the map and episodes last
35 steps, the same critic reaches R2 0.42 and cos 0.28 and its gradient is
worth using. The critic earns its place on one world and not the other, and the
honest thing is to measure which.

WHAT REPLACES IT. The fused kernel evaluates a 400-episode rollout in 1.5 ms, so
a law can be searched the way `search.py` searches action sequences -- sample,
score, keep the elite, refit -- except the samples are POLICY PARAMETERS and the
score is the real objective rather than a horizon-truncated proxy. Sixty-four
candidates per iteration cost a tenth of a second.

    theta_{k+1} = mean of the elite of  theta_k + sigma * N(0, I)

This is the oldest idea in the repository (cross-entropy search) pointed at the
controller instead of at a plan. It needs no labels, no expert and no model of
the value function; it needs a simulator and the willingness to run it.

EVERY ACCEPTED STEP STILL CLEARS THE PAIRED TEST, with an effect-size floor. A
deterministic world gives that test no noise to reject, so significance alone
would wave through anything.
"""
import numpy as np

from .structure import accept, score


def _with_law(bank, arm, th):
    from .kernlaw import constrain
    th = constrain(bank, th)
    if arm < 0:
        return dict(bank, default=th)
    laws = list(bank["laws"])
    laws[arm] = th
    return dict(bank, laws=laws)


def cem_law(env, bank, arm, pol_fn, n_iter=4, K=64, elite_frac=0.25,
            sigma0=0.35, sigma_floor=0.05, n_ep=400, T=400, seed=777,
            rng=None, init=None, verbose=False):
    """Cross-entropy search over one arm's law. Returns (theta, trace).

    `init` seeds the mean; None starts from the law the arm already has, so a
    search that finds nothing returns exactly what it was given.
    """
    rng = rng or np.random.default_rng(0)
    base = bank["default"] if arm < 0 else bank["laws"][arm]
    from .kernlaw import constrain
    mu = constrain(bank, base if init is None else init)
    sigma = np.full(mu.shape, sigma0)
    if bank.get("prior") == "const":
        sigma[:-1] = 0.0                  # only the intercept is a parameter
    n_el = max(4, int(round(elite_frac * K)))
    trace = []
    for it in range(n_iter):
        cand = mu[None] + sigma[None] * rng.standard_normal((K,) + mu.shape)
        cand[0] = mu                                   # keep the incumbent
        g = np.array([score(env, _with_law(bank, arm, c), pol_fn, n_ep, T,
                            seed).mean() for c in cand])
        idx = np.argsort(-g)[:n_el]
        mu = cand[idx].mean(0)
        sigma = np.maximum(cand[idx].std(0), sigma_floor)
        if bank.get("prior") == "const":
            sigma[:-1] = 0.0
        trace.append(float(g[idx].mean()))
        if verbose:
            print("      cem it %d: elite %.2f  best %.2f" % (it, trace[-1],
                                                              g.max()),
                  flush=True)
    return mu, trace


def improve_laws(env, bank, pol_fn, arms=None, cur=None, n_ep=400, T=400,
                 seed=777, z=2.0, min_gain=0.3, rng=None, verbose=True,
                 **cem_kw):
    """CEM each arm in priority order, keeping only what clears the test.

    In order, because an arm's rows depend on what the arms above it take: a law
    tuned against the old flee behaviour is tuned against a distribution that
    the accepted flee law has already changed.
    """
    rng = rng or np.random.default_rng(0)
    cur = score(env, bank, pol_fn, n_ep, T, seed) if cur is None else cur
    order = (list(range(len(bank["clauses"]))) + [-1] if arms is None
             else list(arms))
    log = []
    for arm in order:
        th, trace = cem_law(env, bank, arm, pol_fn, n_ep=n_ep, T=T, seed=seed,
                            rng=rng, **cem_kw)
        cand = _with_law(bank, arm, th)
        ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed, z)
        keep = bool(ok and d > min_gain)
        log.append(dict(arm=int(arm), delta=d, accepted=keep, trace=trace))
        if keep:
            bank, cur = cand, g
        if verbose:
            print("    arm %2d: cem %+.2f %s" % (arm, d,
                  "accepted" if keep else "rejected"), flush=True)
    return bank, cur, log
