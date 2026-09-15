r"""A value function and a critic for the intersection, learned from the tree's own
rollouts -- the intersection's counterpart of `vhat.py` and `critic.py`.

TWO ESTIMATORS, each scored on held-out data before anything uses it:

    V-hat(z)        the return-to-go of the agent under search -- for a vehicle
                    its OWN per-car return (the trace carries it), for the signal
                    the team return. Fitted value iteration on n-step
                    bootstrapped targets over the policy's trajectories, exactly
                    `vhat.fit_vhat_buffer`, on Monte-Carlo targets by default. Scored against the Monte-Carlo
                    return-to-go of trajectories from episodes it never saw.

    A-hat(z, a, k)  the advantage of taking action a (a level, for a continuous
                    leaf) for k ticks from state z, then following the tree. The
                    TARGETS ARE EXACT: `explore.deviations` pairs each episode
                    with the same episode deviated, so every training row is a
                    measured counterfactual, not a TD estimate. What the fit adds
                    is GENERALISATION -- a few thousand deviations become a
                    prediction at every state the tree visits. V-hat(z) is one of
                    its inputs: how much an action matters depends on how much is
                    at stake. Scored on held-out deviations by rank correlation
                    and by how often it gets the sign of a large advantage right.

WHAT THE CRITIC IS FOR, and what it is not. The forage world's action-quadratic
critic reached cos ~0.2 against a reference gradient field, and its docstring
records that it was at chance on NestWorld, so here the critic PROPOSES and the
rollout DISPOSES: its proposals -- inducing points at states where it predicts
an action beats what the leaf does, with that action as the target -- go through
the same paired test as the deviations' own proposals (`kernsearch.py`). A bad
critic costs screening time, never a wrong tree.

CENTRALISED TRAINING, DECENTRALISED EXECUTION. The trees read their own
observation and nothing else -- that is the whole point of a distributed
controller. Their ESTIMATORS are allowed more, and need it: an independent
critic that sees only one agent's observation cannot know how heavy the episode
is or which partner is driving the other agent, and both decide what an action
is worth. Measured without them, on this run's own logs: the signal's V-hat
scored R^2 -3.78 and its A-hat got the sign of a large advantage right 10% of
the time -- worse than chance -- while the vehicle's reached 78-95%. Both
estimators now also see, per episode:

    demand     the number of cars the episode schedules -- its draw from the
               condition distribution, which the trees are deliberately not told
    partner    which partner ran the episode (its block index when a population
               is in force), so a critic fitted across partners can tell them
               apart instead of averaging them

This is the centralised critic of MADDPG/COMA, in the weak form this world
needs; nothing here reaches the controller.

A CRITIC THAT IS NOT USEFUL DOES NOT PROPOSE, and it takes three measures
agreeing to say so, because they disagree often. Held out, on this world:

    agent, demand        rank corr   top-decile lift   sign of large
    vehicle 16-50 veh/h     0.33          0.4x             84%
    vehicle 100-600         0.21          2.1x             64%
    signal  16-50           0.08          0.9x              9%
    signal  100-600         0.16          0.7x             10%

The vehicle critic at 16-50 ranks DEVIATIONS badly (lift 0.4x) while its
PROPOSALS were the highest-yield source in the run (45% of its point-sets were
kept): ranking a random action's value is not the same task as finding a state
where some command beats the leaf's. So a critic is gated out only when the
rank correlation, the top-decile lift and the sign of large advantages ALL say
noise, which is the signal's case at every demand measured. Then `make_critic`
returns None, the round falls back to deviations and failure anchors, and the
log says so.

ELAPSED TIME IS AN INPUT TO BOTH, AND TO NOTHING ELSE. Episodes are truncated,
so a return-to-go depends on how much episode is left, which the trees are
deliberately not shown (`t_norm` is de-phased). An estimator that cannot see it
predicts a blend of early and late states: measured, the signal's V-hat scored
R^2 -2.62 without it. The estimators are learning aids and never run in the
tree, so giving them the clock leaks nothing into the controller.

HOW THEY ARE SCORED. R^2, and the rank correlation, on held-out episodes. For a
car the rank is the number to read: a car's return is heavy-tailed (a few
percent of trajectories end in a -200 crash) and R^2 is dominated by those
tails, while what a proposal needs is which states are better than which.

FITTING A LAW TO THE CRITIC (`point_sets`). One point at a time, each having to
pay on its own, meets the same deceptive landscape that blocked arm growth: a
red stop needs its brake AND its go-on-green, following needs "brake when
closing" AND "go when the gap opens", and either half alone is neutral or worse.
So the critic also proposes SETS: over the states a law owns where A-hat says
some command beats the leaf's, cluster in the kernel's own columns (k = 2, 3),
take each cluster's medoid as a point and A-hat's best command there as its
target, and offer the set as ONE candidate. That is the policy-improvement step
of actor-critic -- fit the actor to the critic's argmax -- with the paired
rollout, not the critic, deciding whether the fitted law is kept.

WHY NOT A CRITIC IN THE COMPILED LOOP. Guards and laws run inside numba per car
per tick; a boosted model cannot. V-hat is therefore not offered as a guard
column here (it is on the Python-path worlds); it informs proposals instead.
"""
import time

import numpy as np

from . import explore as EX
from .memory import mem_names
from .structure import starts
from .tick import trace_array
from .vhat import TrajBuffer, ValueHat, fit_vhat_buffer, mc_return_to_go


def context(env, s):
    """Per-episode context the CRITICS see and the trees never do:
    (cars scheduled, partner block index)."""
    n = len(s)
    cars = np.isfinite(s[:, 4 * env.N:5 * env.N]).sum(1).astype(np.float32)
    other = env.signal_bank if env.agent == "vehicle" else env.vehicle_bank
    k = len(other) if isinstance(other, list) else 1
    edges = np.linspace(0, n, k + 1).round().astype(int)
    who = np.zeros(n, np.float32)
    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        who[a:b] = j
    return np.stack([cars, who], 1)


def _own(env):
    """Both estimators are fitted on the agent's OWN return, whatever objective
    the round is accepting on (`score_reward`)."""
    from contextlib import nullcontext
    return env.reward_as(None) if hasattr(env, "reward_as") else nullcontext()


def record(env, bank, n_ep=60, seed=3, max_slots=None):
    """Trajectories of the agent under search from kernel traces:
    OB (n, T, n_obs + 2), RW (n, T) its own reward, AL, LAW (n, T).

    The two extra columns are this episode's context (see `context`)."""
    from .envs import intersection_fast as IF
    rng = np.random.default_rng(seed)
    s = env.sample_starts(n_ep, rng)
    T = env.duration
    vb = bank if env.agent == "vehicle" else (env.vehicle_bank or env.default_vehicle_bank())
    sb = env.signal_bank if env.agent == "vehicle" else bank
    d = len(mem_names(bank["names"], bank.get("mem"))) + 1
    n_obs = len(env.names)
    ctx = context(env, s)
    OB, RW, AL, LAW = [], [], [], []
    if env.agent == "vehicle":
        # only slots some episode actually fills: an unscheduled slot is a
        # rollout that records nothing
        dep = s[:, 4 * env.N:5 * env.N]
        who = [q for q in range(env.N) if np.isfinite(dep[:, q]).mean() > 0.3]
    else:
        who = [-1]
    if max_slots is not None and len(who) > max_slots:
        who = sorted(rng.choice(who, max_slots, replace=False))
    for q in who:
        dev = np.zeros((n_ep, 4))
        dev[:, 3] = q
        tr = trace_array(n_ep, T, d)
        # ON ITS OWN RETURN, whatever the round is accepting on: an estimator
        # of the agent's value has to be fitted in the units it is answerable
        # for (`own_reward`), not in the shared units a move is priced in.
        IF.run(env, vb, sb, s, T, trace=tr, dev=dev, reward=env.own_reward)
        alive = tr[:, :, d - 1] > 0.5
        for i in np.flatnonzero(alive.any(1)):
            OB.append(np.hstack([tr[i, :, :n_obs],
                                 np.tile(ctx[i], (T, 1))]))
            RW.append(tr[i, :, d + 2] * alive[i])
            AL.append(alive[i])
            LAW.append(tr[i, :, d].astype(int))
    return (np.array(OB, np.float32), np.array(RW, np.float32), np.array(AL),
            np.array(LAW))


def with_time(OB, T):
    """OB (n, T, d) -> (n, T, d + 1) with elapsed fraction t / T appended."""
    tt = np.broadcast_to((np.arange(OB.shape[1]) / T).astype(OB.dtype), OB.shape[:2])
    return np.concatenate([OB, tt[..., None]], 2)


def _r2(y, p):
    return float(1.0 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def fit_value(env, bank, n_ep=150, seed=3, n_step=None, sweeps=1, max_slots=32,
              verbose=True):
    """V-hat by fitted value iteration; R^2 on held-out episodes. (vhat, report)

    A ROBUST LOSS, because a car's return is heavy-tailed: 4.6% of trajectories
    at medium demand end in a -200 crash, and with squared loss the trees
    memorised those few trajectories' states -- held-out R^2 -1.69 against
    +0.05 for the same trees with a Huber loss on the Monte-Carlo target.
    """
    from scipy.stats import spearmanr
    t0 = time.time()
    # MONTE-CARLO TARGETS BY DEFAULT (n_step = the episode). Episodes here are
    # short and finite, and bootstrapping 20 ticks at a time compounded the
    # estimator's own bias: the signal's V-hat ranked held-out returns at -0.09
    # bootstrapped against 0.45 on the full return.
    n_step = env.duration if n_step is None else n_step
    OB, RW, AL, _ = record(env, bank, n_ep, seed, max_slots)
    G0 = mc_return_to_go(RW, AL, env.gamma)[AL]
    # the Huber knee at the spread of the returns, so a car (spread ~5) and the
    # signal (spread ~100) are both fitted robustly; a fixed knee of 5 left the
    # signal's fit a constant
    slope = max(1.0, 1.5 * float(np.median(np.abs(G0 - np.median(G0)))))
    vh = _fvi_huber(with_time(OB, env.duration), RW, AL, env.gamma, n_step, sweeps,
                    seed, slope)
    OBh, RWh, ALh, _ = record(env, bank, max(20, n_ep // 3), seed + 1000, max_slots)
    G = mc_return_to_go(RWh, ALh, env.gamma)
    Xh = with_time(OBh, env.duration)
    X, y = Xh.reshape(-1, Xh.shape[2])[ALh.reshape(-1)], G.reshape(-1)[ALh.reshape(-1)]
    p = vh.predict(X)
    rep = dict(r2=_r2(y, p), spearman=float(spearmanr(p, y).correlation),
               n_rows=int(AL.sum()), n_test=int(len(y)), secs=time.time() - t0)
    if verbose:
        print("    V-hat: R2 %.2f, rank corr %.2f on %d held-out states (%d training) [%.0fs]"
              % (rep["r2"], rep["spearman"], rep["n_test"], rep["n_rows"], rep["secs"]),
              flush=True)
    return vh, rep


def _fvi_huber(OB, RW, AL, gamma, n_step, sweeps, seed, slope=5.0, max_rows=200000):
    from xgboost import XGBRegressor
    from .vhat import _chunk_target
    rng = np.random.default_rng(seed)
    vh = None
    for _ in range(sweeps):
        X, y, live = _chunk_target(OB, RW, AL, gamma, n_step, vh)
        X, y = X[live], y[live]
        if len(X) > max_rows:
            k = rng.choice(len(X), max_rows, replace=False)
            X, y = X[k], y[k]
        m = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, verbosity=0,
                         objective="reg:pseudohubererror", huber_slope=slope,
                         base_score=float(np.median(y)))
        m.fit(X, y)
        vh = ValueHat(m)
    return vh


class AdvHat:
    """A-hat(z, t, a, k): boosted trees on [obs, t/T, V-hat, action code, k]."""

    def __init__(self, model, vh, n_obs, head, levels, T):
        self.m, self.vh, self.n_obs, self.head, self.levels = model, vh, n_obs, head, levels
        self.T = T

    def _x(self, Z, t, a, k):
        Z = np.asarray(Z, float)[:, :self.n_obs]
        tt = np.broadcast_to(np.asarray(t, float) / self.T, (len(Z),)).reshape(-1, 1)
        Z = np.hstack([Z, tt])
        v = self.vh.predict(Z)[:, None] if self.vh is not None else np.zeros((len(Z), 1))
        if self.head == "argmax":
            code = np.zeros((len(Z), len(self.levels)))
            code[np.arange(len(Z)), np.asarray(a, int)] = 1.0
        else:
            code = np.asarray(a, float).reshape(-1, 1)
        kk = np.broadcast_to(np.asarray(k, float), (len(Z),)).reshape(-1, 1)
        return np.hstack([Z, v, code, kk]).astype(np.float32)

    def predict(self, Z, t, a, k):
        return np.asarray(self.m.predict(self._x(Z, t, a, k)), float)

    def table(self, Z, t, k):
        """(n, n_levels): the predicted advantage of every action (or level)."""
        return np.stack([self.predict(Z, t, np.full(len(Z), j if self.head == "argmax"
                                                    else lv), k)
                         for j, lv in enumerate(self.levels)], 1)


def fit_advantage(env, bank, vh=None, n_dev=3000, seed=17, ks=(3, 8), verbose=True):
    """A-hat on exact deviation advantages; scored on held-out deviations.

    `explore.deviations` uses `structure.starts(env, n_ep, seed)`, so the same
    call reproduces the episodes the rows came from and their context."""
    from xgboost import XGBRegressor
    t0 = time.time()
    rng = np.random.default_rng(seed)
    head = bank.get("head")
    with _own(env):
        ex = EX.deviations(env, bank, n_ep=n_dev, T=env.duration, seed=seed, rng=rng, ks=ks)
    n_obs = len(env.names) + 2                 # the observation plus the context
    ctx = context(env, starts(env, n_dev, seed))
    if head == "argmax":
        levels = list(range(len(bank.get("actions") or [])))
        a = ex["a"][:, 0].astype(int)
    else:
        lo, hi = bank.get("u_range") or env.u_range
        levels = list(np.linspace(lo, hi, 9))
        a = ex["a"][:, 0]
    ah = AdvHat(None, vh, n_obs, head, levels, env.duration)
    # the deviations keep every episode, in order, so the context lines up
    X = ah._x(np.hstack([ex["z0"][:, :n_obs - 2], ctx[ex["keep"]]]), ex["t0"], a, ex["k"])
    y = ex["adv"]
    idx = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                     verbosity=0)
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    from scipy.stats import spearmanr
    big = np.abs(y[te]) >= np.quantile(np.abs(y[te]), 0.75)
    # PRECISION AT THE TOP, and its lift over the base rate: of the rows this
    # critic ranks highest, how many actually paid
    top = np.argsort(-p)[:max(10, len(te) // 10)]
    base = float((y[te] > 0).mean())
    prec = float((y[te][top] > 0).mean())
    rep = dict(spearman=float(spearmanr(p, y[te]).correlation) if len(te) > 2 else 0.0,
               sign_big=float((np.sign(p[big]) == np.sign(y[te][big])).mean())
               if big.any() else 0.0,
               prec_top=prec, base_rate=base, lift=prec / max(base, 1e-9),
               n_train=int(n_tr), n_test=int(len(te)), secs=time.time() - t0)
    m.fit(X, y)                                   # the deployed model sees every row
    ah.m = m
    if verbose:
        print("    A-hat: rank corr %.2f, top-decile precision %.0f%% vs %.0f%% base "
              "(lift %.1fx), sign of large %.0f%%, %d held-out (%d training) [%.0fs]"
              % (rep["spearman"], 100 * rep["prec_top"], 100 * rep["base_rate"],
                 rep["lift"], 100 * rep["sign_big"], rep["n_test"], rep["n_train"],
                 rep["secs"]), flush=True)
    return ah, rep


def _kmeans(Xs, kk, rng, n_iter=10):
    """Lloyd's k-means with farthest-point seeding; returns (labels, centres)."""
    cen = Xs[rng.choice(len(Xs), 1)]
    for _ in range(kk - 1):
        d = ((Xs[:, None, :] - cen[None]) ** 2).sum(2).min(1)
        cen = np.vstack([cen, Xs[np.argmax(d)]])
    lab = np.zeros(len(Xs), int)
    for _ in range(n_iter):
        lab = ((Xs[:, None, :] - cen[None]) ** 2).sum(2).argmin(1)
        cen = np.array([Xs[lab == j].mean(0) if (lab == j).any() else cen[j]
                        for j in range(kk)])
    return lab, cen


def point_sets(Z, tab, kern, levels, head, target_fn, rng, ks=(2, 3), min_adv=0.5):
    """Whole point-sets fitted to the critic: (X (M, D), Y (M, n_out), adv sum)."""
    best = tab.max(1)
    sel = np.flatnonzero(best >= min_adv)
    if len(sel) < 4:
        return []
    Xs = Z[sel][:, kern["cols"]] / kern["ls"]
    out = []
    for kk in ks:
        if len(sel) < 2 * kk:
            continue
        lab, cen = _kmeans(Xs, kk, rng)
        rows = [int(sel[np.argmin(((Xs - cen[j]) ** 2).sum(1))]) for j in range(kk)]
        if len({tuple(Z[i, kern["cols"]]) for i in rows}) < kk:
            continue
        X = Z[rows][:, kern["cols"]]
        Y = np.vstack([target_fn(i, int(np.argmax(tab[i])), 1.0) for i in rows])
        out.append((X, Y, float(best[rows].sum()), "critic-set"))
    return out


def make_critic(env, bank, n_ep=150, n_dev=3000, seed=17, k=3, top=24, min_adv=0.5,
                min_lift=1.0, min_rank=0.2, min_sign=0.3, verbose=True):
    """Fit V-hat and A-hat on `bank`, and return the proposal function
    `kernsearch.search_kernels(critic=...)` calls, with both reports.

    Returns (None, reports) when the critic is not measurably better than
    chance at the sign of a large advantage: its proposals would only cost
    screening time."""
    from . import kernlaw as KL
    from .collect import design_matrix
    from .kernsearch import _theta, flat_laws
    vh, vrep = fit_value(env, bank, n_ep=n_ep, seed=seed, verbose=verbose)
    ah, arep = fit_advantage(env, bank, vh, n_dev=n_dev, seed=seed + 1, verbose=verbose)
    if (arep["lift"] < min_lift and arep["spearman"] < min_rank
            and arep["sign_big"] < min_sign):
        if verbose:
            print("    critic not used this round: rank %.2f, lift %.1fx, sign %.0f%% "
                  "-- all below their gates; proposals fall back to deviations and "
                  "failure anchors" % (arep["spearman"], arep["lift"],
                                       100 * arep["sign_big"]), flush=True)
        return None, dict(value=vrep, advantage=arep, used=False)
    OB, _, AL, LAW = record(env, bank, n_ep=max(20, n_ep // 3), seed=seed + 2,
                            max_slots=32)
    Zon = OB.reshape(-1, OB.shape[2])[AL.reshape(-1)]
    Ton = np.broadcast_to(np.arange(OB.shape[1]), AL.shape).reshape(-1)[AL.reshape(-1)]
    Lon = LAW.reshape(-1)[AL.reshape(-1)]

    n_obs_t = len(env.names)

    def critic(b, c, kk, kern, head):
        L = flat_laws(b).index((c, kk))
        rows = np.flatnonzero(Lon == L)
        if not len(rows):
            return []
        rows = rows[np.random.default_rng(seed).permutation(len(rows))[:4000]]
        Z = Zon[rows]
        tab = ah.table(Z, Ton[rows], k)              # (n, levels)
        best = tab.max(1)
        th = _theta(b, c, kk)

        def target(i, j, f):
            """The target a point at row i gets for the critic's best level j:
            a discrete leaf lifts action j above the leaf's preferences; a
            continuous one moves the fraction f from the leaf's command to it."""
            zi = np.hstack([Z[i, :n_obs_t], np.zeros(len(th) - 1 - n_obs_t)])[None]
            u0 = KL.evaluate(th, kern if KL.n_points(kern) else None, design_matrix(zi),
                             bounds=KL.bounds_of(b))[0]
            if head == "argmax":
                y = u0.copy()
                y[j] = u0.max() + 1.0 + 0.25 * (u0.max() - u0.min())
                return y
            return np.array([u0[0] + f * (ah.levels[j] - u0[0])])
        out = []
        for i in np.argsort(-best):
            if best[i] < min_adv or len(out) >= top:
                break
            x = Z[i, kern["cols"]]
            if any((np.abs(o[0] - x) / kern["ls"]).sum() < 0.5 for o in out):
                continue
            j = int(np.argmax(tab[i]))
            if head == "argmax":
                out.append((x, target(i, j, 1.0), float(best[i])))
            else:
                for f in (1.0, 0.5):
                    out.append((x, target(i, j, f), float(best[i])))
        out += point_sets(Z, tab, kern, ah.levels, head, target,
                          np.random.default_rng(seed + 3), min_adv=min_adv)
        return out
    return critic, dict(value=vrep, advantage=arep, used=True)
