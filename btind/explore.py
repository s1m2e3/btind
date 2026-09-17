"""Exploration by counterfactual deviation: what would one different action buy?

The search proposes guards from states and laws from a vocabulary, and a
rollout decides. What it never had was a signal about WHICH action a state
wants -- the fitted critic was measured at chance on NestWorld, and the planner
that gave stage 1 its labels was privileged and is gone. On a deterministic
simulator that signal can be measured exactly instead of estimated:

    A(s_t0, a, k) = G( tree, but take `a` for k ticks from t0 ) - G( tree )

Same start, same world, common random numbers, one action sequence differing.
This is the deterministic policy gradient's dQ/du with the model in the loop,
except it is not an estimate and it needs no fit -- and it is what epsilon-greedy
exploration produces when the perturbed and unperturbed episodes are paired
rather than mixed.

TWO LEVELS, as the request was made. `k = 1` is an ACTION-level deviation:
the value of one different action at one tick. `k > 1` is a TRAJECTORY-level
one: the value of committing to a manoeuvre for several ticks -- which is a
step, or a sticky arm, before the tree has a name for it. Both anchor the
same proposals:

    HOT ROWS     the observations at t0 of the deviations that paid most. A
                 guard proposed on those rows is proposed where the tree is
                 measurably leaving return on the table -- the on-policy version
                 of `failure_states`, which anchors on where it dies.
    LAWS         the action that paid there is a constant preference the law
                 vocabulary already offers; the anchor tells the grower where to
                 try it rather than adding anything to what it may try.

WHAT THIS IS NOT. Not an expert: the deviation is a random action from the
world's own action set, and the tree continues afterwards, so the counterfactual
is measured under the CURRENT controller. Not a label the search fits to: the
paired rollout still accepts or rejects every arm that comes out of it. It is
the exploration noise an RL agent would inject, scored exactly and kept as
evidence instead of being averaged into a return.

WHAT THE HEAD CHANGES. On an argmax head a deviation is an action index; on a
vector head it is a unit heading drawn from a fixed compass, because a random
direction in the plane is a heading and the tree's laws emit headings.
"""
import numpy as np

from .memory import mem_names
from .structure import fast_rollout, kernel_for, starts
from .tick import trace_array

COMPASS = np.array([[np.cos(a), np.sin(a)]
                    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])


def _actions(bank, env):
    #  deviates on its COMMAND, column 0; the declaration is not a thing
    # a counterfactual probe can meaningfully randomise on its own.
    if bank.get("head") in ("scalar", "duration", "pass"):
        # a continuous leaf deviates to one of a few levels across its range:
        # what a random discrete action is to an argmax head
        lo, hi = bank.get("u_range", getattr(env, "u_range", (-1.0, 1.0)))
        return [np.array([lo + f * (hi - lo), 0.0]) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if bank.get("head") == "argmax":
        n_act = (env.sig_n_act if getattr(env, "agent", "vehicle") == "signal"
                 else env.n_act)
        return [np.array([float(k), 0.0]) for k in range(int(n_act))]
    return [u for u in COMPASS]


def _agents(env, s, rng):
    """Column 3 of `dev`: which unit of each episode deviates and is traced.

    On a single-agent world it is unused. On the intersection it names a
    vehicle slot -- one drawn among the slots that are actually scheduled to
    enter -- or -1 for the signal when the signal is the agent under search.
    """
    if type(env).__name__ != "IntersectionBatch":
        return np.zeros(len(s))
    if env.agent == "signal":
        return np.full(len(s), -1.0)
    N = env.N
    dep = s[:, 4 * N:5 * N]
    out = np.zeros(len(s))
    for i in range(len(s)):
        ok = np.flatnonzero(np.isfinite(dep[i]) & (dep[i] < 0.7 * env.T_end))
        out[i] = rng.choice(ok) if len(ok) else 0
    return out


def deviations(env, bank, n_ep=400, T=400, seed=11, rng=None, ks=(1, 3, 8),
               same_start=True):
    """Paired deviated and undeviated rollouts. One deviation per episode.

    Returns a dict of per-episode arrays: `z0` the observation layout at the
    deviation tick, `a` the deviation (action index in column 0, or a heading),
    `k` its length, `t0` its tick, `adv` its exact return difference, and the
    undeviated return `G0`. Episodes that ended before t0 are dropped.
    """
    if kernel_for(env, bank) is None:
        return None
    rng = rng or np.random.default_rng(seed)
    env.seed_kernels(seed)
    s = starts(env, n_ep, seed if same_start else int(rng.integers(2 ** 31)))
    d = len(mem_names(bank["names"], bank.get("mem"))) + 1
    tr = trace_array(len(s), T, d)
    who = _agents(env, s, rng)
    base_dev = np.zeros((len(s), 4))
    base_dev[:, 3] = who
    G0 = fast_rollout(env, bank, s, T, trace=tr, dev=base_dev)
    alive = tr[:, :, d - 1] > 0.5                  # the intercept is written
    length = alive.sum(1)
    acts = _actions(bank, env)
    dev = np.zeros((len(s), 4))
    t0 = np.zeros(len(s), int)
    # DEVIATE WHERE THE AGENT ACTS, not merely where it exists. `alive` marks
    # the ticks the agent was present for; for a controller that is asked for a
    # command only now and then, the rest are no-ops whose advantage is exactly
    # zero -- spent rollouts that also flatten every estimator fitted on them.
    acting = env.acting_ticks(tr) if hasattr(env, "acting_ticks") else None
    for i in range(len(s)):
        on = np.flatnonzero(alive[i] if acting is None else (alive[i] & acting[i]))
        if not len(on):                       # never asked: fall back to present
            on = np.flatnonzero(alive[i])
        t0[i] = int(rng.choice(on)) if len(on) else 0
    k = np.asarray(ks)[rng.integers(len(ks), size=len(s))]
    a_idx = rng.integers(len(acts), size=len(s))
    dev[:, 0] = t0
    dev[:, 1] = k
    for i in range(len(s)):
        dev[i, 2:4] = acts[a_idx[i]]
    if type(env).__name__ == "IntersectionBatch":
        dev[:, 3] = who                      # a heading never applies here
    G1 = fast_rollout(env, bank, s, T, dev=dev)
    keep = length > 0
    z0 = tr[np.arange(len(s)), np.minimum(t0, T - 1), :d - 1]
    G1 = np.where(keep, G1, G0)
    return dict(z0=z0[keep], a=dev[keep, 2:4], k=k[keep], t0=t0[keep], keep=keep,
                adv=(G1 - G0)[keep], G0=G0[keep], G1=G1[keep],
                law=tr[np.arange(len(s)), np.minimum(t0, T - 1), d][keep])


def local_q(env, bank, n_ep=300, T=400, seed=11, rng=None, levels=None,
            same_start=True):
    """A local sample of Q(s, .): every action tried at the SAME tick.

    `deviations` takes ONE action per episode, which prices that action against
    the incumbent and nothing else. Trying the whole action set at one tick
    gives K returns at one state, and a quadratic through them is a MEASURED
    local Q -- the curvature `valuesplit` weights its law fit by, and the peak
    it fits toward.

    NOTHING HERE PLANS. `search.py`, the CEM trajectory planner these labels
    were originally taken from, is the expert this project does not use. Every
    rollout is the incumbent tree continuing from the deviated action, so the
    sample says what the tree would ACTUALLY get, not what an oracle could --
    and the peak is therefore a one-step improvement on the current policy,
    which is what policy iteration asks for and all a rollout can honestly give.

    Costs K + 1 rollouts of `n_ep` episodes. At the intersection's five green
    levels and 300 episodes that is 1800 episodes, about six seconds.
    """
    if kernel_for(env, bank) is None:
        return None
    rng = rng or np.random.default_rng(seed)
    env.seed_kernels(seed)
    s = starts(env, n_ep, seed if same_start else int(rng.integers(2 ** 31)))
    d = len(mem_names(bank["names"], bank.get("mem"))) + 1
    tr = trace_array(len(s), T, d)
    who = _agents(env, s, rng)
    base_dev = np.zeros((len(s), 4))
    base_dev[:, 3] = who
    G0 = fast_rollout(env, bank, s, T, trace=tr, dev=base_dev)
    alive = tr[:, :, d - 1] > 0.5
    length = alive.sum(1)
    # THE SAME TICK FOR EVERY ACTION, and one the agent actually acts on --
    # K returns at K different states are not a sample of Q(s, .) at all.
    acting = env.acting_ticks(tr) if hasattr(env, "acting_ticks") else None
    t0 = np.zeros(len(s), int)
    for i in range(len(s)):
        on = np.flatnonzero(alive[i] if acting is None else (alive[i] & acting[i]))
        if not len(on):
            on = np.flatnonzero(alive[i])
        t0[i] = int(rng.choice(on)) if len(on) else 0
    acts = list(_actions(bank, env)) if levels is None else list(levels)
    width = 1 if bank.get("head") in ("scalar", "duration") else 2
    G = np.zeros((len(s), len(acts)))
    a0 = np.zeros((len(s), len(acts), width))
    for j, a in enumerate(acts):
        dev = np.zeros((len(s), 4))
        dev[:, 0] = t0
        dev[:, 1] = 1                                  # one tick: a FIRST action
        dev[:, 2:4] = a
        if type(env).__name__ == "IntersectionBatch":
            dev[:, 3] = who
        G[:, j] = fast_rollout(env, bank, s, T, dev=dev)
        a0[:, j, :] = np.asarray(a, float)[:width]
    keep = length > 0
    z0 = tr[np.arange(len(s)), np.minimum(t0, T - 1), :d - 1]
    return dict(z0=z0[keep], a0=a0[keep], G=G[keep], G0=G0[keep], t0=t0[keep])


def hot_rows(ex, frac=0.25, min_adv=0.0):
    """Indices into `ex["z0"]` of the deviations that paid most.

    Positive advantage means the tree's own action at that state was worth
    less than a random one -- the rows where a new arm has the most to gain.
    """
    if ex is None or not len(ex["adv"]):
        return np.zeros(0, int)
    adv = ex["adv"]
    thr = max(float(np.quantile(adv, 1 - frac)), min_adv)
    return np.flatnonzero(adv >= thr)


def summary(ex, actions=None):
    """What the deviations found, per action and per horizon, for the log."""
    if ex is None or not len(ex["adv"]):
        return {}
    out = dict(n=int(len(ex["adv"])), mean_adv=float(ex["adv"].mean()),
               frac_positive=float((ex["adv"] > 0).mean()),
               best=float(ex["adv"].max()))
    by_k = {}
    for kk in np.unique(ex["k"]):
        m = ex["k"] == kk
        by_k[int(kk)] = dict(mean=float(ex["adv"][m].mean()),
                             positive=float((ex["adv"][m] > 0).mean()))
    out["by_k"] = by_k
    if actions is not None:
        by_a = {}
        for ai in np.unique(ex["a"][:, 0].astype(int)):
            m = ex["a"][:, 0].astype(int) == ai
            by_a[actions[ai]] = dict(mean=float(ex["adv"][m].mean()),
                                     positive=float((ex["adv"][m] > 0).mean()),
                                     n=int(m.sum()))
        out["by_action"] = by_a
    return out


class Deviate:
    """The Python-path twin of the kernel's `dev`: a policy wrapper that
    overrides the action at ticks t0 <= t < t0 + k. For the equivalence test."""

    def __init__(self, policy, dev, argmax):
        self.p, self.dev, self.argmax, self.t = policy, np.asarray(dev), argmax, 0

    def reset(self, n):
        self.p.reset(n)
        self.t = 0

    def act(self, obs):
        a = self.p.act(obs)
        dv = self.dev
        m = (dv[:, 1] > 0) & (self.t >= dv[:, 0]) & (self.t < dv[:, 0] + dv[:, 1])
        if m.any():
            if self.argmax:
                a = a.copy()
                a[m] = dv[m, 2].astype(int)
            else:
                a = a.copy()
                a[m] = dv[m, 2:4]
        self.t += 1
        return a
