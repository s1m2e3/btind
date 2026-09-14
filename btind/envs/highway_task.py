"""highway-env as a task for the BT search, with a discrete preference head.

WHY THIS WORLD. NestWorld answered what it could. Memory is settled there
(+10.71, planted-distractor store worth 0.00) but beta is not, and the audit
said why: `t_capture` and the other predictive columns already carry what a
latch would remember, so a termination has nothing to sell. A world where the
primitives are all simultaneously load-bearing needs manoeuvres that take
several steps to complete, and lane changes are exactly that.

THE ACTION IS DISCRETE, AND THAT IS THE POINT. highway-env's DiscreteMetaAction
offers five named manoeuvres -- LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER --
and a law now scores them rather than emitting a heading. Three things follow.

    THE LAW CLASS IS UNCHANGED. `theta` is still affine in the features; only
    its width changes, from 2 outputs to 5, and the head from a normalise to an
    argmax. `improve_laws` is CEM over theta scored by rollout, so the fitting
    machinery does not notice. The gradient path would notice, but it was
    measured at chance on this project and the search has been driven by
    rollouts throughout.

    THE TREE BECOMES READABLE. `u = K x + b` is the least legible thing this
    project emits. "In this region, prefer SLOWER unless the gap exceeds x" is a
    sentence a traffic engineer can agree or disagree with, which is the whole
    claim of the readability goal and has never been testable before.

    DITHERING STOPS PAYING. On NestWorld the Fallback alternated arms every two
    ticks to synthesise a heading neither law could emit, and it was PRODUCTIVE
    -- which is why latching always lost. Alternating LANE_LEFT and IDLE
    synthesises nothing; it aborts the manoeuvre halfway. Commitment becomes
    necessary rather than optional, so beta finally has work to do.

NO PREDICTIVE COLUMNS, DELIBERATELY. The observation here is raw kinematics --
positions and velocities, relative to the ego. It would be easy to add a time-
headway or time-to-collision column, and on NestWorld exactly that kind of
column is what made hysteresis worthless (guarding on `t_capture` scored 22.82
against 19.87 for a distance guard with its best hysteresis). Adding them here
before beta has been given a chance would repeat the experiment we already ran.
They belong in a later ablation, as a manipulation, not in the default layout.

THE ABSENT-VEHICLE SENTINEL. highway-env pads unoccupied observation rows with
ZEROS, so a vehicle that does not exist reports `dx = 0` -- indistinguishable
from one directly on top of the ego, and the single most dangerous state there
is. Absent rows are remapped to a far sentinel instead, which is the same shape
as `d_food = 1.5` on the masked NestWorld, and `thresholds.grid` already knows
how to keep a sentinel from eating the quantile grid.

SPEED. This class steps ONE environment at a time through highway-env's own
Python loop, which is fast enough for premise tests, controls and rendering, and
nowhere near fast enough for the search -- a run prices ~50k proposals at 600
episodes each. The fused numba port of the kinematic bicycle and the IDM/MOBIL
traffic is what makes the search affordable, and this class is what it will be
verified against, the way `test_equivalence` held MemBank to LandscapeBank at
exactly 0.0.
"""
import numpy as np

FEATURES = ["presence", "x", "y", "vx", "vy"]
ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
FAR = 200.0          # the absent-vehicle sentinel, in metres


def feature_names(n_veh):
    """Named columns, which is what makes a guard readable.

    The ego contributes its own state; every other row is RELATIVE to the ego,
    so `v1_dx` is a gap and `v1_dvx` a closing speed -- both quantities a guard
    can be written on and a person can check.
    """
    out = ["ego_x", "ego_y", "ego_vx", "ego_vy"]
    for i in range(1, n_veh):
        out += ["v%d_seen" % i, "v%d_dx" % i, "v%d_dy" % i,
                "v%d_dvx" % i, "v%d_dvy" % i]
    return out


def flatten_obs(o):
    """(V, F) kinematics -> one row of named columns, absent vehicles pushed out.

    The zero padding highway-env uses would otherwise read as a vehicle at zero
    distance. `dx` and `dy` go to the sentinel and the relative speeds to zero,
    so an absent vehicle is far away and not closing.
    """
    o = np.asarray(o, float)
    ego = o[0, 1:5]
    rest = []
    for i in range(1, len(o)):
        seen = float(o[i, 0])
        if seen > 0.5:
            rest += [1.0, o[i, 1], o[i, 2], o[i, 3], o[i, 4]]
        else:
            rest += [0.0, FAR, FAR, 0.0, 0.0]
    return np.concatenate([ego, np.asarray(rest, float)])


class HighwayTask:
    """One highway-env scenario, driven by a bank with a discrete head."""

    def __init__(self, name="merge-v0", n_veh=6, policy_freq=1, duration=40,
                 gamma=0.99, render=False, extra=None):
        import gymnasium as gym
        import highway_env                                       # noqa: F401
        self.name, self.n_veh, self.gamma = name, n_veh, gamma
        self.names = feature_names(n_veh)
        self.n_act = len(ACTIONS)
        cfg = {"observation": {"type": "Kinematics", "vehicles_count": n_veh,
                               "features": FEATURES, "absolute": False,
                               "normalize": False},
               "action": {"type": "DiscreteMetaAction"},
               "policy_frequency": policy_freq, "duration": duration,
               "offscreen_rendering": not render}
        cfg.update(extra or {})
        self.env = gym.make(name, render_mode="rgb_array", config=cfg)

    def close(self):
        self.env.close()

    # -- one episode ---------------------------------------------------------
    def episode(self, policy, seed, T=None, render=False, trace=False):
        """Run `policy` (anything with .act(obs) -> action index) for one episode.

        `policy.reset(1)` is called when it has one, so a bank carrying a
        blackboard starts each episode with an empty latch -- the same contract
        `evaluate` uses on NestWorld.
        """
        o, _ = self.env.reset(seed=int(seed))
        if hasattr(policy, "reset"):
            policy.reset(1)
        G, disc, frames, arms, steps = 0.0, 1.0, [], [], 0
        T = T if T is not None else 10 ** 6
        for _ in range(T):
            row = flatten_obs(o)[None, :]
            a = int(np.asarray(policy.act(row)).reshape(-1)[0])
            if trace:
                Z = policy.z(row) if hasattr(policy, "z") else row
                arms.append(int(np.asarray(policy.arbitrate(Z)).reshape(-1)[0])
                            if hasattr(policy, "arbitrate") else -1)
            o, r, term, trunc, info = self.env.step(a)
            G += disc * float(r)
            disc *= self.gamma
            steps += 1
            if render:
                frames.append(self.env.render())
            if term or trunc:
                break
        crashed = bool(info.get("crashed", False))
        return dict(G=G, steps=steps, crashed=crashed, frames=frames, arms=arms)

    def evaluate(self, policy, n_ep=60, seed0=0, T=None):
        """Mean discounted return over `n_ep` episodes, plus what went wrong."""
        g, st, cr = [], [], []
        for i in range(n_ep):
            r = self.episode(policy, seed0 + i, T=T)
            g.append(r["G"])
            st.append(r["steps"])
            cr.append(r["crashed"])
        g = np.asarray(g)
        return dict(G=float(g.mean()), ci=float(1.96 * g.std() / np.sqrt(len(g))),
                    steps=float(np.mean(st)), crash=float(np.mean(cr)))

    # -- observation coverage, for building the threshold alphabet ------------
    def sample_obs(self, policy, n_ep=40, seed0=1000, T=None):
        """Observations visited under a policy -- the rows a guard is drawn from."""
        rows = []
        for i in range(n_ep):
            o, _ = self.env.reset(seed=int(seed0 + i))
            if hasattr(policy, "reset"):
                policy.reset(1)
            for _ in range(T if T is not None else 10 ** 6):
                row = flatten_obs(o)
                rows.append(row)
                a = int(np.asarray(policy.act(row[None, :])).reshape(-1)[0])
                o, _, term, trunc, _ = self.env.step(a)
                if term or trunc:
                    break
        return np.asarray(rows)


def law_width(names):
    """Rows a law needs: the Z layout plus the intercept, NOT the raw columns.

    A law multiplies `design_matrix(z)`, and `z` is `names + [V_hat, leverage]`,
    so sizing theta by `len(names) + 1` is short by two and every coefficient
    after the raw columns binds to the wrong feature. That mismatch has cost
    this project three separate sites and it is silent whenever the widths
    happen to agree, so the width is computed in one place from the layout
    itself rather than counted by hand.
    """
    from ..memory import base_names
    return len(base_names(names)) + 1


def constant_bank(names, action, n_act=len(ACTIONS)):
    """A one-arm bank that always prefers one manoeuvre.

    The simplest possible discrete-head bank, and the null every search starts
    from: `theta` is zero except for a bias on the chosen action, so the argmax
    is that action everywhere. CEM refines from here.
    """
    th = np.zeros((law_width(names), n_act))
    th[-1, action] = 1.0
    return dict(names=list(names), clauses=[], laws=[], default=th,
                laws_on_z=True, head="argmax", actions=list(ACTIONS))
