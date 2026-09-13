"""ForageWorld: batched continuous 2-D foraging sim with a pursuing threat.

Written batched from the start -- every function takes a (B, STATE_DIM) array and
advances the whole population at once, because random-shooting search needs tens
of millions of env-steps and a Python loop over envs would make that impossible.

No policy is authored anywhere in this file. The task is defined by the reward
only; whatever behavioural structure exists has to be discovered from search.
"""
import numpy as np

from .kernels import step_kernel, rollout_kernel, params_array, seed_kernel

# --- state layout -----------------------------------------------------------
PX, PY, FX, FY, TX, TY, EN, TT, NZ = range(9)
STATE_DIM = 9

# --- observation layout -----------------------------------------------------
OBS_NAMES = [
    "d_threat",       # 0  causally relevant
    "bear_threat_x",  # 1
    "bear_threat_y",  # 2
    "energy",         # 3  causally relevant
    "d_food",         # 4
    "bear_food_x",    # 5
    "bear_food_y",    # 6
    "pos_x",          # 7  distractor
    "pos_y",          # 8  distractor
    "t_norm",         # 9  distractor
    "noise",          # 10 pure null -- calibrates the test
]
OBS_DIM = len(OBS_NAMES)
DISTRACTORS = [7, 8, 9, 10]
RELEVANT = [0, 3]

_EPS = 1e-9


class ForageWorld:
    def __init__(
        self,
        agent_speed=0.06,
        threat_speed=0.022,   # slower than the agent: fleeing WORKS, so failing
        catch_r=0.06,         # to flee is a real and punished mistake
        food_r=0.06,
        e_decay=0.010,        # ~100 steps from full to starved
        eat_r=5.0,
        caught_r=-20.0,
        starve_r=-20.0,
        step_cost=0.02,
        gamma=0.98,
        max_t=400.0,
    ):
        self.agent_speed = agent_speed
        self.threat_speed = threat_speed
        self.catch_r = catch_r
        self.food_r = food_r
        self.e_decay = e_decay
        self.eat_r = eat_r
        self.caught_r = caught_r
        self.starve_r = starve_r
        self.step_cost = step_cost
        self.gamma = gamma
        self.max_t = max_t

    # -- state construction --------------------------------------------------
    def sample_states(self, n, rng):
        """Stratified over the *features* we care about, not over positions.

        Sampling agent/threat/food positions uniformly would pile d_threat up
        around 0.4-0.5 and leave the close-range decision region almost empty.
        We instead sample distance and bearing directly so coverage in d_threat
        and energy -- the axes any boundary would live on -- is roughly flat.
        """
        s = np.zeros((n, STATE_DIM))
        s[:, PX] = rng.uniform(0.2, 0.8, n)
        s[:, PY] = rng.uniform(0.2, 0.8, n)

        d_t = rng.uniform(0.08, 0.80, n)   # 0.08 > catch_r: not already caught
        a_t = rng.uniform(0.0, 2 * np.pi, n)
        s[:, TX] = np.clip(s[:, PX] + d_t * np.cos(a_t), 0.0, 1.0)
        s[:, TY] = np.clip(s[:, PY] + d_t * np.sin(a_t), 0.0, 1.0)

        d_f = rng.uniform(0.06, 0.45, n)   # reachable inside the search horizon
        a_f = rng.uniform(0.0, 2 * np.pi, n)
        s[:, FX] = np.clip(s[:, PX] + d_f * np.cos(a_f), 0.0, 1.0)
        s[:, FY] = np.clip(s[:, PY] + d_f * np.sin(a_f), 0.0, 1.0)

        s[:, EN] = rng.uniform(0.05, 1.00, n)
        s[:, TT] = rng.uniform(0.0, 200.0, n)
        s[:, NZ] = rng.normal(0.0, 1.0, n)
        return s

    # -- observation ---------------------------------------------------------
    def observe(self, s):
        n = s.shape[0]
        o = np.zeros((n, OBS_DIM))

        dt = s[:, [TX, TY]] - s[:, [PX, PY]]
        d_threat = np.linalg.norm(dt, axis=1)
        o[:, 0] = d_threat
        o[:, 1:3] = dt / np.maximum(d_threat, _EPS)[:, None]

        o[:, 3] = s[:, EN]

        df = s[:, [FX, FY]] - s[:, [PX, PY]]
        d_food = np.linalg.norm(df, axis=1)
        o[:, 4] = d_food
        o[:, 5:7] = df / np.maximum(d_food, _EPS)[:, None]

        o[:, 7] = s[:, PX]
        o[:, 8] = s[:, PY]
        o[:, 9] = s[:, TT] / self.max_t
        o[:, 10] = s[:, NZ]
        return o

    # -- dynamics ------------------------------------------------------------
    @property
    def _p(self):
        if getattr(self, "_pcache", None) is None:
            self._pcache = params_array(self)
        return self._pcache

    def step(self, s, a, rng=None):
        """One batched step, IN PLACE. Returns (same array, reward, done).

        Mutates `s` rather than copying it: at K=256 the copy alone was a 92 MB
        allocation per step. Every caller reassigns the returned array, and the
        one that needs the pre-step value (evaluate, for energy) copies the
        column it needs first.
        """
        n = s.shape[0]
        r = np.empty(n)
        done = np.empty(n, dtype=np.bool_)
        step_kernel(s, np.ascontiguousarray(a), self._p, r, done)
        return s, r, done

    def rollout_returns(self, states, segs, H):
        """Fused H-step horizon over (n, n_seg, 2) piecewise-constant plans.

        The whole trajectory stays in registers inside the kernel -- no per-step
        state array, and dead rows exit the loop immediately.
        """
        G = np.empty(states.shape[0])
        rollout_kernel(np.ascontiguousarray(states), np.ascontiguousarray(segs),
                       H, max(1, H // segs.shape[1]), self._p, G)
        return G

    @staticmethod
    def seed_kernels(k):
        seed_kernel(k)
