"""NestWorld: ForageWorld with a carry-and-deposit cycle.

The API is ForageWorld's exactly -- `sample_states`, `observe`, `step`,
`rollout_returns`, `seed_kernels`, `gamma` -- so every existing component runs
against it unchanged: the CEM planner, DAgger, `rsfi`, the critic, the boundary
term. Only the task is different, which is the point of building it this way.

OBSERVATION. Three columns are added to ForageWorld's eleven and the distractors
are kept, because the distractor control is the only thing that tells us a
criterion is measuring the task rather than the controller:

    11 carrying     0/1, the phase the agent is in
    12 d_nest       distance to the nest
    13/14 bear_nest unit vector toward it

`carrying` is OBSERVED on purpose. With every mask off the problem is a fully
observed MDP, so a memoryless bank can in principle be optimal, and anything a
behaviour tree gains is gained from TEMPORAL EXTENT or from the limits of the
policy class -- not from memory.

THE MASKS ARE WHAT CHANGE THAT. With `vision_r` or `threat_vision_r` set, the
Markov state stops being observable and the world becomes a POMDP: no function
of the current observation is optimal, and history is required rather than
merely convenient. Two settings make that bite instead of merely holding:

    food_persistent   food returns to the SAME site after a delivery. With
                      teleporting food, losing sight of it costs nothing --
                      the right response is to search again, which is
                      memoryless. With a persistent site, the location is
                      information the agent HAD and can only keep by
                      remembering.
    threat_vision_r   the threat also disappears beyond a radius, so fleeing
                      has to be driven by where it was last seen.

THE NEST MOVES EVERY EPISODE TOO, and that is not cosmetic. With the nest pinned
at (0.5, 0.5) a position literal encodes the bearing home, `pos_x`/`pos_y` become
causally relevant, and the distractor control -- the only check in this project
that says a criterion is measuring the task -- silently stops working. Measured:
with a fixed nest the pipeline emitted three arms, all thresholds within 0.03 of
0.5, and scored -5.4 against an oracle of 31.8. Resampling the nest per episode
puts the information back where it belongs, in `d_nest` and `bear_nest`.

THE TWO SUBGOALS ARE GEOMETRICALLY OPPOSED, which is what makes this more than a
relabelling: at the same position, carrying and not carrying want motion in
different directions, so no single affine law in position can serve both, and
the partition has to discover the phase boundary rather than a radius.

THE DISCOUNT IS PART OF THE TASK DEFINITION, and inheriting it was a bug. At
gamma 0.98 -- ForageWorld's value, calibrated to 35-step episodes -- the horizon
half-life is 34 steps while a NestWorld episode runs 235, so everything past the
first cycle was priced at nothing. Measured against a hand-written control pair,
the value of MEMORY under masking was 0.38 at gamma 0.98, 3.35 at 0.99 and 8.78
at 0.995 (half-life 138). The task did not change; only what the objective was
willing to look at. 0.995 is the default here.

AND THE SHUTTLE HAS TO BE LONG, which the first version got wrong. With the
agent crossing the unit square in 17 steps and food spawned near the agent, a
leg of the near-optimal policy lasted 5.7 steps -- so `carrying` was discovered
as a partition, 98.6% of arm switches were phase-locked to it, and none of that
demonstrated anything about temporal structure, because nothing had time to
persist. The agent is now slower (0.025, so the map is ~40 steps wide) and food
is placed 0.35-0.60 from the NEST rather than beside the agent, which makes a
leg 15-25 steps and a full cycle 30-50. Energy decay is rescaled to match, so
the constraint stays "finish the trip" rather than "survive the trip".
"""
import numpy as np

from .nest_kernels import (params_array, rollout_kernel, seed_kernel,
                           step_kernel)

PX, PY, FX, FY, TX, TY, EN, TT, CR, NX, NY, NZ = range(12)
STATE_DIM = 12

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
    "carrying",       # 11 the phase
    "d_nest",         # 12
    "bear_nest_x",    # 13
    "bear_nest_y",    # 14
    "is_night",       # 15 the threat sleeps; the regime, not the clock
    "t_to_switch",    # 16 steps to dawn/dusk, normalised -- a deadline
    "food_seen",      # 17 0 when food is outside the sensing radius
    "threat_seen",    # 18 0 when the threat is outside the sensing radius
    # --- PREDICTIVE: what a model of the dynamics tells you about the future.
    # Closed form here because this world's dynamics are known, which is exactly
    # what a LEARNED dynamics model would have to approximate elsewhere. They
    # are the quantities a person writes behaviour-tree guards on -- not "how
    # far is the bear" but "how long have I got".
    "t_capture",      # 19 steps until the threat reaches me if I stand still
    "t_starve",       # 20 steps until energy runs out at the current drain
    "t_food",         # 21 steps to reach the food at full speed
    "t_nest",         # 22 steps to reach the nest at full speed
    "slack_food",     # 23 t_capture - t_food : can I get there before it gets me
    "slack_nest",     # 24 t_capture - t_nest : can I get home before it does
]
OBS_DIM = len(OBS_NAMES)
DISTRACTORS = [7, 8, 9, 10]
RELEVANT = [0, 3, 11]

_EPS = 1e-9


class NestWorld:
    def __init__(
        self,
        agent_speed=0.025,   # a leg is 20-30 steps, not 6 -- see the docstring
        threat_speed=0.009,  # ratio to agent_speed held at 0.37, as in Forage
        catch_r=0.06,
        food_r=0.06,
        e_decay=0.006,
        eat_r=1.0,          # pickup pays little: the delivery is the reward
        caught_r=-20.0,
        starve_r=-20.0,
        step_cost=0.02,
        gamma=0.995,
        max_t=400.0,
        nest_r=0.07,
        deposit_r=7.0,
        carry_decay=0.002,  # carrying costs energy, so hoarding a trip is not
        nest_fixed=False,   # free and the return leg has a deadline
        day_len=0.0,        # 0 disables the cycle entirely
        night_threat_speed=0.0,
        vision_r=0.0,       # 0 disables masking: everything is always visible
        night_vision_r=None,
        threat_vision_r=0.0,
        food_persistent=False,
        food_dist=(0.35, 0.60),
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
        self.nest_r = nest_r
        self.deposit_r = deposit_r
        self.carry_decay = carry_decay
        self.nest_fixed = nest_fixed
        self.day_len = float(day_len)
        self.night_threat_speed = night_threat_speed
        self.vision_r = float(vision_r)
        self.night_vision_r = (vision_r if night_vision_r is None
                               else float(night_vision_r))
        self.threat_vision_r = float(threat_vision_r)
        self.food_persistent = 1.0 if food_persistent else 0.0
        self.food_dist = tuple(food_dist)

    # -- state construction --------------------------------------------------
    def sample_states(self, n, rng):
        """Stratified over the features a boundary could live on, as in Forage.

        `carrying` is sampled half and half rather than always starting empty:
        the partition is defined over the state space the planner is asked
        about, and a teleported set that never carries would leave the entire
        return phase unlabelled.
        """
        s = np.zeros((n, STATE_DIM))
        s[:, PX] = rng.uniform(0.15, 0.85, n)
        s[:, PY] = rng.uniform(0.15, 0.85, n)

        d_t = rng.uniform(0.08, 0.80, n)
        a_t = rng.uniform(0.0, 2 * np.pi, n)
        s[:, TX] = np.clip(s[:, PX] + d_t * np.cos(a_t), 0.0, 1.0)
        s[:, TY] = np.clip(s[:, PY] + d_t * np.sin(a_t), 0.0, 1.0)

        if self.nest_fixed:
            s[:, NX] = 0.5
            s[:, NY] = 0.5
        else:
            s[:, NX] = rng.uniform(0.25, 0.75, n)
            s[:, NY] = rng.uniform(0.25, 0.75, n)

        # FOOD IS PLACED RELATIVE TO THE NEST, NOT TO THE AGENT. Sampling it
        # near the agent made the shuttle short by construction: measured, the
        # legs of a near-optimal controller lasted 5.7 steps, and an option that
        # persists for six ticks cannot demonstrate anything a reactive policy
        # does not already do. Placing it a fixed distance from the nest makes
        # the round trip the length of the task.
        d_f = rng.uniform(*self.food_dist, size=n)
        a_f = rng.uniform(0.0, 2 * np.pi, n)
        s[:, FX] = np.clip(s[:, NX] + d_f * np.cos(a_f), 0.02, 0.98)
        s[:, FY] = np.clip(s[:, NY] + d_f * np.sin(a_f), 0.02, 0.98)

        s[:, EN] = rng.uniform(0.05, 1.00, n)
        s[:, TT] = rng.uniform(0.0, 200.0, n)
        s[:, CR] = (rng.random(n) < 0.5).astype(float)
        s[:, NZ] = rng.normal(0.0, 1.0, n)
        return s

    def sample_starts(self, n, rng):
        """Episode starts, which are NOT the same distribution as label states.

        MEMORY AND EXPLORATION ARE DIFFERENT PROBLEMS AND THIS SEPARATES THEM.
        Measured on the masked world with generic starts, a hand-written memory
        controller beat a reactive one by 0.04 return -- not because memory is
        worthless, but because neither controller ever found food (0.6 pickups
        per episode against 6.4 unmasked), so there was never anything to
        remember. Starting the agent inside the food site's sensing radius means
        the location is ACQUIRED in the first few ticks; keeping it after the
        return leg, when the site is out of range, is then the only thing memory
        is being asked to do.

        `sample_states` stays broad, because the planner labels and the guard
        alphabet need coverage of states the controller reaches later -- at the
        nest, far from food, carrying and not.
        """
        s = self.sample_states(n, rng)
        d = rng.uniform(0.02, 0.12, n)
        a = rng.uniform(0.0, 2 * np.pi, n)
        s[:, PX] = np.clip(s[:, FX] + d * np.cos(a), 0.0, 1.0)
        s[:, PY] = np.clip(s[:, FY] + d * np.sin(a), 0.0, 1.0)
        s[:, CR] = 0.0                      # every episode starts a full cycle
        s[:, TT] = 0.0
        dt = rng.uniform(0.25, 0.60, n)     # threat not already on top of us
        at = rng.uniform(0.0, 2 * np.pi, n)
        s[:, TX] = np.clip(s[:, PX] + dt * np.cos(at), 0.0, 1.0)
        s[:, TY] = np.clip(s[:, PY] + dt * np.sin(at), 0.0, 1.0)
        s[:, EN] = 1.0
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

        o[:, 11] = s[:, CR]
        dn = s[:, [NX, NY]] - s[:, [PX, PY]]
        d_nest = np.linalg.norm(dn, axis=1)
        o[:, 12] = d_nest
        o[:, 13:15] = dn / np.maximum(d_nest, _EPS)[:, None]

        # --- the day/night regime -----------------------------------------
        # `is_night` is CYCLIC and `t_norm` is monotone, so the distractor
        # stays a distractor: a guard on `t_norm` can only ever capture one
        # phase of one cycle, which is exactly what a spurious feature does.
        if self.day_len > 0:
            ph = np.mod(s[:, TT], 2.0 * self.day_len)
            night = ph >= self.day_len
            o[:, 15] = night.astype(float)
            o[:, 16] = (np.where(night, 2.0 * self.day_len - ph,
                                 self.day_len - ph) / self.day_len)
        else:
            night = np.zeros(n, bool)
            o[:, 16] = 1.0

        # --- limited sensing ----------------------------------------------
        # When food is out of range the agent is told SO, not lied to: the
        # distance is pinned at the horizon and the bearing zeroed. A memoryless
        # controller then has nothing to steer by, which is the whole point --
        # the information exists only in the past, so only a controller with
        # memory can use it.
        scale = np.where(night, self.night_vision_r / max(self.vision_r, _EPS),
                         1.0) if self.vision_r > 0 else np.ones(n)
        if self.vision_r > 0:
            seen = d_food <= self.vision_r * scale
            o[:, 17] = seen.astype(float)
            o[~seen, 4] = 1.5
            o[~seen, 5:7] = 0.0
        else:
            o[:, 17] = 1.0

        if self.threat_vision_r > 0:
            t_seen = d_threat <= self.threat_vision_r * scale
            o[:, 18] = t_seen.astype(float)
            o[~t_seen, 0] = 1.5
            o[~t_seen, 1:3] = 0.0
        else:
            o[:, 18] = 1.0

        # --- predictive columns, from the observation the agent actually has
        CAP = 100.0
        drain = self.e_decay + self.carry_decay * o[:, 11]
        o[:, 19] = np.minimum((o[:, 0] - self.catch_r)
                              / max(self.threat_speed, _EPS), CAP)
        o[:, 20] = np.minimum(o[:, 3] / np.maximum(drain, _EPS), CAP)
        o[:, 21] = np.minimum(o[:, 4] / max(self.agent_speed, _EPS), CAP)
        o[:, 22] = np.minimum(o[:, 12] / max(self.agent_speed, _EPS), CAP)
        o[:, 23] = o[:, 19] - o[:, 21]
        o[:, 24] = o[:, 19] - o[:, 22]
        return o

    def observe_full(self, s):
        """The same observation with every mask disabled.

        Used only to measure what masking costs. The DYNAMICS are untouched by
        vision, so a masked and an unmasked world differ in the observation
        function alone -- which makes the two controllers comparable on
        identical episodes rather than on two different tasks.
        """
        keep = (self.vision_r, self.threat_vision_r)
        self.vision_r, self.threat_vision_r = 0.0, 0.0
        try:
            return self.observe(s)
        finally:
            self.vision_r, self.threat_vision_r = keep

    # -- dynamics ------------------------------------------------------------
    @property
    def _p(self):
        if getattr(self, "_pcache", None) is None:
            self._pcache = params_array(self)
        return self._pcache

    def step(self, s, a, rng=None):
        n = s.shape[0]
        r = np.empty(n)
        done = np.empty(n, dtype=np.bool_)
        step_kernel(s, np.ascontiguousarray(a), self._p, r, done)
        return s, r, done

    def rollout_returns(self, states, segs, H):
        G = np.empty(states.shape[0])
        rollout_kernel(np.ascontiguousarray(states),
                       np.ascontiguousarray(segs), H,
                       max(1, H // segs.shape[1]), self._p, G)
        return G

    @staticmethod
    def seed_kernels(k):
        seed_kernel(k)
