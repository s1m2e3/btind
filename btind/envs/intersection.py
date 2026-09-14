r"""A signalised intersection, every vehicle and the signal controlled by trees.

WHY THIS WORLD. Highway-v0 measured insensitive to sequences (e26): its
meta-actions are already macro-actions a low-level controller completes, and
the world punishes commitment. Here nothing is completed for the agent. A
vehicle approaching a red light has to brake, wait, and go -- three actions in
an order that depends on a signal it does not control -- and the signal has to
hold a phase while a queue drains and switch when the other side has waited.
Sequences, terminations, failure, and two controllers at two timescales are
load-bearing by construction, which is what a test of hierarchy needs.

THE GEOMETRY IS SUMO'S, exported once from the `sumo_test` net (12 movements,
their path polylines, true pairwise conflict points, the four-phase plan) into
`data/intersection_geom.npz`, so this package needs no SUMO. Vehicles advance an
arc-length along their movement's path; a position in the plane is only needed
for rendering. The dynamics are deliberately simpler than that project's GP
kernel controller, because the controller is what the search is supposed to
find: the physics here is kinematic, and safety is the tree's job.

TWO AGENTS, ONE RETURN. The VEHICLE tree is a single policy every vehicle runs
-- the way IDM is one model every car follows -- observing its own kinematics,
its signal, its leader and its worst crossing rival. The SIGNAL tree runs once
per tick over the queues and chooses EXTEND or SWITCH. Both are ordinary banks
with an argmax head; the world is told which one is being searched (`agent`)
and holds the other fixed (`vehicle_bank`, `signal_bank`), so every existing
stage scores one bank at a time against a shared episode return. When no signal
bank is given a fixed-time plan runs; when no vehicle bank is given a
hand-written follower does, so either agent can be studied alone.

WHAT IS REWARDED. A vehicle leaving the network pays +R_EXIT; a collision costs
R_COLL and removes both vehicles; crossing the stop line on red costs R_RED;
and every tick charges DELAY_W * (V0 - v) * dt for each active vehicle AND
the full standstill delay for each vehicle that is due but blocked at its
entrance, so throughput, safety, compliance and delay are all in one number.
The blocked term is not decoration: without it the first search on this world
parked one car per lane at the spawn point, nothing behind it ever entered,
and the controller scored -6.8 against -37 for cruising with zero exits.
Collisions do NOT end the episode -- with forty vehicles one early crash would
erase the rest of the measurement -- they remove the two vehicles involved.

COLLISIONS are decided on the arc-lengths. Rear-end: two vehicles on the same
approach lane (left turns have their own lane; through and right share one),
or on the same exit edge past the box, whose bumper gap goes negative.
Crossing: two vehicles on conflicting movements both inside half a vehicle of
their shared conflict point at once. This is a model of SUMO's oriented-box
test, not a reimplementation of it, and any tree found here is re-scored in
SUMO before a number is reported, as highway trees are in highway-env.

THE PLANTED COLUMNS are here as everywhere: `t_norm` and `noise`, carrying
nothing about the road, offered to both trees exactly like the real columns.
"""
import os

import numpy as np

GEOM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "intersection_geom.npz")

# -- physical constants, sumo_test's --------------------------------------------
V0, A_MAX, B_MAX = 11.0, 2.6, 4.5
L_VEH, W_VEH = 5.0, 1.8
HALF_CONF = 0.5 * (L_VEH + W_VEH)      # occupancy half-width at a conflict point
SPAWN_GAP = 8.0
T_AR, T_MIN, T_MAX = 3.0, 5.0, 60.0
N_PHASES = 4
FAR = 200.0
TAU_MAX = 8.0
QUEUE_D, QUEUE_V = 120.0, 2.0

# -- rewards ---------------------------------------------------------------------
R_EXIT, R_COLL, R_RED, DELAY_W = 1.0, 10.0, 3.0, 0.002

ACCELS = np.array([-B_MAX, -1.5, 0.0, 1.0, A_MAX])
ACTIONS = ["BRAKE_HARD", "BRAKE", "HOLD", "ACCEL", "ACCEL_MAX"]
SIG_ACTIONS = ["EXTEND", "SWITCH"]
FIXED_GREEN = np.array([30.0, 10.0, 30.0, 10.0])

# DISTRIBUTED CONTROL: a vehicle observes ITSELF -- its speed, where it is
# relative to its stop line and its own first conflict point, what its front
# sensor reports -- and what the INTERSECTION tells every approaching car: the
# signal for its movement, how long the phase has run, whether all-red is on,
# and whether it is close enough to the box to be the intersection's concern.
# Nothing about the other vehicles' plans or positions beyond the one ahead:
# the same subtree runs in every car on that car's own information.
NEAR_INT = 60.0
SIG_HIDDEN = -1.0     # what the signal columns read when the car is not near
# `t_sig`: how long the phase has run (continuous mode), or the message's
# time until MY light changes, in ticks (event mode)
VEH_NAMES = ["v", "d_stop", "near_int", "green", "t_sig", "all_red",
             "lead_gap", "lead_dv", "has_lead", "d_conf", "is_left", "is_right",
             "t_norm", "noise"]
SIG_NAMES = (["q%d" % k for k in range(N_PHASES)]
             + ["n%d" % k for k in range(N_PHASES)]
             + ["ph%d" % k for k in range(N_PHASES)]
             + ["t_phase", "all_red", "t_norm", "noise"])


def _phase(noise):
    """A per-episode phase in [0, 1) derived from the episode's `noise` draw."""
    return np.mod(np.abs(noise) * 7.31, 1.0)


def load_geometry(path=GEOM_PATH):
    g = np.load(path, allow_pickle=False)
    out = {k: g[k] for k in g.files}
    out["lane_group"] = out["appr"] * 2 + (out["dirs"] == 2)     # left alone
    out["s_cp"] = np.nan_to_num(out["s_cp"], nan=-1e9)
    return out


class IntersectionBatch:
    """Every episode advanced together over array columns; N slots per episode.

    STATE LAYOUT per episode, flat:
        [0:N)     movement index of each slot
        [N:2N)    arc-length s
        [2N:3N)   speed v
        [3N:4N)   status: 0 pending, 1 active, 2 done
        [4N:5N)   scheduled depart time
        5N + 0..9 phase, t_phase, in_ar, ar_t, tick, noise, n_coll, n_red,
                  n_rear, n_cross
        5N + 10   the signal's planned green time (duration head), 0 = none yet
        [5N+11 : 6N+11)  told: the slot has received its message (event mode)

    HEADS. Each agent's leaf may be discrete or continuous, per bank:
        vehicle  `argmax` over the five named accelerations, or `scalar`: an
                 affine acceleration clipped to [-B_MAX, A_MAX]
        signal   `argmax` over EXTEND/SWITCH each tick, or `duration`: an
                 affine green time clipped to [T_MIN, T_MAX], read once at the
                 first live tick of each phase
    The arbitration above the leaf -- guards, latch, steps, fail -- is discrete
    either way. `veh_head` and `sig_head` are the world's defaults for a cold
    start; a bank carries its own `head`, and the world runs whatever it is.

    SIGNAL MODES. `continuous`: a car within NEAR_INT of its stop line reads
    the light every tick. `event`: the intersection SPEAKS ONCE -- the tick a
    car enters the zone it receives (my light now, ticks until it changes,
    all-red) and then the columns go back to the sentinel. The car has to
    remember. Time-to-change is computed from the fixed-time plan, which is
    what a V2I broadcast would carry; when a signal tree runs instead, the
    message is the plan's promise and the tree may break it.

    OCCLUSION. `occlude=(lo, hi)` blinds the front sensor while a car is
    between `hi` and `lo` metres before its stop line: `has_lead` reads 0, the
    gap FAR, the closing speed 0, as if there were no one ahead -- a crest, a
    bend, a parked truck. The leader is still there and the collision test
    still sees it. A car that wants to survive the zone has to HOLD what it
    saw before entering it, which is the third thing this world can ask a
    blackboard for. None by default.
    """
    N_MISC = 11

    def __init__(self, n_max=40, T_end=100.0, dt=0.5, vph=(200.0, 60.0, 60.0),
                 gamma=0.995, spawn_back=90.0, exit_after=40.0, seed=0,
                 sig_mode="continuous", veh_head="argmax", sig_head="argmax",
                 occlude=None):
        assert sig_mode in ("continuous", "event")
        self.occlude_lo, self.occlude_hi = ((float(occlude[0]), float(occlude[1]))
                                            if occlude else (-1.0, -1.0))
        self.occluded = occlude is not None
        assert veh_head in ("argmax", "scalar") and sig_head in ("argmax", "duration")
        self.veh_head, self.sig_head = veh_head, sig_head
        self.sig_mode = sig_mode
        self.event = sig_mode == "event"
        self.N, self.dt, self.gamma = int(n_max), float(dt), float(gamma)
        self.T_end, self.duration = float(T_end), int(round(T_end / dt))
        self.vph = tuple(float(x) for x in vph)                 # (s, l, r)
        self.spawn_back, self.exit_after = float(spawn_back), float(exit_after)
        self.geom = load_geometry()
        self.M = len(self.geom["path_len"])
        self.veh_names, self.sig_names = list(VEH_NAMES), list(SIG_NAMES)
        self.veh_actions, self.sig_actions = list(ACTIONS), list(SIG_ACTIONS)
        self.veh_n_act, self.sig_n_act = len(ACTIONS), len(SIG_ACTIONS)
        self.agent = "vehicle"
        self.vehicle_bank = None
        self.signal_bank = None
        self._other = None            # the fixed agent's policy on the Python path
        self._kseed = seed
        g = self.geom
        self.s_spawn = g["s_stop"] - self.spawn_back
        self.s_exit = g["s_junc"] + self.exit_after
        self.lane_group = g["lane_group"]
        # my first conflict point ahead is a property of the GEOMETRY: for
        # each movement, the arc-lengths of every point where a conflicting
        # movement crosses it, sorted
        self.cps = [np.sort(g["s_cp"][m][g["conf"][m]]) for m in range(self.M)]
        self.k = 6 * self.N + self.N_MISC

    def seed_kernels(self, seed):
        self._kseed = int(seed)

    # -- which agent the generic interface speaks for ------------------------
    # `names`, `n_act`, `actions`, `observe` and `step` all describe the agent
    # under search, so every stage that was written for one controller runs
    # unchanged; the other agent is driven by its fixed bank (or the built-in
    # fallback) inside `step`.
    def set_agent(self, agent):
        assert agent in ("vehicle", "signal")
        self.agent = agent
        self._other = None
        return self

    @property
    def names(self):
        return self.veh_names if self.agent == "vehicle" else self.sig_names

    @property
    def head(self):
        return self.veh_head if self.agent == "vehicle" else self.sig_head

    @property
    def n_act(self):
        if self.head in ("scalar", "duration"):
            return 1
        return self.veh_n_act if self.agent == "vehicle" else self.sig_n_act

    @property
    def actions(self):
        if self.head in ("scalar", "duration"):
            return None
        return self.veh_actions if self.agent == "vehicle" else self.sig_actions

    @property
    def u_range(self):
        """The continuous leaf's range: acceleration for a car, green time for
        the signal. The world clips every continuous command to it."""
        return (-B_MAX, A_MAX) if self.agent == "vehicle" else (T_MIN, T_MAX)

    def row_alive(self, s):
        """Which observation rows are live units this tick: active slots for
        the vehicle agent (n * N rows), every episode for the signal (n)."""
        if self.agent == "vehicle":
            return (s[:, 3 * self.N:4 * self.N] == 1.0).reshape(-1)
        return np.ones(len(s), bool)

    def default_vehicle_bank(self):
        """The built-in follower: brake for a close leader, stop for a visible
        red, otherwise accelerate. Hand-written, used only as the fixed vehicle
        controller when the SIGNAL is under search and no vehicle bank exists."""
        from ..memory import mem_names
        N = self.veh_names
        ix = N.index
        d = len(mem_names(N, None)) + 1

        def P(k):
            th = np.zeros((d, self.veh_n_act))
            th[-1, k] = 1.0
            return th
        close = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 10.0, True]]
        closing = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 25.0, True],
                   [ix("lead_dv"), 0.5, False]]
        red = lambda lo, hi: [[ix("green"), 0.5, True], [ix("green"), -0.5, False],
                              [ix("d_stop"), lo, False], [ix("d_stop"), hi, True]]
        return dict(names=list(N), laws_on_z=True, head="argmax",
                    actions=list(self.veh_actions),
                    clauses=[close, red(0.0, 6.0), red(0.0, 45.0), closing],
                    laws=[P(0), P(0), P(1), P(1)], default=P(4))

    # -- state ---------------------------------------------------------------
    def sample_states(self, n, rng):
        return self.sample_starts(n, rng)

    def sample_starts(self, n, rng):
        """Episodes: a Poisson arrival schedule per movement, sorted by time.

        Slots beyond the demand are left pending forever (depart = inf). The
        signal starts in phase 0 at a random point of its fixed plan so a
        fixed-time controller's cycle is not aligned with the episode.
        """
        N, M = self.N, self.M
        s = np.zeros((n, self.k))
        rate = np.array([self.vph[{0: 2, 1: 0, 2: 1}[int(d)]] for d in
                         self.geom["dirs"]]) / 3600.0        # dirs: r=0,s=1,l=2
        for i in range(n):
            ev = []
            for m in range(M):
                t = 0.0
                while True:
                    t += rng.exponential(1.0 / rate[m])
                    if t >= self.T_end:
                        break
                    ev.append((t, m))
            ev.sort()
            ev = ev[:N]
            k = len(ev)
            s[i, 0:k] = [e[1] for e in ev]
            s[i, 4 * N:4 * N + k] = [e[0] for e in ev]
            s[i, 4 * N + k:5 * N] = np.inf
            s[i, N:2 * N] = self.s_spawn[s[i, 0:N].astype(int)]
        b = 5 * N
        s[:, b + 0] = 0.0
        s[:, b + 1] = rng.uniform(0.0, FIXED_GREEN[0], n)
        s[:, b + 5] = rng.normal(0.0, 1.0, n)
        return s

    def _unpack(self, s):
        N = self.N
        return (s[:, 0:N].astype(int), s[:, N:2 * N], s[:, 2 * N:3 * N],
                s[:, 3 * N:4 * N], s[:, 4 * N:5 * N],
                s[:, 5 * N:5 * N + self.N_MISC])

    def _told(self, s):
        N = self.N
        return s[:, 5 * N + self.N_MISC:6 * N + self.N_MISC]

    def _t_change(self, X, m):
        """Ticks until movement m's light changes, under the fixed-time plan.

        Green now: the rest of this phase. Red now: the rest of this phase,
        the all-red, then every phase up to my next green, each with its
        all-red. Zero on the tick the change happens.
        """
        g = self.geom["green"]
        ph = X[:, 0].astype(int)
        in_ar = X[:, 2] > 0.5
        rest = np.where(in_ar, 0.0, FIXED_GREEN[ph] - X[:, 1])
        ar_left = np.where(in_ar, T_AR - X[:, 3], T_AR)
        gm = np.take_along_axis(g[ph], m, 1) & ~in_ar[:, None]     # green now
        t = np.where(gm, rest[:, None], (rest + ar_left)[:, None])
        # red: walk the phases until mine is green
        acc = np.zeros_like(t)
        pending = ~gm
        for k in range(1, N_PHASES + 1):
            q = (ph + k) % N_PHASES
            green_q = np.take_along_axis(g[q], m, 1)
            add = pending & ~green_q
            acc = acc + np.where(add, (FIXED_GREEN[q] + T_AR)[:, None], 0.0)
            pending = pending & ~green_q
        t = t + acc
        return np.maximum(np.round(t / self.dt), 0.0)

    # -- observation ---------------------------------------------------------
    def _green_now(self, X):
        """(n, M) bool: which movements have right of way this tick."""
        ph = X[:, 0].astype(int)
        g = self.geom["green"][ph]
        return g & (X[:, 2:3] < 0.5)

    def _leaders(self, m, s, act):
        """Gap to and speed of the vehicle ahead on my lane, per slot.

        Same approach lane before the box; same exit edge past it. Vehicles on
        different movements have different arc-lengths on a shared exit, so
        the exit ordering uses distance to the path end instead.
        """
        n, N = s.shape
        g = self.geom
        lg = self.lane_group[m]
        te = g["to_edge"][m]
        past = s > g["s_junc"][m] + 2.0
        x_out = g["path_len"][m] - s                       # distance to the end
        A = act[:, :, None] & act[:, None, :]
        same_in = (lg[:, :, None] == lg[:, None, :]) & ~past[:, :, None] & ~past[:, None, :]
        same_out = (te[:, :, None] == te[:, None, :]) & past[:, :, None] & past[:, None, :]
        ahead_in = s[:, None, :] > s[:, :, None]
        ahead_out = x_out[:, None, :] < x_out[:, :, None]
        d_in = np.where(A & same_in & ahead_in, s[:, None, :] - s[:, :, None], np.inf)
        d_out = np.where(A & same_out & ahead_out, x_out[:, :, None] - x_out[:, None, :], np.inf)
        d = np.minimum(d_in, d_out)
        idx = np.arange(N)
        d[:, idx, idx] = np.inf
        j = np.argmin(d, axis=2)
        best = np.take_along_axis(d, j[:, :, None], 2)[:, :, 0]
        has = np.isfinite(best)
        return has, np.where(has, best - L_VEH, FAR), j

    def _d_conf(self, m, s):
        """Distance to MY nearest conflict point still ahead: geometry only."""
        d_conf = np.full(s.shape, FAR)
        for mm in range(self.M):
            rows = m == mm
            if rows.any() and len(self.cps[mm]):
                dd = self.cps[mm][None, :] - s[rows][:, None]
                dd = np.where(dd > 0.0, dd, FAR)
                d_conf[rows] = dd.min(axis=1)
        return np.minimum(d_conf, FAR)

    def observe(self, s):
        """The agent under search's observation: vehicle rows or signal rows."""
        return (self.observe_vehicles(s) if self.agent == "vehicle"
                else self.observe_signal(s))

    def observe_vehicles(self, s):
        """One row per (episode, slot): (n * N, len(VEH_NAMES)). Inactive
        slots are all-zero rows; their actions are ignored by `step`."""
        m, S, V, st, dep, X = self._unpack(s)
        n, N = S.shape
        act = st == 1.0
        g = self.geom
        green = self._green_now(X)
        gm = np.take_along_axis(green, m, 1)
        has, gap, j = self._leaders(m, S, act)
        v_lead = np.take_along_axis(V, j, 1)
        d_conf = self._d_conf(m, S)
        d_stop = g["s_stop"][m] - S
        if self.occluded:
            blind = (d_stop >= self.occlude_lo) & (d_stop <= self.occlude_hi)
            has = has & ~blind
            gap = np.where(blind, FAR, gap)
        o = np.zeros((n, N, len(VEH_NAMES)))
        o[:, :, 0] = V
        o[:, :, 1] = d_stop
        near = (d_stop > 0.0) & (d_stop <= NEAR_INT)
        o[:, :, 2] = near
        # THE INTERSECTION SPEAKS ONLY TO CARS THAT ARE NEAR IT. Beyond
        # NEAR_INT the signal columns read a sentinel: what a car knows about
        # the light is intersection information, handed over on approach, not
        # a global variable. In continuous mode a near car (or one past the
        # line) hears it every tick; in event mode it hears it ONCE, on the
        # tick it enters the zone, and the message carries the time to change.
        if self.event:
            hear = near & (self._told(s) < 0.5)
            t_col = self._t_change(X, m)
        else:
            hear = near | (d_stop <= 0.0)
            t_col = np.broadcast_to(X[:, 1:2], d_stop.shape)
        o[:, :, 3] = np.where(hear, gm, SIG_HIDDEN)
        o[:, :, 4] = np.where(hear, t_col, SIG_HIDDEN)
        o[:, :, 5] = np.where(hear, X[:, 2:3], SIG_HIDDEN)
        o[:, :, 6] = gap
        o[:, :, 7] = np.where(has, V - v_lead, 0.0)
        o[:, :, 8] = has
        o[:, :, 9] = d_conf
        o[:, :, 10] = g["dirs"][m] == 2
        o[:, :, 11] = g["dirs"][m] == 0
        # `t_norm` IS A PLANTED DISTRACTOR AND HAS TO STAY ONE. Elapsed time
        # in a truncated episode predicts whether a car can still exit before
        # the end, so the raw fraction was bought by both trees as a horizon
        # cue. Offset by a per-episode phase drawn from `noise` and wrapped,
        # it is a clock that carries nothing about the remaining time.
        o[:, :, 12] = np.mod(X[:, 4:5] / self.duration + _phase(X[:, 5:6]), 1.0)
        o[:, :, 13] = X[:, 5:6]
        o[~act] = 0.0
        return o.reshape(n * N, -1)

    def observe_signal(self, s):
        m, S, V, st, dep, X = self._unpack(s)
        n, N = S.shape
        act = st == 1.0
        g = self.geom
        d_stop = g["s_stop"][m] - S
        appr = act & (d_stop > 0.0) & (d_stop < QUEUE_D)
        queued = appr & (V < QUEUE_V)
        o = np.zeros((n, len(SIG_NAMES)))
        for k in range(N_PHASES):
            in_k = g["green"][k][m]
            o[:, k] = (queued & in_k).sum(1)
            o[:, N_PHASES + k] = (appr & in_k).sum(1)
        o[np.arange(n), 2 * N_PHASES + X[:, 0].astype(int)] = 1.0
        b = 3 * N_PHASES
        o[:, b] = X[:, 1]
        o[:, b + 1] = X[:, 2]
        o[:, b + 2] = np.mod(X[:, 4] / self.duration + _phase(X[:, 5]), 1.0)
        o[:, b + 3] = X[:, 5]
        return o

    # -- one tick -----------------------------------------------------------
    def fixed_signal_action(self, s):
        """The fixed-time plan, as SWITCH/EXTEND decisions."""
        X = s[:, 5 * self.N:]
        return (X[:, 1] >= FIXED_GREEN[X[:, 0].astype(int)]).astype(int)

    def _fixed_agent_actions(self, s):
        """Actions of the agent NOT under search, from its fixed bank.

        The bank runs on a persistent MemBank so its latch and blackboard
        survive between ticks; it is reset whenever a fresh batch begins (every
        episode at tick 0) or the batch size changes. A vehicle bank also gets
        its rows reset when a slot spawns, as the reference rollout does.
        """
        from ..memory import MemBank
        n, N = len(s), self.N
        fresh_batch = bool((s[:, 5 * N + 4] == 0.0).all())
        if self.agent == "vehicle":
            if self.signal_bank is None:
                return None
            if self._other is None or fresh_batch or len(self._other.latch) != n:
                self._other = MemBank(self.signal_bank, len(self.sig_names))
                self._other.reset(n)
            return self._other.act(self.observe_signal(s))
        vb = self.vehicle_bank or self.default_vehicle_bank()
        if self._other is None or fresh_batch or len(self._other.latch) != n * N:
            self._other = MemBank(vb, len(self.veh_names))
            self._other.reset(n * N)
            self._other._prev = np.zeros(n * N, bool)
        active = (s[:, 3 * N:4 * N] == 1.0).reshape(-1)
        fresh = active & ~self._other._prev
        pol = self._other
        pol.latch[fresh], pol.step[fresh], pol.have[fresh] = -1, 0, False
        pol.age[fresh] = 0
        if pol.slots is not None and pol.slots.shape[1]:
            pol.slots[fresh] = 0.0
        pol._prev = active
        return pol.act(self.observe_vehicles(s))

    _veh_head_now = "argmax"
    _sig_head_now = "argmax"

    def _heads(self, vehicle_bank, signal_bank):
        """Record which head each agent's actions are in, for `step_both`."""
        vb = vehicle_bank if vehicle_bank is not None else (
            self.vehicle_bank if self.agent == "signal" else None)
        self._veh_head_now = (vb or {}).get("head", self.veh_head) if vb else self.veh_head
        self._sig_head_now = (signal_bank or {}).get("head", self.sig_head) \
            if signal_bank else self.sig_head

    def step(self, s, a, rng=None):
        """Advance every episode one tick with the agent under search's
        actions `a`; the other agent acts from its fixed bank. (s, r, done)."""
        other = self._fixed_agent_actions(s)
        if self.agent == "vehicle":
            self._heads(self._search_bank, self.signal_bank)
            return self.step_both(s, a, other)
        self._heads(self.vehicle_bank or self.default_vehicle_bank(),
                    self._search_bank)
        return self.step_both(s, other, a)

    _search_bank = None

    def step_both(self, s, a_veh, a_sig=None, rng=None):
        """Advance every episode one tick. `a_veh` is (n * N,) action indices,
        `a_sig` (n,) or None for the fixed-time plan. Returns (s, r, done)."""
        s = s.copy()
        N, dt, g = self.N, self.dt, self.geom
        m, S, V, st, dep, X = self._unpack(s)
        n = len(s)
        a_veh = np.asarray(a_veh, float).reshape(n, N)
        # the heads the actions are in were recorded by `_heads` -- by `step`
        # for the generic interface, by `python_rollout` for the reference
        veh_scalar = self._veh_head_now == "scalar"
        acc = (np.clip(a_veh, -B_MAX, A_MAX) if veh_scalar
               else ACCELS[np.clip(np.rint(a_veh), 0, len(ACCELS) - 1).astype(int)])
        sig_duration = (self._sig_head_now == "duration") if a_sig is not None else False
        if a_sig is None:
            a_sig = self.fixed_signal_action(s)
        a_sig = np.asarray(a_sig, float).reshape(n)
        r = np.zeros(n)
        t = X[:, 4] * dt

        # -- signal: all-red runs out, or a switch is taken ------------------
        in_ar = X[:, 2] > 0.5
        X[in_ar, 3] += dt
        ends = in_ar & (X[:, 3] >= T_AR - 1e-9)
        X[ends, 0] = np.mod(X[ends, 0] + 1, N_PHASES)
        X[ends, 1] = 0.0
        X[ends, 2] = 0.0
        X[ends, 3] = 0.0
        live = ~in_ar
        if sig_duration:
            # the plan is read at the first live tick of a phase, then the
            # phase runs until it is out; the tree keeps ticking meanwhile
            need = live & (X[:, 10] <= 0.0)
            X[need, 10] = np.clip(a_sig[need], T_MIN, T_MAX)
            X[live, 1] += dt
            switch = live & (X[:, 1] >= X[:, 10] - 1e-9)
            X[switch, 10] = 0.0
        else:
            X[live, 1] += dt
            switch = live & ((a_sig >= 0.5) & (X[:, 1] >= T_MIN) | (X[:, 1] >= T_MAX))
        X[switch, 2] = 1.0
        X[switch, 3] = 0.0
        green = self._green_now(X)                         # after the update
        gm = np.take_along_axis(green, m, 1)

        # -- spawn: earliest due slot per lane whose entry is clear ----------
        # A vehicle spawned THIS tick was never observed, so no action applies
        # to it yet: it enters at V0 and is first driven next tick. Kinematics,
        # red-running and delay use the pre-spawn set; collisions and exits the
        # post-spawn set.
        act = st == 1.0
        was = act.copy()
        lg = self.lane_group[m]
        for i in range(n):
            spawned = set()
            for q in range(N):
                if st[i, q] != 0.0 or dep[i, q] > t[i]:
                    continue
                grp = lg[i, q]
                if grp in spawned:
                    continue
                same = act[i] & (lg[i] == grp)
                ok = True
                if same.any():
                    ahead = S[i, same] - self.s_spawn[m[i, q]]
                    ahead = ahead[ahead >= -1e-9]
                    ok = (not len(ahead)) or ahead.min() >= 2.0 * L_VEH + SPAWN_GAP
                if ok:
                    st[i, q] = 1.0
                    S[i, q] = self.s_spawn[m[i, q]]
                    V[i, q] = V0
                    act[i, q] = True
                    spawned.add(grp)

        # -- kinematics: the tree's acceleration, clipped ---------------------
        before = S.copy()
        v_new = np.clip(V + acc * dt, 0.0, V0)
        S[was] = S[was] + v_new[was] * dt
        V[was] = v_new[was]

        # -- red-light running: crossing the stop line without right of way -
        s_stop = g["s_stop"][m]
        ran = was & (before < s_stop) & (S >= s_stop) & ~gm
        X[:, 7] += ran.sum(1)
        r -= R_RED * ran.sum(1)

        # -- delay ------------------------------------------------------------
        # A VEHICLE THAT IS DUE BUT BLOCKED AT THE ENTRANCE IS DELAYED TOO.
        # Measured: without this the search parked the first car of every
        # lane at its spawn point, nothing behind it ever entered, and the
        # controller scored -6.8 against -37 for cruising with ZERO exits --
        # an optimum of the reward, not of the road. Pending vehicles whose
        # depart time has passed pay the full standstill delay.
        blocked = (st == 0.0) & (dep <= t[:, None])
        r -= DELAY_W * (((V0 - V) * was).sum(1) + V0 * blocked.sum(1)) * dt

        # -- collisions: rear-end on a shared lane, crossing at a conflict --
        has, gap, j = self._leaders(m, S, act)
        rear = act & has & (gap < 0.0)
        conf = g["conf"][m[:, :, None], m[:, None, :]]
        idx = np.arange(N)
        conf[:, idx, idx] = False
        s_me = g["s_cp"][m[:, :, None], m[:, None, :]]
        s_rv = g["s_cp"][m[:, None, :], m[:, :, None]]
        both = (act[:, :, None] & act[:, None, :] & conf
                & (np.abs(s_me - S[:, :, None]) < HALF_CONF)
                & (np.abs(s_rv - S[:, None, :]) < HALF_CONF))
        cross = both.any(2)
        hit = rear | cross
        # each rear-end also removes the leader it hit
        lead_hit = np.zeros_like(hit)
        ii, qq = np.nonzero(rear)
        lead_hit[ii, j[ii, qq]] = True
        hit |= lead_hit
        # count events: rear-ends once each, crossings once per pair
        n_rear = rear.sum(1)
        n_cross = np.triu(both, 1).sum((1, 2))
        n_ev = n_rear + n_cross
        r -= R_COLL * n_ev
        X[:, 6] += n_ev
        X[:, 8] += n_rear
        X[:, 9] += n_cross
        st[hit] = 2.0
        V[hit] = 0.0

        # -- exits -------------------------------------------------------------
        out = (st == 1.0) & (S >= self.s_exit[m])
        st[out] = 2.0
        r += R_EXIT * out.sum(1)

        # -- the message has been delivered to every car that OBSERVED from
        # inside the zone this tick: the position at observation time, not
        # after the move, or a car entering the zone is marked told before
        # it has heard anything and the message is never seen at all
        if self.event:
            told = self._told(s)
            d_obs = g["s_stop"][m] - before
            told[was & (d_obs > 0.0) & (d_obs <= NEAR_INT)] = 1.0

        X[:, 4] += 1.0
        done = X[:, 4] >= self.duration
        return s, r, done

    # -- the reference rollout -------------------------------------------------
    def python_rollout(self, vehicle_bank, signal_bank, s, T=None, veh_dev=None):
        """Both trees on the numpy model. The vehicle tree runs on n * N rows,
        one per slot, with its latch reset when a slot spawns; the signal tree
        on n rows. `signal_bank` None means the fixed-time plan. Returns G."""
        from ..memory import MemBank
        T = self.duration if T is None else min(T, self.duration)
        n, N = len(s), self.N
        self._heads(vehicle_bank, signal_bank)
        pv = MemBank(vehicle_bank, len(self.veh_names))
        pv.reset(n * N)
        ps = None
        if signal_bank is not None:
            ps = MemBank(signal_bank, len(self.sig_names))
            ps.reset(n)
        G, disc = np.zeros(n), 1.0
        prev_active = np.zeros(n * N, bool)
        for t in range(T):
            st = s[:, 3 * N:4 * N].reshape(-1)
            active = st == 1.0
            fresh = active & ~prev_active
            # a vehicle that has just entered has no history: empty latch, step
            # and blackboard, exactly as the kernel starts a slot
            pv.latch[fresh] = -1
            pv.step[fresh] = 0
            if pv.slots is not None and pv.slots.shape[1]:
                pv.slots[fresh] = 0.0
            pv.have[fresh] = False
            pv.age[fresh] = 0
            prev_active = active
            a_v = pv.act(self.observe_vehicles(s))
            if veh_dev is not None:
                a_v = veh_dev(t, a_v)
            a_s = None if ps is None else ps.act(self.observe_signal(s))
            s, r, done = self.step_both(s, a_v, a_s)
            G += disc * r
            disc *= self.gamma
            if done.all():
                break
        return G


    # -- what the search needs from a rollout --------------------------------
    def record_traces(self, bank, n_ep=100, seed=3, max_traj=4000):
        """Trajectories of the agent under search: OB (n_traj, T, n_obs) and
        AL (n_traj, T) alive flags, from kernel traces -- one trajectory per
        active (episode, slot) for the vehicle agent, one per episode for the
        signal. What `memsearch.latched` replays a write rule over."""
        from . import intersection_fast as IF
        from ..memory import mem_names
        from ..tick import trace_array
        rng = np.random.default_rng(seed)
        s = self.sample_starts(n_ep, rng)
        T = self.duration
        vb = (bank if self.agent == "vehicle"
              else (self.vehicle_bank or self.default_vehicle_bank()))
        sb = self.signal_bank if self.agent == "vehicle" else bank
        d = len(mem_names(bank["names"], bank.get("mem"))) + 1
        n_obs = len(self.names)
        OB, AL = [], []
        if self.agent == "vehicle":
            for q in range(self.N):
                dev = np.zeros((n_ep, 4))
                dev[:, 3] = q
                tr = trace_array(n_ep, T, d)
                IF.run(self, vb, sb, s, T, trace=tr, dev=dev)
                alive = tr[:, :, d - 1] > 0.5
                for i in np.flatnonzero(alive.any(1)):
                    OB.append(tr[i, :, :n_obs])
                    AL.append(alive[i])
        else:
            dev = np.zeros((n_ep, 4))
            dev[:, 3] = -1
            tr = trace_array(n_ep, T, d)
            IF.run(self, vb, sb, s, T, trace=tr, dev=dev)
            for i in range(n_ep):
                OB.append(tr[i, :, :n_obs])
                AL.append(tr[i, :, d - 1] > 0.5)
        OB, AL = np.array(OB), np.array(AL)
        if len(OB) > max_traj:
            k = rng.choice(len(OB), max_traj, replace=False)
            OB, AL = OB[k], AL[k]
        return OB, AL

    def coverage_rows(self, bank, n_ep=200, seed=0, lookback=6):
        """Observation rows the agent under search actually visits, and the
        rows just before things went wrong -- from kernel traces.

        A start state has no active vehicle, so `observe(sample_states(...))` is
        all zeros here; the alphabet has to come from trajectories. For the
        vehicle agent every slot is traced in turn (N rollouts of n_ep
        episodes, a second or two); a slot's failure rows are the `lookback`
        ticks before it was removed by a collision -- it vanished before the
        exit -- or before it crossed the stop line on red. For the signal the
        failure rows are the ticks before a tick that paid for a collision or
        a red run.
        """
        from . import intersection_fast as IF
        from ..tick import trace_array
        rng = np.random.default_rng(seed)
        s = self.sample_starts(n_ep, rng)
        T = self.duration
        vb = (bank if self.agent == "vehicle"
              else (self.vehicle_bank or self.default_vehicle_bank()))
        sb = self.signal_bank if self.agent == "vehicle" else bank
        from ..memory import mem_names
        d = len(mem_names(bank["names"], bank.get("mem"))) + 1
        rows, fails = [], []
        if self.agent == "vehicle":
            n_obs = len(self.veh_names)
            ix = self.veh_names.index
            for q in range(self.N):
                dev = np.zeros((n_ep, 4))
                dev[:, 3] = q
                tr = trace_array(n_ep, T, d)
                IF.run(self, vb, sb, s, T, trace=tr, dev=dev)
                alive = tr[:, :, d - 1] > 0.5
                for i in range(n_ep):
                    on = np.flatnonzero(alive[i])
                    if not len(on):
                        continue
                    z = tr[i, on, :n_obs]
                    rows.append(z)
                    last = on[-1]
                    d_stop = z[:, ix("d_stop")]
                    removed_early = (last < T - 1) and d_stop[-1] > -30.0
                    ran = np.flatnonzero((d_stop[:-1] > 0.0) & (d_stop[1:] <= 0.0)
                                         & (np.abs(z[:-1, ix("green")]) < 0.25))
                    bad = list(ran)
                    if removed_early:
                        bad.append(len(on) - 1)
                    for t in bad:
                        fails.append(z[max(0, t - lookback):t + 1])
        else:
            n_obs = len(self.sig_names)
            dev = np.zeros((n_ep, 4))
            dev[:, 3] = -1
            tr = trace_array(n_ep, T, d)
            IF.run(self, vb, sb, s, T, trace=tr, dev=dev)
            r = tr[:, :, d + 2]
            for i in range(n_ep):
                rows.append(tr[i, :, :n_obs])
                for t in np.flatnonzero(r[i] < -2.0):
                    fails.append(tr[i, max(0, t - lookback):t + 1, :n_obs])
        cov = np.concatenate(rows, 0) if rows else np.zeros((0, n_obs))
        fl = np.concatenate(fails, 0) if fails else np.zeros((0, n_obs))
        return cov, fl
