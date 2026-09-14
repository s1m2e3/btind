"""highway-env's model, stepped in lockstep across episodes.

WHY A PORT AT ALL. Measured, highway-env runs one episode in 140 ms; NestWorld
runs one in 0.015 ms. That is ~9000x, and it is not a tuning gap -- highway-env
steps a single object-graph world through a Python loop, one vehicle method call
at a time. A search that prices ~50k proposals at 600 episodes each needs about
84 s per `accept` at that rate, which is a 75-hour run. Multiprocessing does not
close it either: sixteen cores buys 16x and leaves 9 ms an episode.

WHY NUMPY BEFORE NUMBA. The whole speed-up on NestWorld came in two independent
steps, and the FIRST one is the big structural one: stepping every episode in
lockstep over array columns instead of looping. NestWorld's pure-Python path
does 400 episodes in 95 ms, already 600x highway-env, and it is what lets the
existing machinery run UNCHANGED -- `score`, `accept`, `grow`, `memsearch`,
`betasearch` all speak `observe / step / sample_starts`, so matching that
interface is worth more than raw speed. The numba fusion that took NestWorld the
remaining 150-280x is the same follow-on `nest_fast.py` already is, and it can
be written against this once this is verified.

WHAT IS FAITHFUL, taken from the installed highway-env 1.12.1 rather than
remembered: bicycle kinematics with LENGTH 5.0 and the beta = atan(tan(d)/2)
slip form; the lateral controller with KP_LATERAL 1/0.6, KP_HEADING 1/0.2 and
TAU_PURSUIT 0.1; IDM with d0 = 10.0, tau = 1.5, delta = 4.0, a = 3.0, b = -5.0
and ACC_MAX 6.0; MOBIL with politeness 0.0, minimum gain 0.2 and maximum imposed
braking 2.0; MDPVehicle's discrete target speeds [20, 25, 30]; and the highway
reward -- collision -1, right lane 0.1, high speed 0.4 over the range [20, 30],
summed then mapped from [-1, 0.5] onto [0, 1].

WHAT IS APPROXIMATED, and both are stated here because a port that quietly
differs from the thing it is validated against is worse than no port. Collision
uses an axis-aligned overlap (|dx| < LENGTH and |dy| < WIDTH) where highway-env
intersects rotated rectangles; on a straight road headings stay within a few
degrees, so the two agree except in glancing contact. And the road is straight
with `n_lanes` parallel lanes -- which is exactly `highway-v0`, but not the
curved geometry of `roundabout` or the terminating ramp of `merge`.

THIS IS A MODEL OF highway-env, NOT A REIMPLEMENTATION OF IT, and the difference
is a measurement rather than a matter of opinion: `experiments/e25` runs the same
policy through both and compares return, crash rate and episode length. A tree
discovered here is re-scored in real highway-env before any number is reported,
so a divergence shows up as a transfer gap rather than as a result.
"""
import numpy as np

# -- highway-env 1.12.1 constants ------------------------------------------
LENGTH, WIDTH = 5.0, 2.0
MAX_SPEED, MIN_SPEED = 40.0, -40.0
LANE_W = 4.0
KP_A = 1.0 / 0.6
KP_HEADING = 1.0 / 0.2
KP_LATERAL = 1.0 / 0.6
TAU_PURSUIT = 0.1
MAX_STEER = np.pi / 3.0
ACC_MAX, ACC_COMF, DEC_COMF = 6.0, 3.0, -5.0
D0, TAU_W, DELTA = 10.0, 1.5, 4.0
POLITENESS, MIN_GAIN, MAX_IMPOSED_BRAKING = 0.0, 0.2, 2.0
LANE_CHANGE_DELAY = 1.0
TARGET_SPEEDS = np.array([20.0, 25.0, 30.0])
SPEED_RANGE = (20.0, 30.0)
R_COLLISION, R_RIGHT, R_SPEED = -1.0, 0.1, 0.4

FEATURES = ["presence", "x", "y", "vx", "vy"]
ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
FAR = 200.0


def _nz(x, eps=1e-2):
    """highway-env's `not_zero`: keep a denominator away from zero, signed."""
    return np.where(np.abs(x) > eps, x, np.where(x >= 0, eps, -eps))


def feature_names(n_obs_veh):
    """Named columns, plus two PLANTED DISTRACTORS.

    `t_norm` is the fraction of the episode elapsed and `noise` is a constant
    drawn once per episode. Neither carries anything about the road, and both
    are there so that self-discovery is FALSIFIABLE: a search that guards on
    them, or stores them in the blackboard, is buying junk, and without them in
    the alphabet there is no way to tell that from a search that is working.
    NestWorld carries the same two for the same reason -- the memory stage there
    ranked `noise` at rank 1 of 3774 on an in-sample criterion, which is exactly
    the failure they exist to expose.

    They sit at the END of the layout so adding them cannot shift the index of
    any real column.
    """
    out = ["ego_x", "ego_y", "ego_vx", "ego_vy"]
    for i in range(1, n_obs_veh):
        out += ["v%d_seen" % i, "v%d_dx" % i, "v%d_dy" % i,
                "v%d_dvx" % i, "v%d_dvy" % i]
    return out + ["t_norm", "noise"]


class HighwayBatch:
    """`highway-v0` on a straight road, every episode advanced together.

    STATE LAYOUT, flat so the existing machinery can carry it as one array:

        [0 : 4V)          x, y, heading, speed  interleaved per vehicle
        [4V : 5V)         target lane index per vehicle
        [5V : 6V)         lane-change timer per vehicle (MOBIL's delay)
        5 more            ego speed index, crashed, on-road, t, unused

    Vehicle 0 is the ego. The rest are IDM followers with MOBIL lane changes,
    which is what highway-env populates the road with.
    """

    def __init__(self, n_lanes=4, n_veh=15, n_obs_veh=6, duration=40,
                 policy_freq=1, sim_freq=5, gamma=0.99, density=1.0,
                 seed=0):
        """
        `sim_freq` DEFAULTS BELOW highway-env's 15 Hz, and that is a measured
        choice rather than a shortcut. Against the 15 Hz reference, 5 Hz moves
        the return by at most 0.01 on either of the two extreme constant
        policies and leaves the crash rates at 1.6% and 51.1% unchanged, for 2x
        the speed; 3 Hz costs 0.04 for 3.2x.

        `n_veh` IS NOT A SPEED KNOB, which the same sweep showed: dropping 15
        vehicles to 10 moves G(FASTER) by +2.15 and the crash rate from 51.1%
        to 39.2%. That is not a cheaper simulation of the same task, it is a
        different and easier task, so the traffic stays even though it is the
        term the neighbour search is quadratic in.
        """
        self.n_lanes, self.n_veh, self.n_obs_veh = n_lanes, n_veh, n_obs_veh
        self.duration, self.policy_freq, self.sim_freq = duration, policy_freq, sim_freq
        self.n_sub = max(int(round(sim_freq / policy_freq)), 1)
        self.dt = 1.0 / sim_freq
        self.gamma, self.density = gamma, density
        self.names = feature_names(n_obs_veh)
        self.n_act = len(ACTIONS)
        self._kseed = seed

    # -- bookkeeping the rest of the project expects -------------------------
    def seed_kernels(self, seed):
        """Fix whatever is stochastic in the world, separately from the starts.

        Nothing here is stochastic once the start state is drawn -- the traffic
        is deterministic IDM/MOBIL -- so this only records the seed, and exists
        because every stage calls it.
        """
        self._kseed = int(seed)

    @property
    def k(self):
        return 6 * self.n_veh + 5

    # -- state ---------------------------------------------------------------
    def _empty(self, n):
        return np.zeros((n, self.k))

    def sample_states(self, n, rng):
        """Coverage states: vehicles scattered over the road at mixed speeds.

        Used to build the threshold alphabet, so it is deliberately WIDER than
        the start distribution -- a guard has to be proposable on states the
        controller reaches later in an episode, not only on the ones it starts
        from.
        """
        s = self._empty(n)
        V = self.n_veh
        for i in range(V):
            s[:, 4 * i + 0] = rng.uniform(0.0, 400.0, n) if i else 0.0
            lane = rng.integers(0, self.n_lanes, n)
            s[:, 4 * i + 1] = LANE_W * lane
            s[:, 4 * i + 2] = 0.0
            s[:, 4 * i + 3] = rng.uniform(20.0, 30.0, n)
            s[:, 4 * V + i] = lane
            s[:, 5 * V + i] = rng.uniform(0.0, LANE_CHANGE_DELAY, n)
        s[:, 6 * V + 0] = 1.0                      # ego speed index -> 25 m/s
        s[:, 6 * V + 2] = 1.0                      # on road
        s[:, 6 * V + 4] = rng.normal(0.0, 1.0, n)  # the planted distractor
        return s

    def sample_starts(self, n, rng):
        """Episode starts, laid out the way highway-env resets a road.

        The ego begins at the origin; other vehicles are placed ahead and behind
        at an IDM-ish spacing scaled by density, which is what `other_vehicles`
        does when it walks the lanes dropping cars.
        """
        s = self._empty(n)
        V = self.n_veh
        ego_lane = rng.integers(0, self.n_lanes, n)
        s[:, 1] = LANE_W * ego_lane
        s[:, 3] = rng.uniform(23.0, 25.0, n)
        s[:, 4 * V + 0] = ego_lane
        spacing = 25.0 / max(self.density, 1e-6)
        for i in range(1, V):
            lane = rng.integers(0, self.n_lanes, n)
            # walk outward from the ego so the road ahead is populated first
            step = (i + 1) // 2
            sign = 1.0 if i % 2 else -1.0
            s[:, 4 * i + 0] = sign * step * spacing * rng.uniform(0.7, 1.3, n)
            s[:, 4 * i + 1] = LANE_W * lane
            s[:, 4 * i + 3] = rng.uniform(20.0, 28.0, n)
            s[:, 4 * V + i] = lane
            s[:, 5 * V + i] = rng.uniform(0.0, LANE_CHANGE_DELAY, n)
        s[:, 6 * V + 0] = 1.0
        s[:, 6 * V + 2] = 1.0
        s[:, 6 * V + 4] = rng.normal(0.0, 1.0, n)  # the planted distractor
        return s

    # -- views onto the flat state ------------------------------------------
    def _unpack(self, s):
        V = self.n_veh
        p = s[:, :4 * V].reshape(len(s), V, 4)
        return (p[:, :, 0], p[:, :, 1], p[:, :, 2], p[:, :, 3],
                s[:, 4 * V:5 * V], s[:, 5 * V:6 * V], s[:, 6 * V:])

    # -- observation ---------------------------------------------------------
    def observe(self, s):
        """Ego state plus the nearest `n_obs_veh - 1` vehicles, ego-relative.

        Absent slots go to the FAR sentinel rather than to zero. highway-env
        pads with zeros, which reads as a vehicle at zero distance -- the single
        most dangerous state on the road -- and any guard learned against that
        would be learned against a lie.
        """
        X, Y, H, S, TL, TM, EX = self._unpack(s)
        n = len(s)
        out = np.zeros((n, 4 + 5 * (self.n_obs_veh - 1) + 2))
        out[:, 0], out[:, 1] = X[:, 0], Y[:, 0]
        out[:, 2] = S[:, 0] * np.cos(H[:, 0])
        out[:, 3] = S[:, 0] * np.sin(H[:, 0])
        dx = X[:, 1:] - X[:, :1]
        dy = Y[:, 1:] - Y[:, :1]
        d2 = dx ** 2 + dy ** 2
        order = np.argsort(d2, axis=1)[:, :self.n_obs_veh - 1]
        rows = np.arange(n)[:, None]
        gx, gy = dx[rows, order], dy[rows, order]
        gvx = (S[:, 1:] * np.cos(H[:, 1:]))[rows, order] - out[:, 2:3]
        gvy = (S[:, 1:] * np.sin(H[:, 1:]))[rows, order] - out[:, 3:4]
        seen = (np.abs(gx) < FAR).astype(float)
        for j in range(self.n_obs_veh - 1):
            b = 4 + 5 * j
            out[:, b + 0] = seen[:, j]
            out[:, b + 1] = np.where(seen[:, j] > 0.5, gx[:, j], FAR)
            out[:, b + 2] = np.where(seen[:, j] > 0.5, gy[:, j], FAR)
            out[:, b + 3] = np.where(seen[:, j] > 0.5, gvx[:, j], 0.0)
            out[:, b + 4] = np.where(seen[:, j] > 0.5, gvy[:, j], 0.0)
        base = 4 + 5 * (self.n_obs_veh - 1)
        out[:, base] = EX[:, 3] / max(self.duration, 1)          # t_norm
        out[:, base + 1] = EX[:, 4]                              # noise
        return out

    # -- control -------------------------------------------------------------
    def _steering(self, Y, H, S, target_lane):
        """highway-env's `steering_control`, for a straight lane along x."""
        lat = Y - LANE_W * target_lane
        lat_speed_cmd = -KP_LATERAL * lat
        heading_cmd = np.arcsin(np.clip(lat_speed_cmd / _nz(S), -1.0, 1.0))
        heading_ref = np.clip(heading_cmd, -np.pi / 4, np.pi / 4)
        heading_rate = KP_HEADING * (heading_ref - H)
        slip = np.arcsin(np.clip(LENGTH / 2.0 / _nz(S) * heading_rate, -1.0, 1.0))
        return np.clip(np.arctan(2.0 * np.tan(slip)), -MAX_STEER, MAX_STEER)

    def _front(self, X, Y, S, lane_of, target_lane=None):
        """Nearest vehicle ahead in each vehicle's (target) lane, for ALL vehicles.

        THE PYTHON LOOP OVER VEHICLES WAS THE WHOLE COST. This runs four times a
        simulation substep and there are 600 substeps in an episode, so a loop
        of 15 argmins became 36000 of them and the port came out only 8x faster
        than the thing it was replacing. Broadcasting to (n, V, V) is one pass:
        V is 15, so the intermediate is 225 floats per episode, which is nothing
        against paying an interpreter 36000 times.
        """
        # TWO LANE ARRAYS, NOT ONE. `lane_self` is the lane the vehicle being
        # scored is considering; `lane_other` is where everyone else actually
        # is. Passing a single array meant that asking "what if car q moved
        # left" shifted the WHOLE traffic stream left with it, so MOBIL scored
        # a candidate against a road that does not exist -- and the fused
        # kernel, which compares one candidate against the others' real lanes,
        # disagreed with this model by 25 return units. The kernel was right.
        lane_self = lane_of if target_lane is None else target_lane
        lane_other = lane_of
        dx = X[:, None, :] - X[:, :, None]              # [ego i, other j]
        same = np.abs(lane_other[:, None, :] - lane_self[:, :, None]) < 0.5
        d = np.where(same & (dx > 0.0), dx, np.inf)
        n, V = X.shape
        d[:, np.arange(V), np.arange(V)] = np.inf
        j = np.argmin(d, axis=2)
        best = np.take_along_axis(d, j[:, :, None], 2)[:, :, 0]
        has = np.isfinite(best)
        return (np.where(has, best, FAR),
                np.where(has, np.take_along_axis(S, j, 1), 0.0))

    def _idm(self, S, target_speed, gap, front_speed):
        """IDM acceleration, exactly the two-term form highway-env uses."""
        free = ACC_COMF * (1.0 - np.power(np.maximum(S, 0.0)
                                          / np.abs(_nz(target_speed)), DELTA))
        dv = S - front_speed
        d_star = D0 + S * TAU_W + S * dv / (2.0 * np.sqrt(ACC_MAX * ACC_COMF))
        inter = ACC_COMF * np.power(np.maximum(d_star, 0.0) / _nz(gap), 2.0)
        return free - np.where(gap < FAR, inter, 0.0)

    def _front_one(self, X, S, lane_other, lane_q, q):
        """Gap and speed of the nearest vehicle ahead of vehicle `q`.

        One vehicle against the rest, so the reduction is (n, V) rather than
        (n, V, V) -- which is what makes a sequential sweep over vehicles no
        more expensive than the simultaneous one it replaces.
        """
        dx = X - X[:, q:q + 1]
        same = np.abs(lane_other - lane_q[:, None]) < 0.5
        d = np.where(same & (dx > 0.0), dx, np.inf)
        d[:, q] = np.inf
        j = np.argmin(d, axis=1)
        best = np.take_along_axis(d, j[:, None], 1)[:, 0]
        has = np.isfinite(best)
        return (np.where(has, best, FAR),
                np.where(has, np.take_along_axis(S, j[:, None], 1)[:, 0], 0.0))

    def _mobil(self, X, Y, S, lane, timer):
        """MOBIL, swept over vehicles IN ORDER, which is what highway-env does.

        `road.act()` calls each vehicle's `act()` in a loop, so vehicle q+1
        decides against a road in which q has ALREADY moved. Deciding every
        vehicle simultaneously from the old lanes is a different model, and it
        is the one this had: it agreed with the fused kernel exactly up to three
        vehicles -- where no lane change ever happens -- and diverged by 25
        return units at fifteen. The batching that matters is over EPISODES; the
        vehicle axis is 15 long and sequential.

        Politeness is 0.0 in highway-env's default, so the criterion reduces to
        "the change gains me at least MIN_GAIN".
        """
        new_lane = lane.copy()
        V = X.shape[1]
        for q in range(1, V):
            ready = timer[:, q] >= LANE_CHANGE_DELAY
            if not ready.any():
                continue
            g0, f0 = self._front_one(X, S, new_lane, new_lane[:, q], q)
            a0 = self._idm(S[:, q], np.full(len(X), 30.0), g0, f0)
            moved = np.zeros(len(X), bool)
            for d in (-1.0, 1.0):
                cand = np.clip(new_lane[:, q] + d, 0, self.n_lanes - 1)
                live = ready & ~moved & (cand != new_lane[:, q])
                if not live.any():
                    continue
                g1, f1 = self._front_one(X, S, new_lane, cand, q)
                a1 = self._idm(S[:, q], np.full(len(X), 30.0), g1, f1)
                take = live & ((a1 - a0) > MIN_GAIN)
                new_lane[:, q] = np.where(take, cand, new_lane[:, q])
                moved |= take
        return new_lane

    # -- one policy step -----------------------------------------------------
    def step(self, s, a, rng=None):
        """Advance every episode by one POLICY step (`n_sub` simulation steps).

        `rng` is accepted and unused -- the contract `policies.evaluate` calls
        with. Nothing here is stochastic once the start state is drawn, because
        the traffic is deterministic IDM/MOBIL.
        """
        s = s.copy()
        V = self.n_veh
        X, Y, H, S, TL, TM, EX = self._unpack(s)
        a = np.asarray(a).reshape(-1).astype(int)

        # the ego's meta-action: a lane to aim at, or a speed index to track
        si = EX[:, 0]
        TL[:, 0] = np.clip(TL[:, 0] + (a == 0) * -1.0 + (a == 2) * 1.0,
                           0, self.n_lanes - 1)
        si = np.clip(si + (a == 3) * 1.0 + (a == 4) * -1.0, 0,
                     len(TARGET_SPEEDS) - 1)
        EX[:, 0] = si
        ego_target = TARGET_SPEEDS[si.astype(int)]

        crashed = EX[:, 1] > 0.5
        # MOBIL IS EVALUATED ONCE PER POLICY STEP, not per substep. Its own
        # LANE_CHANGE_DELAY is 1.0 s and the policy runs at 1 Hz, so the timer
        # can admit at most one change per policy step anyway; running it
        # fifteen times to have it refuse fourteen of them is pure cost.
        TM += self.n_sub * self.dt
        new_tl = self._mobil(X, Y, S, TL, TM)
        changed = new_tl != TL
        TL[:] = new_tl
        TM[:] = np.where(changed, 0.0, TM)

        for _ in range(self.n_sub):
            lane_now = np.round(Y / LANE_W).clip(0, self.n_lanes - 1)
            gap, fspeed = self._front(X, Y, S, lane_now)
            tgt = np.full_like(S, 30.0)
            tgt[:, 0] = ego_target
            acc = self._idm(S, tgt, gap, fspeed)
            acc[:, 0] = np.clip(KP_A * (ego_target - S[:, 0]), -ACC_MAX, ACC_MAX)
            acc = np.clip(acc, -ACC_MAX, ACC_MAX)

            steer = self._steering(Y, H, S, TL)
            beta = np.arctan(0.5 * np.tan(steer))
            X += S * np.cos(H + beta) * self.dt
            Y += S * np.sin(H + beta) * self.dt
            H += S * np.sin(beta) / (LENGTH / 2.0) * self.dt
            S += acc * self.dt
            np.clip(S, MIN_SPEED, MAX_SPEED, out=S)

            # collision: axis-aligned overlap against every other vehicle
            dx = np.abs(X[:, 1:] - X[:, :1])
            dy = np.abs(Y[:, 1:] - Y[:, :1])
            hit = ((dx < LENGTH) & (dy < WIDTH)).any(axis=1)
            crashed |= hit

        EX[:, 1] = crashed.astype(float)
        on_road = (Y[:, 0] > -LANE_W * 0.5) & (Y[:, 0] < LANE_W * (self.n_lanes - 0.5))
        EX[:, 2] = on_road.astype(float)
        EX[:, 3] += 1.0

        lane = np.round(Y[:, 0] / LANE_W).clip(0, self.n_lanes - 1)
        fwd = S[:, 0] * np.cos(H[:, 0])
        frac = np.clip((fwd - SPEED_RANGE[0]) / (SPEED_RANGE[1] - SPEED_RANGE[0]),
                       0.0, 1.0)
        raw = (R_COLLISION * crashed
               + R_RIGHT * lane / max(self.n_lanes - 1, 1)
               + R_SPEED * frac)
        r = (raw - R_COLLISION) / ((R_SPEED + R_RIGHT) - R_COLLISION)
        r = r * on_road
        done = crashed | (EX[:, 3] >= self.duration) | ~on_road
        return s, r, done
