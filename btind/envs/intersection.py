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
its signal, its leader and the vehicles around it. The SIGNAL tree runs once
per tick over the queues and chooses EXTEND or SWITCH. Both are ordinary banks
with an argmax head; the world is told which one is being searched (`agent`)
and holds the other fixed (`vehicle_bank`, `signal_bank`), so every existing
stage scores one bank at a time against a shared episode return. When no signal
bank is given a fixed-time plan runs; when no vehicle bank is given a
hand-written follower does, so either agent can be studied alone.

WHAT IS REWARDED: TRAFFIC PERFORMANCE, NOT ONLY SAFETY. Every tick, over the
cars in the network and the cars due but blocked at their entrance:

    + W_SPEED  x  mean over approaches of that approach's mean speed / V0
    - W_DELAY  x  sum of (1 - speed / V0)                     per car-second
    - W_QUEUE  x  sum over approaches of (stopped cars there)^2
    - W_STUCK  x  cars stuck                                  per car-second

and on events: + R_EXIT per car leaving, - R_RED per red-light run, - R_COLL
per collision; at the end of the episode, - R_LEFT per car still in the
network or still waiting to enter.

WHY EACH TERM, since each answers a failure measured on an earlier reward:

    speed, per approach   averaged across approaches before across cars, so a
                          busy moving approach cannot hide a starved one
    queue, squared        four cars waiting on each of two approaches cost 32;
                          eight on one approach cost 64 -- serving one phase
                          until the episode ends is no longer cheap
    stuck                 a car below QUEUE_V that is NOT waiting legitimately
                          -- legitimate is before its stop line AND either
                          queued within QUEUE_GAP behind a car or facing a red.
                          Stopped in the box or past it, or stopped on a green
                          with nobody close ahead, is stuck. The first search on
                          the continuous-leaf world stopped every car before the
                          intersection because stopped cars never crash or run
                          a red; now the first car of each such line pays. A
                          first version also called a car stopped on red more
                          than STOP_ZONE short of the line stuck, which charged
                          cautious stopping 70 per episode (e33) -- too much for
                          what is not "stuck in the middle of the road"
    R_LEFT at the end     the signal tree held one phase to the 60 s maximum
                          and never served the north-south left turn inside a
                          100 s episode: cars never served cost only delay
                          until the episode stopped counting. Every car left
                          waiting now pays, and episodes are 150 s with a
                          discount of 0.999, so the end is not discounted away
    R_COLL = 200          at 50, cruising through every red with three crashes
                          an episode still outscored a signal that starves one
                          phase (e33). At 200 any crash-prone controller ranks
                          below every stable one, and no per-tick term buys a
                          crash back

THE VEHICLE TREE IS SEARCHED ON A PER-CAR REWARD (`veh_reward="car"`). The
return above is a team score over up to 64 cars and 300 ticks, and one car
stopping correctly at one red is a small part of it: stage 1 of e28 on it found
a right-turn arm and not the red stop. Every term of the per-car reward belongs
to one car, for what that car did:

    + R_STOP                  once per car: it came to a stop (below QUEUE_V)
                              within STOP_ZONE of its line while its light was red
    + R_GREEN x v / V0        crossing its stop line on green, at its speed
    - R_RED_CAR               crossing its stop line on red. Higher than the
                              team's R_RED: a red run is priced for the risk
                              it takes, not only for the crashes it happens
                              to cause, and in light traffic it rarely causes
                              one -- at R_RED, running reds out-earned waiting
                              for them with two cars an episode (e34)
    - R_COLL                  per car AT FAULT in a collision: in a rear-end the
                              car that ran into the one ahead, in a crossing both.
                              Charging the car that was hit as well made stopping
                              at a red cost the stopper its own crash: measured at
                              16-50 veh/h, a red stop lost -121 per episode and
                              raised crashes 0.02 -> 0.52, so no red stop could be
                              accepted until following existed, and following had
                              nothing to follow until someone stopped. Fault puts
                              the cost on the law that can remove it. The TEAM
                              reward still counts every collision.
    - W_CAR_DELAY x (1 - v/V0) per car-second, blocked-at-entrance cars included
    - W_COMFORT x (dv/dt / B_MAX)^2   per car-second: a regulariser on steep
                              changes of speed, so a stop is a controlled one
    - W_STUCK_CAR             per car-second stuck, as above, except that a
                              red excuses a stop only within NEAR_INT of the
                              line, where the light can be seen: stopped on
                              red further back with nobody close ahead is
                              stuck. Without it, braking every car to a halt
                              where it spawns scored -703 per car against
                              -1161 for cruising (e34)
    + R_EXIT, - R_LEFT        as above

The signal tree is always searched on the team score: serving phases is a
property of all the cars, not of one. The team score stays the yardstick a
vehicle tree is reported on, so a per-car bonus that is farmed at the traffic's
expense shows up as a drop there (e34).

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
T_AR, T_MIN, T_MAX = 3.0, 5.0, 45.0
# T_MAX IS THE CYCLE'S SCALE. Four phases at the cap plus all-red is a
# 4*(45+3) = 192 s cycle, and an episode has to hold two of them for the
# light to be judged on a repeating pattern rather than a fragment: hence
# T_end=384. At the old 60 s cap the cycle was 252 s and two cycles cost
# 9.4x per episode; 60 s was also where the learned green pinned, so the
# bound was setting the policy.
# NOTE it is compiled INTO the kernel (`intersection_fast` clips the plan
# against it), so the numba cache must be cleared when it changes.
N_PHASES = 4
FAR = 200.0
TAU_MAX = 8.0
QUEUE_D, QUEUE_V = 120.0, 2.0
NEAR_D = 40.0      # the closer look-back band; the search picks the horizon

# -- rewards ---------------------------------------------------------------------
# CROSSING ON RED IS NEAR-CATASTROPHIC. At R_RED=10 the discovered car learned
# to run 52 reds an episode in its own training world against the hand-written
# follower's 2.1 -- it was trading red-running for throughput and the team
# return let it, which then made the SIGNAL's objective 49.5% its partner's
# law-breaking. A red-run is now priced like the collision it risks.
# A CRASH IS THE WORST THING THAT HAPPENS HERE, and a red-run is the risk of
# one, so they are priced in that order: 1000 and 100 against an exit's 1.
R_EXIT, R_COLL, R_RED, R_LEFT = 1.0, 1000.0, 100.0, 5.0
# W_STUCK SCALES WITH THE SAFETY WEIGHTS, and has to. It is the term that
# stops a controller buying safety by refusing to serve traffic -- a
# stopped car never crashes or runs a red. Measured at R_COLL=1000 with
# W_STUCK=1: stopping every car scored -1931 against the hand-written
# follower's -2099, so DOING NOTHING WAS THE BEST POLICY AVAILABLE and the
# search would have found it. Raising the crash price without raising this
# one re-opens the hole the stuck term was written to close.
W_SPEED, W_DELAY, W_QUEUE, W_STUCK = 1.0, 0.05, 0.02, 25.0
# STOPPING IS NOT ONE THING, and one weight for it was charging a car that
# panicked in an empty green lane the same as one that braked early for a red.
# Three cases, in the order they deserve:
#   FREE    stopped on GREEN with nobody close ahead -- nothing is stopping it
#   QUEUE   stopped on GREEN behind a leader -- the queue should be discharging
#   EARLY   stopped on RED with no leader, charged BY DEGREE: the road left
#           empty beyond STOP_ZONE, so waiting at the line is free and
#           abandoning the approach 150 m back is not
# Stopping on red near the line, or behind a queue on red, is free -- that is
# what a car is supposed to do, and charging it cost 70 an episode in e33.
# EARLY IS CHARGED PER TICK, so its weight is a RATE and must be sized as one.
# At 5.0 (x3 for the per-car scale) a car waiting 30 s at a red 40 m back paid
# 900 against the 200 it saved by not running the red, so braking for a red went
# net negative and the planted rule in test_kernsearch stopped paying -- its best
# candidate fell from tens of units to +1.35 +-5.77.
#
# AND IT IS CHARGED BY DEGREE, NOT BY A CLIFF. There is no one distance that is
# "too early": a car at 11 m/s needs 13 m to stop, so stopping at 40 m is careful
# and stopping at 150 m is abandoning the approach, and a threshold between them
# would call the first a fault and the second no worse. The charge grows with how
# much road is left standing empty beyond a normal stopping zone, which is also
# what lets a far car brake smoothly and a near one brake hard without either
# being penalised for the profile it needed.
W_STUCK_FREE, W_STUCK_QUEUE, W_STUCK_EARLY = 25.0, 8.0, 0.5
# GREEN HELD ON A PHASE WITH NOBODY TO SERVE, per second, charged to the signal.
# Modelled on sumo_test's `LAMBDA_OVER * relu(T_k - T_k_floor)`, whose floor is
# ZERO when no vehicle sits within its commit distance -- so any green on an
# empty phase is charged directly rather than paid for later in someone else's
# delay. Measured on this world, delay and queue together were 0.5% of the
# signal's return, so holding 45 s on an empty phase cost it almost nothing and
# that is exactly what it learned to do.
W_OVER, COMMIT_D = 1.0, 60.0
QUEUE_GAP, STOP_ZONE = 12.0, 15.0
LONG_QUEUE = 6           # a phase queue this long anchors the signal's proposals
# the per-car reward (veh_reward="car"): see the module docstring
R_STOP, R_GREEN, W_CAR_DELAY, R_RED_CAR, W_STUCK_CAR = 3.0, 2.0, 0.1, 200.0, 75.0
# A RED-RUN IS PRICED BY THE CONFLICT IT CREATES, not by the fact of it. The
# comment above R_RED says the quiet part: "a red-run is now priced like the
# collision it risks" -- a flat fee standing in for a risk the world already
# measures. Measured on the hand-written follower: 7.83 red-runs an episode,
# 3.83 crashes, and EVERY crash a rear-end -- zero crossing collisions from 47
# red-runs, because a car entering during the 3 s all-red is clear before the
# conflicting movement moves. The flat fee was charging ~1566 an episode for
# harm that did not occur, and it charges the same whether a car slips through
# an empty junction or cuts across an oncoming platoon.
#
# So the car's charge scales with `rival_dt` -- the smallest time gap to a
# vehicle heading for a shared conflict point, which is post-encroachment time,
# the standard surrogate for exactly this. It is the car's OWN observation
# column, deliberately: a car is charged by a number it can see, so the rule
# that avoids the charge is one its tree can express, and it covers both the
# committed crossing of a car that cannot stop and the deliberate one of a car
# waiting at a line with nothing coming, without either being named here.
#
# The floor is not zero. There is a cost to running a red beyond the crash it
# risks, and at R_RED=10 the discovered car ran 52 an episode and made the
# signal's objective 49.5% its partner's law-breaking.
R_RED_FLOOR, T_SAFE = 20.0, 3.0
# COMFORT: each car's change of speed, squared, per car-second, normalised by
# the braking limit -- a full-braking second costs W_COMFORT, a gentle stop at
# half the rate a quarter of that per second. Measured on the discrete tree of
# e35 cycle 0: every stop, for a red or for a leader, was full braking.
W_COMFORT = 1.0

ACCELS = np.array([-B_MAX, -1.5, 0.0, 1.0, A_MAX])
ACTIONS = ["BRAKE_HARD", "BRAKE", "HOLD", "ACCEL", "ACCEL_MAX"]
SIG_ACTIONS = ["EXTEND", "SWITCH"]
FIXED_GREEN = np.array([30.0, 10.0, 30.0, 10.0])

# OPERATING CONDITIONS, drawn per episode when a world is built with
# `conditions=WIDE` (or any dict of the same keys): a controller trained on one
# demand and one plan learns that demand and that plan -- measured, the e34 tree
# carried `t_sig>28.9`, a rule about the fixed plan's 30 s green. Every episode
# of a batch draws its own:
#     approach_vph   total flow on EACH approach, independently   veh/h
#                    (with approach_skew: the junction's BASE flow, one
#                     draw per episode, multiplied per approach)
#     approach_skew  per-approach multiplier on that base, one draw each,
#                    so imbalance varies independently of total load
#     left, right    turning shares on each approach
#     green_through  the fixed plan's green for phases 0 and 2     s
#     green_left     the fixed plan's green for phases 1 and 3     s
WIDE = dict(approach_vph=(100.0, 600.0), left=(0.1, 0.3), right=(0.1, 0.3),
            green_through=(15.0, 45.0), green_left=(5.0, 20.0))

# DISTRIBUTED CONTROL: a vehicle observes ITSELF -- its speed, where it is
# relative to its stop line and its own first conflict point, what its front
# sensor reports -- and what the INTERSECTION tells every approaching car: the
# signal for its movement, how long the phase has run, whether all-red is on,
# and whether it is close enough to the box to be the intersection's concern.
# Nothing about the other vehicles' plans or positions beyond the one ahead:
# the same subtree runs in every car on that car's own information.
NEAR_INT = 60.0
SIG_HIDDEN = -1.0     # what the signal columns read when the car is not near
# `t_sig`, `t_sig_max`: in continuous mode how long the phase has run (and the
# sentinel). In event mode the message's WINDOW: the earliest and the latest
# tick, counted from the message, at which MY light can change -- what a SAE
# J2735 SPaT broadcast carries as minEndTime / maxEndTime.
#
# A RADIUS SENSOR, beside the leader. A car sees every vehicle within SENSE_R
# metres in the plane -- ahead, behind, beside and across the box -- reduced to
# six numbers a guard can read:
#   n_near        how many
#   near_d        distance to the nearest one (FAR when none)
#   near_closing  how fast that nearest one is getting closer, m/s (0 when none)
#   has_rival     some vehicle in range is heading for a conflict point it
#                 shares with me, and neither of us has passed it
#   rival_dt      the smallest gap, s, between when I and such a rival reach
#                 our shared point at current speeds (FAR when none) -- the
#                 gap-acceptance quantity a driver uses at a crossing
#   rival_d       that rival's distance to the shared point
# Measured before it existed (e35 cycle 0): the leader was the only other
# vehicle a car could perceive, so crossing traffic was avoided only by obeying
# the light.
SENSE_R = 50.0
T_V_MIN = 0.5          # the speed floor a time-to-conflict is computed with
VEH_NAMES = ["v", "d_stop", "near_int", "green", "t_sig", "t_sig_max", "all_red",
             "lead_gap", "lead_dv", "has_lead", "d_conf", "is_left", "is_right",
             "t_norm", "noise",
             "n_near", "near_d", "near_closing", "has_rival", "rival_dt", "rival_d"]
# Per phase k, over the vehicles whose movement phase k serves and that are
# approaching (0 < distance to the stop line < QUEUE_D):
#   q<k>a q<k>b  how many are QUEUED (speed below QUEUE_V) on each of the two
#                approaches that phase serves, a = the lower approach index
#   vq<k>a/b     the mean speed OF THOSE QUEUED cars, SIG_EMPTY when none
#   n<k>         how many are approaching within QUEUE_D
#   nn<k>        how many are approaching within NEAR_D -- a second, closer
#                horizon, so how far back to look is a column the search picks
#                rather than a constant chosen here
#   d<k>         the distance of the nearest one to its stop line, FAR when none
# -- what loop detectors and a V2I receiver give an actuated controller.
#
# WHY PER APPROACH. A phase serves two opposing approaches and they discharge in
# PARALLEL, so the green it needs is set by its LONGER queue, not by the total:
# summed, 22 is ambiguous between 11+11 (clears in ~11) and 20+2 (needs ~20).
# `approach_skew` draws 0.25-1.75 independently per approach, so that imbalance
# is common by construction and was invisible.
#
# WHY THE QUEUE'S OWN SPEED. The old v<k> averaged every APPROACHING car, free-
# flowing ones 100 m back included, so it could not say whether a queue was
# discharging or standing still -- which is the thing a controller acts on.
SIG_EMPTY = -1.0
# The first four blocks are the original layout, unchanged and at the same
# offsets so nothing that indexes them by position or name moves. The three
# appended blocks are the new information.
Q_OFF, N_OFF, V_OFF, D_OFF, PH_OFF = 0, N_PHASES, 2 * N_PHASES, 3 * N_PHASES, 4 * N_PHASES
MISC_OFF = 5 * N_PHASES
QS_OFF, VA_OFF, NN_OFF = 5 * N_PHASES + 4, 7 * N_PHASES + 4, 9 * N_PHASES + 4
SIG_NAMES = (["q%d" % k for k in range(N_PHASES)]
             + ["n%d" % k for k in range(N_PHASES)]
             + ["v%d" % k for k in range(N_PHASES)]
             + ["d%d" % k for k in range(N_PHASES)]
             + ["ph%d" % k for k in range(N_PHASES)]
             + ["t_phase", "all_red", "t_norm", "noise"]
             # --- appended: what the four blocks above cannot express ---------
             # q<k>a/b  the QUEUED count split by the two approaches the phase
             #          serves (a = lower index). They discharge in PARALLEL, so
             #          the green a phase needs is set by the LONGER queue, not
             #          the total: summed, 22 is ambiguous between 11+11 (clears
             #          in ~11) and 20+2 (needs ~20), and `approach_skew` draws
             #          0.25-1.75 per approach so that gap is common.
             # va<k>a/b the mean speed of every vehicle within QUEUE_D that phase
             #          k serves on that approach -- moving or stopped, SIG_EMPTY
             #          when none. SAME GRANULARITY AS THE QUEUE, so through and
             #          left are separable on one approach: (k=0, a) is approach
             #          a's through+right and (k=1, a) its left.
             #          It replaces a mean over QUEUED cars, measured to carry
             #          almost nothing -- 46% of ticks the empty sentinel and the
             #          rest averaging 0.04 m/s against a 1.8 ceiling, because a
             #          car only counts as queued below QUEUE_V=2. That said "is
             #          there a queue", which q<k>a/b already says.
             # nn<k>    approaching within NEAR_D rather than QUEUE_D: a second,
             #          closer horizon, so how far back to look becomes a column
             #          the search picks instead of a constant chosen here.
             + ["q%d%s" % (k, sd) for k in range(N_PHASES) for sd in "ab"]
             + ["va%d%s" % (k, sd) for k in range(N_PHASES) for sd in "ab"]
             + ["nn%d" % k for k in range(N_PHASES)])


def _kind_of(signal_bank):
    """The signal controller a bank describes: no bank means the fixed plan."""
    if signal_bank is None:
        return "fixed"
    return "duration" if signal_bank.get("head") == "duration" else "argmax"


def load_geometry(path=GEOM_PATH):
    g = np.load(path, allow_pickle=False)
    out = {k: g[k] for k in g.files}
    out["lane_group"] = out["appr"] * 2 + (out["dirs"] == 2)     # left alone
    out["s_cp"] = np.nan_to_num(out["s_cp"], nan=-1e9)
    # polylines are NaN-padded to 7 vertices; a padded vertex is never reached
    out["n_vert"] = np.isfinite(out["cum"]).sum(1).astype(np.int64)
    out["cum_f"] = np.where(np.isfinite(out["cum"]), out["cum"], np.inf)
    out["pts_f"] = np.nan_to_num(out["pts"], nan=0.0)
    return out


class IntersectionBatch:
    """Every episode advanced together over array columns; N slots per episode.

    STATE LAYOUT per episode, flat:
        [0:N)     movement index of each slot
        [N:2N)    arc-length s
        [2N:3N)   speed v; for a pending slot, the speed it will enter at
        [3N:4N)   status: 0 pending, 1 active, 2 done
        [4N:5N)   scheduled depart time
        5N + 0..9 phase, t_phase, in_ar, ar_t, tick, noise, n_coll, n_red,
                  n_rear, n_cross
        5N + 10   the signal's planned green time (duration head), 0 = none yet
        5N + 11   the planted clock's phase, uniform in [0, 1), NEVER observed
        [5N+12 : 6N+12)  told: the slot has received its message (event mode)
        [6N+12 : 7N+12)  paid: the slot has collected its red-stop bonus
        [7N+12 : 7N+16)  this episode's fixed-plan green time per phase

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
    N_MISC = 12
    # Bumped whenever an observation column changes meaning. It is part of the
    # world signature, so the store and the proposal weights of an older
    # observation are never resumed on this one -- a checkpoint that learned
    # to read elapsed time off `t_norm` must not warm-start a world where
    # `t_norm` carries nothing.
    OBS_VERSION = 8
    REWARD_VERSION = 6          # part of the store key, like OBS_VERSION

    def __init__(self, n_max=64, T_end=150.0, dt=0.5, vph=(200.0, 60.0, 60.0),
                 gamma=0.999, spawn_back=90.0, exit_after=40.0, seed=0,
                 sig_mode="continuous", veh_head="argmax", sig_head="argmax",
                 occlude=None, veh_reward="team", entry_v=None, conditions=None):
        assert sig_mode in ("continuous", "event")
        self.cond = dict(conditions) if conditions else None
        # a string, so the condition distribution is part of the store key
        self.conditions = (";".join("%s=%s" % (k, ",".join("%g" % x for x in v))
                                    for k, v in sorted(self.cond.items()))
                           if self.cond else "")
        assert veh_reward in ("team", "car")
        self.veh_reward = veh_reward
        # ENTRY SPEEDS. By default every car enters at V0, so a cold start never
        # sees a slow car and cannot tell HOLD (keep this speed) from ACCEL:
        # measured, CEM picked HOLD on one seed in three, and a tree that brakes
        # for a red then never moves again. `entry_v=(lo, hi)` draws each car's
        # entry speed, a trajectory-level diversity of starts, in the key.
        self.entry_lo, self.entry_hi = ((float(entry_v[0]), float(entry_v[1]))
                                        if entry_v else (V0, V0))
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
        # a string, so the demand is in the store key (tuples are not)
        self.demand = "%g/%g/%g" % self.vph
        self.spawn_back, self.exit_after = float(spawn_back), float(exit_after)
        self.geom = load_geometry()
        self.M = len(self.geom["path_len"])
        self.veh_names, self.sig_names = list(VEH_NAMES), list(SIG_NAMES)
        self._phase_appr = None
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
        self.k = 7 * self.N + self.N_MISC + N_PHASES
        self.obs_version = self.OBS_VERSION
        self.reward_version = self.REWARD_VERSION
        self.terms = None               # set to {} to collect per-term totals

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

    @property
    def u_null(self):
        """The continuous command that expresses no opinion, for a cold start.

        NOT the middle of the range. For acceleration the middle is -0.95 m/s^2,
        and a tree starting there brakes every car to a halt before it reaches
        the zone where the light is heard -- measured, the memory stage then
        found no message to store at all. Zero acceleration holds the entry
        speed; for green time the middle of the range is a plain split.
        """
        return 0.0 if self.agent == "vehicle" else 0.5 * (T_MIN + T_MAX)

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
        plans = np.tile(FIXED_GREEN, (n, 1))
        cd = self.cond
        for i in range(n):
            if cd:
                # this episode's conditions, drawn before its arrivals
                U = lambda key, size=None: rng.uniform(cd[key][0], cd[key][1], size)
                # HOW BUSY, AND HOW UNEVENLY, ARE DRAWN SEPARATELY when
                # `approach_skew` is given: one load for the junction times a
                # multiplier per approach. Four independent draws from one
                # range confound the two -- a heavily loaded episode and a
                # lopsided one are the same event -- and a signal only earns
                # its keep where the approaches disagree, so the skew has to
                # vary at a fixed total and the total at a fixed skew.
                if "approach_skew" in cd:
                    q_app = float(U("approach_vph")) * U("approach_skew", 4)
                else:
                    q_app = U("approach_vph", 4)
                sh_l, sh_r = U("left", 4), U("right", 4)
                share = np.stack([sh_r, 1.0 - sh_l - sh_r, sh_l], 1)   # by dirs r, s, l
                rate = np.array([q_app[a] * share[a, int(dd)] for a, dd in
                                 zip(self.geom["appr"], self.geom["dirs"])]) / 3600.0
                gt, gl = U("green_through", 2), U("green_left", 2)
                plans[i] = [gt[0], gl[0], gt[1], gl[1]]
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
        # THE STARTING PHASE IS DRAWN, not always 0. An episode is 150 s and a
        # cycle is 80 s on the fixed plan, so only 3.6 phase changes happen in
        # one; starting every episode at phase 0 meant the order 0->1->2->3 cut
        # phase 3 off -- reached in 60% of episodes, 4% of ticks, against phase
        # 2's 41%. The light could only learn distinctions among the phases it
        # was shown, and a longer green it learns makes the tail phases rarer
        # still. Drawing the start spreads the exposure evenly at no cost.
        s[:, b + 0] = (0.0 if not cd else
                       rng.integers(0, N_PHASES, n).astype(float))
        s[:, b + 1] = (rng.uniform(0.0, FIXED_GREEN[0], n) if not cd
                       else rng.uniform(0.0, 1.0, n) * plans[:, 0])
        s[:, 7 * N + self.N_MISC:7 * N + self.N_MISC + N_PHASES] = plans
        s[:, b + 5] = rng.normal(0.0, 1.0, n)
        # THE PLANTED CLOCK'S PHASE IS DRAWN INDEPENDENTLY AND NEVER OBSERVED.
        # `t_norm` was the raw fraction of the episode elapsed, and on a
        # truncated episode that is a real horizon cue: it predicts whether a
        # car can still exit and how much delay is left to charge, and both
        # trees bought it. A first fix offset it by a phase computed from
        # `noise` -- but `noise` is observed, so a conjunction of the two could
        # still decode elapsed time. A phase from its own draw, held in state
        # and never shown, makes `t_norm` uniform and independent of the tick,
        # of `noise`, and of everything else a tree can read.
        s[:, b + 11] = rng.uniform(0.0, 1.0, n)
        s[:, 2 * N:3 * N] = V0
        if self.entry_lo < V0 or self.entry_hi < V0:
            # drawn last, so the default world's stream is unchanged
            s[:, 2 * N:3 * N] = rng.uniform(self.entry_lo, self.entry_hi, (n, N))
        return s

    def _unpack(self, s):
        N = self.N
        return (s[:, 0:N].astype(int), s[:, N:2 * N], s[:, 2 * N:3 * N],
                s[:, 3 * N:4 * N], s[:, 4 * N:5 * N],
                s[:, 5 * N:5 * N + self.N_MISC])

    def _told(self, s):
        N = self.N
        return s[:, 5 * N + self.N_MISC:6 * N + self.N_MISC]

    def _paid(self, s):
        N = self.N
        return s[:, 6 * N + self.N_MISC:7 * N + self.N_MISC]

    def plan(self, s):
        """(n, 4): each episode's fixed-plan green times."""
        b = 7 * self.N + self.N_MISC
        return s[:, b:b + N_PHASES]

    n_cond_groups = 3           # raise it to resolve the demand axis more finely

    def acting_ticks(self, tr):
        """Boolean (n, T): the ticks where the agent under search sets an action
        that something reads. None means "every tick", which is the car's case.

        A CAR COMMANDS AN ACCELERATION EVERY TICK; THE LIGHT DOES NOT. It picks
        a green time when a phase begins and is not asked again until that phase
        ends, so a deviation placed at any other tick overrides a command nobody
        reads and returns an advantage of exactly zero. Measured on e37's best
        pair, 91.3% of 3000 signal deviations came back exactly 0.0 against the
        car's 23%: the light was learning from a ninth of its budget, and the
        zeros also flattened the critic's training targets and broke the metrics
        that are supposed to notice (`intersection_critic`).

        A new phase is where `t_phase` stops increasing.
        """
        if self.agent != "signal":
            return None
        # THE PHASE INDICATOR, not `t_phase`. Read off t_phase ("it stopped
        # increasing") this over-flagged by 4.7x -- 16.8 ticks an episode
        # against 3.6 real phase starts -- because t_phase also plateaus inside
        # the all-red interval. The phase index changing is exact.
        ph = np.stack([tr[:, :, self.sig_names.index("ph%d" % k)]
                       for k in range(N_PHASES)], -1).argmax(-1)
        new = np.zeros(ph.shape, bool)
        new[:, 0] = True
        new[:, 1:] = ph[:, 1:] != ph[:, :-1]
        return new

    def condition_groups(self, s, n_groups=None):
        """Episode labels by demand (cars scheduled), for per-condition
        acceptance; None when the world has no condition distribution.

        TRAINING THE WHOLE DEMAND RANGE IN ONE BATCH puts light and heavy
        traffic in the same test, and the per-group rule in `structure.accept`
        is what stops one paying for the other -- so the number of groups is
        the resolution of that protection.

        FEWER GROUPS MAY COME BACK THAN ASKED FOR. The label is the count of
        cars scheduled, an integer, so on a narrow demand range two quantile
        cuts can land on the same count and the groups between them are empty.
        `accept` iterates the labels that actually occur, so this is safe --
        but it means the resolution is bounded by the spread of the demand,
        not by this number alone.
        """
        n_groups = self.n_cond_groups if n_groups is None else n_groups
        if not self.cond:
            return None
        cars = np.isfinite(s[:, 4 * self.N:5 * self.N]).sum(1)
        cuts = np.unique(np.quantile(cars, np.linspace(0, 1, n_groups + 1)[1:-1]))
        return np.searchsorted(cuts, cars, side="right")

    _reward_override = None
    # THE OBJECTIVE A CANDIDATE IS PRICED ON, when it differs from the one the
    # agent's own proposals are generated from: joint training accepts every
    # move -- a car's and the signal's -- on the TEAM return, so the two agents'
    # gains are the same quantity, while the vehicle's deviations and critic
    # keep the per-car return, where credit assignment lives.
    score_reward = None

    def reward_as(self, mode):
        """Context manager: price rollouts on `mode` ("car"/"team"/None) here."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            was = self.score_reward
            self.score_reward = mode
            try:
                yield self
            finally:
                self.score_reward = was
        return _cm()

    @property
    def own_reward(self):
        """The return the agent under search is RESPONSIBLE for: the per-car
        reward for the vehicle tree when `veh_reward` is "car", the team score
        otherwise. A reference run can name either (`reward=`)."""
        if self._reward_override is not None:
            return self._reward_override
        return "car" if (self.agent == "vehicle" and self.veh_reward == "car") else "team"

    @property
    def reward_mode(self):
        """The return a rollout is PRICED on here. Its own, unless a round has
        named another with `reward_as` -- joint training accepts both agents'
        moves on the team return. One property, so the kernel (which reads it
        when no explicit `reward=` is given) and the Python fallback in
        `structure.score` cannot disagree about which objective is in force."""
        if self._reward_override is not None:
            return self._reward_override
        return self.score_reward or self.own_reward

    _sig_kind_override = None

    def _sig_kind(self):
        """Which controller the light is under: 'fixed', 'duration' or 'argmax'.

        The message has to describe THAT controller. `python_rollout` names it
        explicitly; the generic interface reads it off the banks in force.
        """
        if self._sig_kind_override is not None:
            return self._sig_kind_override
        sb = self.signal_bank if self.agent == "vehicle" else self._search_bank
        return _kind_of(sb)

    def _t_window(self, X, m, kind, greens=None):
        """(earliest, latest) ticks until movement m's light is OBSERVED to change.

        A TRUTHFUL MESSAGE, NOT A PROMISE. The first version computed the time
        from the fixed-time plan whatever controller was running, so once a
        signal tree set its own green times the cars stored a number that was
        false. A real broadcast (SAE J2735 SPaT) sends a window instead, because
        an actuated controller has not decided its future yet:

            fixed      exact: every future phase is known
            duration   the running phase is exact once its green time is
                       committed; phases not yet started lie in [T_MIN, T_MAX]
            argmax     a per-tick switcher only guarantees [T_MIN, T_MAX]

        Counted in ticks from the dynamics' own switching rules, not from
        seconds: the fixed plan decides before elapsed time is incremented and
        the tree controllers after, and an all-red lasts ceil(T_AR / dt) ticks.
        `tests/test_intersection_spat.py` checks that the change always lands
        inside the window, exactly, for all three controllers.
        """
        dt, eps = self.dt, 1e-9
        g = self.geom["green"]
        n = len(X)
        ph = X[:, 0].astype(int)
        tau, in_ar, ar_t, plan = X[:, 1], X[:, 2] > 0.5, X[:, 3], X[:, 10]

        def live(bound, t0):
            k = np.ceil((bound - t0) / dt - eps)
            return (np.maximum(0.0, k) + 1.0 if kind == "fixed"
                    else np.maximum(1.0, k))
        if kind == "fixed":
            P = np.tile(FIXED_GREEN, (n, 1)) if greens is None else greens
            cur_min = cur_max = live(P[np.arange(n), ph], tau)
            fut_min = fut_max = live(P, 0.0)                    # (n, 4)
        else:
            known = (plan > 0.0) if kind == "duration" else np.zeros(n, bool)
            cur_min = np.where(known, live(plan, tau), live(T_MIN, tau))
            cur_max = np.where(known, live(plan, tau), live(T_MAX, tau))
            fut_min = np.full((n, N_PHASES), live(T_MIN, 0.0))
            fut_max = np.full((n, N_PHASES), live(T_MAX, 0.0))
        ar_full = float(np.ceil(T_AR / dt - eps))
        ar_now = np.maximum(1.0, np.ceil((T_AR - ar_t) / dt - eps))
        gm = np.take_along_axis(g[ph], m, 1) & ~in_ar[:, None]
        nxt_min = np.where(in_ar, ar_now, cur_min + ar_full)
        nxt_max = np.where(in_ar, ar_now, cur_max + ar_full)
        tmin = np.where(gm, cur_min[:, None], nxt_min[:, None])
        tmax = np.where(gm, cur_max[:, None], nxt_max[:, None])
        pending = ~gm
        for k in range(1, N_PHASES + 1):
            q = (ph + k) % N_PHASES
            green_q = np.take_along_axis(g[q], m, 1)
            add = pending & ~green_q
            fq_min = np.take_along_axis(fut_min, q[:, None], 1)[:, 0]
            fq_max = np.take_along_axis(fut_max, q[:, None], 1)[:, 0]
            tmin = tmin + np.where(add, (fq_min + ar_full)[:, None], 0.0)
            tmax = tmax + np.where(add, (fq_max + ar_full)[:, None], 0.0)
            pending = pending & ~green_q
        return tmin, tmax

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

    def _xy(self, m, S):
        """Position (x, y) in the plane and unit heading (hx, hy) of each slot,
        from its movement's polyline and its arc-length."""
        g = self.geom
        cum, pts, nv = g["cum_f"], g["pts_f"], g["n_vert"]
        k = (S[..., None] >= cum[m]).sum(-1) - 1
        k = np.minimum(np.maximum(k, 0), nv[m] - 2)
        c0 = np.take_along_axis(cum[m], k[..., None], -1)[..., 0]
        c1 = np.take_along_axis(cum[m], (k + 1)[..., None], -1)[..., 0]
        P = pts[m]                                          # (n, N, 7, 2)
        p0 = np.take_along_axis(P, k[..., None, None].repeat(2, -1), -2)[..., 0, :]
        p1 = np.take_along_axis(P, (k + 1)[..., None, None].repeat(2, -1), -2)[..., 0, :]
        seg = c1 - c0
        f = (S - c0) / seg
        x = p0[..., 0] + f * (p1[..., 0] - p0[..., 0])
        y = p0[..., 1] + f * (p1[..., 1] - p0[..., 1])
        hx = (p1[..., 0] - p0[..., 0]) / seg
        hy = (p1[..., 1] - p0[..., 1]) / seg
        return x, y, hx, hy

    def _rival_dt(self, m, S, V, act):
        """`rival_dt` per slot: the smallest time gap to a vehicle heading for a
        shared conflict point, FAR when there is none.

        The fifth column of `_sense`, computed on its own so the reward can
        charge a red-run by the conflict it creates. The definition is copied
        exactly -- same conflict test, same sensor range, same clamps -- because
        the point is that the charge equals the number the car observed.
        """
        g = self.geom
        x, y, _, _ = self._xy(m, S)
        dx = x[:, None, :] - x[:, :, None]
        dy = y[:, None, :] - y[:, :, None]
        dist = np.sqrt(dx * dx + dy * dy)
        eye = np.eye(S.shape[1], dtype=bool)[None]
        inR = act[:, :, None] & act[:, None, :] & ~eye & (dist <= SENSE_R)
        mq, mo = m[:, :, None], m[:, None, :]
        d_me = g["s_cp"][mq, mo] - S[:, :, None]
        d_rv = g["s_cp"][mo, mq] - S[:, None, :]
        ok = inR & g["conf"][mq, mo] & (d_me > -HALF_CONF) & (d_rv > -HALF_CONF)
        t_me = np.maximum(d_me, 0.0) / np.maximum(V[:, :, None], T_V_MIN)
        t_rv = np.maximum(d_rv, 0.0) / np.maximum(V[:, None, :], T_V_MIN)
        gap_t = np.where(ok, np.abs(t_rv - t_me), np.inf)
        best = gap_t.min(2)
        return np.where(np.isfinite(best), best, FAR)

    def _sense(self, m, S, V, act):
        """The radius sensor's six columns per slot (see VEH_NAMES)."""
        g = self.geom
        n, N = S.shape
        x, y, hx, hy = self._xy(m, S)
        dx = x[:, None, :] - x[:, :, None]                  # (n, me, other)
        dy = y[:, None, :] - y[:, :, None]
        dist = np.sqrt(dx * dx + dy * dy)
        eye = np.eye(N, dtype=bool)[None]
        inR = act[:, :, None] & act[:, None, :] & ~eye & (dist <= SENSE_R)
        n_near = inR.sum(2).astype(float)
        dd = np.where(inR, dist, np.inf)
        jn = np.argmin(dd, 2)
        best = np.take_along_axis(dd, jn[:, :, None], 2)[:, :, 0]
        has_near = np.isfinite(best)
        vx, vy = V * hx, V * hy
        dvx = vx[:, None, :] - vx[:, :, None]
        dvy = vy[:, None, :] - vy[:, :, None]
        rate = np.take_along_axis(dx * dvx + dy * dvy, jn[:, :, None], 2)[:, :, 0]
        closing = np.where(has_near & (best > 1e-9), -(rate / np.where(best > 1e-9, best, 1.0)),
                           0.0)
        mq, mo = m[:, :, None], m[:, None, :]
        conf = g["conf"][mq, mo]
        d_me = g["s_cp"][mq, mo] - S[:, :, None]
        d_rv = g["s_cp"][mo, mq] - S[:, None, :]
        ok = inR & conf & (d_me > -HALF_CONF) & (d_rv > -HALF_CONF)
        t_me = np.maximum(d_me, 0.0) / np.maximum(V[:, :, None], T_V_MIN)
        t_rv = np.maximum(d_rv, 0.0) / np.maximum(V[:, None, :], T_V_MIN)
        gap_t = np.where(ok, np.abs(t_rv - t_me), np.inf)
        jr = np.argmin(gap_t, 2)
        rbest = np.take_along_axis(gap_t, jr[:, :, None], 2)[:, :, 0]
        has_rival = np.isfinite(rbest)
        r_d = np.take_along_axis(d_rv, jr[:, :, None], 2)[:, :, 0]
        return (n_near, np.where(has_near, best, FAR), closing, has_rival.astype(float),
                np.where(has_rival, rbest, FAR), np.where(has_rival, r_d, FAR))

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
            t_min, t_max = self._t_window(X, m, self._sig_kind(), self.plan(s))
        else:
            hear = near | (d_stop <= 0.0)
            t_min = np.broadcast_to(X[:, 1:2], d_stop.shape)
            t_max = np.full(d_stop.shape, SIG_HIDDEN)
        o[:, :, 3] = np.where(hear, gm, SIG_HIDDEN)
        o[:, :, 4] = np.where(hear, t_min, SIG_HIDDEN)
        o[:, :, 5] = np.where(hear, t_max, SIG_HIDDEN)
        o[:, :, 6] = np.where(hear, X[:, 2:3], SIG_HIDDEN)
        o[:, :, 7] = gap
        o[:, :, 8] = np.where(has, V - v_lead, 0.0)
        o[:, :, 9] = has
        o[:, :, 10] = d_conf
        o[:, :, 11] = g["dirs"][m] == 2
        o[:, :, 12] = g["dirs"][m] == 0
        # `t_norm` IS A PLANTED DISTRACTOR AND HAS TO STAY ONE: a clock whose
        # phase is a hidden per-episode draw (see `sample_starts`), so no
        # observed column can recover elapsed time from it.
        o[:, :, 13] = np.mod(X[:, 4:5] / self.duration + X[:, 11:12], 1.0)
        o[:, :, 14] = X[:, 5:6]
        for c_i, col in enumerate(self._sense(m, S, V, act)):
            o[:, :, 15 + c_i] = col
        o[~act] = 0.0
        return o.reshape(n * N, -1)

    @property
    def phase_appr(self):
        """(a, b) approach indices each phase serves, a the lower. A phase
        covers exactly two opposing approaches; they discharge in parallel, so
        the queue that matters is the longer of the two, not their sum."""
        if self._phase_appr is None:
            g, out = self.geom, []
            for k in range(N_PHASES):
                aps = sorted({int(a) for a, ok in zip(g["appr"], g["green"][k]) if ok})
                out.append((aps[0], aps[-1]))
            self._phase_appr = out
        return self._phase_appr

    def observe_signal(self, s):
        m, S, V, st, dep, X = self._unpack(s)
        n, N = S.shape
        act = st == 1.0
        g = self.geom
        d_stop = g["s_stop"][m] - S
        ap_m = g["appr"][m]
        appr = act & (d_stop > 0.0) & (d_stop < QUEUE_D)
        queued = appr & (V < QUEUE_V)
        o = np.zeros((n, len(SIG_NAMES)))
        for k in range(N_PHASES):
            in_k = appr & g["green"][k][m]
            cnt = in_k.sum(1)
            o[:, Q_OFF + k] = (queued & g["green"][k][m]).sum(1)
            o[:, N_OFF + k] = cnt
            vsum = np.where(in_k, V, 0.0).sum(1)
            o[:, V_OFF + k] = np.where(cnt > 0, vsum / np.maximum(cnt, 1), SIG_EMPTY)
            o[:, D_OFF + k] = np.where(in_k, d_stop, FAR).min(1)
            o[:, NN_OFF + k] = (in_k & (d_stop < NEAR_D)).sum(1)
            for side, a in enumerate(self.phase_appr[k]):
                on = in_k & (ap_m == a)
                o[:, QS_OFF + 2 * k + side] = (on & (V < QUEUE_V)).sum(1)
                c = on.sum(1)
                o[:, VA_OFF + 2 * k + side] = np.where(
                    c > 0, np.where(on, V, 0.0).sum(1) / np.maximum(c, 1), SIG_EMPTY)
        o[np.arange(n), PH_OFF + X[:, 0].astype(int)] = 1.0
        b = MISC_OFF
        o[:, b] = X[:, 1]
        o[:, b + 1] = X[:, 2]
        o[:, b + 2] = np.mod(X[:, 4] / self.duration + X[:, 11], 1.0)
        o[:, b + 3] = X[:, 5]
        return o

    # -- one tick -----------------------------------------------------------
    def fixed_signal_action(self, s):
        """The fixed-time plan, as SWITCH/EXTEND decisions."""
        X = s[:, 5 * self.N:]
        return (X[:, 1] >= self.plan(s)[np.arange(len(s)), X[:, 0].astype(int)]
                ).astype(int)

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
        fixed = self.signal_bank if self.agent == "vehicle" else self.vehicle_bank
        if isinstance(fixed, list):
            return self._population_actions(s, fixed, fresh_batch)
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

    def _population_actions(self, s, pop, fresh_batch):
        """The fixed agent is a POPULATION: the episodes are split into the
        same consecutive blocks `intersection_fast.run` uses, one partner each.
        A fixed-plan partner (None) among signal trees writes NaN, which
        `step_both` replaces by the plan's own decision. What the reference
        cannot express it refuses: a green-time tree mixed with the plan, and
        event mode, whose message depends on each block's controller."""
        from ..memory import MemBank
        n, N = len(s), self.N
        edges = np.linspace(0, n, len(pop) + 1).round().astype(int)
        if self.agent == "vehicle":
            if self.event:
                raise NotImplementedError("signal populations on the Python path "
                                          "in event mode")
            if fresh_batch or getattr(self, "_pop", None) is None or self._pop_n != n:
                self._pop = [None if b is None else MemBank(b, len(self.sig_names))
                             for b in pop]
                for pb, a, b in zip(self._pop, edges[:-1], edges[1:]):
                    if pb is not None:
                        pb.reset(b - a)
                self._pop_n = n
            out = np.full(n, np.nan)
            o = self.observe_signal(s)
            for pb, a, b in zip(self._pop, edges[:-1], edges[1:]):
                if pb is not None and b > a:
                    out[a:b] = pb.act(o[a:b])
            return out
        vbs = [b or self.default_vehicle_bank() for b in pop]
        if fresh_batch or getattr(self, "_pop", None) is None or self._pop_n != n:
            self._pop = [MemBank(b, len(self.veh_names)) for b in vbs]
            for pb, a, b in zip(self._pop, edges[:-1], edges[1:]):
                pb.reset((b - a) * N)
                pb._prev = np.zeros((b - a) * N, bool)
            self._pop_n = n
        o = self.observe_vehicles(s)
        out = np.zeros(n * N)
        for pb, a, b in zip(self._pop, edges[:-1], edges[1:]):
            if b <= a:
                continue
            active = (s[a:b, 3 * N:4 * N] == 1.0).reshape(-1)
            fresh = active & ~pb._prev
            pb.latch[fresh], pb.step[fresh], pb.have[fresh] = -1, 0, False
            pb.age[fresh] = 0
            if pb.slots is not None and pb.slots.shape[1]:
                pb.slots[fresh] = 0.0
            pb._prev = active
            out[a * N:b * N] = pb.act(o[a * N:b * N])
        return out

    _veh_head_now = "argmax"
    _sig_head_now = "argmax"

    def _heads(self, vehicle_bank, signal_bank):
        """Record which head each agent's actions are in, for `step_both`."""
        first = lambda b: (next((x for x in b if x is not None), None)
                           if isinstance(b, list) else b)
        vehicle_bank, signal_bank = first(vehicle_bank), first(signal_bank)
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
        fixmask = np.isnan(a_sig)
        if fixmask.any():
            # episodes whose partner in a population is the fixed plan
            a_sig = np.where(fixmask, self.fixed_signal_action(s), a_sig)
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
            need = live & ~fixmask & (X[:, 10] <= 0.0)
            X[need, 10] = np.clip(a_sig[need], T_MIN, T_MAX)
            X[live, 1] += dt
            switch = live & ~fixmask & (X[:, 1] >= X[:, 10] - 1e-9)
            X[switch, 10] = 0.0
            # the fixed-plan partners of a population switch by their own rule
            switch = switch | (live & fixmask & ((a_sig >= 0.5) & (X[:, 1] >= T_MIN)
                                                 | (X[:, 1] >= T_MAX)))
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
                    # V already holds the slot's entry speed (sample_starts)
                    act[i, q] = True
                    spawned.add(grp)

        # -- kinematics: the tree's acceleration, clipped ---------------------
        before = S.copy()
        v_pre = V.copy()
        v_new = np.clip(V + acc * dt, 0.0, V0)
        jerk = np.where(was, ((v_new - V) / dt / B_MAX) ** 2, 0.0)
        S[was] = S[was] + v_new[was] * dt
        V[was] = v_new[was]

        # -- red-light running: crossing the stop line without right of way -
        s_stop = g["s_stop"][m]
        ran = was & (before < s_stop) & (S >= s_stop) & ~gm
        X[:, 7] += ran.sum(1)
        car = self.reward_mode == "car"
        if car:
            # the gap the crossing car SAW, on the state it observed and acted
            # from -- pre-move, which is what `observe_vehicle` reads too
            riv = self._rival_dt(m, before, v_pre, act)
            risk = np.clip(1.0 - riv / T_SAFE, 0.0, 1.0)
            red_c = np.where(ran, R_RED_FLOOR
                             + (R_RED_CAR - R_RED_FLOOR) * risk, 0.0)
            r -= red_c.sum(1)
        else:
            r -= R_RED * ran.sum(1)
        t_comfort = np.zeros(n)
        if car:
            crossed_g = was & (before < s_stop) & (S >= s_stop) & gm
            r += R_GREEN * np.where(crossed_g, V / V0, 0.0).sum(1)
            t_comfort = W_COMFORT * jerk.sum(1) * dt
            r -= t_comfort

        # -- collisions: rear-end on a shared lane, crossing at a conflict --
        has, gap, j = self._leaders(m, S, act)

        # -- traffic performance, before the collided cars are removed -------
        # (see the module docstring for why each term is there)
        blocked = (st == 0.0) & (dep <= t[:, None])
        ap = g["appr"][m]
        vf = np.where(act, V / V0, 0.0)
        in_net = act | blocked
        ms, na = np.zeros(n), np.zeros(n)
        for k_a in range(4):
            mem = in_net & (ap == k_a)
            cnt = mem.sum(1)
            ms = ms + np.where(cnt > 0, np.where(mem, vf, 0.0).sum(1)
                               / np.maximum(cnt, 1), 0.0)
            na = na + (cnt > 0)
        t_speed = W_SPEED * np.where(na > 0, ms / np.maximum(na, 1), 0.0) * dt
        # RELU, as sumo_test charges relu(V0 - v): a car above free-flow used to
        # earn a delay BONUS here, since 1 - v/V0 goes negative above V0
        delay_sum = np.where(in_net, np.maximum(1.0 - vf, 0.0), 0.0).sum(1)
        t_delay = (W_CAR_DELAY if car else W_DELAY) * delay_sum * dt
        if not car:
            r += t_speed
        r -= t_delay
        d_now = g["s_stop"][m] - S
        stopped = act & (V < QUEUE_V)
        # ALL-RED IS RED, and the flag has to be read AFTER the phase switch.
        # `gm` here is plan-green only, and `in_ar` further up was computed
        # before the switch that may have just started an all-red; the kernel
        # asks `in_ar or not green[phase, m]` at this point in the tick, so the
        # two paths disagreed during every all-red. A LOCAL mask, because `gm`
        # itself is shared with the step path, which wants plan-green.
        ar_now = X[:, 2] > 0.5
        gm_eff = gm & ~ar_now[:, None]
        before = stopped & (d_now > 0.0)
        led = has & (gap < QUEUE_GAP)
        past = stopped & ~(d_now > 0.0)          # stopped in or past the box
        w_free = before & gm_eff & ~led
        w_queue = before & gm_eff & led
        # stopped on red with nobody ahead: charged by the road left empty
        # beyond a normal stopping zone, zero at the line and growing with it
        early = np.where(before & ~gm_eff & ~led,
                         np.maximum(d_now - STOP_ZONE, 0.0) / NEAR_INT, 0.0)
        scale = (W_STUCK_CAR / W_STUCK) if car else 1.0
        t_stuck = scale * dt * (W_STUCK_FREE * (w_free | past).sum(1)
                                + W_STUCK_QUEUE * w_queue.sum(1)
                                + W_STUCK_EARLY * early.sum(1))
        r -= t_stuck
        waiting = (stopped & (d_now > 0.0)) | blocked
        qsq = np.zeros(n)
        for k_a in range(4):
            qa = (waiting & (ap == k_a)).sum(1)
            qsq = qsq + qa * qa
        t_queue = W_QUEUE * qsq * dt
        # OVER-GREEN: green held on a phase with nothing to serve. `t_green` is
        # the count of cars this phase could discharge that are actually within
        # COMMIT_D of their stop line; when that is zero every second of green
        # is charged, which is sumo_test's floor-of-zero case.
        if not car:
            ph_now = X[:, 0].astype(int)
            servable = np.zeros(n, bool)
            for k_ph in range(N_PHASES):
                sel = ph_now == k_ph
                if not sel.any():
                    continue
                in_k = act[sel] & g["green"][k_ph][m[sel]]
                near = in_k & (d_now[sel] > 0.0) & (d_now[sel] < COMMIT_D)
                servable[sel] = near.any(1)
            t_over = W_OVER * (~servable & (X[:, 2] < 0.5)) * dt
            r -= t_over
        else:
            t_over = np.zeros(n)
        if car:
            # the red-stop bonus, once per car (a car cannot creep and re-stop
            # its way to a second one)
            paid = self._paid(s)
            stop_red = stopped & (d_now > 0.0) & (d_now <= STOP_ZONE) & ~gm & (paid < 0.5)
            paid[stop_red] = 1.0
            r += R_STOP * stop_red.sum(1)
        else:
            r -= t_queue
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
        # the per-car reward charges the cars at fault: the striker of a
        # rear-end, both parties of a crossing
        r -= R_COLL * ((rear | cross).sum(1) if car else n_ev)
        t_red = R_RED * ran.sum(1)
        X[:, 6] += n_ev
        X[:, 8] += n_rear
        X[:, 9] += n_cross
        st[hit] = 2.0
        V[hit] = 0.0

        # -- exits -------------------------------------------------------------
        out = (st == 1.0) & (S >= self.s_exit[m])
        st[out] = 2.0
        r += R_EXIT * out.sum(1)
        t_left = np.zeros(n)

        # -- the end of the episode: every car still waiting pays -----------
        end = X[:, 4] + 1.0 >= self.duration
        if end.any():
            left = ((st == 1.0) | ((st == 0.0) & (dep <= t[:, None]))).sum(1)
            t_left = np.where(end, R_LEFT * left, 0.0)
            r -= t_left
        if self.terms is not None:
            # DIAGNOSTIC ONLY: undiscounted per-term totals, per episode, so a
            # ranking the reward gets wrong can be traced to the term behind it
            if car:
                t_speed = np.zeros(n)
                t_queue = np.zeros(n)
            for key, val in (("over_green", -t_over), ("speed", t_speed), ("delay", -t_delay),
                             ("queue", -t_queue), ("stuck", -t_stuck),
                             ("red", -t_red), ("crash", -R_COLL * n_ev),
                             ("red_car", -red_c.sum(1) if car else np.zeros(n)),
                             ("n_red", ran.sum(1).astype(float)),
                             ("exit", R_EXIT * out.sum(1)), ("left", -t_left),
                             ("comfort", -t_comfort),
                             ("n_rear", n_rear.astype(float)), ("n_cross", n_cross.astype(float))):
                self.terms[key] = self.terms.get(key, 0.0) + val

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
    def python_rollout(self, vehicle_bank, signal_bank, s, T=None, veh_dev=None,
                       reward=None):
        """Both trees on the numpy model. The vehicle tree runs on n * N rows,
        one per slot, with its latch reset when a slot spawns; the signal tree
        on n rows. `signal_bank` None means the fixed-time plan. `reward`
        "team" or "car" overrides the agent's own. Returns G."""
        from ..memory import MemBank
        T = self.duration if T is None else min(T, self.duration)
        n, N = len(s), self.N
        self._heads(vehicle_bank, signal_bank)
        self._sig_kind_override = _kind_of(signal_bank)
        self._reward_override = reward
        try:
            return self._python_rollout(vehicle_bank, signal_bank, s, T, veh_dev)
        finally:
            self._sig_kind_override = None
            self._reward_override = None

    def _python_rollout(self, vehicle_bank, signal_bank, s, T, veh_dev):
        from ..memory import MemBank
        n, N = len(s), self.N
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
        exit -- before it crossed the stop line on red, and before each spell
        of being STUCK, judged from what the car itself observed (stopped, and
        neither close behind a car nor at its line without a green). For the
        signal they are the ticks before a collision and before a phase's
        queue reaches LONG_QUEUE -- the two things the reward now charges
        most, so the grower is pointed at them and not only at crashes.
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
                    v_obs = z[:, ix("v")]
                    near_red = (d_stop > 0.0) & (z[:, ix("green")] < 0.5)
                    behind = (z[:, ix("has_lead")] > 0.5) &                         (z[:, ix("lead_gap")] < QUEUE_GAP)
                    stuck = (v_obs < QUEUE_V) & ~((d_stop > 0.0) & (behind | near_red))
                    starts = np.flatnonzero(stuck & ~np.r_[False, stuck[:-1]])
                    bad = list(ran) + list(starts)
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
            qcols = [self.sig_names.index("q%d%s" % (k, sd))
                     for k in range(N_PHASES) for sd in "ab"]
            for i in range(n_ep):
                rows.append(tr[i, :, :n_obs])
                long_q = tr[i, :, qcols].T.max(1) >= LONG_QUEUE
                starts = np.flatnonzero(long_q & ~np.r_[False, long_q[:-1]])
                crash = np.flatnonzero(r[i] < -0.5 * R_COLL)
                for t in sorted(set(starts) | set(crash)):
                    fails.append(tr[i, max(0, t - lookback):t + 1, :n_obs])
        cov = np.concatenate(rows, 0) if rows else np.zeros((0, n_obs))
        fl = np.concatenate(fails, 0) if fails else np.zeros((0, n_obs))
        return cov, fl
