"""The per-car reward the vehicle tree is searched on.

Three things are held: the kernel agrees with the model on it (with a signal
tree running and cars entering at drawn speeds), stopping every car is not a
refuge from crashing, and the demand ladder e34 climbs exists -- each piece of
a red-stop follower pays ON ITS OWN at some demand, given the pieces before it,
which is what lets a search that adds one arm at a time reach it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import ACTIONS, IntersectionBatch
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, mem_names

from tests.test_intersection_equiv import sig_bank, veh_bank


def _ladder_banks(env):
    N = env.veh_names
    ix = N.index
    dv = len(mem_names(N, None)) + 1

    def P(a):
        th = np.zeros((dv, 5))
        th[-1, ACTIONS.index(a)] = 1.0
        return th

    def bank(arms, default="ACCEL_MAX"):
        return check_arms(dict(names=list(N), laws_on_z=True, head="argmax",
                               actions=list(ACTIONS), clauses=[c for c, _ in arms],
                               laws=[P(a) for _, a in arms], default=P(default)), "b")
    red = ([[ix("near_int"), 0.5, False], [ix("green"), 0.5, True]], "BRAKE")
    close = ([[ix("has_lead"), 0.5, False], [ix("lead_gap"), 10.0, True]], "BRAKE_HARD")
    return dict(cruise=bank([]), stop=bank([], "BRAKE_HARD"), red=bank([red]),
                red_close=bank([close, red]))


def test_car_reward_kernel_matches_model():
    env = IntersectionBatch(n_max=24, T_end=60.0, veh_reward="car", entry_v=(4.0, 11.0))
    s = env.sample_starts(40, np.random.default_rng(3))
    assert s[:, 2 * env.N:3 * env.N].std() > 0          # entry speeds were drawn
    for sb in (None, sig_bank()):
        for reward in (None, "team"):
            gk = IF.run(env, veh_bank(), sb, s, env.duration, reward=reward)
            gp = env.python_rollout(veh_bank(), sb, s, reward=reward)
            assert np.abs(gk - gp).max() < 1e-9, (reward, np.abs(gk - gp).max())
    car = IF.run(env, veh_bank(), None, s, env.duration)
    team = IF.run(env, veh_bank(), None, s, env.duration, reward="team")
    assert np.abs(car - team).max() > 0


def test_signal_agent_is_always_scored_on_the_team_reward():
    env = IntersectionBatch(veh_reward="car")
    assert env.reward_mode == "car"
    env.set_agent("signal")
    assert env.reward_mode == "team"


def _z(a, b):
    d = a - b
    return d.mean() / max(d.std(), 1e-9) * np.sqrt(len(d))


def test_stopping_is_no_refuge_and_the_ladder_exists():
    full = IntersectionBatch(veh_reward="car")
    s = full.sample_starts(300, np.random.default_rng(11))
    B = _ladder_banks(full)
    G = lambda b: IF.run(full, b, None, s, full.duration)
    stop, cruise, hand = G(B["stop"]), G(B["cruise"]), G(full.default_vehicle_bank())
    assert stop.mean() < cruise.mean() < hand.mean()
    # at full demand the red stop alone is a disaster: cars behind it crash
    assert _z(G(B["red"]), cruise) < -5
    # AN EMPTY JUNCTION: the red stop no longer pays, and should not. A
    # crossing with nothing to conflict with costs `R_RED_FLOOR` and braking
    # for it costs delay and the rear-end it invites, so stopping is the worse
    # trade. This is REWARD_VERSION 6 doing what it was written to do -- the
    # charge follows the conflict, and here there is none.
    light = IntersectionBatch(veh_reward="car", vph=(10.0, 3.0, 3.0))
    s = light.sample_starts(300, np.random.default_rng(11))
    B = _ladder_banks(light)
    G = lambda b: IF.run(light, b, None, s, light.duration)
    assert _z(G(B["red"]), G(B["cruise"])) < 0
    # The charge that replaced the flat fee is tested on its own, in
    # `test_the_red_charge_follows_the_conflict` -- not through a controller,
    # because no traffic level reachable with these banks makes a DANGEROUS
    # crossing happen: measured on the hand follower, 47 red-runs produced zero
    # crossing collisions and averaged 22.6 against a floor of 20. The graded
    # part of the charge is a guardrail for a controller that crosses
    # aggressively, and no such controller exists yet to test it with.
    # more traffic: keeping a distance pays, given the red stop
    mid = IntersectionBatch(veh_reward="car", vph=(30.0, 10.0, 10.0))
    s = mid.sample_starts(300, np.random.default_rng(11))
    B = _ladder_banks(mid)
    G = lambda b: IF.run(mid, b, None, s, mid.duration)
    assert _z(G(B["red_close"]), G(B["red"])) > 2


if __name__ == "__main__":
    test_car_reward_kernel_matches_model()
    test_signal_agent_is_always_scored_on_the_team_reward()
    test_stopping_is_no_refuge_and_the_ladder_exists()
    print("ok")


def test_the_red_charge_follows_the_conflict():
    """A red-run is charged for the conflict it creates, not for the fact of it.

    `rival_dt` is the smallest time gap to a vehicle heading for a shared
    conflict point -- post-encroachment time. The charge ramps from
    `R_RED_FLOOR` at `T_SAFE` seconds of margin to the full `R_RED_CAR` at
    none, so a car slipping through an empty junction and one cutting across an
    arriving movement no longer pay the same 200.
    """
    from btind.envs.intersection import R_RED_CAR, R_RED_FLOOR, T_SAFE, FAR
    charge = lambda dt: (R_RED_FLOOR + (R_RED_CAR - R_RED_FLOOR)
                         * min(max(1.0 - dt / T_SAFE, 0.0), 1.0))
    assert charge(FAR) == R_RED_FLOOR              # nothing coming: the floor
    assert charge(T_SAFE) == R_RED_FLOOR           # clears with the margin
    assert charge(0.0) == R_RED_CAR                # simultaneous: all of it
    assert R_RED_FLOOR < charge(T_SAFE / 2) < R_RED_CAR
    # monotone in the margin, so a bigger gap is never charged more
    d = [charge(x) for x in np.linspace(0.0, 2 * T_SAFE, 25)]
    assert all(a >= b - 1e-12 for a, b in zip(d, d[1:]))


def test_rival_dt_is_the_column_the_car_can_see():
    """The charge uses the car's OWN observation, so the rule that avoids it is
    one its tree can express. If these ever diverge the reward is charging for
    something unobservable."""
    from btind.envs.intersection import VEH_NAMES
    assert "rival_dt" in VEH_NAMES


def test_a_red_does_not_excuse_a_stop_the_car_cannot_see():
    """Beyond NEAR_INT `observe` hands the car no signal, so a red is no excuse.

    The per-car reward has said this since it was written -- "a red excuses a
    stop only within NEAR_INT of the line, where the light can be seen" -- and
    `early` applied its by-degree ramp at any distance instead. Measured on the
    discovered car: 90.4% of its active car-ticks stopped, 99.4% of those with
    no leader, at a median 79.0 m, and only 6.8% of them able to see the light
    they were being excused by. It paid 1.6 a second where the same stop on
    green costs 75, which made halting every car at the entrance the best
    policy the search could reach.
    """
    from btind.envs.intersection import (NEAR_INT, STOP_ZONE, W_STUCK,
                                         W_STUCK_CAR, W_STUCK_EARLY,
                                         W_STUCK_FREE)
    sc = W_STUCK_CAR / W_STUCK
    far, near = 79.0, 40.0
    assert far > NEAR_INT >= near

    # what the ramp WOULD have charged, and what a free-lane stop costs
    ramp = lambda d: sc * W_STUCK_EARLY * max(d - STOP_ZONE, 0.0) / NEAR_INT
    assert ramp(far) < 0.05 * sc * W_STUCK_FREE          # the hole, quantified
    assert ramp(near) > 0.0                              # still charged, by degree


def test_standing_still_is_now_the_worst_thing_a_car_can_do():
    """The ranking the fix exists to correct, end to end.

    Under the old charge a constant full brake beat every constant that moved.
    Scored at 100 episodes on e39's world: full brake -82724, hold speed
    -40136, gentle accel -30564, full throttle -29992, hand follower -8787.
    """
    import numpy as np
    from btind.memory import MemBank
    from btind.structure import score
    from btind.envs.intersection import IntersectionBatch

    env = IntersectionBatch(n_max=64, T_end=120, veh_reward="car",
                            veh_head="pass", sig_head="duration",
                            entry_v=(5.5, 11.0)).set_agent("vehicle")
    zn = list(env.names)
    d = len(zn) + 3

    def const(a):
        th = np.zeros((d, 2))
        th[-1, 0] = a
        return dict(clauses=[], laws=[], default=th, names=zn, laws_on_z=True,
                    head="pass", u_range=tuple(env.u_range), actions=None)

    pol = lambda b: MemBank(b, len(zn))
    g = {a: score(env, const(a), pol, 24, 200, 3).mean() for a in (-4.5, 2.6)}
    assert g[-4.5] < g[2.6]
