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
    # light traffic: the red stop pays alone
    light = IntersectionBatch(veh_reward="car", vph=(10.0, 3.0, 3.0))
    s = light.sample_starts(300, np.random.default_rng(11))
    B = _ladder_banks(light)
    G = lambda b: IF.run(light, b, None, s, light.duration)
    assert _z(G(B["red"]), G(B["cruise"])) > 2
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
