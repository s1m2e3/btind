"""Re-weighting WHERE the search trains, without moving the judge.

`band_w` weights equal sub-bands of `approach_vph` so a round can spend its
episodes where the car is losing. Three things are held: uniform sampling is
untouched (bit-exact, same rng draws), a weighted world really does move its
demand, and the weights `e39` derives from a round's report put the episodes on
the deficit while leaving every band a floor -- which is what the per-group
acceptance guard needs in order to still see the bands the car is winning.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import WIDE, IntersectionBatch
from experiments.e39_vehicle_only import FOCUS_FLOOR, SPAN, focus_weights


def _world(cond=None):
    # e39's own span by default; `WIDE` covers the other branch of the draw,
    # the one without `approach_skew` where each approach draws its own load
    return IntersectionBatch(n_max=256, T_end=60.0, veh_reward="car",
                             conditions=SPAN if cond is None else cond)


def _cars(env, n=200, seed=0):
    s = env.sample_starts(n, np.random.default_rng(seed))
    return (s[:, 4 * env.N:5 * env.N] < np.inf).sum(1)


def test_uniform_sampling_is_untouched():
    for cond in (None, WIDE):
        a = _cars(_world(cond))
        b = _cars(_world(cond).set_band_weights(None))
        assert np.array_equal(a, b), cond


def test_weighting_moves_the_demand_it_draws():
    for cond in (None, WIDE):
        lo = _cars(_world(cond).set_band_weights([1.0, 0.0, 0.0]))
        hi = _cars(_world(cond).set_band_weights([0.0, 0.0, 1.0]))
        mid = _cars(_world(cond))
        assert hi.mean() > 2 * lo.mean(), (cond, lo.mean(), hi.mean())
        # the uniform world sits between them: both are sub-bands of it
        assert lo.mean() < mid.mean() < hi.mean(), cond


ROWS = {"light 300-500": dict(vs_hand=+3868.5),
        "heavy 600-800": dict(vs_hand=-5126.7),
        "peak 800-1000": dict(vs_hand=-15436.8),
        # round 8's car, measured: level at a low left share, far behind at a high one
        "few lefts .10-.25": dict(vs_hand=+22.7),
        "many lefts .25-.40": dict(vs_hand=-10871.0)}


def test_left_share_weighting_moves_the_turns_it_draws():
    def lefts(w):
        env = _world().set_left_weights(w)
        s = env.sample_starts(200, np.random.default_rng(0))
        m = env.geom["dirs"] == 2                      # left-turn movements
        mv = s[:, :env.N].astype(int)
        live = s[:, 4 * env.N:5 * env.N] < np.inf
        return (m[mv] & live).sum() / max(live.sum(), 1)
    assert lefts([0.0, 1.0]) > 1.5 * lefts([1.0, 0.0])
    assert abs(lefts(None) - lefts([1.0, 1.0])) < 0.05


def test_focus_follows_the_deficit_and_floors_every_band():
    # round 5's measured report: ahead at light, behind at heavy, far behind at peak
    rows = ROWS
    w = focus_weights(rows)["demand"]
    assert abs(sum(w) - 1.0) < 1e-12
    assert w[2] > w[1] > w[0]                      # the deficit orders them
    assert min(w) >= FOCUS_FLOOR - 1e-9            # nothing is starved
    # A BAND THE CAR IS WINNING KEEPS ITS FLOOR AND NO MORE: `light` is +3868
    # ahead, so it holds episodes only because `accept`'s per-group guard needs
    # them, not because there is anything left to find there.
    assert abs(w[0] - FOCUS_FLOOR) < 1e-9
    # ahead everywhere: nothing to chase, so train the span as it is
    even = focus_weights({k: dict(vs_hand=+100.0) for k in rows})["demand"]
    assert max(even) - min(even) < 1e-12


def test_the_two_axes_are_weighted_independently():
    f = focus_weights(ROWS)
    assert set(f) == {"demand", "left"} and len(f["left"]) == 2
    # level at a low left share, 10871 behind at a high one: the floor and the rest
    assert abs(f["left"][0] - FOCUS_FLOOR) < 1e-9
    assert f["left"][1] > f["left"][0]
    assert abs(sum(f["left"]) - 1.0) < 1e-12


if __name__ == "__main__":
    test_uniform_sampling_is_untouched()
    test_weighting_moves_the_demand_it_draws()
    test_focus_follows_the_deficit_and_floors_every_band()
    print("ok")
