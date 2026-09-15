"""The radius sensor: what a car perceives of the vehicles around it.

Geometry first, on hand-placed cars: two cars on crossing movements heading for
their shared conflict point see each other as rivals with the right time gap,
and a car out of range is not seen. Then the kernel and the model must agree to
the bit on a tree whose guards read every sensor column -- equal returns on a
tree that never reads them would prove nothing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import ACTIONS, SENSE_R, WIDE, IntersectionBatch
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, mem_names


def _state(env, cars):
    """One episode with the given (movement, arc-length, speed) cars active."""
    N = env.N
    s = env.sample_starts(1, np.random.default_rng(0))
    s[:, 3 * N:4 * N] = 0.0
    s[:, 4 * N:5 * N] = np.inf
    for q, (mm, ss, vv) in enumerate(cars):
        s[0, q], s[0, N + q], s[0, 2 * N + q], s[0, 3 * N + q] = mm, ss, vv, 1.0
    return s


def test_crossing_rival_and_range():
    env = IntersectionBatch(n_max=8)
    g = env.geom
    ix = env.veh_names.index
    a, b = [(i, j) for i in range(env.M) for j in range(env.M) if g["conf"][i, j]][0]
    # both 20 m short of their shared point; a at 10 m/s, b at 5 m/s
    sa, sb = g["s_cp"][a, b] - 20.0, g["s_cp"][b, a] - 20.0
    s = _state(env, [(a, sa, 10.0), (b, sb, 5.0), (a, 100.0, 8.0)])
    o = env.observe_vehicles(s)
    assert o[0, ix("has_rival")] == 1.0 and o[1, ix("has_rival")] == 1.0
    assert abs(o[0, ix("rival_dt")] - abs(20.0 / 5.0 - 20.0 / 10.0)) < 1e-6
    assert abs(o[0, ix("rival_d")] - 20.0) < 1e-4
    x, y, _, _ = env._xy(s[:, :env.N].astype(int), s[:, env.N:2 * env.N])
    d01 = np.hypot(x[0, 0] - x[0, 1], y[0, 0] - y[0, 1])
    assert abs(o[0, ix("near_d")] - min(d01, np.hypot(x[0, 0] - x[0, 2], y[0, 0] - y[0, 2]))) < 1e-6
    far = np.hypot(x[0, 2] - x[0, 1], y[0, 2] - y[0, 1]) > SENSE_R
    if far:
        assert o[1, ix("n_near")] == 1.0


def _sensor_bank(env):
    N = env.veh_names
    ix = N.index
    d = len(mem_names(N, None)) + 1

    def P(a):
        th = np.zeros((d, 5))
        th[-1, ACTIONS.index(a)] = 1.0
        return th
    return check_arms(dict(
        names=list(N), laws_on_z=True, head="argmax", actions=list(ACTIONS),
        clauses=[[[ix("has_rival"), 0.5, False], [ix("rival_dt"), 1.5, True],
                  [ix("rival_d"), 25.0, True]],
                 [[ix("near_d"), 12.0, True], [ix("near_closing"), 0.5, False]],
                 [[ix("n_near"), 6.0, False]],
                 [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 10.0, True]]],
        laws=[P("BRAKE_HARD"), P("BRAKE"), P("HOLD"), P("BRAKE_HARD")],
        default=P("ACCEL_MAX")), "sensor")


def test_kernel_matches_model_on_a_tree_that_reads_the_sensor():
    env = IntersectionBatch(n_max=48, T_end=60.0, conditions=WIDE, veh_reward="car")
    s = env.sample_starts(30, np.random.default_rng(4))
    vb = _sensor_bank(env)
    gk = IF.run(env, vb, None, s, env.duration)
    gp = env.python_rollout(vb, None, s)
    assert np.abs(gk - gp).max() < 1e-9, np.abs(gk - gp).max()
    plain = env.default_vehicle_bank()
    assert np.abs(gk - IF.run(env, plain, None, s, env.duration)).max() > 0


if __name__ == "__main__":
    test_crossing_rival_and_range()
    test_kernel_matches_model_on_a_tree_that_reads_the_sensor()
    print("ok")
