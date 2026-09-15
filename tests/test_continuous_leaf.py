"""A continuous leaf: bounded by construction, a constant prior, and a comfort cost.

The command a bounded kernel leaf produces lies inside its range everywhere --
between anchors, beyond them, with targets pushed to the limits -- and is still
the target at an anchor. With a constant prior the tuner leaves nothing but the
intercept. The per-car reward charges a car for steep changes of speed, and a
population of green-time signal trees mixed with the fixed plan runs the same on
the Python path as in the kernel.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind import kernlaw as KL
from btind.collect import design_matrix
from btind.envs.intersection import A_MAX, B_MAX, WIDE, IntersectionBatch
from btind.envs import intersection_fast as IF
from btind.lawcem import cem_law
from btind.memory import MemBank, check_arms, mem_names

from tests.test_intersection_heads import duration_signal, scalar_follower


def test_bounded_leaf_stays_in_range_and_hits_its_anchors():
    rng = np.random.default_rng(0)
    lo, hi = -B_MAX, A_MAX
    d = 8
    th = np.zeros((d, 1))
    th[-1, 0] = 1.5
    X = rng.normal(size=(5, 2)) * 3
    Y = np.array([[lo], [hi], [0.3], [-2.0], [2.2]])
    kern = KL.make([2, 5], X, Y, [0.4, 0.4])          # short lengthscales: overshoot-prone
    Z = rng.normal(size=(4000, d - 1)) * 6
    u = KL.evaluate(th, kern, design_matrix(Z), bounds=(lo, hi))[:, 0]
    assert u.min() >= lo and u.max() <= hi
    at = np.zeros((5, d - 1))
    at[:, 2], at[:, 5] = X[:, 0], X[:, 1]
    ua = KL.evaluate(th, kern, design_matrix(at), bounds=(lo, hi))[:, 0]
    tol = 1e-3
    assert np.abs(ua - Y[:, 0]).max() < tol, np.abs(ua - Y[:, 0])


def test_constant_prior_tunes_only_the_intercept():
    env = IntersectionBatch(n_max=24, T_end=40.0, veh_head="scalar", veh_reward="car")
    env.set_agent("vehicle")
    b = dict(scalar_follower(), prior="const")
    b = dict(b, default=KL.constrain(b, b["default"]))
    th, _ = cem_law(env, b, -1, lambda x: MemBank(x, len(env.names)), n_iter=2, K=8,
                    n_ep=10, T=env.duration, rng=np.random.default_rng(1))
    assert not np.any(th[:-1]) and th[-1, 0] != 0.0


def test_comfort_charges_steep_changes_of_speed():
    env = IntersectionBatch(n_max=24, T_end=60.0, veh_head="scalar", veh_reward="car")
    N = env.veh_names
    d = len(mem_names(N, None)) + 1
    s = env.sample_starts(20, np.random.default_rng(2))

    def stop_at(a):
        th = np.zeros((d, 1))
        th[-1, 0] = a
        return check_arms(dict(names=list(N), laws_on_z=True, head="scalar",
                               u_range=(-B_MAX, A_MAX), clauses=[], laws=[], default=th), "c")
    comfort = {}
    for a in (-1.0, -4.5):
        env.terms = {}
        env.python_rollout(stop_at(a), None, s)
        comfort[a] = float(np.mean(env.terms["comfort"]))
    assert comfort[-4.5] < comfort[-1.0] < 0.0


def test_green_time_population_python_path_matches_kernel():
    env = IntersectionBatch(n_max=32, T_end=60.0, conditions=WIDE, veh_head="scalar",
                            sig_head="duration", veh_reward="car")
    s = env.sample_starts(12, np.random.default_rng(6))
    env.set_agent("vehicle")
    env.signal_bank = [None, duration_signal()]
    vb = scalar_follower()
    env._search_bank = vb
    pol = MemBank(vb, len(env.veh_names))
    pol.reset(len(s) * env.N)
    G, disc, prev = np.zeros(len(s)), 1.0, np.zeros(len(s) * env.N, bool)
    st = s.copy()
    for t in range(env.duration):
        active = (st[:, 3 * env.N:4 * env.N] == 1.0).reshape(-1)
        fresh = active & ~prev
        pol.latch[fresh], pol.step[fresh], pol.have[fresh] = -1, 0, False
        prev = active
        st, r, done = env.step(st, pol.act(env.observe(st)))
        G += disc * r
        disc *= env.gamma
    gk = IF.run(env, vb, env.signal_bank, s, env.duration)
    assert np.abs(G - gk).max() < 1e-9, np.abs(G - gk).max()


if __name__ == "__main__":
    test_bounded_leaf_stays_in_range_and_hits_its_anchors()
    test_constant_prior_tunes_only_the_intercept()
    test_comfort_charges_steep_changes_of_speed()
    test_green_time_population_python_path_matches_kernel()
    print("ok")
