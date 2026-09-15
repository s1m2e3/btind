"""Per-episode operating conditions and partner populations.

Each episode of a batch draws its own approach flows, turning shares and
fixed-plan greens; the kernel must agree with the model on that (continuous and
event V2I, where the SPaT window reads the episode's own plan), the conditions
must actually vary, and a population run must be exactly the concatenation of
its partners' runs so paired tests over it stay paired.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import WIDE, IntersectionBatch
from btind.envs import intersection_fast as IF

from tests.test_intersection_equiv import sig_bank, veh_bank


def test_conditions_vary_and_kernel_matches_model():
    for mode in ("continuous", "event"):
        env = IntersectionBatch(n_max=48, T_end=60.0, conditions=WIDE, sig_mode=mode,
                                veh_reward="car", entry_v=(5.5, 11.0))
        s = env.sample_starts(30, np.random.default_rng(8))
        cars = np.isfinite(s[:, 4 * env.N:5 * env.N]).sum(1)
        assert cars.max() >= 2 * max(cars.min(), 1)
        P = env.plan(s)
        assert P[:, 0].std() > 3 and (P[:, 1] <= 20).all() and (P[:, 0] >= 15).all()
        for sb in (None, sig_bank()):
            gk = IF.run(env, veh_bank(), sb, s, env.duration)
            gp = env.python_rollout(veh_bank(), sb, s)
            assert np.abs(gk - gp).max() < 1e-9, (mode, np.abs(gk - gp).max())


def test_default_world_keeps_the_fixed_plan():
    env = IntersectionBatch(n_max=24, T_end=60.0)
    s = env.sample_starts(5, np.random.default_rng(1))
    assert np.array_equal(env.plan(s), np.tile([30.0, 10.0, 30.0, 10.0], (5, 1)))
    assert env.condition_groups(s) is None


def test_population_is_the_concatenation_of_its_partners():
    env = IntersectionBatch(n_max=48, T_end=60.0, conditions=WIDE, veh_reward="car")
    s = env.sample_starts(31, np.random.default_rng(3))
    pop = [None, sig_bank()]
    G = IF.run(env, veh_bank(), pop, s, env.duration)
    edges = np.linspace(0, len(s), 3).round().astype(int)
    for g, a, b in zip(pop, edges[:-1], edges[1:]):
        assert np.array_equal(G[a:b], IF.run(env, veh_bank(), g, s[a:b], env.duration))
    groups = env.condition_groups(s)
    assert set(np.unique(groups)) == {0, 1, 2}


def test_python_path_population_matches_kernel():
    from btind.memory import MemBank
    from btind.policies import evaluate
    env = IntersectionBatch(n_max=32, T_end=40.0, conditions=WIDE, veh_reward="car")
    s = env.sample_starts(12, np.random.default_rng(6))
    env.set_agent("vehicle")
    env.signal_bank = [None, sig_bank()]
    vb = veh_bank()
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
    test_conditions_vary_and_kernel_matches_model()
    test_default_world_keeps_the_fixed_plan()
    test_population_is_the_concatenation_of_its_partners()
    print("ok")
