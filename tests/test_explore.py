"""Deviations: the kernel's counterfactual must equal the Python path's, exactly.

A deviation of length zero is the undeviated rollout; a deviation that starts
after the episode ended changes nothing; and a deviation that does fire changes
the traced action at exactly the ticks it names and nowhere else. Then the
whole thing is held to the Python path on both worlds.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.explore import Deviate, deviations, hot_rows
from btind.memory import MemBank
from btind.policies import evaluate
from btind.structure import fast_rollout, starts
from btind.tick import trace_array


def _highway_bank():
    from btind.envs.highway_batch import ACTIONS, HighwayBatch
    from btind.envs.highway_task import constant_bank, law_width
    env = HighwayBatch()
    N = env.names
    d = law_width(N)

    def pref(k):
        th = np.zeros((d, 5))
        th[-1, k] = 1.0
        return th
    ix = N.index
    b = dict(constant_bank(N, 3), actions=list(ACTIONS),
             clauses=[[[ix("v1_dx"), 20.0, True]]], laws=[pref(4)],
             sticky=[True], betas=[[[ix("v1_dx"), 30.0, False]]])
    return env, b


def test_zero_length_and_late_deviations_are_no_ops():
    env, b = _highway_bank()
    s = starts(env, 200, 11)
    g0 = fast_rollout(env, b, s, env.duration)
    dev = np.zeros((len(s), 4))
    assert np.array_equal(g0, fast_rollout(env, b, s, env.duration, dev=dev))
    dev[:, 0], dev[:, 1], dev[:, 2] = 10 ** 6, 3, 0        # starts after the end
    assert np.array_equal(g0, fast_rollout(env, b, s, env.duration, dev=dev))


def test_deviation_changes_exactly_the_named_ticks():
    env, b = _highway_bank()
    s = starts(env, 50, 11)
    d = len(env.names) + 3
    tr0 = trace_array(len(s), env.duration, d)
    fast_rollout(env, b, s, env.duration, trace=tr0)
    dev = np.zeros((len(s), 4))
    dev[:, 0], dev[:, 1], dev[:, 2] = 2, 3, 0               # LANE_LEFT at ticks 2..4
    tr1 = trace_array(len(s), env.duration, d)
    fast_rollout(env, b, s, env.duration, trace=tr1, dev=dev)
    alive = tr1[:, :, d - 1] > 0.5
    a1 = tr1[:, :, d + 1]
    assert np.all(a1[:, 2:5][alive[:, 2:5]] == 0)
    # before the deviation the two runs are identical, tick for tick
    assert np.array_equal(tr0[:, :2], tr1[:, :2])


def test_kernel_deviation_matches_python_path_highway():
    env, b = _highway_bank()
    s = starts(env, 300, 11)
    rng = np.random.default_rng(3)
    dev = np.zeros((len(s), 4))
    dev[:, 0] = rng.integers(0, 30, len(s))
    dev[:, 1] = rng.choice([1, 3, 8], len(s))
    dev[:, 2] = rng.integers(0, 5, len(s))
    gk = fast_rollout(env, b, s, env.duration, dev=dev)
    pol = Deviate(MemBank(b, len(env.names)), dev, argmax=True)
    pol.reset(len(s))
    ss, alive, G, disc = s.copy(), np.ones(len(s), bool), np.zeros(len(s)), 1.0
    for _ in range(env.duration):
        ss, r, done = env.step(ss, pol.act(env.observe(ss)))
        G += disc * r * alive
        disc *= env.gamma
        alive &= ~done
        if not alive.any():
            break
    assert np.abs(G - gk).max() < 1e-9, np.abs(G - gk).max()


def test_kernel_deviation_matches_python_path_nest():
    from btind.envs.nest import NestWorld, OBS_NAMES
    from btind.memory import mem_names
    env = NestWorld(food_persistent=True)
    zn = mem_names(OBS_NAMES, None)
    d = len(zn) + 1
    ix = OBS_NAMES.index

    def to(a, bb, sgn=1.0):
        th = np.zeros((d, 2))
        th[ix(a), 0] = sgn
        th[ix(bb), 1] = sgn
        return th
    b = dict(names=list(OBS_NAMES), laws_on_z=True, head="vector",
             clauses=[[[ix("t_capture"), 8.0, True]], [[ix("carrying"), 0.5, False]]],
             laws=[to("bear_threat_x", "bear_threat_y", -1.0),
                   to("bear_nest_x", "bear_nest_y")],
             default=to("bear_food_x", "bear_food_y"), sticky=[True, False],
             betas=[[[ix("t_capture"), 20.0, False]], None])
    n_ep, T, seed = 200, 150, 11
    s = starts(env, n_ep, seed)
    rng = np.random.default_rng(4)
    ang = rng.uniform(0, 2 * np.pi, n_ep)
    dev = np.zeros((n_ep, 4))
    dev[:, 0] = rng.integers(0, 100, n_ep)
    dev[:, 1] = rng.choice([1, 3, 8], n_ep)
    dev[:, 2], dev[:, 3] = np.cos(ang), np.sin(ang)
    env.seed_kernels(seed)
    gk = fast_rollout(env, b, s, T, dev=dev)
    env.seed_kernels(seed)
    gp = evaluate(env, Deviate(MemBank(b, len(OBS_NAMES)), dev, argmax=False),
                  n_ep=n_ep, T=T, seed=seed)["G"]
    assert np.abs(gp - gk).max() < 1e-9, np.abs(gp - gk).max()


def test_deviations_report_shapes_and_hot_rows():
    env, b = _highway_bank()
    ex = deviations(env, b, n_ep=300, T=env.duration, seed=11,
                    rng=np.random.default_rng(0))
    n = len(ex["adv"])
    assert n > 0 and ex["z0"].shape == (n, len(env.names) + 2)
    assert set(np.unique(ex["k"])) <= {1, 3, 8}
    h = hot_rows(ex, frac=0.25)
    assert 0 < len(h) <= n and np.all(ex["adv"][h] >= np.quantile(ex["adv"], 0.75) - 1e-12)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
