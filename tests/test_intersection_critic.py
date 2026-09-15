"""The intersection's value function and critic.

The per-car rewards in the trace must add up, over every car, to the episode's
per-car return -- otherwise V-hat is fitted to a reward nobody is paid. Then both
estimators must carry real signal on held-out data: V-hat ranks the car's own
return-to-go, and A-hat ranks exact deviation advantages better than chance.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import IntersectionBatch
from btind.envs import intersection_fast as IF
from btind.intersection_critic import fit_advantage, fit_value
from btind.memory import mem_names
from btind.tick import trace_array


def test_traced_per_car_rewards_sum_to_the_return():
    env = IntersectionBatch(n_max=24, T_end=60.0, veh_reward="car", entry_v=(5.5, 11.0))
    vb = env.default_vehicle_bank()
    s = env.sample_starts(30, np.random.default_rng(2))
    T = env.duration
    d = len(mem_names(env.veh_names, None)) + 1
    G = IF.run(env, vb, None, s, T)
    disc = env.gamma ** np.arange(T)
    tot = np.zeros(len(s))
    for q in range(env.N):
        dev = np.zeros((len(s), 4))
        dev[:, 3] = q
        tr = trace_array(len(s), T, d)
        assert np.array_equal(IF.run(env, vb, None, s, T, trace=tr, dev=dev), G)
        tot += (tr[:, :, d + 2] * disc).sum(1)
    assert np.abs(tot - G).max() < 1e-8, np.abs(tot - G).max()


def test_value_and_advantage_carry_signal_out_of_sample():
    env = IntersectionBatch(vph=(60.0, 20.0, 20.0), veh_reward="car", entry_v=(5.5, 11.0))
    env.set_agent("vehicle")
    vb = env.default_vehicle_bank()
    vh, vrep = fit_value(env, vb, n_ep=80, max_slots=24, verbose=False)
    assert vrep["spearman"] > 0.4, vrep
    _, arep = fit_advantage(env, vb, vh, n_dev=2000, verbose=False)
    assert arep["spearman"] > 0.2, arep


if __name__ == "__main__":
    test_traced_per_car_rewards_sum_to_the_return()
    test_value_and_advantage_carry_signal_out_of_sample()
    print("ok")


def test_critic_proposes_whole_point_sets():
    from btind import kernlaw as KL
    from btind.intersection_critic import make_critic
    from btind.memory import check_arms
    env = IntersectionBatch(vph=(60.0, 20.0, 20.0), veh_reward="car", entry_v=(5.5, 11.0),
                            veh_head="scalar")
    env.set_agent("vehicle")
    N = env.names
    d = len(mem_names(N, None)) + 1
    th = np.zeros((d, 1))
    th[-1, 0] = 1.0
    bank = check_arms(dict(names=list(N), laws_on_z=True, head="scalar", u_range=env.u_range,
                           prior="const", clauses=[], laws=[], default=th), "b")
    critic, rep = make_critic(env, bank, n_ep=30, n_dev=800, verbose=False)
    ix = N.index
    kern = KL.make([ix("d_stop"), ix("green")], np.zeros((0, 2)), np.zeros((0, 1)), [20.0, 0.25])
    props = critic(bank, -1, 0, kern, "scalar")
    sets = [t for t in props if len(t) > 3 and t[3] == "critic-set"]
    assert sets, "no point-set proposed"
    for X, Y, adv, _ in sets:
        assert X.ndim == 2 and X.shape[1] == 2 and Y.shape == (X.shape[0], 1)
        assert (Y >= env.u_range[0]).all() and (Y <= env.u_range[1]).all()
    singles = [t for t in props if len(t) == 3]
    assert singles


def test_context_columns_and_the_gate():
    import btind.intersection_critic as IC
    from btind.envs.intersection import WIDE
    env = IntersectionBatch(n_max=48, T_end=60.0, conditions=WIDE, veh_reward="car",
                            veh_head="scalar", sig_head="duration")
    env.set_agent("vehicle")
    env.signal_bank = [None, None, None]
    s = env.sample_starts(9, np.random.default_rng(0))
    ctx = IC.context(env, s)
    assert ctx.shape == (9, 2)
    assert list(ctx[:, 1].astype(int)) == [0, 0, 0, 1, 1, 1, 2, 2, 2]   # partner blocks
    assert ctx[:, 0].std() > 0                                          # demand varies
    OB, RW, AL, LAW = IC.record(env, env.default_vehicle_bank(), n_ep=6, seed=1, max_slots=4)
    assert OB.shape[2] == len(env.names) + 2
    # the gate needs all three measures to agree before it silences a critic
    good = dict(lift=0.4, spearman=0.33, sign_big=0.84)
    noise = dict(lift=0.9, spearman=0.08, sign_big=0.09)
    gate = lambda r: (r["lift"] < 1.0 and r["spearman"] < 0.2 and r["sign_big"] < 0.3)
    assert not gate(good) and gate(noise)
