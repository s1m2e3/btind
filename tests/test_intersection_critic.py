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
