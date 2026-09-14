"""`churn` on a world with many rows per episode must count units, not episodes."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.betasearch import churn
from btind.envs.intersection import IntersectionBatch
from btind.memory import MemBank


def test_churn_runs_per_vehicle_and_per_signal():
    env = IntersectionBatch(n_max=16, T_end=40.0)
    vb = env.default_vehicle_bank()
    vb = dict(vb, sticky=[False, True, False, False],
              betas=[None, [[env.veh_names.index("green"), 0.5, False]], None, None])
    env.set_agent("vehicle")
    env.signal_bank = None
    env._search_bank = vb
    ch = churn(env, vb, lambda b: MemBank(b, len(env.veh_names)), n_ep=20,
               T=env.duration)
    assert set(ch) == {0, 1, 2, 3}
    assert all(0.0 <= v["share"] <= 1.0 for v in ch.values())
    assert sum(v["share"] for v in ch.values()) <= 1.0 + 1e-9
    assert ch[1]["dwell"] >= 0.0
    # signal agent: one row per episode, still fine
    env.set_agent("signal")
    env.vehicle_bank = vb
    from tests.test_intersection_equiv import sig_bank
    sb = sig_bank()
    env._search_bank = sb
    ch2 = churn(env, sb, lambda b: MemBank(b, len(env.sig_names)), n_ep=20,
                T=env.duration)
    assert set(ch2) == {0, 1}
    env.set_agent("vehicle")
    env.vehicle_bank = None


if __name__ == "__main__":
    test_churn_runs_per_vehicle_and_per_signal()
    print("ok")
