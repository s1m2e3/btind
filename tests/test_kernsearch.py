"""The inducing-point search finds a rule the tree measurably lacks, on the columns
that express it, for a discrete and a continuous leaf.

Planted: in light traffic every car accelerates flat out, and running reds costs
50 per car. A single rule "near the light, on red -> slow down" is worth tens of
return units, and it needs the light column, which the statistic that ranks
columns does not put first. The column set is chosen by rollout.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind import kernlaw as KL
from btind.envs.intersection import ACTIONS, IntersectionBatch
from btind.kernsearch import search_kernels
from btind.memory import MemBank, check_arms, mem_names
from btind.structure import score


def _run(head):
    env = IntersectionBatch(vph=(10.0, 3.0, 3.0), veh_reward="car", entry_v=(5.5, 11.0),
                            veh_head=head)
    env.set_agent("vehicle")
    N = env.names
    d = len(mem_names(N, None)) + 1
    if head == "argmax":
        th = np.zeros((d, 5))
        th[-1, ACTIONS.index("ACCEL_MAX")] = 1.0
        extra = dict(actions=list(ACTIONS))
    else:
        th = np.zeros((d, 1))
        th[-1, 0] = 2.6
        extra = dict(u_range=env.u_range)
    bank = check_arms(dict(names=list(N), laws_on_z=True, head=head, clauses=[], laws=[],
                           default=th, **extra), "b")
    pol = lambda b: MemBank(b, len(N))
    # a continuous leaf's single braking point is worth ~+5 against a per-episode
    # spread of ~50 in light traffic once smooth driving is charged for, so the
    # paired test needs more episodes to see it than the discrete leaf's
    n_ep = 300 if head == "argmax" else 800
    cur = score(env, bank, pol, n_ep, env.duration, 11)
    b2, cur2, _ = search_kernels(env, bank, N, mem_names(N, None), pol, cur, env.duration,
                                 11, min_gain=1.0, n_ep=n_ep, n_laws=1, max_points=2,
                                 verbose=False)
    kern = KL.kern_of(b2, -1, 0)
    return cur.mean(), cur2.mean(), kern, N


def test_discrete_leaf_learns_a_red_stop_point():
    g0, g1, kern, N = _run("argmax")
    assert g1 > g0 + 10.0
    assert N.index("green") in kern["cols"]


def test_continuous_leaf_learns_a_braking_point():
    g0, g1, kern, N = _run("scalar")
    assert g1 > g0 + 10.0
    assert N.index("green") in kern["cols"]
    assert (kern["Y"][:, 0] < 0).any()


if __name__ == "__main__":
    test_discrete_leaf_learns_a_red_stop_point()
    test_continuous_leaf_learns_a_braking_point()
    print("ok")
