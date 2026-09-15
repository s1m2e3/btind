"""The traffic reward must rank the controllers the searches have exploited.

Both exploits found so far were optima of the reward, not of the road: a
continuous vehicle leaf that stopped every car (never crashes, never runs a red)
and a signal that held each phase to the maximum and never served the last one
before the episode ended. This test holds the ranking that rules both out,
together with the two rankings a traffic engineer would insist on.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import ACTIONS, SIG_ACTIONS, IntersectionBatch
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, mem_names


def test_reward_ranks_stopping_crashing_and_starving_below_control():
    env = IntersectionBatch()
    N, SN = env.veh_names, env.sig_names
    dv, ds = len(mem_names(N, None)) + 1, len(mem_names(SN, None)) + 1

    def P(k, d, n):
        th = np.zeros((d, n))
        th[-1, k] = 1.0
        return th
    veh = lambda a: check_arms(dict(names=list(N), laws_on_z=True, head="argmax",
                                    actions=list(ACTIONS), clauses=[], laws=[],
                                    default=P(ACTIONS.index(a), dv, 5)), a)
    sx = SN.index
    extend = check_arms(dict(names=list(SN), laws_on_z=True, head="argmax",
                             actions=list(SIG_ACTIONS), clauses=[], laws=[],
                             default=P(0, ds, 2)), "extend")
    actuated = check_arms(dict(
        names=list(SN), laws_on_z=True, head="argmax", actions=list(SIG_ACTIONS),
        clauses=[[[sx("ph%d" % k), 0.5, False], [sx("n%d" % k), 0.5, True]]
                 for k in range(4)],
        laws=[P(1, ds, 2)] * 4, default=P(0, ds, 2)), "actuated")
    follower = env.default_vehicle_bank()
    s = env.sample_starts(300, np.random.default_rng(5))
    G = lambda v, sg: IF.run(env, v, sg, s, env.duration).mean()
    stop, cruise = G(veh("BRAKE_HARD"), None), G(veh("HOLD"), None)
    fixed, starve, act = G(follower, None), G(follower, extend), G(follower, actuated)
    assert stop < min(cruise, starve, fixed, act)           # stopping everything is worst
    assert max(cruise, starve) < fixed - 100.0              # crashing and starving lose big
    assert fixed < act                                      # serving demand beats a fixed plan


if __name__ == "__main__":
    test_reward_ranks_stopping_crashing_and_starving_below_control()
    print("ok")
