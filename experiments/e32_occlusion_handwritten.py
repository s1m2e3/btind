r"""e32 -- is the blackboard load-bearing when the front sensor goes blind?

Continuous-mode intersection with `occlude=(15, 40)`: between 40 m and 15 m
before the stop line the car's front sensor reports no leader. Queues form
exactly there when the light is red, so a car that trusts its sensor drives
into the car ahead. The e24 protocol: hand-written controllers with and
without a blackboard against the unoccluded reference, same episodes.

    memoryless             the e27 follower: brakes only when it SEES a leader
    memoryless, cautious   also brakes on entering the blind zone
    blackboard             stores (lead_gap, lead_dv) when a leader is seen and
                           brakes in the zone if the remembered gap was short
                           and the memory is fresh

WHAT CAME OUT, 600 episodes, fixed-time plan:

    reference, no occlusion                          -0.37
    memoryless, same follower                       -25.44   the zone costs 25
    memoryless, brake through the zone              -26.19   caution alone does not help
    blackboard, remembered short gap -> brake       -18.91   +6.5 recovered

THE WORLD IS SENSITIVE TO MEMORY HERE TOO, and a hand-written rule recovers a
quarter of the loss. The rest is in what the rule does not do: a leader seen
at 30 m and closing will be a stopped queue by the time the blind car reaches
it, and predicting that from a remembered gap and closing speed is a law on
the memory columns, not a threshold -- which is what the search's continuous
leaf and CEM are for. Reported as measured, not tuned further by hand.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btind.envs.intersection import ACTIONS, FAR, IntersectionBatch
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, emit, mem_names

N_EP, SEED = 600, 11
ZONE = (15.0, 40.0)


def main(n_ep=N_EP):
    oc = IntersectionBatch(occlude=ZONE)
    co = IntersectionBatch()
    N = oc.veh_names
    ix = N.index

    def P(k, d):
        th = np.zeros((d, 5))
        th[-1, k] = 1.0
        return th

    def vb(clauses, laws, mem=None, default="ACCEL_MAX"):
        zn = mem_names(N, mem)
        d = len(zn) + 1
        A = {a: P(i, d) for i, a in enumerate(ACTIONS)}
        return check_arms(dict(names=list(N), laws_on_z=True, head="argmax",
                               actions=list(ACTIONS), clauses=clauses,
                               laws=[A[l] for l in laws], default=A[default],
                               mem=mem), "vb"), zn.index

    close = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 10.0, True]]
    closing = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 25.0, True],
               [ix("lead_dv"), 0.5, False]]
    red = lambda lo, hi: [[ix("green"), 0.5, True], [ix("green"), -0.5, False],
                          [ix("d_stop"), lo, False], [ix("d_stop"), hi, True]]
    zone = [[ix("d_stop"), ZONE[0], False], [ix("d_stop"), ZONE[1], True]]

    V = {}
    V["reference: follower, no occlusion"], _ = vb(
        [close, red(0, 6), red(0, 45), closing], ["BRAKE_HARD", "BRAKE_HARD", "BRAKE", "BRAKE"])
    V["memoryless: same follower"], _ = vb(
        [close, red(0, 6), red(0, 45), closing], ["BRAKE_HARD", "BRAKE_HARD", "BRAKE", "BRAKE"])
    V["memoryless: brake through the blind zone"], _ = vb(
        [close, red(0, 6), zone + [[ix("v"), 4.0, False]], red(0, 45), closing],
        ["BRAKE_HARD", "BRAKE_HARD", "BRAKE", "BRAKE", "BRAKE"])
    seen = dict(cols=[ix("lead_gap"), ix("lead_dv")],
                write=[[ix("has_lead"), 0.5, False]], clear=None)
    b, zi = vb([], [], mem=seen)
    recent = [[zi("have_mem"), 0.5, False], [zi("mem_age"), 12.0, True]]
    V["blackboard: remembered short gap -> brake in the zone"], _ = vb(
        [close,
         zone + recent + [[zi("mem_lead_gap"), 22.0, True]],
         zone + recent + [[zi("mem_lead_gap"), 40.0, True], [zi("mem_lead_dv"), 0.5, False]],
         red(0, 6), red(0, 45), closing],
        ["BRAKE_HARD", "BRAKE_HARD", "BRAKE", "BRAKE_HARD", "BRAKE", "BRAKE"], mem=seen)

    s = oc.sample_starts(n_ep, np.random.default_rng(SEED))
    t0 = time.time()
    print("occluded intersection, blind zone %s m before the line, %d episodes"
          % (ZONE, n_ep))
    ref = None
    for name, b in V.items():
        env = co if name.startswith("reference") else oc
        G = IF.run(env, b, None, s, env.duration)
        ref = G if ref is None else ref
        se = float((G - ref).std() / np.sqrt(len(G)))
        print("   %-58s G %8.2f   (%+7.2f se %.2f)" % (name, G.mean(), (G - ref).mean(), se))
    print("\n" + emit(V["blackboard: remembered short gap -> brake in the zone"], N))
    print("[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_EP)
