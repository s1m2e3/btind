r"""e27 -- what the intersection world rewards, measured with hand-written trees.

Before any search runs here, the e24 protocol: is the WORLD sensitive to the
constructs the search is supposed to find? Hand-written trees for both agents,
their flattened ablations, and the fixed-time plan, all on the same episodes.

WHAT IS HAND-WRITTEN. Vehicles: a follower that brakes for a close leader,
stops for a red it can see, and otherwise accelerates -- flat and reactive.
Then the same follower with the stop as a SEQUENCE (brake until stopped, hold
until green, go), with a FAIL clause (abort the go if the leader is too close),
and with stickiness. Signals: the fixed-time plan, a reactive queue-based
switcher, and a switcher with a minimum-hold step.

WHAT THE NUMBERS SAID, 600 episodes, 40 slots, 100 s, fixed-time plan:

    cruise                                   -37.21
    flat follower                            -37.51   car-following alone is nothing
    flat follower + red stop                  -0.24   +37: the red stop is the task
    flat + red stop + creep                   -0.08
    sticky red stop, no fail                  -9.01   a latched stop kept braking
                                                      PAST the line and blocked the box
    sequence (brake until stopped, hold)     -39.08   same fault, worse: no hard-brake
                                                      arm above it
    sequence + fail past the line             -0.30   ties the flat tree (se 0.04)
    sticky + fail past the line               -0.24   ties exactly

    signal: fixed plan                        -0.08
    signal: switch when own approach empty    +7.18   +7.3 over the plan
    signal: ... or held 25 s                  +6.74

BOTH AGENTS MATTER AND NEITHER NEEDS A SEQUENCE. The vehicle's task is worth
+37 and a queue-actuated signal +7 more, so the world rewards discovery on both
sides. But every temporal construct at best TIES the reactive tree, and the
fail clause was only needed to undo the damage stickiness did. The reason is the
same one e24 found on NestWorld and e26 on highway: the progress of every
manoeuvre here is OBSERVABLE -- position relative to the line, speed, the
light while near it -- so a guard can read what a step index would remember.
Sequences and memory become load-bearing when progress or information is
TRANSIENT: a light seen once, a message received and gone, a leader that
disappears. That is the design decision the next version of this world makes,
and it is recorded here rather than tuned into the numbers.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btind.envs.intersection import (ACTIONS, SIG_ACTIONS, IntersectionBatch,
                                     NEAR_INT)
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, emit, mem_names

N_EP, SEED = 600, 11


def main(n_ep=N_EP):
    env = IntersectionBatch()
    N, SN = env.names, env.sig_names
    ix, sx = N.index, SN.index
    dv = len(mem_names(N, None)) + 1
    ds = len(mem_names(SN, None)) + 1

    def P(k, d=dv, n=5):
        th = np.zeros((d, n))
        th[-1, k] = 1.0
        return th
    A = {a: P(i) for i, a in enumerate(ACTIONS)}
    SA = {a: P(i, ds, 2) for i, a in enumerate(SIG_ACTIONS)}

    def vb(clauses, laws, default="ACCEL_MAX", **kw):
        b = dict(names=list(N), laws_on_z=True, head="argmax", actions=list(ACTIONS),
                 clauses=clauses, laws=laws, default=A[default])
        b.update(kw)
        return check_arms(b, "vb")

    def sb(clauses, laws, default="EXTEND", **kw):
        b = dict(names=list(SN), laws_on_z=True, head="argmax",
                 actions=list(SIG_ACTIONS), clauses=clauses, laws=laws,
                 default=SA[default])
        b.update(kw)
        return check_arms(b, "sb")

    close = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 10.0, True]]
    closing = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 25.0, True],
               [ix("lead_dv"), 0.5, False]]
    red_near = [[ix("green"), 0.5, True], [ix("green"), -0.5, False],
                [ix("d_stop"), 0.0, False], [ix("d_stop"), 45.0, True]]
    red_at = [[ix("green"), 0.5, True], [ix("green"), -0.5, False],
              [ix("d_stop"), 0.0, False], [ix("d_stop"), 6.0, True]]
    stopped = [[ix("v"), 0.5, True]]
    is_green = [[ix("green"), 0.5, False]]

    V = {}
    V["cruise"] = vb([], [])
    V["flat follower"] = vb([close, closing], [A["BRAKE_HARD"], A["BRAKE"]])
    V["flat follower + red stop"] = vb(
        [close, red_at, red_near, closing],
        [A["BRAKE_HARD"], A["BRAKE_HARD"], A["BRAKE"], A["BRAKE"]])
    V["flat + red stop + creep"] = vb(
        [close, red_at, red_near + [[ix("v"), 3.0, False]], closing],
        [A["BRAKE_HARD"], A["BRAKE_HARD"], A["BRAKE"], A["BRAKE"]])
    # the stop as a SEQUENCE: brake until stopped, hold until green, then go
    seq = vb([close, red_near, closing],
             [A["BRAKE_HARD"], A["BRAKE"], A["BRAKE"]],
             sticky=[False, True, False], betas=[None, is_green, None],
             steps=[None, [(stopped, A["HOLD"])], None])
    V["sequence: brake until stopped, hold until green"] = seq
    V["sequence + fail on close leader"] = dict(seq, fails=[None, close, None])
    V["sticky red stop (no steps)"] = vb(
        [close, red_near, closing], [A["BRAKE_HARD"], A["BRAKE"], A["BRAKE"]],
        sticky=[False, True, False], betas=[None, is_green, None])

    S = {"fixed plan": None}
    q_other = lambda k: [[sx("q%d" % k), 2.0, False]]
    own_empty = lambda k: [[sx("ph%d" % k), 0.5, False], [sx("n%d" % k), 0.5, True]]
    S["reactive: switch when own approach empty"] = sb(
        [own_empty(k) for k in range(4)], [SA["SWITCH"]] * 4)
    S["reactive: switch when own empty or held 25s"] = sb(
        [own_empty(k) for k in range(4)] + [[[sx("t_phase"), 25.0, False]]],
        [SA["SWITCH"]] * 5)

    s = env.sample_starts(n_ep, np.random.default_rng(SEED))
    g = env.geom
    print("opposing lefts conflict in the geometry: %s / %s"
          % (bool(g["conf"][2, 5]), bool(g["conf"][8, 11])))
    print("signal visible within %.0f m of the stop line\n" % NEAR_INT)

    def run(v, sg):
        return IF.run(env, v, sg, s, env.duration)

    t0 = time.time()
    print("VEHICLE trees under the fixed-time plan (%d episodes):" % n_ep)
    ref = None
    best_v = None
    for name, v in V.items():
        G = run(v, None)
        ref = G if ref is None else ref
        se = float((G - ref).std() / np.sqrt(len(G)))
        print("   %-50s G %8.2f   (%+7.2f se %.2f)" % (name, G.mean(), (G - ref).mean(), se))
        if best_v is None or G.mean() > best_v[1]:
            best_v = (name, G.mean(), v)
    print("\nSIGNAL trees with the best vehicle tree (%s):" % best_v[0])
    ref = None
    for name, sg in S.items():
        G = run(best_v[2], sg)
        ref = G if ref is None else ref
        se = float((G - ref).std() / np.sqrt(len(G)))
        print("   %-50s G %8.2f   (%+7.2f se %.2f)" % (name, G.mean(), (G - ref).mean(), se))
    print("\n" + emit(V["sequence + fail on close leader"], N))
    print("\n[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_EP)
