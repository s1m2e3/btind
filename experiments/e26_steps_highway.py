r"""e26 -- is the highway world sensitive to STEPS and FAILURE at all?

The e24 lesson, applied before a single step is searched for: ask the WORLD
first, with hand-written controllers, whether a Sequence of several actions or
a child that can FAIL buys anything a flat guarded action cannot. If the flat
ablations are within noise of the hand-written sequences, this world cannot test
the machinery and the search's silence on it would mean nothing.

WHAT A STEP CAN DO HERE THAT A GUARD CANNOT. `LANE_LEFT` is a one-shot command:
it decrements the target lane and the low-level controller completes the move.
A flat arm whose guard holds for several ticks re-issues it every tick and
drives the ego to the leftmost lane and keeps it there. A two-step arm issues it
once -- step 0 runs until the ego is actually moving sideways, then step 1 takes
over with a different manoeuvre -- and a fail clause hands the state back to
the arms below when someone cuts in. Neither is expressible as a flat guard on
the current observation because the observation does not contain the target
lane; the step index is the memory of having issued the command.

THREE HAND-WRITTEN TREES AND THEIR ABLATIONS, all measured on the same 3000
episodes with the fused kernel:

    flat        the guards with one law each, reactive              (today's class)
    sticky      the same arms latched with a termination            (today's class)
    steps       arm 0 as a two-step Sequence                        (new)
    steps+fail  the same with a fail clause on the sequence         (new)
    -steps      steps+fail with the second step removed             (ablation)
    -fail       steps+fail with the fail clause removed             (ablation)

The number that matters is steps+fail against the best of flat and sticky. If it
is not clearly positive, the world is the blocker, not the search.

WHAT CAME OUT, 3000 episodes, T = 40, fused kernel:

    flat: brake when the lead is close and closing     27.50   (10.2% end early)
    flat: lane-left when close, else brake             20.48   (27%)
    sticky lane-left until clear                       16.37 - 17.97   (54%)
    2 steps: lane-left once, then faster/idle           17.82 - 18.45   (57-60%)
    2 steps + fail on cut-in                            17.99 - 20.98   (33%)
    3 steps: left, idle until clear, right              18.46   (step 2 reached 2.6%)
    braking-regime sequences (slower once, then idle)  20.39 - 22.42   (24-52%)
    flat 2 arms: near -> slower, mid -> idle           23.67

THE MECHANISM WORKS AND THE WORLD DOES NOT WANT IT. The two-step arm reached its
second step in 68% of episodes, so steps run; and the fail clause was worth
+2.5 to +3.0 inside the family of committed overtakes, so status propagation
does something. But every sequence, every hysteresis and every fail variant
lost to the flat reactive brake tree, by 4 to 11 return units, and the losses
are crashes: any strategy that commits to a lane change ends 30-60% of
episodes early against 10% for braking. The meta-actions are already macro-
actions the low-level controller completes, and the world punishes commitment
-- the same finding e24 made about hysteresis on the masked NestWorld.

So on highway-v0 the search's silence on steps would say nothing about the
search, which is the reason this was measured before any step was looked for.
The one thing this world can test is the FAIL clause repairing a committed arm
(tests/test_stepsearch.py plants exactly that), and the world where sequences
are load-bearing by construction is the intersection.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btind.envs.highway_batch import ACTIONS, HighwayBatch
from btind.envs.highway_task import constant_bank, law_width
from btind.memory import MemBank, check_arms, emit
from btind.structure import score

N_EP, T, SEED = 3000, 40, 11


def main():
    env = HighwayBatch()
    N = env.names
    ix = N.index
    d = law_width(N)
    pol_fn = lambda b: MemBank(b, len(N))

    def pref(k):
        th = np.zeros((d, 5))
        th[-1, k] = 1.0
        return th
    A = {a: pref(i) for i, a in enumerate(ACTIONS)}

    def G(bank):
        check_arms(bank, "e26")
        return score(env, bank, pol_fn, N_EP, T, SEED)

    # --- the base: cruise FASTER; slow down when the lead is close and closing
    base = dict(constant_bank(N, ACTIONS.index("FASTER")), actions=list(ACTIONS))
    close = [[ix("v1_dx"), 25.0, True], [ix("v1_dx"), 0.0, False],
             [ix("v1_dvx"), 0.0, True]]                  # lead ahead, closing
    clear = [[ix("v1_dx"), 30.0, False]]                 # lead far ahead
    cut_in = [[ix("v1_dx"), 8.0, True], [ix("v1_dx"), 0.0, False]]
    moving_left = [[ix("ego_vy"), -0.3, True]]           # lane change under way
    lane_ok = [[ix("ego_y"), 2.0, False]]                # not already in lane 0

    trees = {}
    # FLAT: one arm, brake when close
    trees["flat: brake"] = dict(base, clauses=[close], laws=[A["SLOWER"]])
    # FLAT: two arms, change lane when close (re-issued every tick), else brake
    trees["flat: lane-left, brake"] = dict(
        base, clauses=[close + lane_ok, close], laws=[A["LANE_LEFT"], A["SLOWER"]])
    # STICKY: the lane-left arm latched until the lead is far
    trees["sticky: lane-left until clear, brake"] = dict(
        trees["flat: lane-left, brake"], sticky=[True, False], betas=[clear, None])
    # STEPS: lane-left ONCE (until moving sideways), then FASTER until clear
    steps_bank = dict(
        base, clauses=[close + lane_ok, close],
        laws=[A["LANE_LEFT"], A["SLOWER"]], sticky=[True, False],
        betas=[clear, None],
        steps=[[(moving_left, A["FASTER"])], None])
    trees["steps: lane-left once, faster until clear"] = steps_bank
    # STEPS + FAIL: abort the overtake if someone cuts in; the brake arm takes it
    trees["steps+fail: ... fail on cut-in"] = dict(steps_bank, fails=[cut_in, None])
    # ABLATIONS of steps+fail
    trees["-steps (fail only)"] = dict(steps_bank, steps=None, fails=[cut_in, None])
    trees["-fail (steps only)"] = steps_bank
    # a three-step version: lane-left once, faster until clear-ish, lane-right once
    trees["3 steps: left, faster, right"] = dict(
        base, clauses=[close + lane_ok, close],
        laws=[A["LANE_LEFT"], A["SLOWER"]], sticky=[True, False],
        betas=[[[ix("ego_vy"), 0.3, False]], None],      # released once moving right
        steps=[[(moving_left, A["FASTER"]), (clear, A["LANE_RIGHT"])], None],
        fails=[cut_in, None])

    print("constant actions:")
    for a in ACTIONS:
        print("   %-10s %6.2f" % (a, G(constant_bank(N, ACTIONS.index(a))).mean()))
    print("\nhand-written trees, %d episodes, T=%d:" % (N_EP, T))
    t0 = time.time()
    ref = None
    out = {}
    for name, b in trees.items():
        g = G(b)
        out[name] = g
        if ref is None:
            ref = g
        se = float((g - ref).std() / np.sqrt(len(g)))
        print("   %-46s %6.2f   vs first %+6.2f  (se %.2f)"
              % (name, g.mean(), (g - ref).mean(), se))
    flat_best = max(out[k].mean() for k in out if k.startswith(("flat", "sticky")))
    sf = out["steps+fail: ... fail on cut-in"]
    print("\nsteps+fail against the best flat/sticky tree: %+.2f" % (sf.mean() - flat_best))
    print("fail is worth   %+.2f" % (sf.mean() - out["-fail (steps only)"].mean()))
    print("steps are worth %+.2f" % (sf.mean() - out["-steps (fail only)"].mean()))
    print("\n" + emit(trees["3 steps: left, faster, right"], N))
    print("\n[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main()
