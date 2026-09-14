r"""e29 -- is memory load-bearing when the intersection speaks once?

Event mode: a car entering the 60 m zone receives (my light, ticks until it
changes, all-red) on that tick and nothing afterwards. The e24 protocol again,
before any search: hand-written controllers with and without a blackboard,
against the continuous-mode reference, on the same episodes.

    memoryless in event mode      the best a tree can do that cannot remember:
                                  it sees the light for ONE tick and must
                                  decide from ego state alone afterwards
    blackboard, no countdown      store the message; guards on the stored
                                  light and on the age of the message
    blackboard + countdown        store the message; guards on `left_t_sig`,
                                  the remembered clock

If the blackboard controllers approach the continuous-mode ceiling and the
memoryless ones do not, memory is worth the difference and the search has
something to find that no reactive tree can express.

WHAT CAME OUT, 600 episodes, fixed-time plan, same arrival schedules:

    reference: continuous-mode follower               -0.37
    memoryless, same follower, event mode            -37.62   = cruising
    memoryless, stop at the line and creep           -44.91
    memoryless, leader-cued                          -39.70
    blackboard, stored light + message age           -24.96
    blackboard + countdown (remembered clock)         -1.45   within 1.1 of the ceiling
    ... without the leader arms                      -50.59

MEMORY IS WORTH +36 HERE AND THE CLOCK +23 ON TOP OF IT. A tree that cannot
remember sees the light for one tick and is then a car that knows nothing, and
no arrangement of reactive guards recovers that: the three memoryless variants
span -37 to -45. Storing the message recovers a third of the loss; storing it
with a countdown recovers almost all of it, because "how long until my light
changes" is a quantity that has to be READ OFF A CLOCK the car carries. This
is the first world in this project where the blackboard is load-bearing for
a reason other than occlusion of a target, and it is the world the memory
search runs on next.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btind.envs.intersection import ACTIONS, IntersectionBatch, SIG_HIDDEN
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, emit, mem_names

N_EP, SEED = 600, 11


def main(n_ep=N_EP):
    ev = IntersectionBatch(sig_mode="event")
    co = IntersectionBatch()
    N = ev.veh_names
    ix = N.index

    def P(k, d):
        th = np.zeros((d, 5))
        th[-1, k] = 1.0
        return th

    def vb(clauses, laws, mem=None, default="ACCEL_MAX", **kw):
        zn = mem_names(N, mem)
        d = len(zn) + 1
        A = {a: P(i, d) for i, a in enumerate(ACTIONS)}
        b = dict(names=list(N), laws_on_z=True, head="argmax", actions=list(ACTIONS),
                 clauses=clauses, laws=[A[l] for l in laws], default=A[default],
                 mem=mem)
        b.update(kw)
        return check_arms(b, "vb"), zn.index

    close = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 10.0, True]]
    closing = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 25.0, True],
               [ix("lead_dv"), 0.5, False]]
    red = lambda lo, hi: [[ix("green"), 0.5, True], [ix("green"), -0.5, False],
                          [ix("d_stop"), lo, False], [ix("d_stop"), hi, True]]
    near = lambda lo, hi: [[ix("d_stop"), lo, False], [ix("d_stop"), hi, True]]

    V = {}
    V["reference: continuous-mode follower"], _ = vb(
        [close, red(0, 6), red(0, 45), closing],
        ["BRAKE_HARD", "BRAKE_HARD", "BRAKE", "BRAKE"])
    # memoryless in event mode: the same tree (sees the light for one tick)
    V["memoryless: same follower"], _ = vb(
        [close, red(0, 6), red(0, 45), closing],
        ["BRAKE_HARD", "BRAKE_HARD", "BRAKE", "BRAKE"])
    # memoryless, cautious: always stop at the line, creep through slowly
    V["memoryless: stop at line, creep"], _ = vb(
        [close, near(0, 8) + [[ix("v"), 2.0, False]], near(0, 45) + [[ix("v"), 5.0, False]], closing],
        ["BRAKE_HARD", "BRAKE_HARD", "BRAKE", "BRAKE"])
    V["memoryless: follower, leader-cued"], _ = vb(
        [close, near(0, 8) + [[ix("has_lead"), 0.5, True]], closing],
        ["BRAKE_HARD", "BRAKE_HARD", "BRAKE"])

    msg = dict(cols=[ix("green"), ix("t_sig")],
               write=[[ix("green"), SIG_HIDDEN + 0.5, False]], clear=None)
    # blackboard without countdown: stored light and message age (ticks)
    b, zi = vb([], [], mem=msg)
    have = [[zi("have_mem"), 0.5, False]]
    told_red = have + [[zi("mem_green"), 0.5, True]]
    told_green = have + [[zi("mem_green"), 0.5, False]]
    V["blackboard, no countdown: told red -> stop until age > 60"], _ = vb(
        [close,
         told_red + [[zi("mem_age"), 60.0, True]] + near(0, 45),
         told_green + [[zi("mem_age"), 40.0, False]] + near(0, 45),
         closing],
        ["BRAKE_HARD", "BRAKE", "BRAKE", "BRAKE"], mem=msg)
    # blackboard with countdown: the remembered clock
    msg_cd = dict(msg, countdown=True)
    b, zi = vb([], [], mem=msg_cd)
    have = [[zi("have_mem"), 0.5, False]]
    told_red = have + [[zi("mem_green"), 0.5, True]]
    told_green = have + [[zi("mem_green"), 0.5, False]]
    V["blackboard + countdown"], _ = vb(
        [close,
         told_red + [[zi("left_t_sig"), 0.0, False]] + near(0, 6),        # red, wait
         told_red + [[zi("left_t_sig"), 0.0, False]] + near(0, 45),
         told_green + [[zi("left_t_sig"), 8.0, True]] + near(0, 45),      # about to turn
         closing],
        ["BRAKE_HARD", "BRAKE_HARD", "BRAKE", "BRAKE", "BRAKE"], mem=msg_cd)
    V["blackboard + countdown, no leader arms"], _ = vb(
        [told_red + [[zi("left_t_sig"), 0.0, False]] + near(0, 6),
         told_red + [[zi("left_t_sig"), 0.0, False]] + near(0, 45),
         told_green + [[zi("left_t_sig"), 8.0, True]] + near(0, 45)],
        ["BRAKE_HARD", "BRAKE", "BRAKE"], mem=msg_cd)

    s = ev.sample_starts(n_ep, np.random.default_rng(SEED))
    t0 = time.time()
    print("event-mode intersection, %d episodes, fixed-time plan" % n_ep)
    ref = None
    for name, b in V.items():
        env = co if name.startswith("reference") else ev
        G = IF.run(env, b, None, s, env.duration)
        ref = G if ref is None else ref
        se = float((G - ref).std() / np.sqrt(len(G)))
        print("   %-58s G %8.2f   (%+7.2f se %.2f)" % (name, G.mean(), (G - ref).mean(), se))
    print("\n" + emit(V["blackboard + countdown"], N))
    print("\n[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_EP)
