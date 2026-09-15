r"""e33 -- does the traffic-performance reward rank controllers the way a traffic
engineer would?

Before any search runs on the new reward, the e24 protocol again: hand-written
controllers the reward must separate, on the same episodes, 150 s, 64 slots.

    stop every car            the dead end the continuous-leaf search fell into
    cruise                    runs every red
    follower, fixed plan      the hand-written reference vehicle tree
    follower, always extend   a signal that holds each phase to T_MAX: starves
                              the last phase, what the discovered signal did
    follower, actuated        switch when the served approach is empty

The reward is right if stopping and starving are clearly worse than the fixed
plan, the actuated signal is at least as good, and a crash dominates.
Each component is reported per controller so a wrong ranking can be traced to
the term that caused it.

WHAT CAME OUT, 600 episodes (G, discounted) and 80 episodes (terms, per
episode, undiscounted), after two corrections the first pass forced:

                          G   speed  delay  queue  stuck   red  crash  exit  left
    stop every car    -1165      3   -198   -774   -216     0      0     0  -266
    cruise             -675    144     -1      0      0  -357   -573    44   -21
    follower, fixed    -227     62    -60    -89    -21   -15   -110    38   -71
    follower, starving -626     47   -103   -249    -11    -5   -345    26  -119
    follower, actuated  -82     76    -42    -51    -21     0    -40    41   -60

The starving signal never serves phase 3 in any episode and pays for it in
queue, leftover and the rear-ends its spillback causes; the actuated signal
serves every phase and beats the fixed plan by 145.

THE TWO CORRECTIONS. At R_COLL = 50, cruising (2.9 crashes, 36 red runs an
episode) scored -294 and outranked the starving signal at -478: a crash was
cheaper than a long queue. At 200 it is not. And "stuck" first included cars
stopped on red more than 15 m short of the line, charging the hand-written
follower 72 per episode for cautious stopping; stuck now means stopped in or
past the box, or on a green with nobody close ahead, and the follower's 21 is
the second or two a car takes to get moving when the light turns.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btind.envs.intersection import ACTIONS, SIG_ACTIONS, IntersectionBatch
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, mem_names

N_EP, SEED = 600, 11


def served_phases(env, vb, sb, s):
    """Share of episodes in which each phase gets at least one live tick."""
    from btind.tick import trace_array
    SN = env.sig_names
    probe = sb or dict(names=list(SN), laws_on_z=True, head="argmax", clauses=[],
                       laws=[], default=np.zeros((len(mem_names(SN, None)) + 1, 2)))
    ds = len(mem_names(SN, probe.get("mem"))) + 1
    dev = np.zeros((len(s), 4))
    dev[:, 3] = -1
    tr = trace_array(len(s), env.duration, ds)
    IF.run(env, vb, sb, s, env.duration, trace=tr, dev=dev)
    ph = np.argmax(tr[:, :, [SN.index("ph%d" % k) for k in range(4)]], 2)
    live = tr[:, :, SN.index("all_red")] < 0.5
    return [float(((ph == k) & live).any(1).mean()) for k in range(4)]


def main(n_ep=N_EP):
    env = IntersectionBatch()
    N, SN = env.veh_names, env.sig_names
    ix, sx = N.index, SN.index
    dv = len(mem_names(N, None)) + 1
    ds = len(mem_names(SN, None)) + 1

    def P(k, d=dv, n=5):
        th = np.zeros((d, n))
        th[-1, k] = 1.0
        return th

    def vb(clauses, laws, default):
        return check_arms(dict(names=list(N), laws_on_z=True, head="argmax",
                               actions=list(ACTIONS), clauses=clauses,
                               laws=[P(ACTIONS.index(a)) for a in laws],
                               default=P(ACTIONS.index(default))), "vb")

    def sb(clauses, laws, default):
        return check_arms(dict(names=list(SN), laws_on_z=True, head="argmax",
                               actions=list(SIG_ACTIONS), clauses=clauses,
                               laws=[P(SIG_ACTIONS.index(a), ds, 2) for a in laws],
                               default=P(SIG_ACTIONS.index(default), ds, 2)), "sb")

    follower = env.default_vehicle_bank()
    own_empty = lambda k: [[sx("ph%d" % k), 0.5, False], [sx("n%d" % k), 0.5, True]]
    cases = [
        ("stop every car", vb([], [], "BRAKE_HARD"), None),
        ("cruise", vb([], [], "HOLD"), None),
        ("follower, fixed plan", follower, None),
        ("follower, always extend (starves)", follower, sb([], [], "EXTEND")),
        ("follower, actuated", follower,
         sb([own_empty(k) for k in range(4)], ["SWITCH"] * 4, "EXTEND")),
    ]
    s = env.sample_starts(n_ep, np.random.default_rng(SEED))
    t0 = time.time()
    print("150 s, %d slots, %d episodes, gamma %.3f\n" % (env.N, n_ep, env.gamma))
    print("%-36s %9s  %s" % ("controller", "G", "phases served (share of episodes)"))
    ref = None
    for name, v, sg in cases:
        G = IF.run(env, v, sg, s, env.duration)
        ph = served_phases(env, v, sg, s)
        print("%-36s %9.1f  %s" % (name, G.mean(), " ".join("%.2f" % x for x in ph)))
    print("\n[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_EP)
