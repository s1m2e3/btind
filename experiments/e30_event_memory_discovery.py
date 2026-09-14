r"""e30 -- can the search find the message, the clock and the guards that read them?

Event-mode intersection, fixed-time plan. The start is the hand-written
memoryless follower, which scores like cruising here (-37.6, e29) because it
sees the light for one tick and cannot remember it. Nothing names the message:
`discover_transient` is handed the follower's own recorded trajectories and the
whole observation, planted `t_norm` and `noise` included, and has to choose
what to store, when to write, whether it is a clock, and which guards read it.

The hand-written ceiling is -1.45 with the clock and -25 without it (e29). The
run reports what was found, what it is worth, and whether a planted column was
stored or guarded on.

WHAT CAME OUT (pool 25, 2 arms per candidate, 2096 s, 2026-09-14):

    memoryless follower, event mode      held-out  -37.19
    after the memory stage               held-out  -16.24   (+20.95)

    stored:  t_sig + all_red   written when  t_sig > -0.588   with the clock
    arm:     left_t_sig > -1.000  ->  a CEM-tuned preference over the widened z
    planted columns stored or guarded:  none

Every one of 26 candidates that produced an arm was worth +17 to +21 -- the
message is redundant across its three columns, and even a car's OWN speed
dipping below V0 (the follower's one-tick brake on the message tick) encodes
"I was told", so the search had several roads to the same fact. It took the
best: the message's time-to-change and the all-red flag, as a clock. That is
+21 of the +36 a hand-written clock controller recovers, from one memory
stage with a two-arm budget; the rest is guards and laws on the remembered
columns, which is what the rounds after the memory stage refine. The clock
was chosen over the plain store on 4 of the 6 message-shaped candidates.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btind.envs.intersection import IntersectionBatch
from btind.memory import MemBank, emit, mem_names
from btind.memtransient import discover_transient
from btind.structure import heldout

DISTRACT = ("t_norm", "noise")


def main(pool=40, confirm_ep=400, screen_ep=120, seed=0):
    env = IntersectionBatch(sig_mode="event")
    names = env.veh_names
    n_obs = len(names)
    pol_fn = lambda b: MemBank(b, n_obs)
    bank = env.default_vehicle_bank()
    t0 = time.time()
    m0 = heldout(env, bank, n_ep=600)
    print("memoryless follower, event mode: held-out G %.2f +-%.2f" % (m0["G"], m0["ci"]))
    OB, AL = env.record_traces(bank, n_ep=80, seed=3)
    print("recorded %d trajectories" % len(OB), flush=True)
    out, log = discover_transient(env, bank, names, pol_fn, OB, AL, n_obs,
                                  screen_ep=screen_ep, confirm_ep=confirm_ep,
                                  T=env.duration, seed=11, z=2.0, min_gain=0.5,
                                  max_arms=2, rng=np.random.default_rng(seed),
                                  verbose=True, pool=pool)
    m1 = heldout(env, out, n_ep=600)
    print("\nafter the memory stage: held-out G %.2f +-%.2f  (%+.2f)  [%.0fs]"
          % (m1["G"], m1["ci"], m1["G"] - m0["G"], time.time() - t0))
    print(emit(out, names))
    if out.get("mem"):
        zn = mem_names(names, out["mem"])
        stored = [names[j] for j in out["mem"]["cols"]]
        used = sorted({zn[l[0]] for c in out["clauses"] for l in c})
        print("\nstored: %s   clock: %s" % (", ".join(stored), bool(out["mem"].get("countdown"))))
        print("guards read: %s" % ", ".join(used))
        print("planted columns stored or guarded: %s"
              % (", ".join(sorted(set(stored + used) & set(DISTRACT))) or "none"))
    return out


if __name__ == "__main__":
    main(pool=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
