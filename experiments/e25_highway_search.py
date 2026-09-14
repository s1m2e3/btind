r"""e25 -- the search, on highway, against a null that prefers nothing.

WHAT IS BEING TESTED. Not whether a good highway controller can be written --
one can, in ten lines, and writing it would answer nothing. The question is
whether the same machinery that found a partition, a blackboard and a set of
control laws on NestWorld finds them on a borrowed world whose action set is
discrete, whose observation is raw kinematics, and which nobody here designed.

THE START PREFERS NOTHING. The null bank has one law, all five manoeuvres tied
at zero, and no arms. It is not the best constant action, and it is not a random
draw: the constant actions span 20.2 to 25.9 on this world, so a lucky
initialisation would be worth points the search would then appear to have found.

THE REFERENCE IS THE BEST CONSTANT ACTION, which is the strongest thing that can
be said without any state at all:

    LANE_RIGHT 25.89 (11.6% crash)    IDLE    24.98 (11.1%)
    LANE_LEFT  23.78 (12.2%)          SLOWER  22.87 ( 1.7%)
                                      FASTER  20.24 (50.7%)

Beating 25.89 means the partition is buying something a fixed manoeuvre cannot.

THE DISTRACTOR CONTROL IS IN THE ALPHABET, not bolted on afterwards. `t_norm`
and `noise` are two planted columns carrying nothing about the road, and they
are offered to every stage exactly like the real ones -- proposable as guards,
as thresholds, and as single-column law terms. A search that guards on them is
buying junk, and the run reports how often they were tried and how often
accepted beside the real columns. Without them there is no way to tell a working
search from a lucky one.

WHAT IS NOT HERE YET, said plainly rather than left to be noticed: the
blackboard. `memory_primitives` builds `mem_c - c` as a two-component direction,
which means something for a holonomic agent and nothing as a preference over
named manoeuvres, so the memory stage skips itself on a discrete head. Beta and
stickiness DO run, and this world is the reason -- on NestWorld a latch had
nothing to sell because `t_capture` already carried what it would remember,
while a lane change here takes several steps and aborting it halfway is the
failure mode. Whether beta finally earns its place is the open question this run
exists to answer.

TRANSFER IS THE LAST WORD. Everything here is measured on the fused port; the
emitted tree is then re-scored inside real highway-env, because the port is a
MODEL of that world and the difference between the two numbers is the honest
error bar on all of this.
"""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("experiments", 1)[0])
from btind.envs.highway_batch import ACTIONS, HighwayBatch
from btind.envs.highway_task import constant_bank, law_width
from btind.memory import MemBank, emit, mem_names
from btind.rlfit import fit
from btind.runlog import RunLog
from btind.structure import score
from btind import proposal as PR

DISTRACTORS = ("t_norm", "noise")


def baselines(env, n_ep=3000, T=40, seed=11):
    """The best fixed manoeuvre -- what the search has to beat with no state."""
    pol_fn = lambda b: MemBank(b, len(env.names))
    out = {}
    for i, a in enumerate(ACTIONS):
        out[a] = float(score(env, constant_bank(env.names, i), pol_fn,
                             n_ep, T, seed).mean())
    return out


def distractor_report(env, zn):
    """How often the planted columns were proposed, and how often accepted.

    Read against the real columns in the same table. The claim "it discovered
    the partition" is only worth something if the junk was on offer and lost.
    """
    w = PR.load(env, len(zn) + 8)
    rows = []
    for j, nm in enumerate(zn):
        if j < w.n and w.tried[j] > 0:
            rows.append((nm, int(w.acc[j]), int(w.tried[j]),
                         w.acc[j] / max(w.tried[j], 1)))
    rows.sort(key=lambda r: -r[3])
    return rows


def main(rounds=4, seed=0, n_ep=600):
    rl = RunLog("e25-highway-search")
    env = HighwayBatch()
    N = env.names
    zn = mem_names(N, None)
    rl.config(env, dict(rounds=rounds, seed=seed, n_ep=n_ep))

    print("constant-action reference (3000 episodes):")
    base = baselines(env)
    for a, g in sorted(base.items(), key=lambda kv: -kv[1]):
        print("   %-10s %7.2f" % (a, g))
    best_const = max(base.values())
    print("   %-10s %7.2f  <- the number to beat\n" % ("best", best_const))

    t0 = time.time()
    bank, log, m = fit(env, list(N), rounds=rounds, warm=True, run_seed=seed,
                       tag="e25", cfg=dict(n_ep=n_ep, T=40, seed=11,
                                           val_ep=1200, mem_at=99))
    print("\n" + emit(bank, N))
    print("\nheld out G %.2f +-%.2f   vs best constant %.2f   (%+.2f)  [%.0fs]"
          % (m["G"], m["ci"], best_const, m["G"] - best_const, time.time() - t0))

    print("\nDISTRACTOR CONTROL -- planted columns are marked; they were offered")
    print("to every stage exactly like the real ones.")
    print("   %-12s %8s %8s %9s" % ("column", "accepted", "tried", "rate"))
    for nm, acc, tried, rate in distractor_report(env, zn)[:12]:
        print("   %-12s %8d %8d %8.3f %s"
              % (nm, acc, tried, rate, "   <- PLANTED" if nm in DISTRACTORS else ""))
    used = [zn[l[0]] for c in bank["clauses"] for l in c]
    bad = [u for u in used if u in DISTRACTORS]
    print("\n   guards in the emitted tree: %s" % ", ".join(used))
    print("   planted columns among them: %s" % (", ".join(bad) if bad else "none"))

    rl.bank(bank, zn, bt=emit(bank, N))
    rl.finish(dict(G=m["G"], ci=m["ci"], best_constant=best_const,
                   gain=m["G"] - best_const, rounds=rounds, seed=seed,
                   distractors_in_tree=bad), bt=emit(bank, N))
    print("\nlogged to %s" % rl.dir)
    return bank


if __name__ == "__main__":
    main(rounds=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
         seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
