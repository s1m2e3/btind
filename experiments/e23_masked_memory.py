r"""e23 -- memory that actually fires, on the masked world.

WHAT THIS RUN IS TESTING. Every previous masked run reported a memory gain of
exactly +0.00, on every candidate, at every position. That was not a null
result about memory: the emitted tree carried `is_night > -0.040` and `is_night`
takes the values {0, 1}, so that guard matched 100% of states, the default was
reached on 0.0% of ticks, and every arm below it -- which is where the memory
search appends -- was dead code. All 10752 candidates scored identically zero
because none of them ever ran.

FOUR CHANGES, each of which had to be made before the next one could be seen:

    absorb_universal    an arm whose guard matches everything IS the default.
                        Its law becomes the default's law and the dead arms
                        below it are removed, which costs nothing (the move is
                        priced for non-inferiority, and its true delta is 0.00)
                        and makes the end of the tree reachable again.

    refine_guard        `have_mem > 0.5` is true forever once the blackboard has
                        been written, so as a singleton the arm either overrides
                        the whole tree from the top or inherits leftovers at the
                        bottom. A SECOND literal is searched from the same
                        alphabet as any other guard -- not written by hand, and
                        the task word "food" appears nowhere in it.

    the honest reference    scored against the bank with the unrefined guard,
                        the winning conjunct is one that is never true: measured,
                        `slack_food <= -45.377` scored +15.14 that way and +0.00
                        against no memory, because all it did was switch the arm
                        off. The reference is the tree WITHOUT the arm.

    joint screening     a literal is only good somewhere. Ranking literals at
                        one position and position-searching the survivors drops
                        the winner before it is ever placed.

CONTROLS. The same tree with the discovered store replaced by the planted
`t_norm` and `noise`, which must be worth nothing or less; and the unmasked
world.

THE UNMASKED CONTROL DOES NOT SAY WHAT IT WAS WRITTEN TO SAY, and the run that
first passed the distractor control is what showed it. "Memory must be worth
zero when nothing is hidden" is true of memory used for RECALL -- latch where
the food was, walk back to it -- and that is the only use this project had in
mind. It is false of memory in general. The search found a different use: it
stored `bear_threat` while the threat was far and steered on
`mem_bear_threat - bear_threat`, which is a FINITE DIFFERENCE over time. The
observation has 25 columns and not one of them is a velocity, so the blackboard
is supplying information the observation does not contain in any masking
condition, and it pays in both -- measured, +7.01 masked and +8.73 unmasked.

That is a real capability of a controller with state rather than a leak, so the
control is reported as it came out rather than tightened until it passes. What
it no longer certifies is that the discovered memory is about observability;
for that, read the distractor row, which is exact (5.97 with the planted store
against 5.97 with no blackboard at all).
"""
import sys, time
import numpy as np

sys.path.insert(0, __file__.rsplit("experiments", 1)[0])
from btind.envs.nest import NestWorld, OBS_NAMES
from btind.memory import MemBank, emit, mem_names, relayout, without_memory
from btind.pipeline import evaluate_bank
from btind.rlfit import fit
from btind.runlog import RunLog

MASK = dict(vision_r=0.30, night_vision_r=0.15, threat_vision_r=0.35)
BASE = dict(day_len=80.0, food_persistent=True)


def main(rounds=4, seed=0):
    rl = RunLog("e23-masked-memory")
    env = NestWorld(**dict(BASE, **MASK))
    rl.config(env, dict(rounds=rounds, seed=seed))
    n_obs = len(OBS_NAMES)
    t0 = time.time()
    bank, log, m = fit(env, list(OBS_NAMES), rounds=rounds, warm=True,
                       run_seed=seed, tag="e23")
    print("\n" + emit(bank, OBS_NAMES))
    print("\nmasked, held out:  G %.2f +-%.2f  pickups %.2f  [%.0fs]"
          % (m["G"], m["ci"], m["eaten"], time.time() - t0))

    rows = [("masked, discovered tree", m["G"], m["ci"])]
    if bank.get("mem"):
        nom = without_memory(bank, OBS_NAMES)
        mn = evaluate_bank(env, nom, n_obs=n_obs)
        rows.append(("  same tree, blackboard removed", mn["G"], mn["ci"]))
        # DISTRACTOR STORE: same tree, same write event, planted columns.
        I = {n: i for i, n in enumerate(OBS_NAMES)}
        dmem = dict(bank["mem"], cols=[I["t_norm"], I["noise"]])
        zn_o = mem_names(OBS_NAMES, bank["mem"])
        zn_d = mem_names(OBS_NAMES, dmem)
        db = dict(bank, mem=dmem,
                  laws=[relayout(l, zn_o, zn_d) for l in bank["laws"]],
                  default=relayout(bank["default"], zn_o, zn_d))
        md = evaluate_bank(env, db, n_obs=n_obs)
        rows.append(("  same tree, storing t_norm+noise", md["G"], md["ci"]))
    # UNMASKED: the blackboard arm must be worth nothing when nothing is hidden.
    ue = NestWorld(**BASE)
    mu = evaluate_bank(ue, bank, n_obs=n_obs)
    rows.append(("unmasked, same tree", mu["G"], mu["ci"]))
    if bank.get("mem"):
        mun = evaluate_bank(ue, without_memory(bank, OBS_NAMES), n_obs=n_obs)
        rows.append(("  unmasked, blackboard removed", mun["G"], mun["ci"]))

    print("\n%-36s %8s %7s" % ("control", "G", "+-"))
    for nm, g, ci in rows:
        print("%-36s %8.2f %7.2f" % (nm, g, ci))
    rl.bank(bank, mem_names(OBS_NAMES, bank.get("mem")),
            bt=emit(bank, OBS_NAMES))
    rl.finish(dict(G=m["G"], ci=m["ci"], eaten=m["eaten"], rounds=rounds,
                   seed=seed, controls={n: float(g) for n, g, _ in rows}),
              bt=emit(bank, OBS_NAMES))
    print("\nlogged to %s" % rl.dir)


if __name__ == "__main__":
    main(rounds=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
         seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
