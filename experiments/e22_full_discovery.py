r"""e22 -- everything from zero on a partially observed world.

No hand-built scaffold. The guards, the laws, the arm count and the blackboard
are all discovered in one run, from a planner that has never been told what the
agent can see.

    stage 1   imitation      DAgger against the CEM planner, which is PRIVILEGED:
                             it plans from the true state, including the food the
                             controller cannot see. That is not a mistake to fix,
                             it is the only teacher available -- but it means the
                             student can never match it, and stage 1's number
                             should be read as a starting point, not a ceiling.
    stage 2   scheduled RL   six rounds, each mechanism on its own cadence:

                  every round     V-hat by fitted value iteration, Q-hat by
                                  probing, then the region-restricted policy
                                  gradient on the laws and boundary-term moves
                                  on the thresholds
                  every 3 rounds  law library search (the discrete jump a
                                  gradient cannot make), drop/reorder
                  rounds 2, 5     beta and stickiness, ranked by which arms the
                                  controller keeps abandoning and re-entering
                  round 1         memory: what to store and when to write it,
                                  searched jointly with an arm that reads it

              Running everything every round is unaffordable (the memory sweep
              costs ~300s against ~20s for a gradient step) and wrong: a move
              proposed against stale laws is priced against a counterfactual the
              next law step destroys.

THREE CONTROLS, all reported, none of them optional.

  UNMASKED. The same emitted tree, run on the world with nothing hidden. Memory
  must be worth ZERO there: `food_seen` is always true, so the blackboard arm can
  never fire. A non-zero gain would mean the machinery is buying something that
  has nothing to do with observability.

  DISTRACTOR STORE. The same tree with the same write event, storing `t_norm` and
  `noise` instead of whatever was discovered. Measured on a hand-built version of
  this controller it scored -1.34 against a no-memory 2.06: storing the wrong
  thing is not neutral, it is worse than remembering nothing.

  THE REFERENCES. A hand-written reactive controller scores 2.47 masked and 11.25
  unmasked; a hand-written blackboard controller scores 11.25 masked. So 8.78 is
  what memory is worth on this world, and the question is how much of it a search
  that was told nothing can recover.

WHY DAgger STOPS AFTER STAGE 1. Measured in e19 over four runs, re-labelling
on-policy states with the planner and refitting was worth -3.0 return where a
gradient step from the critic on the same rows was worth +1.8. The planner is
optimal for a controller that replans every tick and ours does not -- and under
masking its labels are a PRIVILEGED controller's answers, which the student
cannot reproduce even in principle. It is a warm start, not a teacher to follow.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btind.envs.nest import NestWorld, OBS_NAMES
from btind.pipeline import DEFAULTS, imitate, evaluate_bank
from btind.memory import MemBank, mem_names, widen, emit
from btind.memsearch import record, discover
from btind import loop as LP
from btind.lawsearch import search_laws
from btind.structure import accept, drop_arm, reorder, score
from btind.landscape import materialise_default
from btind.policies import evaluate
from btind.runlog import RunLog
from btind import qwire as Q

WORLD = dict(day_len=80.0, food_persistent=True)
MASK = dict(vision_r=0.30, night_vision_r=0.15, threat_vision_r=0.35)
DOMAIN = dict(
    cand_names=("d_threat", "energy", "d_food", "carrying", "d_nest",
                "is_night", "t_to_switch", "food_seen", "threat_seen",
                "pos_x", "pos_y", "t_norm", "noise"),
    radial=(0, 3, 4, 12), bearing=((1, 2), (5, 6), (13, 14)),
    search_K=192, search_H=120, search_n_seg=6, T=400)
SEL_EP, SEL_SEED, Z_THR = 600, 777, 2.0
DISTRACT = ("noise", "t_norm")


def main():
    t0 = time.time()
    seed = int(os.environ.get("E22_SEED", "0"))
    rl = RunLog("e22-full")
    env = NestWorld(**dict(WORLD, **MASK))
    n_obs = len(OBS_NAMES)
    cfg = dict(DEFAULTS)
    cfg.update(DOMAIN)
    rl.config(env, cfg, extra=dict(seed=seed))
    zn0 = mem_names(OBS_NAMES, None)
    pol_for = lambda zn: (lambda b: MemBank(b, n_obs))

    print("stage 1 -- imitation against a privileged planner", flush=True)
    env.seed_kernels(seed)
    bank = imitate(env, cfg, OBS_NAMES, np.random.default_rng(4 + seed))
    bank, _ = materialise_default(bank)
    bank = widen(bank, list(OBS_NAMES), zn0)          # laws move onto z
    r1 = evaluate_bank(env, bank, n_obs=n_obs)
    print("   G %.2f +-%.2f  pickups %.2f  [%.0fs]"
          % (r1["G"], r1["ci"], r1["eaten"], time.time() - t0), flush=True)

    print("stage 2 -- scheduled refinement: critic, gradient, library, "
          "structure, terminations, memory", flush=True)
    cfg.update(sel_ep=SEL_EP, z=Z_THR, n_on=12000, memory=True,
               mem_thr=7, mem_screen_ep=120,
               etas=(0.02, 0.05, 0.1, 0.2))
    bank, rlog, state = LP.run(env, bank, OBS_NAMES, cfg, n_rounds=6,
                               rng=np.random.default_rng(5 + seed),
                               runlog=rl, verbose=True,
                               sched=dict(memory_at=1, library_every=3,
                                          structure_every=3, beta_every=3,
                                          beta_offset=2))
    zn = mem_names(OBS_NAMES, bank.get("mem"))
    r4 = evaluate_bank(env, bank, n_obs=n_obs)
    r2 = r3 = r4
    print("   G %.2f +-%.2f  pickups %.2f  [%.0fs]"
          % (r4["G"], r4["ci"], r4["eaten"], time.time() - t0), flush=True)

    # ---------------------------------------------------------- the controls
    print("\ncontrols", flush=True)
    unmasked = NestWorld(**WORLD)
    r_un = evaluate_bank(unmasked, bank, n_obs=n_obs)
    no_mem = dict(bank, mem=None)
    no_mem = dict(no_mem,
                  clauses=[c for c in bank["clauses"]
                           if not any(l[0] >= len(zn0) for l in c)])
    no_mem["laws"] = bank["laws"][:len(no_mem["clauses"])]
    r_nm = evaluate_bank(env, no_mem, n_obs=n_obs)
    r_un_nm = evaluate_bank(unmasked, no_mem, n_obs=n_obs)
    ctrl = None
    if bank.get("mem"):
        d = [OBS_NAMES.index(n) for n in DISTRACT]
        bad = dict(bank, mem=dict(bank["mem"], cols=d))
        ctrl = evaluate_bank(env, bad, n_obs=n_obs)

    print("=" * 74)
    print("%-42s %8s %8s" % ("", "G", "pickups"))
    print("%-42s %8.2f %8.2f" % ("masked, discovered tree", r4["G"], r4["eaten"]))
    print("%-42s %8.2f %8.2f" % ("masked, same tree minus memory", r_nm["G"], r_nm["eaten"]))
    if ctrl:
        print("%-42s %8.2f %8.2f" % ("masked, memory stores noise+t_norm", ctrl["G"], ctrl["eaten"]))
    print("%-42s %8.2f %8.2f" % ("unmasked, discovered tree", r_un["G"], r_un["eaten"]))
    print("%-42s %8.2f %8.2f" % ("unmasked, same tree minus memory", r_un_nm["G"], r_un_nm["eaten"]))
    print("-" * 74)
    print("memory gain, masked   %+.2f   (hand-written reference: +8.78)"
          % (r4["G"] - r_nm["G"]))
    print("memory gain, unmasked %+.2f   (must be 0.00)"
          % (r_un["G"] - r_un_nm["G"]))
    print("=" * 74)
    bt = emit(bank, OBS_NAMES)
    print(bt)
    used = sorted({zn[l[0]] for c in bank["clauses"] for l in c})
    print("\nguard features: %s" % ",".join(used))
    print("distractors in guards: %s"
          % (",".join(f for f in used if f in DISTRACT) or "none"))

    rl.bank(bank, zn, bt)
    rl.finish(dict(stage1=r1, stage2=r2, stage3=r3, stage4=r4,
                   masked_no_mem=r_nm, unmasked=r_un,
                   unmasked_no_mem=r_un_nm, distractor_mem=ctrl,
                   mem=bank.get("mem"), seed=seed), bt=bt)
    with open(os.path.join(ROOT, "data", "e22_summary.json"), "w") as fh:
        json.dump(dict(stage1=r1, stage2=r2, stage3=r3, stage4=r4,
                       masked_no_mem=r_nm, unmasked=r_un,
                       unmasked_no_mem=r_un_nm, distractor_mem=ctrl,
                       bt=bt, seconds=time.time() - t0), fh, indent=1,
                  default=float)
    print("\ntotal %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
