"""e21 -- how much does masking actually cost? The control before the BT test.

The plan is to test whether a behaviour tree's memory primitives earn their
place, and the variant meant to force that is partial observability. Before
building any of it, the premise has to be checked: if hiding food and threat
costs a memoryless controller almost nothing, then the variant is not testing
memory and every result that follows would be about something else.

THE COMPARISON IS FAIR BY CONSTRUCTION, which is the only reason it means
anything. Vision enters `NestWorld.observe` and nothing else, so masked and
unmasked runs are the SAME dynamics, the same reward, the same episodes and the
same policy class -- 19 observation columns either way, with `food_seen` and
`threat_seen` simply pinned to 1 when nothing is hidden. Verified directly: one
step from identical states gives bit-identical successor states under both, and
the CEM oracle, which plans from the true state, scores 31.80 in both.

    masked      food beyond 0.35 (0.18 at night) and threat beyond 0.40 are
                absent from the observation: distance pinned at the horizon,
                bearing zeroed, and a flag saying so
    unmasked    the same world with the radii set to zero

WHAT THE GAP MEANS, and what it does not. The oracle is unaffected by masking,
so oracle-minus-controller is not the quantity of interest; the quantity is
UNMASKED-minus-MASKED, both memoryless, which is the information the observation
function destroys. A large gap says a memoryless policy cannot recover what it
lost, and leaves room for memory to pay. A small gap says the hidden information
was not worth much and the variant is a bad test bed.

TWO SETTINGS MAKE THE HIDING BITE. Food returns to the SAME site after a
delivery, so losing sight of it destroys something the agent HAD -- with
teleporting food the correct response to blindness is to search again, which is
memoryless and would understate the gap. And the site is resampled every
episode, so `pos_x`/`pos_y` cannot encode it and stay distractors.

A SIDE EFFECT WORTH RECORDING: with a persistent food site the simulator has no
random draw left anywhere, so it is fully deterministic (verified: two runs from
the same starting states differ by 0.0). Every paired acceptance test in the
pipeline now has a zero noise floor, where on ForageWorld the respawn draw left
0.57 return units of irreducible disagreement on 300 episodes.

THE EXPERT IS PRIVILEGED IN BOTH ARMS, and that is a real caveat rather than a
detail. `CEMPolicy` and the DAgger labels are computed from the true state,
including the food the masked controller cannot see, so stage 1 under masking is
imitation of an expert with information the student can never have. It is
unbiased as a COMPARISON -- both arms imitate the same privileged expert -- but
the masked arm's stage-1 number should not be read as the best a memoryless
controller could do; stage 2, which learns from the controller's own returns,
is the one to trust.
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btind.envs.nest import NestWorld, OBS_NAMES
from btind.pipeline import fit_bt
from btind.policies import CEMPolicy, evaluate

WORLD = dict(day_len=40.0, food_persistent=True)
MASK = dict(vision_r=0.35, night_vision_r=0.18, threat_vision_r=0.40)
DOMAIN = dict(
    cand_names=("d_threat", "energy", "d_food", "carrying", "d_nest",
                "is_night", "t_to_switch", "food_seen", "threat_seen",
                "pos_x", "pos_y", "t_norm", "noise"),
    radial=(0, 3, 4, 12), bearing=((1, 2), (5, 6), (13, 14)))
DISTRACTORS = ("noise", "pos_x", "pos_y", "t_norm")


def main():
    t0 = time.time()
    reps = int(os.environ.get("E21_REPEATS", "2"))
    arms = (("unmasked (MDP)", dict(WORLD)),
            ("masked (POMDP)", dict(WORLD, **MASK)))

    NestWorld.seed_kernels(0)
    oracle = {}
    for name, kw in arms:
        env = NestWorld(**kw)
        env.seed_kernels(11)
        r = evaluate(env, CEMPolicy(env, np.random.default_rng(3)), n_ep=200,
                     T=200, seed=11, from_state=True)
        oracle[name] = float(r["G"].mean())
    print("CEM oracle (plans from true state, unaffected by masking): "
          "%.2f / %.2f" % tuple(oracle.values()))

    runs = []
    for k in range(reps):
        row = {}
        for name, kw in arms:
            env = NestWorld(**kw)
            print("\n### run %d -- %s" % (k, name), flush=True)
            r = fit_bt(env, OBS_NAMES, seed=k, **DOMAIN)
            feats = r["features"]
            row[name] = dict(stage1=r["imitation"]["G"], stage1_ci=r["imitation"]["ci"],
                             stage2=r["final"]["G"], stage2_ci=r["final"]["ci"],
                             meals=r["final"]["eaten"], feats=feats,
                             distractors=[f for f in feats if f in DISTRACTORS],
                             n_clauses=len(r["bank"]["clauses"]), bt=r["bt"])
            print(r["bt"])
        runs.append(row)

    print("\n" + "=" * 76)
    print("%-18s %9s %9s %9s %9s" % ("", "stage 1", "stage 2", "meals",
                                     "clauses"))
    agg = {}
    for name, _ in arms:
        s1 = np.array([r[name]["stage1"] for r in runs])
        s2 = np.array([r[name]["stage2"] for r in runs])
        agg[name] = (s1.mean(), s2.mean())
        print("%-18s %9.2f %9.2f %9.2f %9.1f"
              % (name, s1.mean(), s2.mean(),
                 np.mean([r[name]["meals"] for r in runs]),
                 np.mean([r[name]["n_clauses"] for r in runs])))
    gap1 = agg["unmasked (MDP)"][0] - agg["masked (POMDP)"][0]
    gap2 = agg["unmasked (MDP)"][1] - agg["masked (POMDP)"][1]
    print("-" * 76)
    print("information destroyed by masking: %.2f (stage 1), %.2f (stage 2)"
          % (gap1, gap2))
    print("oracle, unaffected by masking:    %.2f" % oracle["masked (POMDP)"])
    print("=" * 76)
    ds = sum(len(r[n]["distractors"]) for r in runs for n, _ in arms)
    print("distractor literals bought: %s" % (ds if ds else "none"))

    with open(os.path.join(ROOT, "data", "e21_summary.json"), "w") as fh:
        json.dump(dict(runs=runs, oracle=oracle, gap_stage1=gap1,
                       gap_stage2=gap2, reps=reps,
                       seconds=time.time() - t0), fh, indent=1, default=float)
    figure(runs, oracle, arms)
    print("\ntotal %.0fs" % (time.time() - t0))


def figure(runs, oracle, arms):
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
    fig.suptitle("NestWorld: what the observation function destroys\n"
                 "identical dynamics, identical policy class -- only the "
                 "observation differs", fontsize=12.5, fontweight="bold")

    a = ax[0]
    names = [n for n, _ in arms]
    x = np.arange(2)
    s1 = [np.mean([r[n]["stage1"] for r in runs]) for n in names]
    s2 = [np.mean([r[n]["stage2"] for r in runs]) for n in names]
    e1 = [np.ptp([r[n]["stage1"] for r in runs]) / 2 for n in names]
    e2 = [np.ptp([r[n]["stage2"] for r in runs]) / 2 for n in names]
    a.bar(x - 0.2, s1, 0.38, yerr=e1, color="#8c8c8c",
          label="stage 1 (imitation)")
    a.bar(x + 0.2, s2, 0.38, yerr=e2, color="#55a868",
          label="stage 2 (improvement)")
    a.axhline(list(oracle.values())[0], color="#4c72b0", ls=":", lw=2,
              label="CEM oracle (sees the true state)")
    a.set_xticks(x)
    a.set_xticklabels(names)
    a.set_ylabel("held-out return")
    a.set_title("memoryless bank, both worlds", fontsize=11)
    a.legend(fontsize=9)
    a.grid(alpha=0.3, axis="y")

    a = ax[1]
    for k, r in enumerate(runs):
        a.plot([0, 1], [r[names[0]]["stage2"], r[names[1]]["stage2"]],
               marker="o", lw=2.2, label="run %d" % k)
    a.set_xticks([0, 1])
    a.set_xticklabels(["unmasked", "masked"])
    a.set_ylabel("stage 2 held-out return")
    a.set_title("paired per run: the cost of hiding", fontsize=11)
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    out = os.path.join(ROOT, "figs", "e21_observability.png")
    fig.savefig(out, dpi=140)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
