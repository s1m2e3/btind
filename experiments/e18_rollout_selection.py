r"""e18 -- select arms by measured return. The proxy comes out of the loop entirely.

FIVE PROXIES HAVE NOW FAILED ON THIS TASK, each for the same reason.

  sup-LM instability     e06/e07  the most detectable split (`energy`) was a
                                  planner artifact; banning it improved return
  M-weighted value loss  e14      rgfi won by 35 points out of sample and lost
                                  by 8 return units, eating 1.95 meals vs 3.20
  gap-weighted loss      e14      swapping the weighting recovered 0.4 meals
                                  and left the gap intact
  value-regime loss      e13      the best regionaliser found (47.9% of one law
                                  out of sample) produced policies that
                                  flatlined at -1 to -3 across six rounds
  one-step TD advantage  e17-val  ranks policies BACKWARDS: random -0.87,
                                  global -1.05, bank -1.25, good bank -1.58,
                                  whose return is +10.25

Every one is a LOCAL quantity -- curvature in the action, leverage at a state, a
regression residual, a one-step residual -- and return here is accumulation over
~35 steps. They price not-making-a-mistake-now and never price reward acquired
later, so the harder a method optimises them the more cautious and less hungry
it becomes. That is not a defect in any one of them; it is what local means.

Return itself costs 0.25s for 300 episodes. So stop approximating it: build the
bank a candidate arm would produce, RUN it, and keep the arm only if the
measured return goes up.

THREE THINGS THAT MAKE THIS WORK, two of which were forced by a failure.

  PROPOSE CHEAPLY, SELECT EXPENSIVELY. Rollouts give no per-clause credit, so
  there is nothing to hill-climb per literal. The loss-based machinery is
  demoted to a proposal distribution, where being wrong is free, and every
  accepted decision is made by measured return.

  COMMON RANDOM NUMBERS. All candidates in a fit share the episode seed, so they
  are compared on identical starting states.

  A PAIRED TEST. The first version took the argmax of ~80 noisy estimates and
  immediately selected `pos_y>0.600` -- a declared distractor -- on a measured
  +0.43 that was worth -1.46 held out. Because candidates share starting states,
  per-episode differences are paired and their variance is far below either
  arm's; an arm is accepted only when that paired difference clears 2 standard
  errors. With the test in place the first arm chosen is `d_threat<=0.137`, and
  the round-0 controller scores -0.04 against the loss-selected bank's -0.86.

  SELECTION SEED != EVALUATION SEED. Selection runs 300 episodes on seed 777;
  every number reported below is 3000 episodes on seed 11. Otherwise this would
  repeat the overfitting it was built to avoid.

THE CONTROL LAW IS UNCHANGED. `fit_value_law` stays, because it is already a
Q-maximisation: `u_medoid` IS argmax_a Q(s,a) from the planner's 256-sequence
search, and `M` IS the negative Hessian of return in the action, so the fit is
the best affine approximation to that argmax under a second-order model of the
value lost by deviating. e14 showed that expansion is a bad thing to SELECT on,
not a bad thing to FIT with. Only selection changes here.

THE DISTRACTOR CONTROL IS THE ONE THAT MATTERS. `noise`, `pos_x`, `pos_y` are
causally irrelevant by construction and stay in the alphabet. A criterion that
is really measuring the task should never pay for them.

RESULT. The first thing since e12 that beats the incumbent, and it does it by
building far less.

    round                 0      1      2      3      4      5      6
    loss-selected     -0.82  -0.99   6.06   8.86   2.44   7.20   6.32
    rollout-selected  -1.17   1.69   3.28   7.85   7.94  10.31  10.38
    CEM oracle        12.79

    meals/episode      r0    r6      literals r6    distractors
    loss-selected     3.26  5.94          8         EVERY round
    rollout-selected  3.23  8.51          2         none after r0

Four things, in order of how much they survive the single-seed caveat.

  THE DISTRACTOR CONTROL FINALLY PASSES. Loss selection reached for `pos_x`,
  `pos_y`, `noise` or `t_norm` in all seven rounds. Rollout selection used
  `pos_y` once at round 0 and never again: from round 1 on, every literal it
  buys is `d_threat`. This is structural, not noise -- a criterion measuring the
  task cannot pay for a variable that does not affect the task.

  IT BUILDS TWO LITERALS. The emitted controller is

      Fallback
      |-- Sequence[ d_threat <= 0.100 , Action ]
      |-- Sequence[ d_threat >  0.219 , Action ]
      \-- Action(default)          # the band 0.100 < d_threat <= 0.219

  three regions on one variable, against loss selection's six clauses over four
  variables. And the band it carves brackets 0.15 -- the flee radius at which a
  two-line handwritten rule scores 10.93, measured separately. The structure the
  task actually has is what it found.

  IT EATS. 8.51 meals per episode against 5.94, and more than the handwritten
  rule's 7.72. Meals are the quantity every local proxy priced at zero, and the
  only criterion that prices them is the one that measures them.

  THE CURVE IS MONOTONE. Return and meals rise every single round, where loss
  selection oscillates (8.86 -> 2.44 -> 7.20 -> 6.32). Selection under a
  criterion that is not the task has no reason to improve monotonically in the
  task, and did not.

WHAT DOES NOT SURVIVE THE CAVEAT: the size of the return gap. This is one seed,
and the loss-selected arm has ranged 6.32 to 10.89 across nominally identical
runs (e15 8.47, e16 9.57, e17 10.89, here 6.32), so 10.38 vs 6.32 is flattered
by a bad draw for the incumbent. The parsimony, the distractor cleanliness and
the monotonicity are not explainable that way; the 4-point margin is. Multiple
pipeline seeds are the obvious next step and are not run here.
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
from btind.envs.forage import ForageWorld, OBS_NAMES
from btind.collect import collect, design_matrix
from btind.joint import artifact_ratio, veto_vars
from btind.evotm import evolve_bank, BankPolicy, clause_text
from btind.rollout_select import rsfi, _Bank
from btind.valuesplit import action_curvature
from btind.dagger import collect_onpolicy_states
from btind.search import search
from btind.policies import CEMPolicy, evaluate

ALL = ["d_threat", "energy", "d_food", "pos_x", "pos_y", "t_norm", "noise"]
SV = [OBS_NAMES.index(n) for n in ALL]
DISTRACTORS = ("noise", "pos_x", "pos_y")
N_DAG, N_NEW, N_EP, T, SEED_EVAL = 6, 2000, 3000, 200, 11
MIN_N, GENS, POP = 400, 60, 48
SEL_EP, SEL_T, SEL_SEED, ZTHR = 300, 200, 777, 2.0


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std() / np.sqrt(max(len(x), 1))


def candidates(o, u, v, g, c):
    ratio, r = artifact_ratio(o, u, v, g, c, SV)
    keep, _ = veto_vars(ratio, SV, 1.5)
    return [j for j in keep if r["z_u"][SV.index(j)] > 1.0]


def feats_of(clauses):
    return sorted({OBS_NAMES[l[0]] for cl in clauses for l in cl})


def run_arm(env, pack, mode, rng, log):
    obs, U, V, gap, coh, M = [x.copy() for x in pack]
    out = []
    for it in range(N_DAG + 1):
        sv = candidates(obs, U, V, gap, coh)
        loss_bank = evolve_bank(obs, U, M, sv, OBS_NAMES, C=POP,
                                generations=GENS, max_arity=3, min_n=MIN_N,
                                max_keep=16, seed=0, verbose=False)
        if mode == "loss":
            bank, pol = loss_bank, BankPolicy(loss_bank)
        else:
            m = rsfi(env, obs, U, M, sv, OBS_NAMES, n_arms=6, pool=40,
                     refine=12, max_arity=3, min_n=MIN_N, n_ep=SEL_EP,
                     T=SEL_T, sel_seed=SEL_SEED, seed=0, z=ZTHR,
                     seed_clauses=loss_bank["clauses"], verbose=False)
            bank = m
            pol = _Bank(m["clauses"], m["laws"], m["default"], design_matrix)

        r = evaluate(env, pol, n_ep=N_EP, T=T, seed=SEED_EVAL)
        ft = feats_of(bank["clauses"])
        rec = dict(round=it, G=float(r["G"].mean()), ci=float(ci(r["G"])),
                   eaten=float(r["eaten"].mean()),
                   n_clauses=len(bank["clauses"]),
                   n_literals=sum(len(c) for c in bank["clauses"]),
                   feats=ft, distractors=[f for f in ft if f in DISTRACTORS])
        out.append(rec)
        log(rec)
        if it == N_DAG:
            return out, bank

        S = collect_onpolicy_states(env, pol, N_NEW, rng)
        rs = search(env, S, rng, label="medoid")
        obs = np.vstack([obs, env.observe(S)])
        U = np.vstack([U, rs["u_star"]])
        V = np.concatenate([V, rs["V"]])
        gap = np.concatenate([gap, rs["gap"]])
        coh = np.concatenate([coh, rs["coherence"]])
        M = np.concatenate([M, action_curvature(rs["a0_all"], rs["G_all"])])


def main():
    t0 = time.time()
    ForageWorld.seed_kernels(0)
    env = ForageWorld()
    D = collect(n_states=5000, seed=0, label="medoid")
    pack = (D["obs"], D["u_medoid"], D["V"], D["gap"], D["coherence"], D["M"])

    R, BANKS = {}, {}
    for name, mode in (("loss-selected", "loss"), ("rollout-selected", "roll")):
        t = time.time()
        print("  %s" % name)
        rows, bank = run_arm(
            env, pack, mode, np.random.default_rng(4),
            lambda r: print("    r%d  G %6.2f  eaten %.2f  %2d clauses, %2d "
                            "literals  %s%s"
                            % (r["round"], r["G"], r["eaten"], r["n_clauses"],
                               r["n_literals"], ",".join(r["feats"]),
                               "   DISTRACTORS: " + ",".join(r["distractors"])
                               if r["distractors"] else "")))
        R[name], BANKS[name] = rows, bank
        print("    [%.0fs]" % (time.time() - t))

    cem = evaluate(env, CEMPolicy(env, np.random.default_rng(3)), n_ep=200,
                   T=T, seed=SEED_EVAL, from_state=True)["G"].mean()
    print("\n" + "=" * 72)
    for n, rows in R.items():
        print("%-18s %s" % (n, " ".join("%6.2f" % r["G"] for r in rows)))
    print("%-18s %6.2f" % ("CEM oracle", cem))
    print("=" * 72)
    for n in R:
        print("\n%s\n%s" % (n, "\n".join("  " + clause_text(cl, OBS_NAMES)
                                         for cl in BANKS[n]["clauses"])))

    with open(os.path.join(ROOT, "data", "e18_summary.json"), "w") as fh:
        json.dump(dict(rows=R, cem=float(cem)), fh, indent=1)
    figure(R, cem)
    print("\ntotal %.0fs" % (time.time() - t0))


def figure(R, cem):
    fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.8))
    fig.suptitle("ForageWorld: arms selected by MEASURED RETURN, not by a "
                 "regression loss\n"
                 "five local proxies have failed on this task; this one takes "
                 "the proxy out of the loop entirely",
                 fontsize=12.5, fontweight="bold")
    C = {"loss-selected": "#dd8452", "rollout-selected": "#8c4bbf"}
    a = ax[0]
    for n, rows in R.items():
        a.errorbar([r["round"] for r in rows], [r["G"] for r in rows],
                   yerr=[r["ci"] for r in rows], marker="o", lw=2.4, capsize=4,
                   color=C[n], label="%s (%.2f)" % (n, rows[-1]["G"]))
    a.axhline(cem, color="#4c72b0", ls=":", lw=2, label="CEM oracle")
    a.set_xlabel("DAgger round")
    a.set_ylabel("mean discounted return")
    a.set_title("compared at equal rounds", fontsize=11)
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    a = ax[1]
    for n, rows in R.items():
        a.plot([r["round"] for r in rows], [r["eaten"] for r in rows],
               marker="o", lw=2.4, color=C[n], label=n)
    a.set_xlabel("DAgger round")
    a.set_ylabel("meals eaten per episode")
    a.set_title("the quantity every local proxy priced at zero", fontsize=11)
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    a = ax[2]
    for n, rows in R.items():
        a.plot([r["round"] for r in rows], [r["n_literals"] for r in rows],
               marker="o", lw=2.4, color=C[n], label="%s literals" % n)
    a.set_xlabel("DAgger round")
    a.set_ylabel("literals in the emitted bank")
    a.set_title("artifact size", fontsize=11)
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.87])
    out = os.path.join(ROOT, "figs", "e18_rollout_selection.png")
    fig.savefig(out, dpi=140)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
