"""e19 -- close the loop with a critic of the CONTROLLER, not of the planner.

WHERE e18 LEFT IT. Rollout selection fixed the criterion by deleting it: arms
are proposed blind and accepted only if 300 measured episodes say return went
up. That works, and it is also the reason the loop cannot do anything finer
than accept or reject a whole controller. A trajectory that crosses two regions
returns ONE number, so there is no per-arm signal anywhere in the pipeline:

    laws    fitted to (u*, M) from the PLANNER -- Q*, off-policy, at teleported
            states, every row weighted equally
    guards  proposed by threshold drift, priced by a rollout of the whole bank

This experiment supplies the missing decomposition. The region-restricted
deterministic policy gradient splits one trajectory's return across the arms it
visited,

    grad_{theta_c} J = E_{s ~ d_gamma^pi} [ 1{s in R_c} x(s) dQ^pi/du ]

and the guard derivative is a boundary term -- occupancy on the cut times the
Q-gap between the arm that owns the shell and the arm that would inherit it.
Both need one object the project did not have: Q^pi, the value of THIS bank.

FOUR MEASUREMENTS, in the order they have to hold.

  1. IS THE REFERENCE REAL? Finite differences are NOT usable as ground truth
     here: two independent 4-point FD estimates of the same gradient agree at
     cos 0.03 (eps 0.05), because a +-eps nudge either changes nothing or flips
     a catch/meal worth +-5 to +-20. A quadratic fitted over K=96 probes and 16
     continuations agrees with itself at cos 0.79 / median 0.99. That is the
     reference, and its self-agreement is the ceiling everything else is scored
     against.

  2. IS THE CRITIC BETTER THAN THE PLANNER AT THE THING WE CONSUME? Both give a
     gradient field; only one is on-policy. This is the question the experiment
     answers NEGATIVELY, and it is repeated over independent validation samples
     because a single sample flipped the ranking between two runs.

  3. DOES THE DECOMPOSITION MOVE THE CONTROLLER? One law step, ablated: planner
     labels as a regression target, planner labels as a gradient source, critic
     as a gradient source, with and without occupancy weights. The gradient
     SOURCE and the step SHAPE are varied separately, because varying both at
     once is what made every earlier comparison in this project unreadable.

  4. DOES IT SURVIVE THE ACCEPTANCE TEST? Everything is a PROPOSAL. The paired
     z-test from e18 is unchanged except in episode count -- 1000 rather than
     300, because the respawn draw lives in a per-thread numba stream that no
     seeding can pin, and 300 episodes leave a 0.57-unit noise floor against
     effects of about 1 unit.

RESULT, four independent runs of the whole pipeline (the incumbent is rebuilt
each time; nothing here is one seed):

    run              0      1      2      3     mean
    e18 incumbent  9.34   8.04  10.97   7.85    9.05
    + wiring      12.85   9.29  12.23  12.26   11.66
    delta         +3.51  +1.25  +1.26  +4.41   +2.61     meals +1.64

Positive in 4/4, against a per-run held-out CI of +-0.21 and a spread across
runs of 1.58 half-range on the delta. This is the first per-arm credit signal
anywhere in the pipeline and it is worth about a fifth of the remaining gap to
the CEM oracle.

THREE THINGS THE ABLATION SAYS THAT THE HEADLINE DOES NOT, all averaged over the
same four runs, partition held fixed, one law step only:

  * THE STEP SHAPE IS THE WHOLE STORY. A small line-searched DPG move is worth
    +1.8; the closed-form refit to the critic argmax -- the shape this project
    has always used -- is worth -14.6, and the same refit from planner labels
    -3.0. Direction from a model, distance from measured return; never a solve.

  * THE CRITIC IS NOT WHAT MAKES IT WORK. Q^pi and the planner Q* agree with the
    reference gradient field equally well in DIRECTION (cos 0.21 and 0.24
    against a ceiling of 0.84), and as gradient sources they are worth +1.8 and
    +1.5. The critic is better only in SCALE (median relative error 1.6 against
    2.1). A critic that approached the ceiling would be worth more than this one
    is; today it is an interchangeable supplier of a direction.

  * THE OCCUPANCY WEIGHT IS INERT HERE. gamma^t weighting was the factor the
    derivation said was missing, and dropping it changes the law step by 0.08
    (+1.86 uniform against +1.78 weighted) -- inside the run-to-run spread. The
    weight is still correct; on this task, at these step sizes, it is not what
    was costing return.

WHAT IS NOT CLAIMED. That any of this beats replanning: the CEM oracle scores
about 13. And the guard half is thinly evidenced -- the boundary term is
consistent in sign across runs (it wants the inner threshold raised, dJ/dtau
+2.4 to +4.6) but only three of twelve polish iterations produced a threshold
move that survived the paired test.
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
from btind.evotm import evolve_bank, clause_text
from btind.rollout_select import rsfi, _Bank
from btind.valuesplit import action_curvature
from btind.dagger import collect_onpolicy_states
from btind.search import search
from btind.policies import evaluate
from btind.vhat import fit_vhat
from btind.critic import fit_qhat, local_q_grad, grad_score
from btind.qwire import (onpolicy_states, refit_laws, blend, dpg_step, polish,
                         TabulatedGrad,
                         boundary_grad, law_gradient_norm, paired_accept,
                         assign, _score)

ALL = ["d_threat", "energy", "d_food", "pos_x", "pos_y", "t_norm", "noise"]
SV = [OBS_NAMES.index(n) for n in ALL]
N_DAG, N_NEW = 4, 2000
N_EP, T, EVAL_SEEDS = 3000, 200, (11, 12, 13)
SEL_EP, SEL_SEED, ZTHR = 1000, 777, 2.0
N_ON = 20000


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std() / np.sqrt(max(len(x), 1))


def pol_of(b):
    return _Bank(b["clauses"], b["laws"], b["default"], design_matrix)


def heldout(env, b, seeds=EVAL_SEEDS, n_ep=N_EP):
    g, e = [], []
    for s in seeds:
        env.seed_kernels(s)
        r = evaluate(env, pol_of(b), n_ep=n_ep, T=T, seed=s)
        g.append(r["G"]), e.append(r["eaten"])
    g, e = np.concatenate(g), np.concatenate(e)
    return float(g.mean()), float(ci(g)), float(e.mean())


# ------------------------------------------------------- the e18 incumbent
def build_incumbent(env, verbose=True):
    """e18's winning arm, rebuilt here so the comparison is like for like."""
    D = collect(n_states=5000, seed=0, label="medoid")
    obs, U, V = D["obs"], D["u_medoid"], D["V"]
    gap, coh, M = D["gap"], D["coherence"], D["M"]
    rng = np.random.default_rng(4)
    rows = []
    for it in range(N_DAG + 1):
        ratio, r = artifact_ratio(obs, U, V, gap, coh, SV)
        keep, _ = veto_vars(ratio, SV, 1.5)
        sv = [j for j in keep if r["z_u"][SV.index(j)] > 1.0]
        lb = evolve_bank(obs, U, M, sv, OBS_NAMES, C=48, generations=60,
                         max_arity=3, min_n=400, max_keep=16, seed=0,
                         verbose=False)
        m = rsfi(env, obs, U, M, sv, OBS_NAMES, n_arms=6, pool=40, refine=12,
                 max_arity=3, min_n=400, n_ep=300, T=T, sel_seed=SEL_SEED,
                 seed=0, z=ZTHR, seed_clauses=lb["clauses"], verbose=False)
        bank = dict(clauses=m["clauses"], laws=m["laws"], default=m["default"],
                    names=OBS_NAMES)
        g, c, e = heldout(env, bank, seeds=(11,))
        rows.append(dict(round=it, G=g, ci=c, eaten=e,
                         n_clauses=len(m["clauses"])))
        if verbose:
            print("  r%d  G %6.2f  eaten %.2f  %d clauses"
                  % (it, g, e, len(m["clauses"])), flush=True)
        if it == N_DAG:
            break
        S = collect_onpolicy_states(env, pol_of(bank), N_NEW, rng)
        rs = search(env, S, rng, label="medoid")
        obs = np.vstack([obs, env.observe(S)])
        U = np.vstack([U, rs["u_star"]])
        V = np.concatenate([V, rs["V"]])
        gap = np.concatenate([gap, rs["gap"]])
        coh = np.concatenate([coh, rs["coherence"]])
        M = np.concatenate([M, action_curvature(rs["a0_all"], rs["G_all"])])
    return bank, rows


# ------------------------------------------------------------ measurement 1+2
def validate_gradient(env, bank, vh, qh, rng, n=1000, reps=3):
    """Score the critic and the planner against a re-probed reference field.

    REPEATED over independent validation samples. One sample is not enough: the
    first run of this experiment put the critic ahead of the planner (cos 0.28
    against 0.23) and the second tied them (0.23 against 0.23), which is the
    difference between a headline and an artefact.
    """
    pol = pol_of(bank)
    reps_out = []
    for k in range(reps):
        S, _, _ = onpolicy_states(env, pol, n, np.random.default_rng(99 + k),
                                  n_ep=300, T=T)
        o, u = env.observe(S), pol.act(env.observe(S))
        g1 = local_q_grad(env, pol, S, u, vh, np.random.default_rng(1 + 10 * k),
                          K=96, n_rep=16, h=5, seed=11 + k)
        g2 = local_q_grad(env, pol, S, u, vh, np.random.default_rng(2 + 10 * k),
                          K=96, n_rep=16, h=5, seed=77 + k)
        ref = 0.5 * (g1 + g2)
        rs = search(env, S, rng, label="medoid")
        Mc = action_curvature(rs["a0_all"], rs["G_all"])
        reps_out.append(dict(
            ceiling=grad_score(g1, g2),
            qhat=grad_score(qh.grad_u(o, u), ref),
            planner=grad_score(
                np.einsum("iqm,im->iq", Mc, rs["u_medoid"] - u), ref)))
    agg = {}
    for src in ("ceiling", "planner", "qhat"):
        agg[src] = {m: float(np.mean([r[src][m] for r in reps_out]))
                    for m in ("cos", "cos_med", "agree", "rel")}
        agg[src]["spread"] = float(
            np.ptp([r[src]["cos"] for r in reps_out]) / 2)
    agg["reps"] = reps_out
    return agg


# -------------------------------------------------------------- measurement 3
def ablate_laws(env, bank, obs, w, qh, Uc, Mc, cur):
    """One law step, six ways. Same partition throughout -- only laws move."""
    Uq, Mq = qh.targets(obs)
    full = refit_laws(bank, obs, Uq, Mq, w=w, min_n=150)
    arms = [("incumbent", bank),
            ("planner relabel, uniform w",
             refit_laws(bank, obs, Uc, Mc, w=None, min_n=150)),
            ("planner relabel, gamma^t w",
             refit_laws(bank, obs, Uc, Mc, w=w, min_n=150)),
            ("critic newton, alpha 1.0", full),
            ("critic newton, alpha 0.25", blend(bank, full, 0.25))]
    tab = TabulatedGrad(Mc, Uc)
    arms += [("planner DPG, eta %.2f" % e,
              dpg_step(bank, obs, w, qh, e, grad_fn=tab))
             for e in (0.02, 0.05, 0.1)]
    arms += [("critic DPG, eta %.2f" % e, dpg_step(bank, obs, w, qh, e))
             for e in (0.02, 0.05, 0.1, 0.2)]
    ones = np.ones(len(obs))
    arms += [("critic DPG, uniform w, eta %.2f" % e,
              dpg_step(bank, obs, ones, qh, e)) for e in (0.02, 0.05)]
    out = []
    for name, b in arms:
        ok, d, _ = paired_accept(env, b, cur, SEL_EP, T, SEL_SEED, ZTHR)
        g, c, e = heldout(env, b)
        out.append(dict(name=name, sel_delta=d, accepted=bool(ok), G=g, ci=c,
                        eaten=e))
        print("  %-28s sel %+6.2f %s  held-out %6.2f +-%.2f  eaten %.2f"
              % (name, d, "ACC" if ok else "   ", g, c, e), flush=True)
    return out


def run_once(env, tag, verbose=True):
    """The whole pipeline once: build the e18 incumbent, then wire and polish."""
    t0 = time.time()
    print("\n### run %d" % tag, flush=True)
    print("[1] rebuilding the e18 incumbent (DAgger x %d + rollout selection)"
          % N_DAG)
    bank, rounds = build_incumbent(env)
    g0, c0, e0 = heldout(env, bank)
    print("    incumbent over %d eval seeds: G %.2f +-%.2f  eaten %.2f  [%.0fs]"
          % (len(EVAL_SEEDS), g0, c0, e0, time.time() - t0))

    print("[2] fitting V-hat and the action-quadratic critic Q-hat")
    pol = pol_of(bank)
    rng = np.random.default_rng(5 + tag)
    vh = fit_vhat(env, pol, n_ep=800, T=T, n_step=20, sweeps=2, seed=21)
    S, tt, w = onpolicy_states(env, pol, N_ON, rng)
    obs = env.observe(S)
    qh, info = fit_qhat(env, pol, S, vh, rng, seed=3)
    print("    %d probe rows, %d parameters, within-state R2 %.3f;  "
          "occupancy: median t %d, mean gamma^t %.3f"
          % (info["n_rows"], info["n_par"], info["r2"], np.median(tt), w.mean()))

    print("[3] gradient validation against a re-probed reference field")
    val = validate_gradient(env, bank, vh, qh, np.random.default_rng(7), reps=2)
    for k in ("ceiling", "planner", "qhat"):
        v = val[k]
        print("    %-9s cos %5.2f +-%.2f   median %5.2f   sign agreement "
              "%4.1f%%   median rel err %.2f"
              % (k, v["cos"], v["spread"], v["cos_med"], 100 * v["agree"],
                 v["rel"]))

    print("[4] one law step, ablated (partition held fixed)")
    rs = search(env, S, np.random.default_rng(17), label="medoid")
    Uc, Mcem = rs["u_medoid"], action_curvature(rs["a0_all"], rs["G_all"])
    cur = _score(env, bank, SEL_EP, T, SEL_SEED)
    abl = ablate_laws(env, bank, obs, w, qh, Uc, Mcem, cur)

    print("[5] boundary term on the incumbent guards")
    bg = boundary_grad(bank, obs, w, qh)
    for d in bg:
        print("    %-10s %s %.3f   dJ/dtau %+8.4f   shell n=%d"
              % (OBS_NAMES[d["j"]], "<=" if d["neg"] else "> ", d["thr"],
                 d["grad"], d["n"]))

    print("[6] two-timescale polish: laws, then one guard move, x3")
    lo_hi = {j: (float(obs[:, j].min()), float(obs[:, j].max()))
             for j in range(obs.shape[1])}
    out, plog = polish(
        env, bank,
        vhat_fn=lambda p_: fit_vhat(env, p_, n_ep=800, T=T, n_step=20,
                                    sweeps=2, seed=21),
        qhat_fn=lambda p_, S_, v_: fit_qhat(env, p_, S_, v_,
                                            np.random.default_rng(3),
                                            seed=3)[0],
        rng=np.random.default_rng(5 + tag), n_iter=3, n_states=N_ON,
        n_ep_sel=SEL_EP, T=T, sel_seed=SEL_SEED, z=ZTHR, mode="grad",
        lo_hi=lo_hi)
    g1, c1, e1 = heldout(env, out)
    print("    incumbent %.2f +-%.2f -> wired %.2f +-%.2f   meals %.2f -> %.2f"
          " [%.0fs]" % (g0, c0, g1, c1, e0, e1, time.time() - t0))
    return dict(
        incumbent=dict(G=g0, ci=c0, eaten=e0, rounds=rounds,
                       clauses=[clause_text(c, OBS_NAMES)
                                for c in bank["clauses"]]),
        wired=dict(G=g1, ci=c1, eaten=e1,
                   clauses=[clause_text(c, OBS_NAMES)
                            for c in out["clauses"]]),
        critic=info, gradient=val, ablation=abl, boundary=bg,
        polish=[dict(it=r["it"], G_sel=r["G_sel"],
                     law_accepted=r["law"]["accepted"],
                     law_best=r["law"]["best"], law_tries=r["law"]["tries"],
                     guard=r["guard"]) for r in plog],
        seconds=time.time() - t0)


def main():
    t0 = time.time()
    reps = int(os.environ.get("E19_REPEATS", "3"))
    ForageWorld.seed_kernels(0)
    env = ForageWorld()
    runs = [run_once(env, k) for k in range(reps)]

    d = np.array([r["wired"]["G"] - r["incumbent"]["G"] for r in runs])
    dm = np.array([r["wired"]["eaten"] - r["incumbent"]["eaten"] for r in runs])
    print("\n" + "=" * 74)
    print("%-8s %10s %10s %10s %10s" % ("run", "incumbent", "wired", "delta",
                                        "d meals"))
    for k, r in enumerate(runs):
        print("%-8d %10.2f %10.2f %10.2f %10.2f"
              % (k, r["incumbent"]["G"], r["wired"]["G"], d[k], dm[k]))
    print("%-8s %10.2f %10.2f %10.2f %10.2f"
          % ("mean", np.mean([r["incumbent"]["G"] for r in runs]),
             np.mean([r["wired"]["G"] for r in runs]), d.mean(), dm.mean()))
    print("=" * 74)
    print("per-run held-out CI is about +-0.21; the spread ACROSS runs is the "
          "one that matters,\nand it is %.2f half-range on the delta."
          % (np.ptp(d) / 2 if len(d) > 1 else 0.0))

    names = [a["name"] for a in runs[0]["ablation"]]
    print("\nlaw step, averaged over %d runs (held-out return):" % reps)
    for i, n in enumerate(names):
        v = [r["ablation"][i]["G"] for r in runs]
        base = [r["incumbent"]["G"] for r in runs]
        print("  %-32s %6.2f   (%+.2f vs its own incumbent)"
              % (n, np.mean(v), np.mean(np.array(v) - np.array(base))))

    print("\nemitted controller, last run:")
    for cl in runs[-1]["wired"]["clauses"]:
        print("  " + cl)
    print("  Action(default)")

    summary = dict(runs=runs, delta=list(map(float, d)),
                   delta_meals=list(map(float, dm)), reps=reps,
                   seconds=time.time() - t0)
    with open(os.path.join(ROOT, "data", "e19_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    figure(runs)
    print("\ntotal %.0fs" % (time.time() - t0))


def figure(runs):
    fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.8))
    fig.suptitle("ForageWorld: per-arm credit from a critic of the CONTROLLER\n"
                 "the region-restricted policy gradient for the laws, the "
                 "boundary term for the guards -- both accepted only by "
                 "measured return",
                 fontsize=12.5, fontweight="bold")

    a = ax[0]
    keys = ["planner", "qhat", "ceiling"]
    lab = ["planner\nQ* curvature", "critic\nQ^pi", "reference\nself-agreement"]
    x = np.arange(3)
    med = [np.mean([r["gradient"][k]["cos_med"] for r in runs]) for k in keys]
    agr = [np.mean([r["gradient"][k]["agree"] for r in runs]) for k in keys]
    a.bar(x - 0.2, med, 0.38, color="#4c72b0", label="median cosine vs reference")
    a.bar(x + 0.2, agr, 0.38, color="#c44e52", label="sign agreement")
    a.set_xticks(x)
    a.set_xticklabels(lab, fontsize=9)
    a.set_ylim(0, 1.05)
    a.set_ylabel("agreement with the re-probed gradient field")
    a.set_title("what the law step consumes: the critic is no better\n"
                "than the planner in direction, only in scale", fontsize=10.5)
    a.legend(fontsize=9)
    a.grid(alpha=0.3, axis="y")

    a = ax[1]
    names = [r["name"] for r in runs[0]["ablation"]]
    base = np.mean([r["incumbent"]["G"] for r in runs])
    vals = [np.mean([r["ablation"][i]["G"] for r in runs])
            for i in range(len(names))]
    err = [np.ptp([r["ablation"][i]["G"] for r in runs]) / 2
           for i in range(len(names))]
    col = ["#8c8c8c" if n == "incumbent" else
           ("#dd8452" if n.startswith("planner") else
            ("#b0a160" if "uniform" in n else "#55a868")) for n in names]
    a.barh(np.arange(len(names)), vals, xerr=err, color=col, height=0.62)
    a.axvline(base, color="#8c8c8c", ls=":", lw=2)
    a.set_yticks(np.arange(len(names)))
    a.set_yticklabels(names, fontsize=8.5)
    a.invert_yaxis()
    a.set_xlabel("held-out return, mean over runs (bars: half-range)")
    a.set_title("one law step, same partition:\nthe step SHAPE is what matters",
                fontsize=10.5)
    a.grid(alpha=0.3, axis="x")

    a = ax[2]
    for k, r in enumerate(runs):
        y0, y1 = r["incumbent"]["G"], r["wired"]["G"]
        a.plot([0, 1], [y0, y1], marker="o", lw=2.2,
               color="#55a868" if y1 > y0 else "#c44e52")
        a.annotate("%+.2f" % (y1 - y0), (1, y1), textcoords="offset points",
                   xytext=(8, -3 + 11 * (k % 2)), fontsize=9)
    a.set_xlim(-0.25, 1.45)
    a.set_xticks([0, 1])
    a.set_xticklabels(["e18 incumbent", "+ critic-wired feedback"], fontsize=9)
    a.set_ylabel("held-out return (3 eval seeds x 3000 episodes)")
    a.set_title("paired, one line per independent run\n"
                "the acceptance test refuses a move as often as it takes one",
                fontsize=10.5)
    a.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out = os.path.join(ROOT, "figs", "e19_critic_wiring.png")
    fig.savefig(out, dpi=140)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
