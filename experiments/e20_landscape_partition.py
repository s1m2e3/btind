"""e20 -- partition by the shape of the value mountain, and close the structure loop.

e19 closed the law half: each arm now gets its own gradient from the steps it
owned, worth +2.61 held-out over four runs. It left the guard half half-open --
the boundary term can SLIDE a threshold but cannot add, delete or re-order an
arm, so the partition is still whatever `rsfi` happened to buy with blind random
literals, and nothing in it references value at all.

FOUR CAPABILITIES, added one at a time to the same incumbent, so that a gain can
be attributed to the thing that caused it:

  A  laws + thresholds            e19 exactly. Nothing structural.

THE BAND IS MADE EXPLICIT FIRST, for every capability, so the comparison is not
confounded: `materialise_default` turns the region the Fallback reached by
omission into an arm with its own literals (semantics identical, verified to
bit-equality on 4000 states). Without it the band -- which on this task is the
flee radius the controller is built around -- has no threshold of its own to
move, and every capability below inherits that handicap equally.
  B  + proposed arms              new clauses proposed by GRADIENT DISAGREEMENT
                                  over observations: where the residual field
                                  dQ/du inside one arm points systematically two
                                  ways, one affine law is serving two regimes.
  C  + landscape alphabet         guards may also mention V_hat and leverage
                                  (lambda_max of M), and proposals may aim at a
                                  value REGIME, translated into an observation
                                  literal so the artifact stays causal.
  D  + drop / re-order            an arm must keep earning its place, and
                                  priority becomes a decision rather than the
                                  order `rsfi` bought things in.

THE REGIME CROSS, which is the content of (C). A level cannot decide whether a
region deserves an arm; e15 gated on V_hat alone and lost 1.7 return. Crossing
drift with LEVERAGE can:

    steep,     high leverage   an arm pays: the action sets the rate of climb
    bad flat,  high leverage   an arm pays: a recoverable basin, mishandled
    bad flat,  low  leverage   DISREGARD: doomed, every action scores the same
    good flat, low  leverage   DISREGARD: solved, every action scores the same

`worth_an_arm` applies that as a filter BEFORE any rollout is spent, so the
budget goes to cells that carry both occupancy mass and gradient signal. The
table printed per run is the evidence for whether the cross is real: a cell with
mass and no gradient is already solved, a cell with gradient and no mass cannot
pay for a boundary, and only mass AND gradient justifies an arm.

WHAT WOULD FALSIFY THE WHOLE IDEA. The known answer is a band on `d_threat` at
roughly 0.10-0.15, which is where the value mountain bends. If the landscape
alphabet is measuring anything real it should REDISCOVER that band from drift
and leverage. If instead it buys `noise`, `pos_x` or `pos_y` -- causally
irrelevant by construction, and still in the alphabet for exactly this purpose --
then it is fitting the controller, not the task.

RESULT, three independent runs (incumbent rebuilt each time):

                            G    delta   meals  literals  distractors
    e18 incumbent       10.00       --    8.10       3.0
    A laws+thresholds   11.79    +1.78    9.21       4.0         none
    B +proposed arms    11.11    +1.11    8.80       4.0         none
    C +landscape alpha  11.54    +1.54    9.09       4.0         none
    D +drop/reorder     11.81    +1.80    9.25       2.7         none

WHAT THE TABLE ACTUALLY SHOWS, including the part that undercuts the design.

  THE ADD OPERATOR NEVER FIRED. Zero proposals were accepted in B, C or D
  across all three runs. So B and C are NOT measurements of "proposed arms" or
  "landscape alphabet" at all: with no proposal accepted, no clause ever
  mentions a landscape feature, and the three capabilities differ only in how
  many extra rollouts they spend. Those rollouts perturb the kernel respawn
  stream, so A, B and C are the SAME pipeline measured three times.

  WHICH MAKES A-B-C A NOISE MEASUREMENT, and a useful one. 11.79 / 11.11 /
  11.54 on identical machinery is +-0.34 half-range between nominally identical
  runs, against a per-run held-out CI of +-0.20. Any future capability claiming
  less than about 0.7 return on three runs is claiming nothing. That is the
  number to hold every arm of this experiment to, including A.

  DROP IS THE ONE OPERATOR THAT DID SOMETHING. It fired in 2 of 3 runs and took
  the artifact from 4 literals to 2 with no loss of return (11.81 against A's
  11.79). Nothing else in this project has ever made the emitted tree SMALLER.
  The asymmetric acceptance is why -- an arm must FAIL to prove it is earning
  its place -- and it is the only test here whose burden of proof sits on the
  incumbent rather than on the change.

  NAMING THE BAND IS THE PLAUSIBLE SOURCE OF THE +1.78. Capability A is e19's
  machinery plus `materialise_default`, and it gains more than e19's identical
  law-and-threshold loop did on comparable incumbents. The band -- the flee
  radius the whole controller is organised around -- had no threshold of its own
  until now, so `boundary_grad` could only move its edges by dragging the arms
  on either side. This is a cross-experiment comparison and therefore weak
  evidence; it is a hypothesis with a mechanism, not a measured effect.

  THE REGIME CROSS IS REAL AS A DIAGNOSTIC AND UNPROVEN AS AN ALPHABET.
  Leverage separates gradient signal by about 9x WITHIN a flatness class (bad
  flat, high leverage 0.51; bad flat, low leverage 0.06) -- exactly the
  recoverable-versus-doomed distinction a value level cannot make. But since no
  landscape literal was ever accepted, this experiment says nothing about
  whether a guard mentioning `leverage` helps. One earlier single run, before
  the band was made explicit, did buy `d_threat<=0.103 AND leverage<=6.385` and
  held it. Why proposals stopped being accepted once the band became its own
  arm is not explained here.

  The distractor control passes everywhere: no capability in any run bought
  `noise`, `pos_x`, `pos_y` or `t_norm`, including with two learned features in
  the alphabet.
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
from btind.evotm import evolve_bank
from btind.rollout_select import rsfi
from btind.valuesplit import action_curvature, fit_value_law
from btind.dagger import collect_onpolicy_states
from btind.search import search
from btind.policies import evaluate
from btind.vhat import fit_vhat
from btind.critic import fit_qhat
from btind import qwire as Q
from btind import structure as ST
from btind import landscape as LS

ALL = ["d_threat", "energy", "d_food", "pos_x", "pos_y", "t_norm", "noise"]
SV = [OBS_NAMES.index(n) for n in ALL]
DISTRACTORS = ("noise", "pos_x", "pos_y", "t_norm")
N_DAG, N_NEW = 4, 2000
N_EP, T, EVAL_SEEDS = 3000, 200, (11, 12, 13)
SEL_EP, SEL_SEED, ZTHR = 1000, 777, 2.0
N_ON, N_ITER, MIN_N = 20000, 3, 400
ARMS = ("A laws+thresholds", "B +proposed arms", "C +landscape alphabet",
        "D +drop/reorder")


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std() / np.sqrt(max(len(x), 1))


def heldout(env, bank, pol_fn, seeds=EVAL_SEEDS, n_ep=N_EP):
    g, e = [], []
    for s in seeds:
        env.seed_kernels(s)
        r = evaluate(env, pol_fn(bank), n_ep=n_ep, T=T, seed=s)
        g.append(r["G"]), e.append(r["eaten"])
    g, e = np.concatenate(g), np.concatenate(e)
    return float(g.mean()), float(ci(g)), float(e.mean())


def names_used(bank, names):
    return sorted({names[l[0]] for cl in bank["clauses"] for l in cl})


def bt_text(bank, names):
    out = []
    for cl in bank["clauses"]:
        out.append("  Sequence[ %s , Action ]" % " AND ".join(
            "%s%s%.3f" % (names[j], "<=" if n else ">", t) for j, t, n in cl))
    return "\n".join(out + ["  Action(default)"])


def residual_grad(bank, obs, Z, qhat):
    """Per-row dQ/du at the action the owning arm currently produces."""
    Xd = design_matrix(obs)
    a = Q.assign(bank["clauses"], Z)
    g = np.zeros((len(obs), 2))
    for c in np.unique(a):
        idx = np.flatnonzero(a == c)
        u = np.clip(Xd[idx] @ Q._law(bank, int(c)), -1, 1)
        g[idx] = qhat.grad_u(obs[idx], u)
    return g


def build_incumbent(env, verbose=True):
    """e18's winning arm, rebuilt so every capability starts from the same bank."""
    D = collect(n_states=5000, seed=0, label="medoid")
    obs, U, V = D["obs"], D["u_medoid"], D["V"]
    gap, coh, M = D["gap"], D["coherence"], D["M"]
    rng = np.random.default_rng(4)
    for it in range(N_DAG + 1):
        ratio, r = artifact_ratio(obs, U, V, gap, coh, SV)
        keep, _ = veto_vars(ratio, SV, 1.5)
        sv = [j for j in keep if r["z_u"][SV.index(j)] > 1.0]
        lb = evolve_bank(obs, U, M, sv, OBS_NAMES, C=48, generations=60,
                         max_arity=3, min_n=MIN_N, max_keep=16, seed=0,
                         verbose=False)
        m = rsfi(env, obs, U, M, sv, OBS_NAMES, n_arms=6, pool=40, refine=12,
                 max_arity=3, min_n=MIN_N, n_ep=300, T=T, sel_seed=SEL_SEED,
                 seed=0, z=ZTHR, seed_clauses=lb["clauses"], verbose=False)
        bank = dict(clauses=m["clauses"], laws=m["laws"], default=m["default"],
                    names=OBS_NAMES)
        if it == N_DAG:
            break
        pol = LS.LandscapeBank(bank, None, None, len(OBS_NAMES))
        S = collect_onpolicy_states(env, pol, N_NEW, rng)
        rs = search(env, S, rng, label="medoid")
        obs = np.vstack([obs, env.observe(S)])
        U = np.vstack([U, rs["u_star"]])
        V = np.concatenate([V, rs["V"]])
        gap = np.concatenate([gap, rs["gap"]])
        coh = np.concatenate([coh, rs["coherence"]])
        M = np.concatenate([M, action_curvature(rs["a0_all"], rs["G_all"])])
    return bank


# ----------------------------------------------------------------- the loop
def run_capability(env, bank0, cap, rng, verbose=True):
    """One polish run at a given capability level. cap in 'ABCD'."""
    bank = dict(bank0)
    log, table = [], None
    n_obs = len(OBS_NAMES)
    vh_q = [None, None]          # latest (V_hat, Q_hat), shared with pol_fn

    def pol_fn(b, _c=vh_q):
        return LS.LandscapeBank(b, _c[0], _c[1], n_obs)

    for it in range(N_ITER):
        pol = LS.LandscapeBank(bank, vh_q[0], vh_q[1], n_obs)
        vh = fit_vhat(env, pol, n_ep=800, T=T, n_step=20, sweeps=2, seed=21)
        L = LS.subsample(LS.landscape_rollout(env, pol, vh, rng, n_ep=1200,
                                              T=T, k=10), N_ON, rng)
        qh, _ = fit_qhat(env, pol, L["state"], vh, rng, seed=3)
        vh_q[0], vh_q[1] = vh, qh

        obs, w = L["obs"], L["w"]
        lam = LS.leverage(qh, obs)
        if cap >= "C":
            Z, znames = LS.guard_features(obs, vh, qh, OBS_NAMES)
        else:
            Z, znames = obs, list(OBS_NAMES)
        g = residual_grad(bank, obs, Z, qh)
        R = LS.regimes(L, lam)
        if table is None:
            table = LS.regime_table(L, lam, g, R)

        cur = Q._score(env, bank, SEL_EP, T, SEL_SEED, pol_fn)
        lo_hi = {j: (float(Z[:, j].min()), float(Z[:, j].max()))
                 for j in range(Z.shape[1])}

        bank, cur, li = Q.law_step(env, bank, obs, w, qh, cur, mode="grad",
                                   etas=(0.02, 0.05, 0.1, 0.2), n_ep=SEL_EP,
                                   T=T, seed=SEL_SEED, z=ZTHR, pol_fn=pol_fn,
                                   Z=Z)
        bank, cur, gi = Q.guard_step(env, bank, obs, w, qh, cur, n_try=4,
                                     n_ep=SEL_EP, T=T, seed=SEL_SEED, z=ZTHR,
                                     lo_hi=lo_hi, pol_fn=pol_fn, Z=Z)
        rec = dict(it=it, cap=cap, law=li["accepted"], law_best=li["best"],
                   guard=[x for x in gi if x["accepted"]])

        if cap >= "B":
            ok_rows = (LS.worth_an_arm(R) if cap >= "C"
                       else np.ones(len(obs), bool))
            props = LS.propose(bank, Q.assign, Z, OBS_NAMES, g, w, ok_rows,
                               per_arm=2, min_n=MIN_N, X=design_matrix(obs))
            if cap >= "C":
                props = props + LS.propose_from_regimes(
                    R, Z, w, ok_rows, list(range(Z.shape[1])))
                props = sorted(props, key=lambda d: -d["gain"])

            def fit_law(m, arm, eta, _b=bank, _o=obs, _w=w, _q=qh):
                """Parent law plus a gradient step of size eta on its own rows."""
                Xd = design_matrix(_o)
                th = Q._law(_b, arm)
                idx = np.flatnonzero(m)
                u = np.clip(Xd[idx] @ th, -1, 1)
                gr = (_w[idx, None] * Xd[idx]).T @ _q.grad_u(_o[idx], u)
                return th + eta * gr / max(_w[idx].sum(), 1e-9)

            bank, cur, ai = ST.add_arm(env, bank, props, fit_law, pol_fn, cur,
                                       n_try=6, n_ep=SEL_EP, T=T,
                                       seed=SEL_SEED, z=ZTHR, min_n=MIN_N,
                                       match_fn=LS._match_cols, Z=Z)
            rec["add"] = [x for x in ai if x["accepted"]] or ai[:1]

        if cap >= "D":
            bank, cur, di = ST.drop_arm(env, bank, pol_fn, cur, n_ep=SEL_EP,
                                        T=T, seed=SEL_SEED, z=ZTHR)
            bank, cur, ri = ST.reorder(env, bank, pol_fn, cur, LS._match_cols,
                                       Z, n_ep=SEL_EP, T=T, seed=SEL_SEED,
                                       z=ZTHR)
            rec["drop"] = [x for x in di if x["accepted"]]
            rec["reorder"] = [x for x in ri if x["accepted"]]

        rec["G_sel"] = float(cur.mean())
        rec["n_clauses"] = len(bank["clauses"])
        log.append(rec)
        if verbose:
            moves = []
            if li["accepted"]:
                moves.append("laws")
            if rec["guard"]:
                moves.append("thr:%s" % znames[rec["guard"][0]["j"]])
            for key in ("add", "drop", "reorder"):
                if rec.get(key) and rec[key] and rec[key][0].get("accepted"):
                    tag = (znames[rec[key][0]["col"]] if key == "add"
                           else str(rec[key][0]["arm"]))
                    moves.append("%s:%s" % (key, tag))
            print("    [%s it%d] %-34s G_sel %6.2f  %d clauses"
                  % (cap, it, ",".join(moves) or "nothing accepted",
                     cur.mean(), len(bank["clauses"])), flush=True)
    return bank, log, table, (vh_q[0], vh_q[1])


def run_once(env, tag):
    t0 = time.time()
    print("\n### run %d" % tag, flush=True)
    bank0 = build_incumbent(env)
    bank0, mat = LS.materialise_default(bank0)
    n_obs = len(OBS_NAMES)
    plain = lambda b: LS.LandscapeBank(b, None, None, n_obs)
    g0, c0, e0 = heldout(env, bank0, plain)
    print("  incumbent  G %6.2f +-%.2f  eaten %.2f  %s  (band made explicit: "
          "%s)  [%.0fs]"
          % (g0, c0, e0, ",".join(names_used(bank0, OBS_NAMES)), mat,
             time.time() - t0), flush=True)

    out, first_table = {}, None
    for cap, name in zip("ABCD", ARMS):
        bank, log, table, (vh, qh) = run_capability(
            env, bank0, cap, np.random.default_rng(5 + tag))
        pol_fn = (lambda b, _v=vh, _q=qh: LS.LandscapeBank(b, _v, _q, n_obs))
        g, c, e = heldout(env, bank, pol_fn)
        znames = list(OBS_NAMES) + ["V_hat", "leverage"]
        used = names_used(bank, znames)
        out[name] = dict(G=g, ci=c, eaten=e, log=log,
                         clauses=[[list(map(float, l)) for l in cl]
                                  for cl in bank["clauses"]],
                         bt=bt_text(bank, znames), feats=used,
                         distractors=[f for f in used if f in DISTRACTORS],
                         n_clauses=len(bank["clauses"]),
                         n_literals=sum(len(c) for c in bank["clauses"]))
        first_table = first_table or table
        print("  %-22s G %6.2f +-%.2f  eaten %.2f  %d clauses  %s%s"
              % (name, g, c, e, len(bank["clauses"]), ",".join(used),
                 "   DISTRACTORS: " + ",".join(out[name]["distractors"])
                 if out[name]["distractors"] else ""), flush=True)

    print("  regime cross (occupancy mass / gradient signal):")
    print("    %-10s %-5s %7s %7s %8s %7s %8s"
          % ("regime", "lev", "mass", "V", "drift", "lam", "signal"))
    for r in first_table:
        print("    %-10s %-5s %7.3f %7.2f %+8.4f %7.3f %8.4f"
              % (r["regime"], r["lev"], r["mass"], r["V"], r["drift"],
                 r["lam"], r["signal"]))
    return dict(incumbent=dict(G=g0, ci=c0, eaten=e0,
                               feats=names_used(bank0, OBS_NAMES),
                               n_clauses=len(bank0["clauses"])),
                arms=out, table=first_table, seconds=time.time() - t0)


def main():
    t0 = time.time()
    reps = int(os.environ.get("E20_REPEATS", "3"))
    ForageWorld.seed_kernels(0)
    env = ForageWorld()
    runs = [run_once(env, k) for k in range(reps)]

    print("\n" + "=" * 78)
    print("%-24s %8s %8s %8s %8s %s"
          % ("", "G", "delta", "meals", "literals", "distractors"))
    base = np.array([r["incumbent"]["G"] for r in runs])
    print("%-24s %8.2f %8s %8.2f %8d" %
          ("e18 incumbent", base.mean(), "--",
           np.mean([r["incumbent"]["eaten"] for r in runs]),
           int(np.mean([r["incumbent"]["n_clauses"] for r in runs]))))
    for name in ARMS:
        g = np.array([r["arms"][name]["G"] for r in runs])
        e = np.mean([r["arms"][name]["eaten"] for r in runs])
        li = np.mean([r["arms"][name]["n_literals"] for r in runs])
        ds = sum(len(r["arms"][name]["distractors"]) for r in runs)
        print("%-24s %8.2f %8.2f %8.2f %8.1f %s"
              % (name, g.mean(), (g - base).mean(), e, li,
                 "%d run(s)" % ds if ds else "none"))
    print("=" * 78)
    print("\nemitted controller, last run, capability D:")
    print(runs[-1]["arms"][ARMS[-1]]["bt"])

    with open(os.path.join(ROOT, "data", "e20_summary.json"), "w") as fh:
        json.dump(dict(runs=runs, reps=reps, seconds=time.time() - t0), fh,
                  indent=1, default=float)
    figure(runs)
    print("\ntotal %.0fs" % (time.time() - t0))


def figure(runs):
    fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.8))
    fig.suptitle("ForageWorld: partitioning by the shape of the value mountain\n"
                 "capabilities added one at a time to the same incumbent; every "
                 "move still accepted only by measured return",
                 fontsize=12.5, fontweight="bold")

    a = ax[0]
    base = np.mean([r["incumbent"]["G"] for r in runs])
    xs = np.arange(len(ARMS) + 1)
    ys = [base] + [np.mean([r["arms"][n]["G"] for r in runs]) for n in ARMS]
    err = [np.ptp([r["incumbent"]["G"] for r in runs]) / 2] + \
          [np.ptp([r["arms"][n]["G"] for r in runs]) / 2 for n in ARMS]
    a.bar(xs, ys, yerr=err, color=["#8c8c8c", "#55a868", "#4c72b0", "#c44e52",
                                   "#8172b2"], width=0.62)
    a.axhline(base, color="#8c8c8c", ls=":", lw=2)
    a.set_xticks(xs)
    a.set_xticklabels(["e18\nincumbent"] + [n.replace(" ", "\n", 1)
                                            for n in ARMS], fontsize=8.5)
    a.set_ylabel("held-out return (3 eval seeds x 3000 episodes)")
    a.set_title("cumulative capability", fontsize=11)
    a.grid(alpha=0.3, axis="y")

    a = ax[1]
    tab = runs[0]["table"]
    labs = ["%s\n%s lev" % (r["regime"].replace("_", " "), r["lev"])
            for r in tab]
    x = np.arange(len(tab))
    a.bar(x - 0.2, [r["mass"] for r in tab], 0.38, color="#4c72b0",
          label="occupancy mass")
    sig = np.array([r["signal"] for r in tab])
    a.bar(x + 0.2, sig / max(sig.max(), 1e-9) * max(
        [r["mass"] for r in tab]), 0.38, color="#c44e52",
        label="gradient signal (scaled)")
    a.set_xticks(x)
    a.set_xticklabels(labs, fontsize=8)
    a.set_ylabel("share of discounted occupancy")
    a.set_title("the regime cross: only mass AND gradient\njustifies an arm",
                fontsize=11)
    a.legend(fontsize=9)
    a.grid(alpha=0.3, axis="y")

    a = ax[2]
    for k, r in enumerate(runs):
        ys = [r["incumbent"]["G"]] + [r["arms"][n]["G"] for n in ARMS]
        a.plot(np.arange(len(ys)), ys, marker="o", lw=2.0, alpha=0.85,
               label="run %d" % k)
    a.set_xticks(np.arange(len(ARMS) + 1))
    a.set_xticklabels(["inc"] + list("ABCD"))
    a.set_xlabel("capability")
    a.set_ylabel("held-out return")
    a.set_title("per run, unaveraged", fontsize=11)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out = os.path.join(ROOT, "figs", "e20_landscape_partition.png")
    fig.savefig(out, dpi=140)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
