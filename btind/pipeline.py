"""The pipeline, end to end: imitation warm start, then policy improvement.

TWO STAGES, AND THEY ARE DIFFERENT ALGORITHMS.

  STAGE 1 -- IMITATION (DAgger with a planner as the expert).
      `search` fires 256 action sequences through the sim at each state and
      returns u* = argmax_a Q*(s,a) with the curvature M = -H_a Q*. `rsfi`
      buys guards greedily, each one kept only if 300 measured episodes say
      return rose, and fits each arm law to those labels. The controller is then
      rolled out, the states it actually visits are re-labelled by the planner,
      and the whole thing repeats. This produces a competent bank -- about 10
      return against the planner's own 13 -- and it stops improving there.

  STAGE 2 -- POLICY IMPROVEMENT (actor-critic, no expert at all).
      The planner is dropped. V-hat is fitted to the CONTROLLER by fitted value
      iteration, Q-hat is fitted to it by probing the action space, and the two
      halves of the bank get the gradient the objective actually defines:

          laws     grad_theta_c J = E_{d_gamma^pi}[ 1{s in R_c} x(s) dQ^pi/du ]
          guards   dJ/dtau = int_{x_j=tau} d_gamma^pi [ Q(theta_new) - Q(theta_cur) ]

      plus one structural move -- DROP, on non-inferiority against a stated
      margin -- which is the only thing in this project that makes the emitted
      tree smaller.

WHY STAGE 2 IS NOT MORE OF STAGE 1. Measured in e19 over four runs, re-labelling
with the planner and refitting on-policy is worth -3.0 return; the same rows with
a gradient step from the critic are worth +1.8. The expert is optimal for a
policy that replans every tick, and the bank does not replan. Past the warm start
its labels are a different controller's answers.

DEFAULTS ARE WHAT WAS MEASURED, not what was designed:

  law step      DIRECTION from the critic, DISTANCE from a rollout line search.
                Never the closed-form refit to the critic argmax: same critic,
                same rows, -14.6 return (e19).
  band          `materialise_default` first, so the region the Fallback reaches
                by omission has thresholds of its own to move (e20).
  add_arms      OFF. 27 proposals, 27 rejected across three runs (e20); the
                move is bigger than a threshold slide, so at the same z it has
                more ways to look good on the selection draw. Available, and it
                needs a stiffer test before it should be trusted.
  drop_after    1. Drop cannot run on the first iteration. The band arm that
                `materialise_default` creates starts carrying the DEFAULT law,
                so removing it provably changes nothing and non-inferiority
                accepts it instantly -- deleting the movable edges before any
                law step has made the band differ from its neighbour. Structure
                is the slowest timescale; it has to wait for the fast one.
  landscape     OFF. `V_hat` and `leverage` in the guard alphabet is a real
                diagnostic -- leverage separates gradient signal ~9x inside a
                flatness class -- but no landscape literal ever survived a
                rollout, so gating on it is unproven rather than useful.
"""
import time

import numpy as np

from .collect import collect, design_matrix
from .critic import QFeatures, fit_qhat
from .dagger import collect_onpolicy_states
from .envs.forage import OBS_NAMES
from .evotm import evolve_bank
from .joint import artifact_ratio, veto_vars
from .lawsearch import search_laws
from .landscape import (LandscapeBank, guard_features, landscape_rollout,
                        leverage, materialise_default, propose, regime_table,
                        regimes, subsample, worth_an_arm, _match_cols)
from .policies import evaluate
from .qwire import _law, _score, assign, guard_step, law_step
from .rollout_select import rsfi
from .runlog import (RunLog, STAGE1_MODULES, cache_get, cache_key,
                     cache_put, source_hash)
from .search import search
from .structure import add_arm, drop_arm, reorder
from .valuesplit import action_curvature
from .vhat import fit_vhat

DEFAULTS = dict(
    # --- domain wiring. Every index below is an OBSERVATION COLUMN, so a new
    # world is a new config rather than a new pipeline. `radial` are the axes
    # value bends along (distances, energy) and `bearing` the unit-vector pairs
    # they modulate; the critic tensors one against the other, which is the only
    # part of the machinery that has to know what the columns MEAN.
    cand_names=("d_threat", "energy", "d_food", "pos_x", "pos_y", "t_norm",
                "noise"),
    radial=(0, 3, 4), bearing=((1, 2), (5, 6)),
    # stage 1. The PLANNER HORIZON has to reach the reward it is labelling for:
    # on NestWorld a delivery cycle is 30-50 steps, so a 40-step search sees at
    # most one, and at gamma 0.995 the value it is estimating extends far past
    # that. H is a domain parameter, not a constant.
    search_K=256, search_H=40, search_n_seg=4,
    n_seed=5000, n_dagger=4, n_new=2000, pop=48, generations=60, max_arity=3,
    min_n=400, n_arms=6, pool=40, refine=12, sel_ep_imitate=300,
    # stage 2
    n_polish=3, n_on=20000, n_ep_rollout=1200, drift_k=10,
    etas=(0.02, 0.05, 0.1, 0.2), n_guard=4, eps_frac=0.05,
    law_search=True, law_search_ep=600,
    add_arms=False, landscape_guards=False, drop=True, drop_after=1,
    drop_margin=0.25, reorder=True,
    # shared
    sel_ep=1000, sel_seed=777, z=2.0, T=200,
    # bookkeeping
    log=True, cache=True, tag="fit",
)

# Config entries stage 1 reads. The imitation bank is a pure function of these
# plus the world and the seed, so they are the whole cache key -- a stage-2 knob
# must never invalidate a stage-1 result.
STAGE1_KEYS = ("cand_names", "n_seed", "n_dagger", "n_new", "pop",
               "generations", "max_arity", "min_n", "n_arms", "pool", "refine",
               "sel_ep_imitate", "sel_seed", "z", "T")


def emit(bank, names=OBS_NAMES):
    """The Fallback as text. The default child is a totality guard, not a region.

    A Fallback returns FAILURE if every child fails and a controller has to act
    every tick, so the last child is unconditional by construction. After
    `materialise_default` the region it used to own is an explicit arm above it
    and this line is unreachable -- kept because nothing guarantees the arms
    stay exhaustive once thresholds move.
    """
    zn = list(names) + ["V_hat", "leverage"]
    out = ["Fallback"]
    for cl in bank["clauses"]:
        out.append("|-- Sequence[ %s , Action(u = K x + b) ]" % " AND ".join(
            "%s%s%.3f" % (zn[j], "<=" if n else ">", t) for j, t, n in cl))
    out.append("\\-- Action(default)          # totality guard")
    return "\n".join(out)


def features_used(bank, names=OBS_NAMES):
    zn = list(names) + ["V_hat", "leverage"]
    return sorted({zn[l[0]] for cl in bank["clauses"] for l in cl})


def evaluate_bank(env, bank, vhat=None, qhat=None, seeds=(11, 12, 13),
                  n_ep=3000, T=400, n_obs=None):
    """Held-out return on seeds no stage of the search ever uses.

    THE HORIZON HAS TO MATCH THE ONE BEING OPTIMISED. This defaulted to T=200
    while every `score` call in the search runs T=400, so the reported number
    was measuring a different task -- half an episode, on a world where food is
    gathered over time under a 0.995 discount -- and it read about 4.5 return
    units low as a result. That offset was constant, so the RANKINGS this
    reported were never wrong, but none of its numbers could be compared with a
    number from inside the search or with a hand-written reference measured at
    T=400, and both comparisons were being made.

    The three layers are now distinct and each has a job: the search trains on a
    seed that ROTATES every round, selects the bank it returns on one fixed
    VALIDATION seed it never accepts a move against, and reports here on seeds
    reserved for reporting.
    """
    # MemBank, NOT LandscapeBank: once a bank carries `laws_on_z` its laws are
    # sized for the augmented layout, and a policy that multiplies
    # design_matrix(obs) either raises (if the widths differ) or silently binds
    # every coefficient to the wrong feature (if they happen to match).
    # `test_equivalence` shows MemBank reproduces LandscapeBank bit for bit on a
    # plain bank, so this is a strict generalisation.
    from .memory import MemBank
    n_obs = len(bank["names"]) if n_obs is None else n_obs
    g, e, s_ = [], [], []
    for sd in seeds:
        env.seed_kernels(sd)
        r = evaluate(env, MemBank(bank, n_obs, vhat, qhat), n_ep=n_ep,
                     T=T, seed=sd)
        g.append(r["G"]), e.append(r["eaten"]), s_.append(r["survived"])
    g, e, s_ = np.concatenate(g), np.concatenate(e), np.concatenate(s_)
    return dict(G=float(g.mean()), ci=float(1.96 * g.std() / np.sqrt(len(g))),
                eaten=float(e.mean()), survived=float(s_.mean()))


# --------------------------------------------------------------- stage one
def imitate(env, cfg, names=OBS_NAMES, rng=None, verbose=True):
    """DAgger against the planner. Returns a bank and the labelled pool."""
    rng = rng or np.random.default_rng(4)
    sv_all = [names.index(n) for n in cfg["cand_names"]]
    D = collect(n_states=cfg["n_seed"], seed=0, label="medoid", env=env,
                K=cfg["search_K"], H=cfg["search_H"], n_seg=cfg["search_n_seg"])
    obs, U, V = D["obs"], D["u_medoid"], D["V"]
    gap, coh, M = D["gap"], D["coherence"], D["M"]
    bank = None
    for it in range(cfg["n_dagger"] + 1):
        ratio, r = artifact_ratio(obs, U, V, gap, coh, sv_all)
        keep, _ = veto_vars(ratio, sv_all, 1.5)
        sv = [j for j in keep if r["z_u"][sv_all.index(j)] > 1.0]
        seeds = evolve_bank(obs, U, M, sv, names, C=cfg["pop"],
                            generations=cfg["generations"],
                            max_arity=cfg["max_arity"], min_n=cfg["min_n"],
                            max_keep=16, seed=0, verbose=False)["clauses"]
        m = rsfi(env, obs, U, M, sv, names, n_arms=cfg["n_arms"],
                 pool=cfg["pool"], refine=cfg["refine"],
                 max_arity=cfg["max_arity"], min_n=cfg["min_n"],
                 n_ep=cfg["sel_ep_imitate"], T=cfg["T"],
                 sel_seed=cfg["sel_seed"], seed=0, z=cfg["z"],
                 seed_clauses=seeds, verbose=False)
        bank = dict(clauses=m["clauses"], laws=m["laws"], default=m["default"],
                    names=names)
        if verbose:
            print("  [imitate %d] %d clauses  %s"
                  % (it, len(m["clauses"]), ",".join(features_used(bank, names))
                     or "-"), flush=True)
        if it == cfg["n_dagger"]:
            break
        pol = LandscapeBank(bank, None, None, len(names))
        S = collect_onpolicy_states(env, pol, cfg["n_new"], rng)
        rs = search(env, S, rng, label="medoid", K=cfg["search_K"],
                    H=cfg["search_H"], n_seg=cfg["search_n_seg"])
        obs = np.vstack([obs, env.observe(S)])
        U = np.vstack([U, rs["u_star"]])
        V = np.concatenate([V, rs["V"]])
        gap = np.concatenate([gap, rs["gap"]])
        coh = np.concatenate([coh, rs["coherence"]])
        M = np.concatenate([M, action_curvature(rs["a0_all"], rs["G_all"])])
    return bank


# --------------------------------------------------------------- stage two
def _residual_grad(bank, obs, Z, qhat):
    Xd = design_matrix(obs)
    a = assign(bank["clauses"], Z)
    g = np.zeros((len(obs), 2))
    for c in np.unique(a):
        idx = np.flatnonzero(a == c)
        u = np.clip(Xd[idx] @ _law(bank, int(c)), -1, 1)
        g[idx] = qhat.grad_u(obs[idx], u)
    return g


def improve(env, bank, cfg, names=OBS_NAMES, rng=None, verbose=True,
            runlog=None):
    """Actor-critic polish. Laws, then thresholds, then structure."""
    rng = rng or np.random.default_rng(5)
    n_obs = len(names)
    bank, banded = materialise_default(bank)
    state = [None, None]                      # latest (V_hat, Q_hat)

    def pol_fn(b, _s=state):
        return LandscapeBank(b, _s[0], _s[1], n_obs)

    log = []
    for it in range(cfg["n_polish"]):
        vh = fit_vhat(env, pol_fn(bank), n_ep=800, T=cfg["T"], n_step=20,
                      sweeps=2, seed=21)
        L = subsample(landscape_rollout(env, pol_fn(bank), vh, rng,
                                        n_ep=cfg["n_ep_rollout"], T=cfg["T"],
                                        k=cfg["drift_k"]), cfg["n_on"], rng)
        qh, qinfo = fit_qhat(env, pol_fn(bank), L["state"], vh, rng, seed=3,
                             feats=QFeatures(radial=cfg["radial"],
                                             bearing=cfg["bearing"]))
        state[0], state[1] = vh, qh

        obs, w = L["obs"], L["w"]
        lam = leverage(qh, obs)
        Z = (guard_features(obs, vh, qh, names)[0]
             if cfg["landscape_guards"] else obs)
        g = _residual_grad(bank, obs, Z, qh)
        R = regimes(L, lam)
        rec = dict(it=it, critic_r2=qinfo["r2"],
                   regimes=regime_table(L, lam, g, R))

        cur = _score(env, bank, cfg["sel_ep"], cfg["T"], cfg["sel_seed"],
                     pol_fn)

        # THE DISCRETE JUMP COMES FIRST. A gradient step cannot cross from a
        # regression fit to `-bear_threat`; it can only polish whichever basin
        # it starts in. Library search picks the basin by rollout, the DPG step
        # then refines inside it.
        if cfg["law_search"] and it == 0:
            bank, _, ls = search_laws(env, bank, names, pol_fn,
                                      n_ep=cfg["law_search_ep"], T=cfg["T"],
                                      seed=cfg["sel_seed"], z=cfg["z"],
                                      verbose=verbose)
            # A SCORE IS TIED TO THE EPISODE COUNT THAT PRODUCED IT. The search
            # runs cheaper rollouts, so its reference cannot be handed onward;
            # re-measure at the selection count before anything compares to it.
            cur = _score(env, bank, cfg["sel_ep"], cfg["T"], cfg["sel_seed"],
                         pol_fn)
            rec["law_search"] = dict(bank.get("law_names", {}))
            if runlog:
                for t in ls:
                    runlog.move("law_search", dict(it=it, **t))
        lo_hi = {j: (float(Z[:, j].min()), float(Z[:, j].max()))
                 for j in range(Z.shape[1])}

        bank, cur, li = law_step(env, bank, obs, w, qh, cur, mode="grad",
                                 etas=cfg["etas"], n_ep=cfg["sel_ep"],
                                 T=cfg["T"], seed=cfg["sel_seed"], z=cfg["z"],
                                 pol_fn=pol_fn, Z=Z)
        bank, cur, gi = guard_step(env, bank, obs, w, qh, cur,
                                   n_try=cfg["n_guard"], n_ep=cfg["sel_ep"],
                                   T=cfg["T"], seed=cfg["sel_seed"],
                                   z=cfg["z"], eps_frac=cfg["eps_frac"],
                                   lo_hi=lo_hi, pol_fn=pol_fn, Z=Z)
        rec.update(laws=li["accepted"], law_gain=li["best"],
                   guard=[x for x in gi if x["accepted"]])
        if runlog:
            for t in li["tries"]:
                runlog.move("law", dict(it=it, **t))
            for t in gi:
                runlog.move("guard", dict(it=it, **t))

        if cfg["add_arms"]:
            ok_rows = (worth_an_arm(R) if cfg["landscape_guards"]
                       else np.ones(len(obs), bool))
            props = propose(bank, assign, Z, names, g, w, ok_rows, per_arm=2,
                            min_n=cfg["min_n"], X=design_matrix(obs))

            def fit_law(m, arm, eta, _b=bank, _o=obs, _w=w, _q=qh):
                Xd = design_matrix(_o)
                th = _law(_b, arm)
                idx = np.flatnonzero(m)
                u = np.clip(Xd[idx] @ th, -1, 1)
                gr = (_w[idx, None] * Xd[idx]).T @ _q.grad_u(_o[idx], u)
                return th + eta * gr / max(_w[idx].sum(), 1e-9)

            bank, cur, ai = add_arm(env, bank, props, fit_law, pol_fn, cur,
                                    n_try=6, n_ep=cfg["sel_ep"], T=cfg["T"],
                                    seed=cfg["sel_seed"], z=cfg["z"],
                                    min_n=cfg["min_n"], match_fn=_match_cols,
                                    Z=Z)
            rec["add"] = [x for x in ai if x["accepted"]]
            if runlog:
                for t in ai:
                    runlog.move("add", dict(it=it, **t))
        if cfg["drop"] and it >= cfg["drop_after"]:
            bank, cur, di = drop_arm(env, bank, pol_fn, cur,
                                     n_ep=cfg["sel_ep"], T=cfg["T"],
                                     seed=cfg["sel_seed"], z=cfg["z"],
                                     margin=cfg["drop_margin"])
            rec["drop"] = [x for x in di if x["accepted"]]
            if runlog:
                for t in di:
                    runlog.move("drop", dict(it=it, **t))
        if cfg["reorder"]:
            bank, cur, ri = reorder(env, bank, pol_fn, cur, _match_cols, Z,
                                    n_ep=cfg["sel_ep"], T=cfg["T"],
                                    seed=cfg["sel_seed"], z=cfg["z"])
            rec["reorder"] = [x for x in ri if x["accepted"]]
            if runlog:
                for t in ri:
                    runlog.move("reorder", dict(it=it, **t))

        rec["G_sel"] = float(cur.mean())
        rec["n_clauses"] = len(bank["clauses"])
        log.append(rec)
        if verbose:
            moves = ["laws"] if li["accepted"] else []
            if rec["guard"]:
                moves.append("threshold")
            for k in ("add", "drop", "reorder"):
                if rec.get(k):
                    moves.append(k)
            print("  [improve %d] %-22s G_sel %6.2f  %d clauses  (critic R2 "
                  "%.2f)" % (it, ",".join(moves) or "nothing accepted",
                             cur.mean(), len(bank["clauses"]), qinfo["r2"]),
                  flush=True)
    return bank, dict(log=log, banded=banded, vhat=state[0], qhat=state[1])


# ------------------------------------------------------------------- driver
def fit_bt(env, names=OBS_NAMES, seed=0, verbose=True, **overrides):
    """Imitation warm start, then policy improvement. Returns the bank and log.

    `names` and the `radial`/`bearing`/`cand_names` overrides are the entire
    domain interface. ForageWorld needs none of them; NestWorld passes its own.
    """
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    t0 = time.time()
    rl = RunLog(cfg["tag"], enabled=cfg["log"])
    if cfg["log"]:
        rl.config(env, cfg, extra=dict(names=list(names), seed=seed))
    env.seed_kernels(seed)

    import inspect
    key = cache_key(env, cfg, seed, STAGE1_KEYS,
                    src=source_hash(modules=STAGE1_MODULES,
                                    extra=inspect.getsource(imitate)))
    bank = cache_get(key) if cfg["cache"] else None
    if bank is not None:
        bank["names"] = list(names)
        if verbose:
            print("stage 1 -- imitation: CACHED (%s)" % key)
    else:
        if verbose:
            print("stage 1 -- imitation (DAgger, planner as expert)")
        bank = imitate(env, cfg, names, np.random.default_rng(4 + seed),
                       verbose)
        if cfg["cache"]:
            cache_put(key, bank, meta=dict(seed=seed, tag=cfg["tag"]))
    r1 = evaluate_bank(env, bank, n_obs=len(names))
    if verbose:
        print("  G %.2f +-%.2f  meals %.2f  [%.0fs]\n"
              "stage 2 -- policy improvement (actor-critic, no expert)"
              % (r1["G"], r1["ci"], r1["eaten"], time.time() - t0))
    bank, info = improve(env, bank, cfg, names, np.random.default_rng(5 + seed),
                         verbose, runlog=rl if cfg["log"] else None)
    r2 = evaluate_bank(env, bank, info["vhat"], info["qhat"], n_obs=len(names))
    bt = emit(bank, names)
    if verbose:
        print("  G %.2f +-%.2f  meals %.2f  [%.0fs]"
              % (r2["G"], r2["ci"], r2["eaten"], time.time() - t0))
    out = dict(bank=bank, bt=bt, imitation=r1, final=r2,
               features=features_used(bank, names), log=info["log"],
               vhat=info["vhat"], qhat=info["qhat"],
               seconds=time.time() - t0, cfg=cfg,
               run_id=rl.id if cfg["log"] else None)
    if cfg["log"]:
        rl.bank(bank, names, bt)
        rl.finish(dict(stage1=r1, stage2=r2,
                       features=features_used(bank, names),
                       n_clauses=len(bank["clauses"]), seed=seed), bt=bt)
    return out
