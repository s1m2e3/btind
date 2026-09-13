r"""The outer loop: what gets refined, and how often.

Running every mechanism every round is both unaffordable and wrong. Unaffordable
because the memory sweep costs ~750s against ~20s for a gradient step; wrong
because a move proposed against stale laws is priced against a counterfactual
that will not survive the next law step. So each mechanism gets a CADENCE, and
they are ordered fastest-first within a round:

    every round     critic refit, then the region-restricted policy gradient
                    on the laws, then boundary-term threshold moves
    every 3 rounds  law library search -- the discrete jump a gradient cannot
                    make (regression fit -> `from_threat` is not a small step)
    every 3 rounds  structure: drop and reorder
    every 3 rounds  beta / stickiness, offset so it lands AFTER laws have moved
    once            memory: (what to store, when to write), the expensive one

WHY DAgger DOES NOT CONTINUE PAST STAGE 1. Measured in e19 over four runs,
re-labelling on-policy states with the planner and refitting was worth -3.0
return, where a gradient step from the critic on the same rows was worth +1.8.
The planner is optimal for a controller that replans every tick; ours does not.
Past the warm start its labels are a different controller's answers, and under
partial observability they are a PRIVILEGED controller's answers, which the
student cannot reproduce even in principle.

EVERY MOVE IS STILL A PROPOSAL. Nothing here is applied because a criterion
liked it -- the paired rollout test accepts or rejects each one, and on this
deterministic world its noise floor is zero, so a rejected move is rejected on
its merits rather than on sampling error.
"""
import time

import numpy as np

from .betasearch import search_beta
from .critic import QFeatures, fit_qhat
from .landscape import guard_features, landscape_rollout, subsample
from .lawsearch import search_laws
from .memory import MemBank, mem_names
from .memsearch import discover, record
from .qwire import _score, guard_step, law_step
from .structure import drop_arm, reorder, score
from .vhat import fit_vhat

SCHEDULE = dict(laws_every=1, guards_every=1, library_every=3, structure_every=3,
                beta_every=3, beta_offset=2, memory_at=1)


def _pol_fn(n_obs, state):
    """Policy factory that always reflects the latest critic and value fit."""
    def f(b):
        return MemBank(b, n_obs, state.get("vhat"), state.get("qhat"))
    return f


def run(env, bank, names, cfg, n_rounds=6, rng=None, verbose=True, runlog=None,
        sched=None):
    """Refine a bank for `n_rounds`, each mechanism on its own cadence."""
    sched = dict(SCHEDULE, **(sched or {}))
    rng = rng or np.random.default_rng(5)
    n_obs = len(names)
    state = {}
    pol_fn = _pol_fn(n_obs, state)
    T, sel, seed, z = cfg["T"], cfg["sel_ep"], cfg["sel_seed"], cfg["z"]
    log, t0 = [], time.time()

    for r in range(n_rounds):
        zn = mem_names(names, bank.get("mem"))
        pol = pol_fn(bank)

        # --- critic: V^pi by fitted value iteration, then Q^pi by probing ----
        vh = fit_vhat(env, pol, n_ep=600, T=T, n_step=20, sweeps=2, seed=21)
        state["vhat"] = vh
        L = subsample(landscape_rollout(env, pol_fn(bank), vh, rng, n_ep=600,
                                        T=T, k=10), cfg["n_on"], rng)
        obs = L["obs"]
        w = L["w"]
        memp = pol_fn(bank)
        memp.reset(len(obs))
        zfn = (lambda o, p=memp: p.z(o, update=False))
        qh, qinfo = fit_qhat(env, pol_fn(bank), L["state"], vh, rng, seed=3,
                             feats=QFeatures(radial=cfg["radial"],
                                             bearing=cfg["bearing"]),
                             zfn=zfn)
        state["qhat"] = qh
        Z = zfn(obs)
        cur = _score(env, bank, sel, T, seed, pol_fn)
        rec = dict(round=r, critic_r2=qinfo["r2"], G_in=float(cur.mean()),
                   moves=[])

        # --- fast: laws by gradient, then thresholds by the boundary term ----
        if r % sched["laws_every"] == 0:
            bank, cur, li = law_step(env, bank, obs, w, qh, cur, mode="grad",
                                     etas=cfg["etas"], n_ep=sel, T=T, seed=seed,
                                     z=z, pol_fn=pol_fn, Z=Z)
            rec["moves"].append("laws" if li["accepted"] else "")
            if runlog:
                for t in li["tries"]:
                    runlog.move("dpg", dict(round=r, **t))
        if r % sched["guards_every"] == 0:
            lo_hi = {j: (float(Z[:, j].min()), float(Z[:, j].max()))
                     for j in range(Z.shape[1])}
            bank, cur, gi = guard_step(env, bank, obs, w, qh, cur, n_try=4,
                                       n_ep=sel, T=T, seed=seed, z=z,
                                       lo_hi=lo_hi, pol_fn=pol_fn, Z=Z)
            if any(x["accepted"] for x in gi):
                rec["moves"].append("threshold")
            if runlog:
                for t in gi:
                    runlog.move("boundary", dict(round=r, **t))

        # --- medium: the discrete jump the gradient cannot make --------------
        if r % sched["library_every"] == 0:
            bank, _, ls = search_laws(env, bank, names, pol_fn, n_ep=sel, T=T,
                                      seed=seed, z=z, zn=zn, verbose=False)
            cur = _score(env, bank, sel, T, seed, pol_fn)
            if any(x["accepted"] for x in ls):
                rec["moves"].append("library")
            if runlog:
                for t in ls:
                    runlog.move("library", dict(round=r, **t))

        # --- slow: structure -------------------------------------------------
        if r % sched["structure_every"] == 0:
            bank, cur, di = drop_arm(env, bank, pol_fn, cur, n_ep=sel, T=T,
                                     seed=seed, z=z)
            bank, cur, ri = reorder(env, bank, pol_fn, cur,
                                    lambda c, ZZ: _match(c, ZZ), Z, n_ep=sel,
                                    T=T, seed=seed, z=z)
            for tag, lg in (("drop", di), ("reorder", ri)):
                if any(x["accepted"] for x in lg):
                    rec["moves"].append(tag)
                if runlog:
                    for t in lg:
                        runlog.move(tag, dict(round=r, **t))

        # --- slow: terminations ---------------------------------------------
        if r % sched["beta_every"] == sched["beta_offset"]:
            bank, bl, ch = search_beta(env, bank, zn, Z, pol_fn, cur_G=cur,
                                       n_ep=sel, T=T, seed=seed, z=z,
                                       verbose=verbose)
            cur = _score(env, bank, sel, T, seed, pol_fn)
            if any(x["accepted"] for x in bl):
                rec["moves"].append("beta")
            if runlog:
                for t in bl:
                    runlog.move("beta", dict(round=r, **t))

        # --- once: memory ----------------------------------------------------
        if r == sched["memory_at"] and cfg.get("memory", True):
            env.seed_kernels(3)
            OB, ST, AL = record(env, pol_fn(bank), np.random.default_rng(3),
                                n_ep=200, T=T)
            bank, ml = discover(env, bank, names, lambda _zn: pol_fn, n_obs,
                                OB, AL, n_thr=cfg.get("mem_thr", 7),
                                screen_ep=cfg.get("mem_screen_ep", 120),
                                confirm_ep=sel, n_confirm=25, T=T, seed=seed,
                                z=z, verbose=verbose)
            cur = _score(env, bank, sel, T, seed, pol_fn)
            if bank.get("mem"):
                rec["moves"].append("memory")
            if runlog:
                for t in ml:
                    runlog.move("memory", dict(round=r, **t))

        rec["G_out"] = float(cur.mean())
        rec["n_arms"] = len(bank["clauses"])
        rec["moves"] = [m for m in rec["moves"] if m]
        log.append(rec)
        if verbose:
            print("  [round %d] %-34s G %6.2f -> %6.2f  %d arms  (critic R2 "
                  "%.2f)  [%.0fs]"
                  % (r, ",".join(rec["moves"]) or "nothing accepted",
                     rec["G_in"], rec["G_out"], rec["n_arms"], qinfo["r2"],
                     time.time() - t0), flush=True)
    return bank, log, state


def _match(clause, Z):
    from .landscape import _match_cols
    return _match_cols(clause, Z)
