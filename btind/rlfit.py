"""The pure-RL pipeline: no expert, no labels, warm-started from what we know.

    round 0..n:  grow arms by rollout   (guards proposed randomly and from the
                                         store; laws by CEM on the parameters)
                 CEM every arm's law
                 memory, once           (what to store, when to write)
                 beta, once laws settle (which arms persist, and until when)

Nothing here calls the planner and nothing fits to labels. The only measurement
anywhere is a rollout, which the fused kernel makes cost 1.2 ms.

WARM START, AND WHAT IT IS NOT. A run begins from the best bank previously
stored FOR THIS WORLD -- same controller, same guards, same laws -- and every
move it then makes still has to clear its own paired test. That is resumption,
not cheating: nothing is credited to the search that the search did not win.
What the store also supplies is the guards earlier runs ACCEPTED, and those
enter the candidate pool as PROPOSALS beside the random ones, because a clause's
value depends on the arms above it and importing one imports a number measured
against a tree that no longer exists.

THE WORLD IS THE KEY. Banks are indexed by the environment's full parameter
signature, so a controller grown on the masked world is never resumed on the
unmasked one. They share a class name and nothing else.
"""
import time

import numpy as np

from . import store as ST
from .betasearch import search_beta
from .grow_bt import grow
from .lawcem import cem_law, improve_laws
from .memory import MemBank, emit, mem_names
from .memsearch import discover as mem_discover, record
from .structure import drop_arm, reorder, score

DEFAULTS = dict(
    n_ep=400, T=400, seed=777, z=2.0, min_gain=0.3,
    grow_pool=60, grow_arms=4, max_arity=3, min_n=400,
    cem_iter=12, cem_K=96, cem_sigma=0.35,
    mem_at=1, mem_thr=7, beta_at=2, n_cover=6000,
)


def _cover(env, n, rng):
    """States for proposing guards on: teleported, for breadth.

    This is a simulator affordance, not expert knowledge -- no one labels them.
    On-policy rows alone would define the alphabet on the distribution the
    CURRENT controller induces, which is exactly the coverage problem DAgger
    was built to avoid.
    """
    return env.sample_states(n, rng)


def fit(env, names, rounds=3, warm=True, cfg=None, rng=None, verbose=True,
        tag="rlfit"):
    cfg = dict(DEFAULTS, **(cfg or {}))
    rng = rng or np.random.default_rng(0)
    n_obs = len(names)
    zn0 = mem_names(names, None)
    pol_fn = lambda b: MemBank(b, n_obs)
    t0 = time.time()

    bank, meta = (ST.best(env) if warm else (None, None))
    if bank is not None:
        bank["names"] = list(names)
        if verbose:
            print("  warm start: %d arms, stored G %.2f (%s)"
                  % (len(bank["clauses"]), meta["G"], meta["tag"]), flush=True)
    else:
        th, _ = cem_law(env, dict(clauses=[], laws=[],
                                  default=rng.normal(0, .3, (len(zn0) + 1, 2)),
                                  names=list(names), laws_on_z=True),
                        -1, pol_fn, n_iter=cfg["cem_iter"], K=cfg["cem_K"],
                        sigma0=cfg["cem_sigma"], n_ep=cfg["n_ep"], T=cfg["T"],
                        seed=cfg["seed"], rng=rng)
        bank = dict(clauses=[], laws=[], default=th, names=list(names),
                    laws_on_z=True)
        if verbose:
            print("  cold start: default law by CEM, G %.2f"
                  % score(env, bank, pol_fn, cfg["n_ep"], cfg["T"],
                          cfg["seed"]).mean(), flush=True)

    seeds = ST.clauses(env) if warm else []
    if verbose and seeds:
        print("  store offers %d accepted clauses as proposals" % len(seeds),
              flush=True)

    log = []
    for r in range(rounds):
        zn = mem_names(names, bank.get("mem"))
        obs = env.observe(_cover(env, cfg["n_cover"], np.random.default_rng(r)))
        p = pol_fn(bank)
        p.reset(len(obs))
        Z = p.z(obs, update=False)
        cur = score(env, bank, pol_fn, cfg["n_ep"], cfg["T"], cfg["seed"])
        rec = dict(round=r, G_in=float(cur.mean()), moves=[])

        bank, glog = grow(env, bank, names, zn, pol_fn, obs, Z,
                          max_arms=cfg["grow_arms"], pool=cfg["grow_pool"],
                          max_arity=cfg["max_arity"], min_n=cfg["min_n"],
                          min_gain=cfg["min_gain"], labels=None, qhat=None,
                          seed_clauses=seeds if r == 0 else None,
                          screen_ep=cfg["n_ep"], confirm_ep=cfg["n_ep"],
                          n_confirm=10, T=cfg["T"], seed=cfg["seed"],
                          z=cfg["z"], rng=rng, use_library=False,
                          verbose=verbose)
        bank, cur, llog = improve_laws(env, bank, pol_fn, cur=None,
                                       n_ep=cfg["n_ep"], T=cfg["T"],
                                       seed=cfg["seed"], z=cfg["z"],
                                       min_gain=cfg["min_gain"], rng=rng,
                                       n_iter=cfg["cem_iter"], K=cfg["cem_K"],
                                       sigma0=cfg["cem_sigma"], verbose=verbose)
        if any(l["accepted"] for l in llog):
            rec["moves"].append("cem-laws")

        if r == cfg["mem_at"]:
            env.seed_kernels(3)
            OB, _, AL = record(env, pol_fn(bank), np.random.default_rng(3),
                               n_ep=200, T=cfg["T"])
            bank, mlog = mem_discover(env, bank, names, lambda _z: pol_fn,
                                      n_obs, OB, AL, n_thr=cfg["mem_thr"],
                                      screen_ep=cfg["n_ep"],
                                      confirm_ep=cfg["n_ep"], n_confirm=25,
                                      T=cfg["T"], seed=cfg["seed"],
                                      z=cfg["z"], verbose=verbose)
            if bank.get("mem"):
                rec["moves"].append("memory")
            cur = score(env, bank, pol_fn, cfg["n_ep"], cfg["T"], cfg["seed"])

        if r == cfg["beta_at"]:
            zn = mem_names(names, bank.get("mem"))
            p = pol_fn(bank)
            p.reset(len(obs))
            bank, blog, _ = search_beta(env, bank, zn, p.z(obs, update=False),
                                        pol_fn, cur_G=cur, n_ep=cfg["n_ep"],
                                        T=cfg["T"], seed=cfg["seed"],
                                        z=cfg["z"], verbose=verbose)
            if any(b["accepted"] for b in blog):
                rec["moves"].append("beta")

        bank, cur, _ = drop_arm(env, bank, pol_fn, cur, n_ep=cfg["n_ep"],
                                T=cfg["T"], seed=cfg["seed"], z=cfg["z"])
        rec["G_out"] = float(score(env, bank, pol_fn, cfg["n_ep"], cfg["T"],
                                   cfg["seed"]).mean())
        rec["n_arms"] = len(bank["clauses"])
        log.append(rec)
        if verbose:
            print("  [round %d] %-26s G %6.2f -> %6.2f  %d arms  [%.0fs]"
                  % (r, ",".join(rec["moves"]) or "-", rec["G_in"],
                     rec["G_out"], rec["n_arms"], time.time() - t0), flush=True)

    from .pipeline import evaluate_bank
    m = evaluate_bank(env, bank, n_obs=n_obs)
    ST.save(env, bank, m, names=mem_names(names, bank.get("mem")), tag=tag)
    if verbose:
        print("  held-out G %.2f +-%.2f  pickups %.2f -- stored"
              % (m["G"], m["ci"], m["eaten"]), flush=True)
    return bank, log, m
