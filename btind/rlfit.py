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

from . import explore as EX
from . import proposal as PR
from . import store as ST
from .betasearch import search_beta
from .grow_bt import failure_states, grow
from .lawcem import cem_law, improve_laws
from .memory import MemBank, emit, mem_names
from .memsearch import discover as mem_discover, record
from .escape import kick
from .stepsearch import search_fails, search_steps
from .structure import (absorb_universal, collapse_bottom, drop_arm,
                         polish_thresholds, reorder, score, simplify)

DEFAULTS = dict(
    n_ep=400, T=400, seed=777, z=2.0, min_gain=0.3,
    grow_pool=60, grow_arms=4, max_arity=3, min_n=400,
    cem_iter=12, cem_K=96, cem_sigma=0.35,
    mem_at=1, mem_thr=7, mem_refine=3, beta_at=2, steps_at=2, n_cover=6000,
    subtree_at=(1, 2), subtree_arms=2, subtree_pool=30,
    explore_ep=300, explore_ks=(1, 3, 8), explore_frac=0.25,
    # kernel-interpolation leaves (`kernsearch.py`): rounds, and its settings
    kern_at=(), kern_cfg=None, critic=False, prior=None,
    stall_before_kick=2, kick_size=1, hop_budget=2,
    seed_stride=1009, val_seed=90210, val_ep=1200,
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
        tag="rlfit", run_seed=None, branch=None, transfer_from=None, init_bank=None):
    """One run. `run_seed` varies the PROPOSALS, never the evaluation.

    The rollout seed stays fixed so every candidate in a run is compared on
    identical episodes; what varies between runs is which candidates get
    proposed. Without that, a warm-started run replays itself exactly -- two
    masked runs returned the same tree to the decimal.
    """
    cfg = dict(DEFAULTS, **(cfg or {}))
    # THE HEAD COMES FROM THE WORLD, not from a flag. A world that offers a
    # discrete action set says so (`n_act`), and the law simply gets that many
    # outputs instead of two. Everything downstream -- CEM, grow, simplify,
    # thresholds, beta -- is indifferent, because it scores theta by rollout.
    n_act = int(getattr(env, "n_act", 2))
    # A world that names its head wins over the count: the intersection's
    # signal has exactly two actions, and "two outputs means a heading" would
    # have normalised EXTEND/SWITCH into a unit vector.
    head = getattr(env, "head", None) or ("argmax" if n_act != 2 else "vector")
    run_seed = int(np.random.SeedSequence().entropy % 2**31 if run_seed is None
                   else run_seed)
    rng = rng or np.random.default_rng(run_seed)
    n_obs = len(names)
    zn0 = mem_names(names, None)
    pol_fn = lambda b: MemBank(b, n_obs)
    t0 = time.time()

    # BRANCH, don't always resume the incumbent: a monotone search restarted at
    # its own optimum has nowhere to go. `branch` picks how far down the kept
    # population to start; None means occasionally take a lower-ranked bank.
    if branch is None:
        branch = int(rng.integers(0, 3)) if rng.random() < 0.35 else 0
    # AN EXPLICIT START WINS: a stage cycle resumes each agent from the bank it
    # left, not from the best stored score -- scores stored in earlier cycles
    # were measured against partners that no longer exist
    bank, meta = ((init_bank, dict(G=float("nan"), tag="init_bank"))
                  if init_bank is not None else
                  (ST.best(env, rank=branch) if warm else (None, None)))
    if bank is None and transfer_from is not None:
        bank, meta = ST.transfer(transfer_from, env)
        if bank is not None and verbose:
            print("  transferred a bank from another world: %d arms, it scored "
                  "%.2f there" % (len(bank["clauses"]), meta["G"]), flush=True)
    if bank is not None:
        bank["names"] = list(names)
        from .memory import upgrade_layout
        bank = upgrade_layout(bank, names)
        if verbose:
            print("  warm start: %d arms, stored G %.2f (%s, rank %d) | "
                  "run_seed %d" % (len(bank["clauses"]), meta["G"],
                                   meta["tag"], branch, run_seed), flush=True)
    else:
        # The null for a discrete head is every action tied at zero: it prefers
        # nothing, so CEM starts from no opinion about which manoeuvre is good.
        # A random init would already be an opinion, and on this world the
        # constant actions span 20.2 to 25.9, so a lucky draw is worth points
        # the search would then appear to have found.
        if head == "argmax":
            th0 = np.zeros((len(zn0) + 1, n_act))
        elif head in ("scalar", "duration"):
            # the null command for a continuous leaf: the middle of the range
            th0 = np.zeros((len(zn0) + 1, 1))
            lo, hi = env.u_range
            th0[-1, 0] = getattr(env, "u_null", 0.5 * (lo + hi))
        else:
            th0 = rng.normal(0, .3, (len(zn0) + 1, 2))
        seed_bank = dict(clauses=[], laws=[], default=th0, names=list(names),
                         laws_on_z=True, head=head, n_act=n_act,
                         actions=list(getattr(env, "actions", []) or []) or None)
        if head in ("scalar", "duration"):
            seed_bank["u_range"] = tuple(env.u_range)
        if cfg.get("prior"):
            seed_bank["prior"] = cfg["prior"]
        th, _ = cem_law(env, seed_bank, -1, pol_fn, n_iter=cfg["cem_iter"],
                        K=cfg["cem_K"], sigma0=cfg["cem_sigma"],
                        n_ep=cfg["n_ep"], T=cfg["T"], seed=cfg["seed"], rng=rng)
        bank = dict(seed_bank, default=th)
        if verbose:
            print("  cold start: default law by CEM, G %.2f"
                  % score(env, bank, pol_fn, cfg["n_ep"], cfg["T"],
                          cfg["seed"]).mean(), flush=True)

    weights = PR.load(env, len(mem_names(names, None)) + 8)
    weights.n = len(mem_names(names, None)) + 8
    if verbose:
        top = weights.top(mem_names(names, None) + ["mem"] * 8, k=5)
        if any(t[3] for t in top):
            print("  learned proposal weights: %s"
                  % ", ".join("%s %.2f(%d/%d)" % t for t in top), flush=True)
    # A STORED CLAUSE MAY NOT FIT THIS BANK. The store keeps guards from every
    # bank for this world, including ones that had a blackboard, and those
    # reference memory columns that do not exist in a bank without one --
    # measured, a `have_mem` literal at column 29 proposed into a 27-wide
    # layout. Seeds are filtered to the layout they are being proposed into.
    _w0 = len(mem_names(names, bank.get("mem")))
    seeds = ([c for c in ST.clauses(env) if all(l[0] < _w0 for l in c)]
             if warm else [])
    if verbose and seeds:
        print("  store offers %d accepted clauses as proposals" % len(seeds),
              flush=True)

    # THE SEARCH WAS CLIMBING ONE SAMPLE. Every `accept` in this project is a
    # paired test on the SAME episodes -- common random numbers across the pair,
    # which is what makes the difference low-variance and is right. What was
    # wrong is that the sample never changed: one masked run put roughly fifty
    # thousand proposals through a single draw of 600 episodes at seed 777, and
    # the tree that came out scored 19.53 there against 12.98 on held-out seeds.
    # That 6.5 is not noise, it is the selection bias of choosing thousands of
    # times against one sample, and it made every intermediate number the search
    # believed too high.
    #
    # TWO CHANGES, and they do different jobs.
    #
    #   ROTATE THE TRAINING SEED PER ROUND. CRN is preserved where it matters --
    #   inside a comparison, where cand and incumbent still share episodes -- but
    #   no single draw is climbed indefinitely. A move that only helped on round
    #   0's episodes has to survive round 1's, and the incumbent is re-scored on
    #   the new sample before anything is priced against it.
    #
    #   KEEP ONE SEED THE SEARCH NEVER OPTIMISES AGAINST. The incumbent is
    #   tracked on it, and the revert at the end selects on it. Rotation alone
    #   still lets the best-of-N over rounds be an optimistic pick; choosing the
    #   bank to return on a sample that was never used to accept a move is what
    #   makes the returned number mean what it says.
    def val(b):
        return float(score(env, b, pol_fn, cfg["val_ep"], cfg["T"],
                           cfg["val_seed"]).mean())

    log = []
    best_bank, best_V, stalled, hop_left = bank, None, 0, 0
    for r in range(rounds):
        rseed = cfg["seed"] + cfg["seed_stride"] * r
        zn = mem_names(names, bank.get("mem"))
        # ANCHOR ON FAILURES. The quantile alphabet built from coverage states
        # cannot express the thresholds that matter: d_threat is uniform on
        # [0.08, 0.80] there, so its 5% quantile is 0.11 while the region worth
        # +5.87 is `d_threat <= 0.09`. Observations from the steps before a
        # death put the candidate thresholds where the trouble is.
        if hasattr(env, "coverage_rows"):
            # A MULTI-AGENT WORLD HAS NO OBSERVATION AT A START STATE -- every
            # vehicle is still pending -- so coverage and failure rows both come
            # from traced rollouts of the agent under search (`coverage_rows`).
            cov, fail = env.coverage_rows(bank, n_ep=cfg.get("cover_ep", 150),
                                          seed=r)
            if len(cov) > cfg["n_cover"]:
                cov = cov[np.random.default_rng(r).choice(len(cov), cfg["n_cover"],
                                                          replace=False)]
        else:
            cov = env.observe(_cover(env, cfg["n_cover"], np.random.default_rng(r)))
            fail = failure_states(env, bank, pol_fn, n_ep=300, T=cfg["T"])
        # EXPLORE BY COUNTERFACTUAL. A random action, or a random manoeuvre of a
        # few ticks, taken at a random tick of the tree's own episode and paired
        # against the undeviated episode -- an exact advantage on this world
        # (`explore.py`). The rows where a deviation paid most join the failure
        # rows as anchors: they are where the tree measurably leaves return on
        # the table, which is the on-policy complement of where it dies.
        ex = (EX.deviations(env, bank, n_ep=cfg["explore_ep"], T=cfg["T"],
                            seed=rseed, rng=rng, ks=cfg["explore_ks"])
              if cfg["explore_ep"] else None)
        ex_obs = (ex["z0"][EX.hot_rows(ex, frac=cfg["explore_frac"]), :n_obs]
                  if ex is not None else np.zeros((0, n_obs)))
        anchors = [a for a in (fail, ex_obs) if len(a)]
        obs = np.vstack([cov] + anchors) if anchors else cov
        hot = np.arange(len(cov), len(obs))
        if verbose and ex is not None:
            sm = EX.summary(ex, actions=bank.get("actions"))
            print("  explore: %d deviations, %.0f%% paid, mean %+.2f, best %+.2f%s"
                  % (sm["n"], 100 * sm["frac_positive"], sm["mean_adv"],
                     sm["best"],
                     ("; by k " + " ".join("%d:%+.2f" % (k, v["mean"])
                                          for k, v in sm["by_k"].items()))),
                  flush=True)
        p = pol_fn(bank)
        p.reset(len(obs))
        Z = p.z(obs, update=False)
        cur = score(env, bank, pol_fn, cfg["n_ep"], cfg["T"], rseed)
        v_in = val(bank)
        # TIES GO TO THE NEWER BANK. Capturing the incumbent only on a strict
        # improvement kept a seven-arm tree over the six-arm one a later round
        # had simplified it into at equal return -- and the seven-arm one still
        # carried the universal guard that absorb had just removed, so the
        # revert at the end undid the repair. Equal return, fewer arms, later in
        # the search: the newer bank is the better incumbent.
        if best_V is None or v_in >= best_V:
            best_bank, best_V = bank, v_in
        # THE HOP HAPPENS AFTER A STALLED ROUND, not on a schedule. A round
        # that accepted something has not run out of monotone moves yet, and
        # kicking it would throw away progress the test had already bought.
        if stalled >= cfg["stall_before_kick"]:
            # PROPOSE FROM THE INCUMBENT, always. A hop starts where the search
            # actually is, not where the last abandoned hop left it.
            bank, why = kick(best_bank, Z, rng, n=cfg["kick_size"])
            hop_left = cfg["hop_budget"]
            cur = score(env, bank, pol_fn, cfg["n_ep"], cfg["T"], rseed)
            if verbose:
                print("  [round %d] stalled %d rounds -- kick: %s  (G %.2f -> "
                      "%.2f, to be re-optimised)"
                      % (r, stalled, why, best_V, cur.mean()), flush=True)
            stalled = 0
            rec_kick = why
        else:
            rec_kick = None
        rec = dict(round=r, G_in=float(cur.mean()), V_in=v_in, seed=rseed,
                   moves=[], kick=rec_kick)

        import btind.grow_bt as _GB
        _cp = _GB._clause_pool
        _GB._clause_pool = (lambda rg, al, ZZ, _h, pl, ma, la, sd, *a,
                            _hot=hot, **kw:
                            _cp(rg, al, ZZ, _hot if len(_hot) else None, pl, ma,
                                la, sd, *a, **kw))
        seeds = [c for c in seeds if all(l[0] < Z.shape[1] for l in c)]
        bank, glog = grow(env, bank, names, zn, pol_fn, obs, Z,
                          max_arms=cfg["grow_arms"], pool=cfg["grow_pool"],
                          max_arity=cfg["max_arity"], min_n=cfg["min_n"],
                          min_gain=cfg["min_gain"], labels=None, qhat=None,
                          seed_clauses=seeds if r == 0 else None,
                          screen_ep=cfg["n_ep"], confirm_ep=cfg["n_ep"],
                          n_confirm=10, T=cfg["T"], seed=rseed,
                          z=cfg["z"], rng=rng, use_library=False,
                          weights=weights, verbose=verbose)
        weights.update_many(glog)
        _GB._clause_pool = _cp
        bank, cur, _ = simplify(env, bank, pol_fn, n_ep=cfg["n_ep"],
                                T=cfg["T"], seed=rseed, z=cfg["z"],
                                names=zn, verbose=verbose)
        # BEFORE ANYTHING IS APPENDED, make sure the end of the tree is
        # reachable. A guard that matches every state turns its arm into a
        # second default and every arm below it into dead code -- which is
        # where the memory search puts its arm.
        bank, cur, absorbed = absorb_universal(env, bank, pol_fn, Z, cur_G=cur,
                                               n_ep=cfg["n_ep"], T=cfg["T"],
                                               seed=rseed, z=cfg["z"],
                                               names=zn, verbose=verbose)
        if absorbed:
            rec["moves"].append("absorb")
        # A GUARD THAT IS A PASSENGER GOES. The bottom arm's law is offered
        # as the default; non-inferior means the split was buying nothing.
        # Repeats while it keeps succeeding, because a tree can carry several.
        for _ in range(len(bank["clauses"])):
            bank, cur, did = collapse_bottom(env, bank, pol_fn, cur_G=cur,
                                             n_ep=cfg["n_ep"], T=cfg["T"],
                                             seed=rseed, z=cfg["z"], names=zn,
                                             verbose=verbose)
            if not did:
                break
            rec["moves"].append("collapse")
        bank, cur, llog = improve_laws(env, bank, pol_fn, cur=cur,
                                       n_ep=cfg["n_ep"], T=cfg["T"],
                                       seed=rseed, z=cfg["z"],
                                       min_gain=cfg["min_gain"], rng=rng,
                                       n_iter=cfg["cem_iter"], K=cfg["cem_K"],
                                       sigma0=cfg["cem_sigma"], verbose=verbose)
        if any(l["accepted"] for l in llog):
            rec["moves"].append("cem-laws")
        bank, cur, plog = polish_thresholds(env, bank, pol_fn, Z, cur_G=cur,
                                            n_ep=cfg["n_ep"], T=cfg["T"],
                                            seed=rseed, z=cfg["z"],
                                            names=zn, verbose=verbose)
        if any(x["accepted"] for x in plog):
            rec["moves"].append("thresholds")

        # --- kernel-interpolation leaves: inducing points on the busiest laws --
        # Anchored where deviations paid, columns chosen by rollout, tuned by
        # CEM and pruned (`kernsearch.py`). After the affine laws and thresholds
        # have settled, because a point refines the law it sits on.
        if r in cfg["kern_at"]:
            from .kernsearch import search_kernels
            from . import kernlaw as KL
            zn = mem_names(names, bank.get("mem"))
            n_before = sum(KL.n_points(k) for k in [bank.get("kern_default")] + [
                x for s_ in (bank.get("kerns") or []) for x in (s_ or [])])
            # THE CRITIC IS REFITTED ON THE BANK IT ADVISES: V-hat and A-hat
            # of this round's tree (`intersection_critic.py`), reported on
            # held-out data, proposing points beside the deviations
            critic = None
            if cfg.get("critic") and hasattr(env, "coverage_rows"):
                from .intersection_critic import make_critic
                critic, crep = make_critic(env, bank, seed=rseed, verbose=verbose)
                rec["critic"] = crep
            bank, cur, klog = search_kernels(
                env, bank, names, zn, pol_fn, cur, cfg["T"], rseed, z=cfg["z"],
                min_gain=cfg["min_gain"], n_ep=cfg["n_ep"], rng=rng,
                verbose=verbose, critic=critic, anchors=fail if len(fail) else None,
                **(cfg["kern_cfg"] or {}))
            n_after = sum(KL.n_points(k) for k in [bank.get("kern_default")] + [
                x for s_ in (bank.get("kerns") or []) for x in (s_ or [])])
            if any(e.get("accepted") and e["op"] != "add" for e in klog) or n_after != n_before:
                rec["moves"].append("kernels")
            rec["kernel_sources"] = {s: [sum(1 for e in klog if e.get("op") == "add"
                                             and e.get("src") == s),
                                         sum(1 for e in klog if e.get("op") == "add"
                                             and e.get("src") == s and e["accepted"])]
                                     for s in ("dev", "critic", "critic-set", "joint", "anchor", "bound")}

        # THE MEMORY STAGE IS NOT READY FOR A DISCRETE HEAD, and it says so
        # rather than producing a shaped-wrong law and a silent wrong answer.
        # `memory_primitives` builds `mem_c - c` as a 2-output direction, which
        # is meaningful for a holonomic agent and meaningless as a preference
        # over named manoeuvres. The discrete analogue -- score each action by
        # the gap between a remembered value and the current one -- is the next
        # increment, not a thing to improvise mid-run.
        if r == cfg["mem_at"] and head != "vector" and not bank.get("mem"):
            # THE DISCRETE-HEAD MEMORY STAGE: a message-shaped blackboard --
            # a transient column, its arrival as the event, with or without a
            # countdown -- and the arms that read it, grown on the widened
            # layout and priced against the memoryless tree (`memtransient`).
            from .memtransient import discover_transient
            if hasattr(env, "record_traces"):
                OB, AL = env.record_traces(bank, n_ep=cfg.get("record_ep", 80),
                                           seed=3)
            else:
                env.seed_kernels(3)
                OB, _, AL = record(env, pol_fn(bank), np.random.default_rng(3),
                                   n_ep=200, T=cfg["T"])
            bank, mlog = discover_transient(
                env, bank, names, pol_fn, OB, AL, n_obs,
                screen_ep=min(cfg["n_ep"], 120), confirm_ep=cfg["n_ep"],
                T=cfg["T"], seed=rseed, z=cfg["z"],
                min_gain=max(cfg["min_gain"], 0.5), rng=rng, weights=weights,
                verbose=verbose, pool=cfg.get("mem_pool", 40),
                max_arms=cfg.get("mem_arms", 2))
            if bank.get("mem"):
                rec["moves"].append("memory")
            cur = score(env, bank, pol_fn, cfg["n_ep"], cfg["T"], rseed)
        elif r == cfg["mem_at"] and head != "vector":
            pass                                  # a blackboard already exists
        elif r == cfg["mem_at"]:
            env.seed_kernels(3)
            OB, _, AL = record(env, pol_fn(bank), np.random.default_rng(3),
                               n_ep=200, T=cfg["T"])
            bank, mlog = mem_discover(env, bank, names, lambda _z: pol_fn,
                                      n_obs, OB, AL, n_thr=cfg["mem_thr"],
                                      screen_ep=cfg["n_ep"],
                                      confirm_ep=cfg["n_ep"], n_confirm=25,
                                      T=cfg["T"], seed=rseed,
                                      z=cfg["z"], verbose=verbose,
                                      n_refine=cfg["mem_refine"],
                                      weights=weights, rng=rng)
            if bank.get("mem"):
                rec["moves"].append("memory")
            cur = score(env, bank, pol_fn, cfg["n_ep"], cfg["T"], rseed)

        if r == cfg["beta_at"]:
            zn = mem_names(names, bank.get("mem"))
            p = pol_fn(bank)
            p.reset(len(obs))
            bank, blog, _ = search_beta(env, bank, zn, p.z(obs, update=False),
                                        pol_fn, cur_G=cur, n_ep=cfg["n_ep"],
                                        T=cfg["T"], seed=rseed,
                                        z=cfg["z"], verbose=verbose)
            if any(b["accepted"] for b in blog):
                rec["moves"].append("beta")
            cur = score(env, bank, pol_fn, cfg["n_ep"], cfg["T"], rseed)

        # --- steps and failure conditions: the Sequence and the status --------
        # After beta, on the same round: a step lives inside a latch, so the
        # arms have to have been offered stickiness first. One step and one fail
        # clause per round at most, each priced on the tree the other left.
        if r == cfg["steps_at"]:
            zn = mem_names(names, bank.get("mem"))
            p = pol_fn(bank)
            p.reset(len(obs))
            Zs = p.z(obs, update=False)
            bank, slog = search_steps(env, bank, zn, Zs, pol_fn, cur_G=cur,
                                      screen_ep=min(cfg["n_ep"], 120),
                                      confirm_ep=cfg["n_ep"], T=cfg["T"],
                                      seed=rseed, z=cfg["z"],
                                      min_gain=cfg["min_gain"], rng=rng,
                                      weights=weights, verbose=verbose)
            if any(s["accepted"] for s in slog):
                rec["moves"].append("steps")
                cur = score(env, bank, pol_fn, cfg["n_ep"], cfg["T"], rseed)
            bank, flog = search_fails(env, bank, zn, Zs, pol_fn, cur_G=cur,
                                      n_ep=cfg["n_ep"], T=cfg["T"], seed=rseed,
                                      z=cfg["z"], min_gain=cfg["min_gain"],
                                      rng=rng, weights=weights, verbose=verbose)
            if any(f["accepted"] for f in flog):
                rec["moves"].append("fail")
                cur = score(env, bank, pol_fn, cfg["n_ep"], cfg["T"], rseed)
            weights.update_many(slog + flog)

        # --- nested subtrees: grow children INSIDE an existing child ---------
        # On that child's own rows, with an alphabet fitted there and its law
        # as the starting point (`subtree.py`). After steps and terminations,
        # because a latched or multi-step child is not offered as a parent.
        if r in cfg["subtree_at"]:
            from .subtree import search_subtrees
            zn = mem_names(names, bank.get("mem"))
            p = pol_fn(bank)
            p.reset(len(obs))
            Zt = p.z(obs, update=False)
            bank, tlog = search_subtrees(
                env, bank, names, zn, pol_fn, obs, Zt, verbose=verbose,
                max_arms=cfg["subtree_arms"], pool=cfg["subtree_pool"],
                min_gain=cfg["min_gain"], screen_ep=min(cfg["n_ep"], 120),
                confirm_ep=cfg["n_ep"], T=cfg["T"], seed=rseed, z=cfg["z"],
                rng=rng, weights=weights, cem_iter=max(2, cfg["cem_iter"] // 3),
                cem_K=max(12, cfg["cem_K"] // 4))
            if any(e.get("accepted") for e in tlog):
                rec["moves"].append("subtree")
                cur = score(env, bank, pol_fn, cfg["n_ep"], cfg["T"], rseed)
            weights.update_many(tlog)

        bank, cur, _ = drop_arm(env, bank, pol_fn, cur, n_ep=cfg["n_ep"],
                                T=cfg["T"], seed=rseed, z=cfg["z"])
        rec["G_out"] = float(score(env, bank, pol_fn, cfg["n_ep"], cfg["T"],
                                   rseed).mean())
        rec["n_arms"] = len(bank["clauses"])
        rec["V_out"] = val(bank)
        if best_V is None or rec["V_out"] >= best_V:
            best_bank, best_V = bank, rec["V_out"]
            hop_left = 0
        elif hop_left:
            hop_left -= 1
            if not hop_left:
                # THE HOP IS ABANDONED, not merely noted. Without this the run
                # continues from whatever the kick left and spends every
                # remaining round climbing back towards a bank it already had.
                if verbose:
                    print("  hop did not pay within %d rounds (%.2f < %.2f)"
                          " -- back to the incumbent"
                          % (cfg["hop_budget"], rec["V_out"], best_V),
                          flush=True)
                bank = best_bank
                rec["moves"].append("hop-abandoned")
        stalled = 0 if rec["moves"] else stalled + 1
        log.append(rec)
        # CHECKPOINT. A run killed in round 4 used to leave nothing behind, and
        # the next run rediscovered everything it had. Every round's best bank
        # goes to the store with its validation-seed score, under this run's
        # tag, so a resume (warm start, rank 0) picks up where this left off.
        ST.save(env, best_bank, dict(G=best_V), tag="%s@r%d" % (tag, r),
                names=mem_names(names, best_bank.get("mem")))
        PR.save(env, weights)
        if verbose:
            print("  [round %d] %-26s G %6.2f -> %6.2f  val %6.2f (gap %+.2f)"
                  "  %d arms  [%.0fs]"
                  % (r, ",".join(rec["moves"]) or "-", rec["G_in"],
                     rec["G_out"], rec["V_out"], rec["G_out"] - rec["V_out"],
                     rec["n_arms"], time.time() - t0), flush=True)

    # A HOP IS KEPT ONLY IF IT PAID. Reverting here is what preserves the
    # monotone guarantee at the outer level: the run can never return a
    # controller worse than the best one it held, however the kicks landed.
    final = val(bank)
    if best_V is not None and final < best_V:
        if verbose:
            print("  final bank is worse on the validation seed (%.2f < %.2f)"
                  " -- reverting to the best bank held" % (final, best_V),
                  flush=True)
        bank = best_bank

    from .structure import heldout
    m = None
    if hasattr(env, "coverage_rows"):
        m = heldout(env, bank, n_ep=1000, T=cfg["T"])
    if m is None:
        from .pipeline import evaluate_bank
        m = evaluate_bank(env, bank, n_obs=n_obs, T=cfg["T"])
    ST.save(env, bank, m, names=mem_names(names, bank.get("mem")), tag=tag)
    PR.save(env, weights)
    if verbose:
        print("  held-out G %.2f +-%.2f%s -- stored"
              % (m["G"], m["ci"], ("  pickups %.2f" % m["eaten"]) if "eaten" in m
                 else ""), flush=True)
    return bank, log, m


def critic_quality(env, bank, names, rng=None, n=300, h=15):
    """How well a fitted critic agrees with a re-probed reference field.

    Returns (median cosine, ceiling). The ceiling is the reference's agreement
    with itself, so a score near it means the critic is as good as the
    measurement allows and a score near zero means the gradient is noise. Call
    it before deciding whether gradient steps belong in the loop.
    """
    from .critic import QFeatures, fit_qhat, grad_score, local_q_grad
    from .landscape import landscape_rollout, subsample
    from .vhat import fit_vhat
    rng = rng or np.random.default_rng(5)
    n_obs = len(names)
    pol_fn = lambda b: MemBank(b, n_obs)
    vh = fit_vhat(env, pol_fn(bank), n_ep=400, T=400, n_step=20, sweeps=2,
                  seed=21)
    L = subsample(landscape_rollout(env, pol_fn(bank), vh, rng, n_ep=400,
                                    T=400, k=10), 8000, rng)
    S = L["state"][:n]
    p = pol_fn(bank)
    p.reset(len(S))
    u = p.act(env.observe(S))
    g1 = local_q_grad(env, pol_fn(bank), S, u, vh, np.random.default_rng(1),
                      K=96, n_rep=8, h=h, seed=11)
    g2 = local_q_grad(env, pol_fn(bank), S, u, vh, np.random.default_rng(2),
                      K=96, n_rep=8, h=h, seed=77)
    qh, info = fit_qhat(env, pol_fn(bank), L["state"], vh, rng, seed=3, h=h,
                        n_rep=3)
    sc = grad_score(qh.grad_u(env.observe(S), u), 0.5 * (g1 + g2))
    return dict(median_cos=sc["cos_med"], agree=sc["agree"], r2=info["r2"],
                ceiling=grad_score(g1, g2)["cos_med"], qhat=qh, vhat=vh)
