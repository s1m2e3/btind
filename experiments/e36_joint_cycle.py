r"""e36 -- both trees trained in ONE loop, on ONE objective, over the condition
distribution: joint rounds instead of alternating stages.

WHAT CHANGES FROM e35, and why each part changed:

    ONE OBJECTIVE. Every move -- a car's and the signal's -- is accepted on the
    TEAM return (`env.reward_as("team")`), so the two agents' gains are the same
    quantity and a move that helps one at the other's expense cannot be bought.
    IN PRACTICE THIS CHANGES THE VEHICLE ONLY: `own_reward` is already "team"
    for the signal, so e35's signal stage was on the team return too. What was
    per-car, and is now team at acceptance, is the vehicle's.

    SO THE TEAM RETURN IS NOT WHAT COLLAPSED e35's SIGNAL. That move -- a flat
    5.64 s green, worth -770 on the target distribution where the same vehicle
    tree with the plain fixed plan is worth -626 -- was already priced on the
    team return. It was priced on the team return AT ITS OWN RUNG, 16-50 veh/h,
    where cutting both left phases to the 5 s minimum really is free. Sharing
    the objective does not fix a move that is correct in the demand it was
    measured in and wrong in the demand that counts.

    WHAT FIXES IT IS PER-AGENT CREDIT ON THE TARGET DISTRIBUTION (`credit`).
    After each agent's pass, that pass alone is re-priced on unseen episodes of
    the target distribution with THE PARTNER HELD FIXED -- a paired difference
    that is attributable to that agent and to nothing else -- and a pass that
    loses ground there is reverted on its own. A pair-level revert cannot do
    this: in e35's cycle 3 the signal's collapse and a large vehicle gain landed
    in the same cycle, and the pair improved (-1385 -> -770), so a test on the
    pair sees a good cycle and keeps the flat green inside it.

    PER-AGENT CREDIT FOR PROPOSALS. The vehicle's deviations and both estimators
    still measure its OWN per-car return, with collisions charged to the car at
    fault: an exact counterfactual is only informative in the units the agent is
    responsible for. Only acceptance is shared (`kernsearch._own_reward`,
    `intersection_critic._own`).

    NO STALE PARTNER. A round is one pass of the vehicle tree and one of the
    signal, each against the other AS IT IS NOW -- not a population standing in
    for a partner that has moved on. Populations were e35's patch for staging;
    joint rounds do not need them.

    CENTRALISED CRITICS, DECENTRALISED TREES. Each agent keeps its own V-hat and
    A-hat; they see the episode's demand and the partner index, the trees never
    do. A critic whose rank correlation, top-decile lift and sign accuracy all
    say noise is not asked for proposals that round.

THE CURRICULUM is e35's: demand widens a rung when a round accepts nothing for
both agents twice in a row, or after `per_rung` rounds. Episodes per test scale
with the rung, so every rung gets the same simulation time.

THE BEST PAIR, by the team return on unseen episodes of the target
distribution, is stored after every round (`runs/e36_best.json`) and a round
that falls more than two standard errors below it is reverted -- both trees.

RESUMABLE: `runs/e36_state.json` after every round.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btind.envs.intersection import WIDE, IntersectionBatch
from btind.rlfit import fit
from btind.memory import emit
from btind.runlog import RUNS, bank_from_json, bank_json

from btind.envs import intersection_fast as IF

from experiments.e35_condition_cycle import (EP_SCALE, RUNGS, behaviour,
                                             evaluate, world)

STATE = os.path.join(RUNS, "e36_state.json")
BEST = os.path.join(RUNS, "e36_best.json")


def config_key():
    env = world(RUNGS[0], "vehicle")
    return dict(joint=True, veh_head=env.veh_head, sig_head=env.sig_head, prior="const",
                rungs=[list(r["approach_vph"]) for r in RUNGS],
                obs=env.OBS_VERSION, reward=env.REWARD_VERSION)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            st = json.load(fh)
        if st.get("config") != config_key():
            raise SystemExit("%s was written under a different configuration; move it "
                             "aside to start fresh" % STATE)
        return st
    return dict(round=0, rung=0, on_rung=0, stalls=0, veh=None, sig=None, history=[],
                config=config_key())


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh)
    os.replace(tmp, STATE)


def update_best(st, rows, r):
    cur, se = rows["train"]["discovered"], rows["train"]["se"]
    best = st.get("best")
    if best is None or cur > best["train"]:
        st["best"] = dict(round=r, train=cur, se=se, veh=st["veh"], sig=st["sig"])
        os.makedirs(os.path.dirname(BEST), exist_ok=True)
        with open(BEST, "w", encoding="utf-8") as fh:
            json.dump(st["best"], fh)
        return False
    return cur < best["train"] - 2.0 * max(se, best["se"])


def one_pass(env, agent, bank, other, cfg, tag, seed, rounds=1):
    """One agent's moves for this round, accepted on the team return.

    THE PROPOSALS ARE STILL THE AGENT'S OWN. `reward_as` moves only the price a
    candidate is accepted at; the deviations that generate proposals and the
    two estimators re-enter `own_reward` themselves (`kernsearch._own_reward`,
    `intersection_critic._own`), so the vehicle's advantages stay per-car with
    collisions charged to the car at fault.
    """
    env.set_agent(agent)
    if agent == "vehicle":
        env.signal_bank = other
    else:
        env.vehicle_bank = other
    with env.reward_as("team"):
        return fit(env, list(env.names), rounds=rounds, warm=bank is None,
                   init_bank=bank, run_seed=seed, tag=tag,
                   cfg=dict(cfg, T=env.duration), branch=0)


def target_G(tgt, vb, sb, s):
    """Team return per episode on the target distribution."""
    return IF.run(tgt, vb, sb, s, tgt.duration, reward="team")


def credit(tgt, s, new, old, who, z=2.0, g0=None):
    """Did THIS agent's pass lose ground on the target distribution?

    Paired, with the partner held fixed on both sides, so the difference is
    that agent's and no one else's -- the measurement a pair-level test cannot
    make. `g0` is the baseline's return when the caller already has it: the
    signal is judged against the vehicle tree this round leaves, which is
    exactly what the vehicle's own test just measured, so the round costs three
    of these rollouts and not four. Returns (keep, the kept side's return, mean
    difference, its standard error).
    """
    g1 = target_G(tgt, *new, s)
    g0 = target_G(tgt, *old, s) if g0 is None else g0
    d = g1 - g0
    se = float(d.std() / np.sqrt(len(d))) or 1e-9
    keep = float(d.mean()) > -z * se
    print("   %s on the target distribution: %+.1f +-%.1f%s"
          % (who, d.mean(), se, "" if keep else "   -- reverting this pass"), flush=True)
    return keep, (g1 if keep else g0), float(d.mean()), se


def main(rounds=16, n_ep=500, per_rung=4, critic=1, seed=0, verbose=1, n_guard=400,
         diag_every=4):
    rounds, n_ep, per_rung = int(rounds), int(n_ep), int(per_rung)
    n_guard, diag_every = int(n_guard), int(diag_every)
    critic, seed, verbose = bool(int(critic)), int(seed), bool(int(verbose))
    st = load_state()
    t0 = time.time()
    base0 = dict(seed=11, mem_at=99, beta_at=0, steps_at=0, grow_arms=2, min_n=200,
                 n_cover=6000, cover_ep=80, cem_iter=4, cem_K=32, grow_pool=40,
                 min_gain=1.0, subtree_at=(0,), kern_at=(0,), critic=critic,
                 prior="const", kern_cfg=dict(n_laws=2, max_points=4, dev_ep=3000))
    while st["round"] < rounds:
        r, rung = st["round"], st["rung"]
        cond = RUNGS[rung]
        ne = int(round(n_ep * EP_SCALE[rung]))
        cfg = dict(base0, n_ep=ne, val_ep=int(1.2 * ne), explore_ep=max(200, ne // 2))
        print("\n==== round %d, rung %d (approach %g..%g veh/h), %d episodes a test [%.0fs]"
              % (r, rung, cond["approach_vph"][0], cond["approach_vph"][1], ne,
                 time.time() - t0), flush=True)
        env = world(cond, "vehicle")
        vb = bank_from_json(st["veh"]) if st["veh"] else None
        sb = bank_from_json(st["sig"]) if st["sig"] else None
        # THE JUDGE: unseen episodes of the target distribution, redrawn every
        # round so no pass can be kept for fitting one sample of it.
        tgt = world(WIDE, "vehicle")
        s_t = tgt.sample_starts(n_guard, np.random.default_rng(4242 + r))
        moved, cred, g_base = [], {}, None
        banks = {"vehicle": vb, "signal": sb}
        names = {"vehicle": env.veh_names, "signal": env.sig_names}
        tags = {"vehicle": "e36-veh", "signal": "e36-sig"}
        keys = {"vehicle": "veh", "signal": "sig"}
        metas = {}
        # NEITHER AGENT IS SYSTEMATICALLY FIRST. Whoever passes second adapts to
        # a partner that has already moved this round, which is an advantage;
        # with a fixed order it is always the same agent's advantage, every
        # round. Both passes share this round's episodes either way -- `rseed`
        # is the same for both and the start states are cached on the world --
        # so the second pass re-simulates nothing the first one built.
        order = ["vehicle", "signal"] if r % 2 == 0 else ["signal", "vehicle"]
        for who in order:
            other = "signal" if who == "vehicle" else "vehicle"
            print("   -- %s, against the %s as it is now" % (who, other), flush=True)
            was = banks[who]
            bank, log, meta = one_pass(env, who, was, banks[other], cfg,
                                       tags[who], seed + r)
            metas[who] = meta
            mv = log[-1]["moves"] if log else []
            if mv and was is not None:
                new = dict(banks)
                new[who] = bank
                keep, g_base, d, se = credit(
                    tgt, s_t, (new["vehicle"], new["signal"]),
                    (banks["vehicle"], banks["signal"]), "%-8s" % who, g0=g_base)
                cred[who] = dict(d=d, se=se, kept=keep)
                if not keep:
                    bank, mv = was, []
            moved += mv
            banks[who] = bank
            st[keys[who]] = bank_json(bank, names[who])
            if verbose:
                print("\n%s tree, round %d (team %.2f)\n%s"
                      % (who, r, meta["G"], emit(bank, names[who])), flush=True)
            save_state(st)
        vb, sb = banks["vehicle"], banks["signal"]
        rows = evaluate(vb, sb)
        # THE BEHAVIOUR TABLE IS THE ONLY THING LEFT ON THE PYTHON ROLLOUT, at
        # about 50 s a call against 6 s for the whole evaluation. It reports;
        # it decides nothing -- `credit` and `update_best` do that, on the
        # kernel -- so it runs on a schedule and at the end, not every round.
        if r % diag_every == 0 or r == rounds - 1:
            behaviour(vb)
        st["history"].append(dict(round=r, rung=rung, eval=rows, moves=moved, credit=cred))
        revert = update_best(st, rows, r)
        b = st["best"]
        print("   best pair so far: round %d, train %.1f +-%.1f%s"
              % (b["round"], b["train"], b["se"],
                 "  -- this round is significantly worse: reverting to it" if revert else ""),
              flush=True)
        if revert:
            st["veh"], st["sig"] = b["veh"], b["sig"]
        st["on_rung"] += 1
        st["stalls"] = st["stalls"] + 1 if not moved else 0
        if rung < len(RUNGS) - 1 and (st["stalls"] >= 2 or st["on_rung"] >= per_rung):
            st["rung"], st["on_rung"], st["stalls"] = rung + 1, 0, 0
            print("   -> widening demand to rung %d" % st["rung"], flush=True)
        st["round"] = r + 1
        save_state(st)
    b = st.get("best")
    if b is not None:
        env = world(WIDE, "vehicle")
        print("\n==== best pair: round %d, target-distribution team %.1f +-%.1f\n%s\n%s"
              % (b["round"], b["train"], b["se"], emit(bank_from_json(b["veh"]), env.veh_names),
                 emit(bank_from_json(b["sig"]), env.sig_names) if b["sig"] else "(fixed plan)"),
              flush=True)
    print("[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main(**dict(a.split("=") for a in sys.argv[1:]))
