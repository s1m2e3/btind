r"""e37 -- both trees, one batch, EVERY demand at once. No ladder.

WHAT THIS DROPS. e35 and e36 trained up a ladder of demand rungs, 4-16 veh/h
first and the target 100-600 last, widening only when a rung stopped paying.
That separated the conditions in time: a move was measured at one demand and
carried into the next, which is exactly how e35's signal came to hold a flat
5.64 s green -- correct at 16-50 veh/h, wrong everywhere that counts.

WHAT REPLACES IT. `sample_starts` already draws EACH EPISODE's demand from the
condition range, so a batch over `approach_vph=(100, 600)` is not one demand,
it is a thousand episodes spread across the whole range. Every candidate is
screened across that range in the same test, and the range it is trained on is
the range it is scored on -- there is no transfer step left to go wrong.

WHY AN AVERAGE IS NOT ENOUGH, and what stops one demand paying for another: an
average gain hides a loss. `structure.accept` splits the batch by demand (cars
scheduled) and REJECTS a move if any group loses by more than z standard errors
of that group, so a candidate that buys throughput at 600 veh/h by starving the
left turns at 150 cannot be bought. The ladder enforced that by training the
conditions one at a time; this enforces it inside every test. `n_cond_groups`
is the resolution -- 5 across 100-600 veh/h.

THE RISK, stated plainly: e34 found that the intersection's pieces do not pay
alone at full demand, which is why the ladder was built -- a first arm that is
worth nothing on its own is never accepted, and the tree never starts. The
ladder gave the search a demand where single pieces did pay. This run bets that
a batch spanning 100-600 contains enough light episodes to make a first piece
pay, with per-condition acceptance stopping it from being bought by the heavy
ones. If both trees sit empty for several rounds, that bet has lost and the
ladder was doing more than guarding against deceptive gains.

EVERYTHING ELSE IS e36's. Joint rounds: each agent passes against the other as
it is now, every move accepted on the TEAM return, proposals still in the units
each agent is answerable for, neither agent systematically first, and each pass
re-priced alone on unseen target-distribution episodes with the partner held
fixed (`credit`) so a regression is reverted on its own.

RESUMABLE: `runs/e37_state.json` after every pass.
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

from experiments.e35_condition_cycle import N_MAX, actuated

STATE = os.path.join(RUNS, "e37_state.json")
BEST = os.path.join(RUNS, "e37_best.json")

# THE ONE CONDITION SET, and it is the TARGET distribution: 300-600 veh/h on
# EACH approach, drawn per episode, so the junction is busy in every episode
# and heavily loaded in many. 48-107 cars an episode against a 112-slot world,
# with no episode reaching the cap. Every batch is a sample across the whole
# range, never a slice of it, and the range trained on is the range scored on.
SPAN = dict(WIDE, approach_vph=(300.0, 600.0))
N_GROUPS = 5                     # demand groups a move must not lose in

# e35's range, kept as a column in the evaluation ONLY so its -769.7 stays
# readable next to this run. Nothing is trained on it.
E35_REF = WIDE


def world(agent, cond=SPAN, groups=N_GROUPS):
    env = IntersectionBatch(n_max=N_MAX, conditions=cond, veh_reward="car",
                            entry_v=(5.5, 11.0), veh_head="scalar",
                            sig_head="duration")
    env.n_cond_groups = groups
    return env.set_agent(agent)


def config_key():
    env = world("vehicle")
    return dict(batched=True, span=list(SPAN["approach_vph"]), groups=N_GROUPS,
                veh_head=env.veh_head, sig_head=env.sig_head, prior="const",
                obs=env.OBS_VERSION, reward=env.REWARD_VERSION)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            st = json.load(fh)
        if st.get("config") != config_key():
            raise SystemExit("%s was written under a different configuration; move "
                             "it aside to start fresh" % STATE)
        return st
    return dict(round=0, veh=None, sig=None, history=[], config=config_key())


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh)
    os.replace(tmp, STATE)


def demand_table(env, n_ep, seed=0):
    """What one training batch actually contains, by demand group."""
    from btind.structure import starts
    s = starts(env, n_ep, seed)
    g = env.condition_groups(s)
    cars = np.isfinite(s[:, 4 * env.N:5 * env.N]).sum(1)
    print("   one batch of %d episodes, by demand group:" % n_ep, flush=True)
    for gi in np.unique(g):
        m = g == gi
        print("     group %d: %4d episodes, %5.1f-%5.1f cars (median %.0f)"
              % (gi, m.sum(), cars[m].min(), cars[m].max(), np.median(cars[m])),
              flush=True)


def evaluate(vb, sb, n_ep=400, seed=999):
    """The pair against the baselines, on the span it trains on.

    `train` IS the training distribution -- with no ladder there is no held-out
    demand range, so the honest report is: this distribution on episodes the
    search never saw, plus two it was never trained on at all.
    """
    tgt = world("vehicle")
    hand = tgt.default_vehicle_bank()
    act = actuated(tgt)
    sets = {"train 300-600": tgt.sample_starts(n_ep, np.random.default_rng(seed)),
            "e35 100-600": world("vehicle", cond=E35_REF)
            .sample_starts(n_ep, np.random.default_rng(seed)),
            "heavy 600-800": world("vehicle", cond=dict(WIDE, approach_vph=(600.0, 800.0)))
            .sample_starts(n_ep, np.random.default_rng(seed))}
    rows = {}
    for name, s in sets.items():
        R = lambda v, g: IF.run(tgt, v, g, s, tgt.duration, reward="team")
        g = R(vb, sb)
        rows[name] = dict(discovered=g.mean(), discovered_veh_fixed=R(vb, None).mean(),
                          hand_fixed=R(hand, None).mean(), hand_actuated=R(hand, act).mean())
        if name.startswith("train"):
            rows[name]["se"] = float(g.std() / np.sqrt(len(g)))
    print("\n%-16s %12s %14s %12s %14s" % ("episodes", "discovered", "disc. veh +",
                                            "hand +", "hand +"))
    print("%-16s %12s %14s %12s %14s" % ("", "pair", "fixed plan", "fixed plan",
                                          "actuated"))
    for name, r in rows.items():
        print("%-16s %12.1f %14.1f %12.1f %14.1f"
              % (name, r["discovered"], r["discovered_veh_fixed"], r["hand_fixed"],
                 r["hand_actuated"]), flush=True)
    return rows


def behaviour(vb, n_ep=60, seed=321):
    """What the vehicle tree does on the training span, against the follower."""
    env = world("vehicle")
    s = env.sample_starts(n_ep, np.random.default_rng(seed))
    out = {}
    key = (n_ep, seed)
    todo = [("discovered", vb)]
    if key not in _HAND:
        todo.append(("hand-written", env.default_vehicle_bank()))
    for name, b in todo:
        env.terms = {}
        env.python_rollout(b, None, s)
        t = {k: float(np.mean(v)) for k, v in env.terms.items()}
        out[name] = dict(crashes=t["crash"] / -200.0, red_runs=t["red"] / -50.0,
                         stuck_s=-t["stuck"] / 3.0, comfort=-t.get("comfort", 0.0),
                         exits=t["exit"])
    out["hand-written"] = _HAND.setdefault(key, out.get("hand-written"))
    print("\n   per episode (%d, 300-600 veh/h, fixed plan):" % n_ep)
    print("   %-14s %8s %9s %9s %9s %7s"
          % ("", "crashes", "red runs", "stuck s", "comfort", "exits"))
    for name, r in out.items():
        print("   %-14s %8.2f %9.2f %9.1f %9.1f %7.1f"
              % (name, r["crashes"], r["red_runs"], r["stuck_s"], r["comfort"],
                 r["exits"]), flush=True)
    return out


_HAND = {}                      # the follower's row, which never changes


def target_G(tgt, vb, sb, s):
    return IF.run(tgt, vb, sb, s, tgt.duration, reward="team")


def credit(tgt, s, new, old, who, z=2.0, g0=None):
    """Did THIS agent's pass lose ground on the target distribution?

    Paired, partner held fixed on both sides, so the difference is that agent's
    and no one else's. `g0` is the baseline when the caller already has it.
    """
    g1 = target_G(tgt, *new, s)
    g0 = target_G(tgt, *old, s) if g0 is None else g0
    d = g1 - g0
    se = float(d.std() / np.sqrt(len(d))) or 1e-9
    keep = float(d.mean()) > -z * se
    print("   %s on the target distribution: %+.1f +-%.1f%s"
          % (who, d.mean(), se, "" if keep else "   -- reverting this pass"),
          flush=True)
    return keep, (g1 if keep else g0), float(d.mean()), se


def one_pass(env, agent, bank, other, cfg, tag, seed, rounds=1):
    """One agent's moves, accepted on the team return over the whole span."""
    env.set_agent(agent)
    if agent == "vehicle":
        env.signal_bank = other
    else:
        env.vehicle_bank = other
    with env.reward_as("team"):
        return fit(env, list(env.names), rounds=rounds, warm=bank is None,
                   init_bank=bank, run_seed=seed, tag=tag,
                   cfg=dict(cfg, T=env.duration), branch=0)


def update_best(st, rows, r):
    cur, se = rows["train 300-600"]["discovered"], rows["train 300-600"]["se"]
    best = st.get("best")
    if best is None or cur > best["train"]:
        st["best"] = dict(round=r, train=cur, se=se, veh=st["veh"], sig=st["sig"])
        os.makedirs(os.path.dirname(BEST), exist_ok=True)
        with open(BEST, "w", encoding="utf-8") as fh:
            json.dump(st["best"], fh)
        return False
    return cur < best["train"] - 2.0 * max(se, best["se"])


def main(rounds=20, n_ep=1200, critic=1, seed=0, verbose=1, n_guard=400,
         diag_every=4):
    rounds, n_ep, seed = int(rounds), int(n_ep), int(seed)
    critic, verbose = bool(int(critic)), bool(int(verbose))
    n_guard, diag_every = int(n_guard), int(diag_every)
    st = load_state()
    t0 = time.time()
    cfg = dict(seed=11, mem_at=99, beta_at=0, steps_at=0, grow_arms=2, min_n=200,
               n_cover=6000, cover_ep=80, cem_iter=4, cem_K=32, grow_pool=40,
               min_gain=1.0, subtree_at=(0,), kern_at=(0,), critic=critic,
               prior="const", kern_cfg=dict(n_laws=2, max_points=4, dev_ep=3000),
               n_ep=n_ep, val_ep=int(1.2 * n_ep), explore_ep=max(200, n_ep // 2))

    env0 = world("vehicle")
    print("==== e37: one batch, approach %g..%g veh/h EACH, %d episodes a test, "
          "%d demand groups" % (SPAN["approach_vph"][0], SPAN["approach_vph"][1],
                                n_ep, N_GROUPS), flush=True)
    demand_table(env0, n_ep, cfg["seed"])

    while st["round"] < rounds:
        r = st["round"]
        print("\n==== round %d of %d  [%.0fs elapsed]" % (r, rounds, time.time() - t0),
              flush=True)
        env = world("vehicle")
        # HELD-OUT EPISODES OF THE SAME DISTRIBUTION, redrawn every round. With
        # no ladder this is no longer a different demand range from the one
        # trained on -- it is fresh draws from it, so `credit` now asks whether
        # a pass survives episodes the search never optimised against, and it
        # still attributes the answer to one agent by holding the partner fixed.
        tgt = world("vehicle")
        s_t = tgt.sample_starts(n_guard, np.random.default_rng(4242 + r))
        vb = bank_from_json(st["veh"]) if st["veh"] else None
        sb = bank_from_json(st["sig"]) if st["sig"] else None

        banks = {"vehicle": vb, "signal": sb}
        names = {"vehicle": env.veh_names, "signal": env.sig_names}
        tags = {"vehicle": "e37-veh", "signal": "e37-sig"}
        keys = {"vehicle": "veh", "signal": "sig"}
        moved, cred, g_base, metas = [], {}, None, {}
        # neither agent is systematically the one that adapts to a partner
        # that has already moved this round
        order = ["vehicle", "signal"] if r % 2 == 0 else ["signal", "vehicle"]
        for who in order:
            other = "signal" if who == "vehicle" else "vehicle"
            print("\n   -- %s pass, against the %s as it is now  [%.0fs]"
                  % (who, other, time.time() - t0), flush=True)
            was = banks[who]
            bank, log, meta = one_pass(env, who, was, banks[other], cfg,
                                       tags[who], seed + r)
            metas[who] = meta
            mv = log[-1]["moves"] if log else []
            print("   %s pass took moves: %s" % (who, ", ".join(mv) or "none"),
                  flush=True)
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
        if r % diag_every == 0 or r == rounds - 1:
            behaviour(vb)
        st["history"].append(dict(round=r, eval=rows, moves=moved, credit=cred))
        revert = update_best(st, rows, r)
        b = st["best"]
        print("   best pair so far: round %d, train %.1f +-%.1f%s"
              % (b["round"], b["train"], b["se"],
                 "   -- this round is significantly worse: reverting to it" if revert
                 else ""), flush=True)
        if revert:
            st["veh"], st["sig"] = b["veh"], b["sig"]
        st["round"] = r + 1
        save_state(st)

    b = st.get("best")
    if b is not None:
        env = world("vehicle")
        print("\n==== best pair: round %d, target-distribution team %.1f +-%.1f\n%s\n%s"
              % (b["round"], b["train"], b["se"],
                 emit(bank_from_json(b["veh"]), env.veh_names),
                 emit(bank_from_json(b["sig"]), env.sig_names) if b["sig"]
                 else "(fixed plan)"), flush=True)
    print("[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main(**dict(a.split("=") for a in sys.argv[1:]))
