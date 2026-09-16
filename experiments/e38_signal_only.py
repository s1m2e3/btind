r"""e38 -- train the LIGHT alone, against e37's discovered car, on demand whose
imbalance varies independently of its total.

WHY ONLY THE SIGNAL. e37's vehicle tree earned its place: four arms, and at
600-800 veh/h it beats the hand-written follower paired with an actuated signal
by +59.2 +-11.7 (5.1 sigma) on demand it never trained on. The light learned
nothing in six rounds -- it grew one arm, collapsed it, and then accepted
nothing four rounds running. Half of every round was being spent on an agent
that could not move, so here the car is FROZEN and every rollout goes to the
light.

WHAT WAS ACTUALLY WRONG WITH THE LIGHT, measured on e37's best pair rather than
guessed. `explore.deviations` placed its deviation at a uniformly random tick
among those where the agent was ALIVE. A car commands an acceleration every
tick; the light is asked for a green time only when a phase begins -- 12.3% of
ticks. So 91.3% of the light's deviations overrode a command nobody read and
came back with an advantage of EXACTLY zero, against the car's 23%.

That one mismatch broke four things downstream, all of which are now fixed:

    the budget       261 informative deviations out of 3000. Deviating where
                     the agent acts makes it 1935 -- 7.4x the evidence for the
                     same rollouts (`IntersectionBatch.acting_ticks`).

    the critic       A-hat trained on 91% exact zeros learns to predict zero.
                     Its proposals were confirmed and never kept, every round.

    the sign metric  "sign of large" took the top quartile by |advantage|, but
                     with 75%+ of them exactly zero that quantile IS zero, so
                     "large" selected every row and sign(p)==sign(0) could never
                     match. It read 5% regardless of the critic. Scored on the
                     informative rows only, the same data gives a 484-row set at
                     40% positive.

    the gate         a critic is silenced only when rank, lift and sign ALL say
                     noise. Lift is precision/base-rate, and the zeros diluted
                     the base rate from 35% to 3%, inflating lift to 3.5x on
                     chance-level ranking. That one spurious number held the
                     gate open. Metrics with too little evidence now abstain.

AND THE DEMAND ITSELF. Four independent draws from one range confound how busy
the junction is with how unevenly it is loaded, and a light only earns its keep
where the approaches disagree. Measured: a smart signal is worth +185 over the
fixed plan on the most balanced episodes and +222 on the most unbalanced, so
imbalance is real headroom but not the whole story. `approach_skew` draws the
junction's load once and multiplies each approach by its own factor, so skew
varies at a fixed total: median imbalance 0.265 against 0.155 before, reaching
0.71. (A +-100 veh/h offset around a shared base was tried first and is WORSE
than what it replaced, 0.131 -- four independent draws already spread further.)

RESUMABLE: `runs/e38_state.json` after every round.
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

STATE = os.path.join(RUNS, "e38_state.json")
BEST = os.path.join(RUNS, "e38_best.json")
VEHICLE = os.path.join(RUNS, "e37_final", "e37_best.json")

# Load and skew drawn separately: `approach_vph` is now the junction's base
# flow, one draw an episode, and each approach multiplies it by its own factor.
SPAN = dict(WIDE, approach_vph=(300.0, 600.0), approach_skew=(0.25, 1.75))
N_GROUPS = 5

# AN EPISODE MUST HOLD TWO FULL CYCLES. At T_MAX=45 plus 3 s all-red a cycle of
# four phases is 192 s, so 384 s holds two of them -- measured, 9.4 phase
# changes an episode against 3.5 before, and the light is judged on a repeating
# pattern instead of a fragment. `n_max` is vehicles PER EPISODE (`ev = ev[:N]`
# truncates arrivals), so it has to grow with the episode: at 112 slots the
# skewed demand was silently dropping cars in 9% of episodes, and 384 leaves
# none. Costs 7.3x an episode, paid for by fewer episodes and a cheaper screen.
T_END, N_MAX_EP = 384.0, 384


def world(agent, cond=SPAN, groups=N_GROUPS):
    env = IntersectionBatch(n_max=N_MAX_EP, T_end=T_END, conditions=cond,
                            veh_reward="car", entry_v=(5.5, 11.0),
                            veh_head="scalar", sig_head="duration")
    env.n_cond_groups = groups
    return env.set_agent(agent)


def load_vehicle(path=VEHICLE):
    """e37's car, frozen. It is the partner, never the thing under search."""
    with open(path, encoding="utf-8") as fh:
        b = json.load(fh)
    vb = bank_from_json(b["veh"])
    print("   frozen car from e37 round %d (%d arms, target-distribution team "
          "%.1f +-%.1f)" % (b["round"], len(b["veh"]["clauses"]), b["train"],
                            b["se"]), flush=True)
    return vb


def config_key():
    env = world("signal")
    return dict(signal_only=True, span=list(SPAN["approach_vph"]),
                skew=list(SPAN["approach_skew"]), groups=N_GROUPS,
                t_end=T_END, n_max=N_MAX_EP, t_max=env.bounds[1],
                sig_head=env.sig_head, prior="const",
                obs=env.OBS_VERSION, reward=env.REWARD_VERSION)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            st = json.load(fh)
        if st.get("config") != config_key():
            raise SystemExit("%s was written under a different configuration; move "
                             "it aside to start fresh" % STATE)
        return st
    return dict(round=0, sig=None, history=[], config=config_key())


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh)
    os.replace(tmp, STATE)


def demand_table(env, n_ep, seed=0):
    """What one training batch contains: how busy, and how unevenly."""
    from btind.structure import starts
    s = starts(env, n_ep, seed)
    g = env.condition_groups(s)
    N = env.N
    appr = np.asarray(env.geom["appr"])
    mv = s[:, 0:N].astype(int)
    live = np.isfinite(s[:, 4 * N:5 * N])
    per = np.stack([((appr[np.clip(mv, 0, len(appr) - 1)] == a) & live).sum(1)
                    for a in range(4)], 1)
    tot = per.sum(1)
    imb = (per.max(1) - per.min(1)) / np.maximum(tot, 1)
    print("   one batch of %d episodes: %d-%d cars (median %d), approach "
          "imbalance median %.3f / p90 %.3f / max %.3f"
          % (n_ep, tot.min(), tot.max(), np.median(tot), np.median(imb),
             np.quantile(imb, 0.9), imb.max()), flush=True)
    for gi in np.unique(g):
        m = g == gi
        print("     demand group %d: %4d episodes, %3d-%3d cars"
              % (gi, m.sum(), tot[m].min(), tot[m].max()), flush=True)


def evaluate(vb, sb, n_ep=600, seed=999):
    """The light against the fixed plan and the actuated rule, car held fixed."""
    tgt = world("signal")
    act = actuated(tgt)
    hand = tgt.default_vehicle_bank()
    sets = {"train 300-600": tgt.sample_starts(n_ep, np.random.default_rng(seed)),
            "heavy 600-800": world("signal", cond=dict(SPAN, approach_vph=(600.0, 800.0)))
            .sample_starts(n_ep, np.random.default_rng(seed))}
    rows = {}
    for name, s in sets.items():
        R = lambda v, g: IF.run(tgt, v, g, s, tgt.duration, reward="team")
        g = R(vb, sb)
        base = R(vb, None)
        rows[name] = dict(discovered=g.mean(), fixed_plan=base.mean(),
                          actuated=R(vb, act).mean(), hand_fixed=R(hand, None).mean())
        d = g - base
        rows[name]["vs_fixed"] = float(d.mean())
        rows[name]["se"] = float(d.std() / np.sqrt(len(d)))
    print("\n%-16s %12s %12s %12s %12s" % ("episodes", "discovered", "fixed plan",
                                            "actuated", "hand car+fix"))
    for name, r in rows.items():
        print("%-16s %12.1f %12.1f %12.1f %12.1f    (vs fixed %+.1f +-%.1f)"
              % (name, r["discovered"], r["fixed_plan"], r["actuated"],
                 r["hand_fixed"], r["vs_fixed"], r["se"]), flush=True)
    return rows


def credit(tgt, s, vb, new_sb, old_sb, z=2.0, g0=None):
    """Did this signal pass lose ground on held-out episodes? Car held fixed."""
    g1 = IF.run(tgt, vb, new_sb, s, tgt.duration, reward="team")
    g0 = IF.run(tgt, vb, old_sb, s, tgt.duration, reward="team") if g0 is None else g0
    d = g1 - g0
    se = float(d.std() / np.sqrt(len(d))) or 1e-9
    keep = float(d.mean()) > -z * se
    print("   signal on held-out episodes: %+.1f +-%.1f%s"
          % (d.mean(), se, "" if keep else "   -- reverting this pass"), flush=True)
    return keep, (g1 if keep else g0), float(d.mean()), se


def update_best(st, rows, r):
    cur, se = rows["train 300-600"]["discovered"], rows["train 300-600"]["se"]
    best = st.get("best")
    if best is None or cur > best["train"]:
        st["best"] = dict(round=r, train=cur, se=se, sig=st["sig"])
        os.makedirs(os.path.dirname(BEST), exist_ok=True)
        with open(BEST, "w", encoding="utf-8") as fh:
            json.dump(st["best"], fh)
        return False
    return cur < best["train"] - 2.0 * max(se, best["se"])


def main(rounds=100, n_ep=50, screen_ep=20, critic=1, seed=0, verbose=1,
         n_guard=400, from_sig="", kern_every=3):
    rounds, n_ep, seed = int(rounds), int(n_ep), int(seed)
    critic, verbose, n_guard = bool(int(critic)), bool(int(verbose)), int(n_guard)
    kern_every, screen_ep = int(kern_every), int(screen_ep)
    st = load_state()
    # START FROM WHAT WAS ALREADY FOUND. The state file is refused across a
    # configuration change -- the world is a different one -- but the tree is
    # still a tree, so it is seeded as the incumbent rather than thrown away.
    if st["sig"] is None and from_sig and os.path.exists(from_sig):
        with open(from_sig, encoding="utf-8") as fh:
            st["sig"] = json.load(fh)["sig"]
        print("   seeded the light from %s" % from_sig, flush=True)
    t0 = time.time()
    cfg = dict(seed=11, mem_at=99, beta_at=0, steps_at=0, grow_arms=2, min_n=200,
               n_cover=6000, cover_ep=80, cem_iter=4, cem_K=32, grow_pool=40,
               min_gain=1.0, subtree_at=(0,), kern_at=(0,), critic=critic,
               prior="const", kern_cfg=dict(n_laws=2, max_points=4, dev_ep=3000),
               n_ep=n_ep, val_ep=int(1.2 * n_ep), explore_ep=max(200, n_ep // 2),
               # MORE DECISIONS, NOT MORE PRECISION PER DECISION. A test of 50
               # episodes costs about 0.1 s here, so a round spends its budget
               # iterating rather than re-measuring one candidate to three
               # decimal places. What 50 buys, measured on this world: the
               # paired test sees gains above ~31 return units, and -- the part
               # that has to be checked rather than assumed -- `accept`'s
               # per-demand-group rule engages in all 5 groups at 9-11 episodes
               # each. It needs >5 in a group, so 20 episodes over 5 groups
               # would switch the protection that replaced the demand ladder off
               # ENTIRELY AND SILENTLY. 20 is therefore the screen and never the
               # verdict: ranking a pool on it is free, deciding on it is not.
               grow_screen_ep=screen_ep, grow_pool=28)

    print("==== e38: the light alone, approach base %g..%g veh/h x skew %g..%g, "
          "%d episodes a test, %d demand groups"
          % (SPAN["approach_vph"][0], SPAN["approach_vph"][1],
             SPAN["approach_skew"][0], SPAN["approach_skew"][1], n_ep, N_GROUPS),
          flush=True)
    vb = load_vehicle()
    demand_table(world("signal"), n_ep, cfg["seed"])

    while st["round"] < rounds:
        r = st["round"]
        print("\n==== round %d of %d  [%.0fs elapsed]" % (r, rounds, time.time() - t0),
              flush=True)
        env = world("signal")
        env.vehicle_bank = vb
        tgt = world("signal")
        s_t = tgt.sample_starts(n_guard, np.random.default_rng(4242 + r))
        sb = bank_from_json(st["sig"]) if st["sig"] else None

        # the kernel stage kept 0 of 16 confirmed points in e37 and e38 while
        # costing a fifth of the round, so it runs periodically, not every round
        rcfg = dict(cfg, T=env.duration,
                    kern_at=(0,) if (kern_every and r % kern_every == 0) else ())
        with env.reward_as("team"):
            new_sb, log, meta = fit(env, list(env.names), rounds=1, warm=sb is None,
                                    init_bank=sb, run_seed=seed + r, tag="e38-sig",
                                    cfg=rcfg, branch=0)
        mv = log[-1]["moves"] if log else []
        print("   moves: %s" % (", ".join(mv) or "none"), flush=True)
        cred = {}
        if mv and sb is not None:
            keep, _, d, se = credit(tgt, s_t, vb, new_sb, sb)
            cred = dict(d=d, se=se, kept=keep)
            if not keep:
                new_sb, mv = sb, []
        sb = new_sb
        st["sig"] = bank_json(sb, env.sig_names)
        if verbose:
            print("\nsignal tree, round %d (team %.2f)\n%s"
                  % (r, meta["G"], emit(sb, env.sig_names)), flush=True)
        save_state(st)

        rows = evaluate(vb, sb)
        st["history"].append(dict(round=r, eval=rows, moves=mv, credit=cred))
        revert = update_best(st, rows, r)
        b = st["best"]
        print("   best signal so far: round %d, train %.1f +-%.1f%s"
              % (b["round"], b["train"], b["se"],
                 "   -- this round is significantly worse: reverting" if revert
                 else ""), flush=True)
        if revert:
            st["sig"] = b["sig"]
        st["round"] = r + 1
        save_state(st)

    b = st.get("best")
    if b is not None:
        env = world("signal")
        print("\n==== best signal: round %d, train %.1f +-%.1f\n%s"
              % (b["round"], b["train"], b["se"],
                 emit(bank_from_json(b["sig"]), env.sig_names)), flush=True)
    print("[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main(**dict(a.split("=") for a in sys.argv[1:]))
