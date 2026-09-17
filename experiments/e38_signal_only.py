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
import btind.structure as ST
from btind.envs.intersection import WIDE, IntersectionBatch, T_MAX
from btind.escape import kick
from btind.memory import MemBank
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
# two dry rounds is the signal that the monotone moves are exhausted; a hop
# then gets two rounds to be re-optimised before it is priced and kept or not
STALL_BEFORE_KICK, KICK_SIZE, HOP_BUDGET = 2, 1, 2
# The law class, at module scope because `config_key` has to report the one
# actually in use. It said prior="const" as a literal while the cfg below moved
# to "sparse", so a bank grown under constants-only would have been resumed
# under a class that can carry slopes -- silently, which is the single thing
# the configuration guard exists to prevent.
PRIOR, PRIOR_K, VALUE_LAWS = "sparse", 10, True


def world(agent, cond=SPAN, groups=N_GROUPS):
    env = IntersectionBatch(n_max=N_MAX_EP, T_end=T_END, conditions=cond,
                            veh_reward="car", entry_v=(5.5, 11.0),
                            veh_head="scalar", sig_head="duration")
    env.n_cond_groups = groups
    return env.set_agent(agent)


def load_vehicle(which="hand", path=VEHICLE):
    """The frozen partner. It is never the thing under search.

    WHY NOT e37's DISCOVERED CAR, which was the obvious choice: it learned to
    run red lights -- 52.1 an episode in its own training world against the
    hand-written follower's 2.1, trading them for throughput because the team
    return let it. Measured, that made 49.5% of the SIGNAL's objective its
    partner's law-breaking and left delay and queue together at 0.5%, so the
    light was being graded almost entirely on something it cannot prevent.

    The follower is a partner, not a teacher: it supplies no targets and never
    seeds a tree. The light still discovers its own from nothing.
    """
    if which == "hand":
        env = world("signal")
        print("   frozen partner: the hand-written follower (~7.5 red-runs an "
              "episode against the discovered car's 133)", flush=True)
        return env.default_vehicle_bank()
    with open(path, encoding="utf-8") as fh:
        b = json.load(fh)
    vb = bank_from_json(b["veh"])
    print("   frozen car from e37 round %d (%d arms) -- WARNING: this one runs "
          "reds" % (b["round"], len(b["veh"]["clauses"])), flush=True)
    return vb


def config_key():
    env = world("signal")
    return dict(signal_only=True, span=list(SPAN["approach_vph"]),
                skew=list(SPAN["approach_skew"]), groups=N_GROUPS,
                t_end=T_END, n_max=N_MAX_EP, t_max=T_MAX,
                sig_head=env.sig_head, prior=PRIOR, prior_k=PRIOR_K,
                value_laws=VALUE_LAWS,
                obs=env.OBS_VERSION, reward=env.REWARD_VERSION)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            st = json.load(fh)
        if st.get("config") != config_key():
            raise SystemExit("%s was written under a different configuration; move "
                             "it aside to start fresh" % STATE)
        return st
    return dict(round=0, sig=None, history=[], config=config_key(),
                stalled=0, hop_left=0, pre_hop=None)


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


def load_best():
    """The best tree any previous run of this configuration reached.

    `runs/e38_best.json` is written every time the best improves and is the one
    file worth carrying between snapshots -- a snapshot starts with no state
    file, so without this a fresh one throws away everything the last one
    found. It is refused across a configuration change for the same reason the
    state file is: the tree would be a tree, but not of this world.
    """
    if not os.path.exists(BEST):
        return None
    with open(BEST, encoding="utf-8") as fh:
        b = json.load(fh)
    if not b.get("sig"):
        return None
    # A MISSING KEY IS A MISMATCH. Files written before the law class was part
    # of the configuration carry no provenance at all, and loading one under a
    # class that can carry slopes when it was grown under constants-only is the
    # exact silent swap this guard exists to stop. `from_sig=` is the explicit
    # override for when that is what you want.
    if b.get("config") != config_key():
        print("   %s was written under a different configuration -- ignored "
              "(pass from_sig=<path> to load it anyway)" % BEST, flush=True)
        return None
    return b


def update_best(st, rows, r):
    cur, se = rows["train 300-600"]["discovered"], rows["train 300-600"]["se"]
    best = st.get("best")
    if best is None or cur > best["train"]:
        st["best"] = dict(round=r, train=cur, se=se, sig=st["sig"],
                          config=config_key())
        os.makedirs(os.path.dirname(BEST), exist_ok=True)
        with open(BEST, "w", encoding="utf-8") as fh:
            json.dump(st["best"], fh)
        return False
    return cur < best["train"] - 2.0 * max(se, best["se"])


def main(rounds=100, n_ep=100, screen_ep=20, critic=1, seed=0, verbose=1,
         n_guard=400, from_sig="", kern_every=3, car="hand", resume="best"):
    rounds, n_ep, seed = int(rounds), int(n_ep), int(seed)
    critic, verbose, n_guard = bool(int(critic)), bool(int(verbose)), int(n_guard)
    kern_every, screen_ep = int(kern_every), int(screen_ep)
    # one line per learning step: every candidate priced against the incumbent
    ST.LOG_STEPS = True
    st = load_state()
    # START FROM WHAT WAS ALREADY FOUND. The state file is refused across a
    # configuration change -- the world is a different one -- but the tree is
    # still a tree, so it is seeded as the incumbent rather than thrown away.
    if st["sig"] is None and from_sig and os.path.exists(from_sig):
        with open(from_sig, encoding="utf-8") as fh:
            st["sig"] = json.load(fh)["sig"]
        print("   seeded the light from %s" % from_sig, flush=True)
    # RESUME FROM THE BEST TREE, NOT THE LAST ONE. `st["sig"]` is wherever the
    # search happened to stop, and a round is kept whenever it is not
    # SIGNIFICANTLY worse, so a run can drift down and did: rounds 3 and 4 of
    # the previous run lost 10.6 and 14.9 against a best that was two rounds
    # behind them. Restarting from that is restarting from the drift.
    if resume != "off" and not from_sig:
        b = st.get("best") or load_best()
        if b and b.get("sig"):
            if resume == "best":
                st["sig"], st["best"] = b["sig"], b
                print("   resuming from the BEST tree held: round %s, train "
                      "%.1f +-%.1f" % (b.get("round"), b["train"], b["se"]),
                      flush=True)
            elif st["sig"] is None:
                st["sig"], st["best"] = b["sig"], b
                print("   no state file; seeded from %s (round %s, train %.1f)"
                      % (BEST, b.get("round"), b["train"]), flush=True)
    t0 = time.time()
    # Beta is back ON: `churn` reads the arm sequence off a kernel trace now
    # (1504x, identical statistics), where it used to step the world in
    # Python -- 779 s for 40 episodes, so ~65 min a round at its default 200.
    # That was the stage a round appeared to hang in after joint_sets.
    cfg = dict(seed=11, mem_at=99, beta_at=0, steps_at=0, grow_arms=2, min_n=200,
               # ONE SIGMA, NOT TWO. At 50 episodes a +37.19 gain was turned away
               # because 2 se was +49.6, so the search could not move at all.
               # The loosening is not free -- a worthless candidate now clears
               # the gain test about one time in six -- and two things absorb
               # it. The SAME z tightens the tests that protect what already
               # exists: the per-demand-group veto fires when a group loses by
               # more than z se, and non-inferiority (collapse, simplify) now
               # demands d > -1 se, so growth gets cheaper while removal gets
               # dearer. And every round is re-priced whole on 400 held-out
               # episodes by `credit`, which reverts a pass that lost ground.
               z=1.0,
               n_cover=6000, cover_ep=80, cem_iter=4, cem_K=32,
               min_gain=1.0, critic=critic,
               # A SPARSE PRIOR, NOT A CONSTANT ONE. `const` keeps only a
               # leaf's intercept so that every state dependence is a named
               # inducing point and "nothing hides in a dense affine prior".
               # That is a readability rule, and it was costing the whole of
               # what a law can express here: the value-fitted default is worth
               # +104.42 on 400 held-out episodes with its slopes and -23.27
               # constrained to its intercept, while the kernel that is meant
               # to carry the state dependence instead has found two points in
               # five rounds. Ten terms, selected and refitted, recover +104.40
               # of it and still read as a sentence --
               #   green = 20.9 + 10.1*ph2 + 9.7*ph0 + 7.0*ph1 + 2.5*q1b + ...
               # which is a phase plan with a queue term, the thing a signal
               # controller actually is. Set prior="const" to go back.
               prior=PRIOR, prior_k=PRIOR_K,
               # NARROWER PROPOSAL SEARCH, SAME ANSWER. The kernel stage was the
               # slowest thing in a round -- 264 scoring calls, ~248 of them
               # screens -- and widening the observation 24 -> 44 widened the
               # column-set search that drives it. Measured on this world:
               # col_pool 5 / n_prop 24 takes 94 s, col_pool 3 / n_prop 12 /
               # n_confirm 5 takes 56 s and returns the IDENTICAL three points
               # and the identical +668 gain. One step narrower (col_pool 2)
               # starts losing gain, so this is the knee.
               #
               # NOT screen_ep, deliberately: the module's own note records that
               # screening cheaply was measured to break this search -- "among
               # 300 proposals the top four screened were never the good one".
               kern_cfg=dict(n_laws=2, max_points=4, dev_ep=3000,
                             col_pool=3, n_prop=12, n_confirm=5),
               n_ep=n_ep, val_ep=int(1.2 * n_ep), explore_ep=max(200, n_ep // 2),
               # MORE DECISIONS, NOT MORE PRECISION PER DECISION -- but the
               # batch still has to resolve the gains on offer, and 50 did not.
               # Measured over 52 logged steps at 50 episodes: nothing accepted,
               # yet the search was finding real gains of +10 to +18 with
               # standard errors of +-12 to +-36, so the best candidate stood at
               # t=0.96 against a 1.0 bar and ten more sat between 0.5 and 1.0.
               # Missing by a factor of ~sqrt(2), so 100 -- which cuts those
               # errors 1.41x and tips the near-misses over, while still costing
               # a twelfth of the 1200 this started from.
               #
               # `accept`'s per-demand-group rule needs >5 episodes in a group;
               # at 100 over 5 groups it engages everywhere. 20 episodes would
               # leave 4 per group and switch that protection -- the one that
               # replaced the demand ladder -- off ENTIRELY AND SILENTLY, which
               # is why 20 screens a pool and never decides.
               screen_ep=screen_ep, grow_pool=28,
               # CEM's inner evaluations only RANK candidates to refit its
               # sampling distribution -- `improve_laws` still puts the result
               # through `accept` at the full n_ep -- and at n_ep they were the
               # largest single cost in a round: K*iter*n_ep = 32*4*100 = 12800
               # episode-units an arm against ~1000 for growing's confirms.
               # Measured 36.6 s -> 6.0 s at 20. Each iteration now draws a
               # fresh sample too, so a small screen cannot be chased.
               cem_ep=screen_ep,
               # FIT LAWS BY VALUE, not only by picking from a vocabulary.
               value_laws=VALUE_LAWS, value_ep=300)

    print("==== e38: the light alone, approach base %g..%g veh/h x skew %g..%g, "
          "%d episodes a test, %d demand groups"
          % (SPAN["approach_vph"][0], SPAN["approach_vph"][1],
             SPAN["approach_skew"][0], SPAN["approach_skew"][1], n_ep, N_GROUPS),
          flush=True)
    vb = load_vehicle(car)
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

        # BASIN-HOP WHEN THE MONOTONE SEARCH RUNS DRY. Every operator here
        # accepts only improvements, so a tree with no accepted move left is
        # finished and running more rounds finds nothing -- which is what
        # rounds 2 and 5 were. `escape.kick` is this project's answer and it
        # has never fired: `fit` owns the stall counter, e38 calls it with
        # rounds=1, so `stalled` was reset to 0 before every single check.
        # The counter belongs out here, where the rounds actually are.
        kicked = None
        if sb is not None and st.get("stalled", 0) >= STALL_BEFORE_KICK:
            cov, _ = env.coverage_rows(sb, n_ep=60, seed=r)
            p = MemBank(sb, len(env.names))
            p.reset(len(cov))
            st["pre_hop"] = bank_json(sb, env.sig_names)
            sb, kicked = kick(sb, p.z(cov, update=False),
                              np.random.default_rng(9000 + r), n=KICK_SIZE)
            st["hop_left"], st["stalled"] = HOP_BUDGET, 0
            print("   stalled %d rounds -- kick: %s  (re-optimising for %d "
                  "rounds, reverted if it does not pay)"
                  % (STALL_BEFORE_KICK, kicked, HOP_BUDGET), flush=True)

        # the kernel stage kept 0 of 16 confirmed points in e37 and e38 while
        # costing a fifth of the round, so it runs periodically, not every round
        # STAGGER THE EXPENSIVE OPTIONAL STAGES so a round pays for at most one.
        # Measured from a real round 0: grow 139 s, kernels 134 s, beta ~25 s,
        # plus subtree and the evaluation. Growing is the stage that actually
        # builds the tree and runs every round; the other three rotate, so each
        # still runs regularly and a round costs roughly grow + one of them.
        rcfg = dict(cfg, T=env.duration,
                    kern_at=(0,) if r % 3 == 0 else (),
                    beta_at=0 if r % 3 == 1 else 99,
                    subtree_at=(0,) if r % 3 == 2 else ())
        with env.reward_as("team"):
            new_sb, log, meta = fit(env, list(env.names), rounds=1, warm=sb is None,
                                    init_bank=sb, run_seed=seed + r, tag="e38-sig",
                                    cfg=rcfg, branch=0)
        mv = log[-1]["moves"] if log else []
        print("   moves: %s" % (", ".join(mv) or "none"), flush=True)
        cred = {}
        # A HOP IS WORSE BY CONSTRUCTION, so the per-round guard is suspended
        # while one is being re-optimised; the whole hop is priced at its end
        # against the tree it started from, which is the outer guarantee
        # basin-hopping actually needs.
        if mv and sb is not None and not st.get("hop_left"):
            keep, _, d, se = credit(tgt, s_t, vb, new_sb, sb)
            cred = dict(d=d, se=se, kept=keep)
            if not keep:
                new_sb, mv = sb, []
        sb = new_sb
        if st.get("hop_left"):
            st["hop_left"] -= 1
            if not st["hop_left"] and st.get("pre_hop"):
                base = bank_from_json(st["pre_hop"])
                keep, _, d, se = credit(tgt, s_t, vb, sb, base)
                cred = dict(d=d, se=se, kept=keep, hop=True)
                print("   hop over: %+.1f +-%.1f against the pre-kick tree -- %s"
                      % (d, se, "kept" if keep else "reverted"), flush=True)
                if not keep:
                    sb, mv = base, []
                st["pre_hop"] = None
        st["sig"] = bank_json(sb, env.sig_names)
        if verbose:
            print("\nsignal tree, round %d (team %.2f)\n%s"
                  % (r, meta["G"], emit(sb, env.sig_names)), flush=True)
        save_state(st)

        rows = evaluate(vb, sb)
        st["stalled"] = 0 if mv else st.get("stalled", 0) + 1
        st["history"].append(dict(round=r, eval=rows, moves=mv, credit=cred,
                                  kick=kicked, stalled=st["stalled"]))
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
