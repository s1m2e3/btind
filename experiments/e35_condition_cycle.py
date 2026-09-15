r"""e35 -- both trees trained over a wide set of operating conditions, the stages
cycled: vehicles, signal, vehicles, signal, ... each against a population of the
other agent's controllers.

WHAT VARIES, per episode (`intersection.WIDE`):

    flow on each approach, independently    100 .. 600 veh/h
    left and right turning shares           10 .. 30 % each
    fixed-plan greens                       through 15 .. 45 s, left 5 .. 20 s
    car entry speeds                        5.5 .. 11 m/s

so a batch of 300 episodes holds light and heavy, balanced and lopsided traffic
under plans the trees were never tuned to. e34's tree read `t_sig>28.9` -- a rule
about one plan's 30 s green -- and a condition distribution is what makes such a
rule stop paying.

THE LEAVES ARE CONTINUOUS. A car's leaf commands an acceleration in
[-4.5, 2.6] m/s^2 and the signal's a green time in [5, 60] s -- no action set
is defined for either; each leaf is a kernel-interpolation law with a CONSTANT
prior, so every state dependence is a learned inducing point, and the output is
clipped to the bounds (`kernlaw.py`). The first points a continuous leaf is
offered are its minimum and maximum response. The per-car reward charges steep
changes of speed (`W_COMFORT`), and every car sees the vehicles within 50 m,
including the most urgent crossing rival (`SENSE_R`).

THE CYCLE. Stage V fits the vehicle tree against a population of signals: the
randomised fixed plan plus the last `pop` discovered signal trees, episodes split
evenly among them (`intersection_fast.run` with a list). Stage S fits the signal
tree against the last `pop` vehicle trees. Then V again, then S. Each stage
resumes the agent's own last tree (`fit(init_bank=...)`), never a stored score
from an earlier cycle: those were measured against partners that are gone.

THE CURRICULUM IS A RULE, NOT A CHOICE. Demand starts in light traffic, where a
red stop pays on its own (e34), and the range widens to the next rung when a
vehicle stage's last round accepts nothing, or after `per_rung` cycles:

    rung 0   4 .. 16     rung 1   16 .. 50     rung 2   50 .. 150
    rung 3   100 .. 300  rung 4   100 .. 600 veh/h on each approach (the target)

Only the upper rung is the target; evaluation always reports it.

WHY THE FIRST RUNG IS THAT LIGHT. With continuous leaves and a constant prior a
red stop pays on its own only where the cars behind are rarely there: measured
with an accelerate-always default, "near a red -> -2 m/s^2" gained +14.7 (z 4.2)
at 4-16 veh/h, was neutral at 8-30, and at 16-100 every red-stop rule lost -- a
car slowing for red was rear-ended (0.54 crashes an episode from none). The
discrete run had escaped this at 16-100 only because its dense affine prior
happened to brake for a close leader.

ACCEPTANCE PER CONDITION. A move whose average gain hides a significant loss in
light, medium or heavy traffic is rejected (`structure.accept`).

RESUMABLE. `runs/e35_state.json` holds the populations, the cycle, the stage and
the rung after every completed stage; rerunning the script continues from there.

EVALUATION after every cycle, team reward, the discovered pair against the
hand-written follower under the randomised fixed plan and under an actuated
signal, on held-out episodes of:

    train     the target distribution (100 .. 600), unseen seeds
    heavy     every approach 600 .. 800 veh/h -- outside the training range
    lopsided  one approach 500 .. 600, the others 100 .. 150
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btind.envs.intersection import SIG_ACTIONS, WIDE, IntersectionBatch
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, emit, mem_names
from btind.rlfit import fit
from btind.runlog import RUNS, bank_from_json, bank_json

RUNGS = [dict(WIDE, approach_vph=lh) for lh in
         ((4.0, 16.0), (16.0, 50.0), (50.0, 150.0), (100.0, 300.0))] + [WIDE]
STATE = os.path.join(RUNS, "e35_state.json")
N_MAX = 112


def world(cond, agent):
    env = IntersectionBatch(n_max=N_MAX, conditions=cond, veh_reward="car",
                            entry_v=(5.5, 11.0), veh_head="scalar", sig_head="duration")
    return env.set_agent(agent)


def actuated(env):
    SN = env.sig_names
    sx = SN.index
    d = len(mem_names(SN, None)) + 1

    def P(k):
        th = np.zeros((d, 2))
        th[-1, k] = 1.0
        return th
    return check_arms(dict(names=list(SN), laws_on_z=True, head="argmax",
                           actions=list(SIG_ACTIONS),
                           clauses=[[[sx("ph%d" % k), 0.5, False], [sx("n%d" % k), 0.5, True]]
                                    for k in range(4)],
                           laws=[P(1)] * 4, default=P(0)), "actuated")


def lopsided_starts(n, seed):
    """One approach heavy, the others light; which one rotates by episode."""
    out = []
    for a in range(4):
        env = IntersectionBatch(n_max=N_MAX, conditions=dict(WIDE, approach_vph=(100.0, 150.0)),
                                veh_reward="car", entry_v=(5.5, 11.0))
        rng = np.random.default_rng(seed + a)
        m = n // 4
        # draw light starts, then add a heavy approach by re-drawing its arrivals
        heavy = IntersectionBatch(n_max=N_MAX, conditions=dict(WIDE, approach_vph=(500.0, 600.0)),
                                  veh_reward="car", entry_v=(5.5, 11.0))
        sl, sh = env.sample_starts(m, rng), heavy.sample_starts(m, rng)
        N = env.N
        appr = env.geom["appr"]
        for i in range(m):
            ev = []
            for src, keep in ((sl, lambda mv: appr[mv] != a), (sh, lambda mv: appr[mv] == a)):
                for q in range(N):
                    if np.isfinite(src[i, 4 * N + q]) and keep(int(src[i, q])):
                        ev.append((src[i, 4 * N + q], int(src[i, q])))
            ev = sorted(ev)[:N]
            row = sl[i].copy()
            k = len(ev)
            row[0:N] = 0
            row[0:k] = [e[1] for e in ev]
            row[4 * N:5 * N] = np.inf
            row[4 * N:4 * N + k] = [e[0] for e in ev]
            row[N:2 * N] = env.s_spawn[row[0:N].astype(int)]
            out.append(row)
    return np.array(out)


def evaluate(vb, sb, n_ep=400, seed=999):
    tgt = world(WIDE, "vehicle")
    hand = tgt.default_vehicle_bank()
    sets = {"train": tgt.sample_starts(n_ep, np.random.default_rng(seed)),
            "heavy": world(dict(WIDE, approach_vph=(600.0, 800.0)), "vehicle")
            .sample_starts(n_ep, np.random.default_rng(seed)),
            "lopsided": lopsided_starts(n_ep, seed)}
    rows = {}
    for name, s in sets.items():
        G = lambda v, g: IF.run(tgt, v, g, s, tgt.duration, reward="team").mean()
        rows[name] = dict(discovered=G(vb, sb), discovered_veh_fixed=G(vb, None),
                          hand_fixed=G(hand, None), hand_actuated=G(hand, actuated(tgt)))
    print("\n%-10s %12s %14s %12s %14s" % ("episodes", "discovered", "disc. veh +", "hand +",
                                            "hand +"))
    print("%-10s %12s %14s %12s %14s" % ("", "pair", "fixed plan", "fixed plan", "actuated"))
    for name, r in rows.items():
        print("%-10s %12.1f %14.1f %12.1f %14.1f" % (name, r["discovered"], r["discovered_veh_fixed"],
                                                     r["hand_fixed"], r["hand_actuated"]))
    return rows


def config_key():
    """What a saved state was produced under. A state from a different setup --
    other leaves, prior or rungs -- is never resumed: measured, a relaunch with
    continuous leaves silently continued a discrete run's cycle 2."""
    env = world(RUNGS[0], "vehicle")
    return dict(veh_head=env.veh_head, sig_head=env.sig_head, prior="const",
                rungs=[list(r["approach_vph"]) for r in RUNGS],
                obs=env.OBS_VERSION, reward=env.REWARD_VERSION)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            st = json.load(fh)
        if st.get("config") != config_key():
            raise SystemExit("%s was written under a different configuration (%s); "
                             "move it aside to start fresh" % (STATE, st.get("config")))
        return st
    return dict(cycle=0, stage="veh", rung=0, on_rung=0, veh=[], sig=[], history=[],
                config=config_key())


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh)
    os.replace(tmp, STATE)


def main(cycles=8, rounds_veh=2, rounds_sig=1, n_ep=500, pop=2, per_rung=2, critic=1,
         seed=0):
    cycles, rounds_veh, rounds_sig, n_ep = int(cycles), int(rounds_veh), int(rounds_sig), int(n_ep)
    pop, per_rung, critic, seed = int(pop), int(per_rung), bool(int(critic)), int(seed)
    st = load_state()
    t0 = time.time()
    base = dict(n_ep=n_ep, seed=11, val_ep=600, mem_at=99, beta_at=1, steps_at=1,
                grow_arms=2, min_n=200, n_cover=6000, explore_ep=200, cover_ep=80,
                cem_iter=4, cem_K=32, grow_pool=40, min_gain=1.0,
                subtree_at=(1,), kern_at=(0, 1), critic=critic, prior="const",
                kern_cfg=dict(n_laws=2, max_points=4, dev_ep=600))
    while st["cycle"] < cycles:
        cyc, rung = st["cycle"], st["rung"]
        cond = RUNGS[rung]
        print("\n==== cycle %d, rung %d (approach %g..%g veh/h), stage %s [%.0fs]"
              % (cyc, rung, cond["approach_vph"][0], cond["approach_vph"][1], st["stage"],
                 time.time() - t0), flush=True)
        vpop = [bank_from_json(b) for b in st["veh"]]
        spop = [bank_from_json(b) for b in st["sig"]]
        if st["stage"] == "veh":
            env = world(cond, "vehicle")
            env.signal_bank = [None] + spop[-pop:]
            print("   partners: fixed plan + %d signal trees" % len(spop[-pop:]), flush=True)
            vb, log, m = fit(env, list(env.names), rounds=rounds_veh, warm=not vpop,
                             init_bank=vpop[-1] if vpop else None, run_seed=seed + cyc,
                             tag="e35-veh-c%d" % cyc, cfg=dict(base, T=env.duration), branch=0)
            stalled = not (log and log[-1]["moves"])
            print("\nvehicle tree, cycle %d (%.2f per car%s)\n%s"
                  % (cyc, m["G"], ", last round accepted nothing" if stalled else "",
                     emit(vb, env.names)), flush=True)
            st["veh"].append(bank_json(vb, env.names))
            st["stage"] = "sig"
            st["veh_stalled"] = stalled
            save_state(st)
            continue
        env = world(cond, "signal")
        env.vehicle_bank = vpop[-pop:]
        print("   partners: %d vehicle trees" % len(vpop[-pop:]), flush=True)
        sb, log, m = fit(env, list(env.names), rounds=rounds_sig, warm=not spop,
                         init_bank=spop[-1] if spop else None, run_seed=seed + cyc,
                         tag="e35-sig-c%d" % cyc, cfg=dict(base, T=env.duration), branch=0)
        print("\nsignal tree, cycle %d (%.2f team)\n%s" % (cyc, m["G"], emit(sb, env.names)),
              flush=True)
        st["sig"].append(bank_json(sb, env.names))
        rows = evaluate(vpop[-1], sb)
        st["history"].append(dict(cycle=cyc, rung=rung, eval=rows))
        # the curriculum rule
        st["on_rung"] += 1
        if rung < len(RUNGS) - 1 and (st.get("veh_stalled") or st["on_rung"] >= per_rung):
            st["rung"], st["on_rung"] = rung + 1, 0
            print("   -> widening demand to rung %d" % st["rung"], flush=True)
        st["stage"], st["cycle"] = "veh", cyc + 1
        save_state(st)
    print("[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main(**dict(a.split("=") for a in sys.argv[1:]))
