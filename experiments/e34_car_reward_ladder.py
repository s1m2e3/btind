r"""e34 -- the vehicle tree on a per-car reward, climbed up a demand ladder.

WHY. e28 on the team reward (stage 1, two rounds) found a right-turn arm and not
the red stop, and the diagnosis was not noise. On the same episodes the red stop
is detected at z 23 (team) and 27 (per car) -- the paired test isolates it
either way. The search could not PROPOSE its way there, because every piece of
the hand-written follower, added alone to a cruising tree, is worse:

    full demand, per-car reward, 300 episodes     G       z of the step
    stop every car                             -2809
    cruise (accelerate)                        -2383
    + stop for a red                           -4849      -50   cars behind crash
    + brake for a close leader                 -2365      +72
    + brake when closing                        -538      +44
    hand-written follower                       -310

Growth adds one arm at a time and keeps it only if it pays, so the red stop is
never kept at full demand. In light traffic the cars behind are rarely there:

    demand (s/l/r veh/h)    the next piece pays alone, given the ones before
    10/3/3                  red stop               z +4.1
    30/10/10                close-leader brake     z +17.5
    60/20/20 and up         closing brake          z +9.7 .. +44

so the ladder is a curriculum a one-arm-at-a-time search can climb, with no
teacher: each level warm-starts from the best bank the level below found
(`fit(transfer_from=...)`), and everything added still clears its own paired
test at the new demand.

THE REWARD CHANGES THIS NEEDED, each measured before any search ran:
    per-car reward       see `intersection.py`; the signal stays on the team one
    R_RED_CAR = 50       at R_RED = 10, with two cars an episode, running reds
                         out-earned stopping for them (z -2 for the red stop)
    stuck beyond sight   a red excuses a stop only within NEAR_INT of the line;
                         otherwise stopping every car beat cruising per car
    W_STUCK_CAR = 3      at 1, once red runs cost 50, stopping every car (-1284)
                         beat cruising (-2383) at full demand again
    entry speeds         cars enter at U(5.5, 11) m/s; with every car at V0 the
                         cold start cannot tell HOLD from ACCEL and picked HOLD
                         on one seed in three -- a tree that stops for a red and
                         never moves again

THE YARDSTICK is the team traffic score on the default world (every car at V0,
full demand), where the hand-written follower scores -227 and cruising -675,
so a per-car bonus farmed at the traffic's expense cannot hide.

WHAT CAME OUT (rounds=2 a level, 300 episodes a test, 2797 s, 2026-09-14, one
seed, fixed-time signal):

    demand      per car   hand-written   team yardstick (hand-written -223.1)
    10/3/3       -15.01       -4.85         -2450.6   red stop, no following
    30/10/10     -11.43      -23.22          -128.2   + following, as default
    60/20/20     -22.81      -74.63          -140.5   + closing brake
    200/60/60    -71.74     -575.28           -95.3   + subtree, beta

    Fallback
    |-- Sequence[ all_red<=-1 , accelerate; brake hard when closing ]  (can't hear the light)
    |-- Sequence[ lead_dv>0.7 , brake hard when closing ]
    |-- KeepRunningUntilFailure( Sequence[ d_conf<=28.5 ,
    |       brake hard on red, accelerate on green  until d_conf<=9.29 ,
    |       ACCEL_MAX ] )     beta: near_int>0   fail: d_conf>26.4
    |-- Sequence[ green>0 , Fallback ]
    |   |-- Sequence[ t_sig>28.9 , brake hard ]      a late green: don't go
    |   \-- ACCEL_MAX
    \-- accelerate; brake hard when closing

(the laws are dense CEM preferences; the readings are their argmax on probe
states). On the default world, 80 episodes, per episode: crashes 0.00 against
0.82 for the hand-written follower, red runs 0.70 against 1.55, stuck -9
against -21. The team score went -646 (e28 stage 1, team reward, no ladder) to
-95 with every piece found by growth; a step, a termination, a fail clause and
a nested subtree are all in the final tree, and no planted column.

WHAT TO DISTRUST. `t_sig>28.9` reads how long the phase has run, and 28.9 s is
the fixed plan's 30 s green: that arm is fitted to this controller and will not
survive a discovered signal. The red stop relies on argmax breaking a tie toward
BRAKE_HARD, the first action -- the action ORDER is doing part of the work. One
seed. The level rungs were chosen after measuring hand-written pieces.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btind.envs.intersection import IntersectionBatch
from btind.envs import intersection_fast as IF
from btind.memory import emit, mem_names
from btind.rlfit import fit
from btind.structure import heldout

LADDER = [(10.0, 3.0, 3.0), (30.0, 10.0, 10.0), (60.0, 20.0, 20.0), (200.0, 60.0, 60.0)]
DISTRACT = ("t_norm", "noise")


def guards(env, bank):
    zn = mem_names(env.names, bank.get("mem"))
    used = [zn[l[0]] for c in bank["clauses"] for l in c]
    bad = [u for u in used if any(d in u for d in DISTRACT)]
    return ", ".join(used) or "-", ", ".join(bad) or "none"


def team_score(bank, n_ep=600):
    """The yardstick: team reward, default world, same episodes for everyone."""
    env = IntersectionBatch()
    s = env.sample_starts(n_ep, np.random.default_rng(123))
    return IF.run(env, bank, None, s, env.duration, reward="team").mean()


def main(rounds=2, seed=0, n_ep=300, levels=None, tag=""):
    rounds, seed, n_ep = int(rounds), int(seed), int(n_ep)
    levels = range(len(LADDER)) if levels is None else [int(x) for x in str(levels).split(",")]
    t0 = time.time()
    cfg = dict(n_ep=n_ep, seed=11, val_ep=800, mem_at=99, beta_at=1, steps_at=1,
               grow_arms=2, min_n=200, n_cover=6000, explore_ep=200, cover_ep=120,
               cem_iter=6, cem_K=48, grow_pool=40, mem_pool=20, mem_arms=2,
               min_gain=1.0)
    hand_team = team_score(IntersectionBatch().default_vehicle_bank())
    print("yardstick (team reward, default world): hand-written follower %.1f" % hand_team,
          flush=True)
    prev, rows, vb = None, [], None
    for k in levels:
        vph = LADDER[k]
        env = IntersectionBatch(vph=vph, veh_reward="car", entry_v=(5.5, 11.0))
        env.set_agent("vehicle")
        if k > 0 and prev is None:
            prev = IntersectionBatch(vph=LADDER[k - 1], veh_reward="car", entry_v=(5.5, 11.0))
            prev.set_agent("vehicle")
        hand = heldout(env, env.default_vehicle_bank(), n_ep=1000)["G"]
        print("\n== level %d, demand %g/%g/%g veh/h -- hand-written follower, per car, %.2f"
              % ((k,) + vph + (hand,)), flush=True)
        vb, log, m = fit(env, list(env.names), rounds=rounds, warm=True,
                         run_seed=seed + k, tag="e34-veh" + tag,
                         cfg=dict(cfg, T=env.duration), branch=0, transfer_from=prev)
        g_team = team_score(vb)
        used, bad = guards(env, vb)
        print("level %d tree, per car %.2f (hand-written %.2f); team yardstick %.1f (hand-written %.1f)\n%s\n   guards: %s\n   planted: %s"
              % (k, m["G"], hand, g_team, hand_team, emit(vb, env.names), used, bad),
              flush=True)
        rows.append((vph, m["G"], hand, g_team))
        prev = env
    print("\n%-16s %10s %10s %12s" % ("demand", "per car", "hand", "team yard."))
    for vph, g, h, gt in rows:
        print("%-16s %10.2f %10.2f %12.1f" % ("%g/%g/%g" % vph, g, h, gt))
    print("%-16s %10s %10s %12.1f  (cruise -675, e28 stage 1 -646)" % ("hand-written", "", "", hand_team))
    print("[%.0fs]" % (time.time() - t0))
    return vb


if __name__ == "__main__":
    main(**dict(a.split("=") for a in sys.argv[1:]))
