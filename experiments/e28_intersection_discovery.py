r"""e28 -- both intersection controllers discovered from zero, in stages.

No expert anywhere. The vehicle tree starts as five manoeuvres tied at zero and
the signal tree as EXTEND and SWITCH tied at zero; every guard, law, step,
termination and fail clause is proposed from the trees' own rollouts and kept
only by the paired test. The hand-written references from e27 are the ceiling
this is read against, never an input to it:

    vehicles, fixed-time plan      flat follower + red stop   -0.2   (cruise -37)
    signal, hand-written vehicles  switch when own approach empty   +7.2

THE STAGES, because a bank is priced against the OTHER controller it will run
with, and that controller changes:

    1  vehicle tree under the fixed-time plan
    2  signal tree over the discovered vehicle tree
    3  vehicle tree again, under the discovered signal

Each stage is a warm-startable `rlfit.fit` on the world with `agent` set; the
store and the proposal weights are keyed by the world's signature, which
includes the agent, so the two searches never read each other's memory.

THE DISTRACTOR CONTROL runs on both trees: `t_norm` and `noise` are in both
alphabets and the run reports whether either was bought.

WHAT CAME OUT (small budget: rounds 2/1/1, two arms a round, 300 episodes a
test, 1618 s, 2026-09-14; checkpointed every round, stage 3 resumed from the
stage-1 checkpoint):

                                           held-out G
    cruise, fixed plan (e27)                  -37.20
    hand-written follower, fixed plan          -0.50
    stage 1  vehicles, fixed plan              -7.31    3 arms, from -20.3 cold
    stage 2  + discovered signal               -5.17    2 arms, +2.1 over the plan
    stage 3  vehicles refit under it           -3.08    5 arms

From five manoeuvres tied at zero and EXTEND/SWITCH tied at zero, both trees
were found by rollout alone: -37 to -3 in 27 minutes, the vehicle tree guarding
on its own speed, its leader and the closing speed, the signal on the queue of
the phase it is in. One step was accepted (a two-action Sequence), no memory
was needed on this fully-observed variant, and the collapse move refused three
passenger guards.

WHAT IS WRONG WITH IT, said plainly: both trees bought `t_norm`, the planted
elapsed-time column -- the vehicle tree's Sequence advances on it, the signal
tree's top arm reads it. On this world it was not a distractor: the episode
is truncated at 100 s, so elapsed time predicts whether a car can still exit
and how long delay keeps being charged, and the search exploited a real
horizon cue.

THE FIRST FIX DID NOT FIX IT. De-phasing the clock by an offset computed from
`noise` left elapsed time decodable, because `noise` is itself observed:
measured on held-out episodes (tests/test_tnorm_distractor.py), a regressor
recovers the tick from (t_norm, noise) at R^2 0.86, against 1.00 for the raw
fraction. The phase is now drawn independently into unobserved state, where
the same regressor scores below zero; `OBS_VERSION` in the world signature
keeps the leaky run's checkpoints from ever being resumed.

HOW MUCH OF THE RESULT ABOVE WAS THE CUE: the stage-3 trees from that run
score -22.0 on the fixed world, against -3.08 on the leaky one. Removing the
vehicle tree's `t_norm` arm there is worth +2.2, and the CEM-tuned laws carried
coefficients on the column as well. The -3.08 was substantially a horizon
exploit and is superseded by the rerun below.

THE RERUN, hidden-phase clock (obs v2), argmax leaves, continuous light, rounds
2/1/1, stage 1 checkpointed and resumed in a frozen snapshot:

                                           held-out G
    hand-written follower, fixed plan          -0.50
    stage 1  vehicles, fixed plan              -0.83    2 children
    stage 2  + discovered signal               -0.01    1 child
    stage 3  vehicles refit                    -0.01    nothing accepted

    vehicle: near_int>0 -> accelerate if green, else brake hard  (exact:
             the law's only coefficient is +1 on accelerate for the light,
             and the argmax breaks the tie on red toward brake hard);
             has_lead>0 -> a dense tuned preference, mostly brake hard;
             default accelerate
    signal:  n1>1 -> extend; default extend, switching early in a phase
             when phase 2 has a queue

The collapse move removed a `t_norm` passenger in round 0; no planted column is
in either tree. Terminations and steps were searched and refused ("nothing beat
staying reactive", "no step cleared +0.30"), as e27 predicts for this variant.

TWO FINDINGS, NOT YET FIXED. The signal tree holds phase 0 to the 60 s maximum
and never serves phase 3 (north-south left) within a 100 s episode, which the
fixed plan serves in every episode: the same horizon exploit as `t_norm`, one
level up, since queued cars that are never served cost only delay until the
episode ends. And the car-ahead leaf needs ten terms, two of them on the
planted columns, to reproduce 95% of its choices: the audit covers guards,
not law coefficients, and CEM leaves laws dense.

EVENT MODE WITH CONTINUOUS LEAVES (obs v3, truthful SPaT window): stage 1 stuck
at -20.4. CEM tuned the cold-start acceleration to stop every car before the
zone -- -20 beats cruising at -37 because stopped cars never run a red or
collide -- and from there no car hears a message, so the memory stage found
nothing to store. A local optimum of the reward, reported as measured.
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
from btind.runlog import RunLog
from btind.structure import heldout

DISTRACT = ("t_norm", "noise")


def report(env, bank, label):
    zn = mem_names(env.names, bank.get("mem"))
    used = [zn[l[0]] for c in bank["clauses"] for l in c]
    if bank.get("mem"):
        used += ["store:" + env.names[j] for j in bank["mem"]["cols"]]
    bad = [u for u in used if any(dn in u for dn in DISTRACT)]
    print("\n%s\n%s\n   guards: %s\n   planted among them: %s"
          % (label, emit(bank, env.names), ", ".join(used) or "-",
             ", ".join(bad) if bad else "none"), flush=True)


def main(rounds=(2, 1, 1), seed=0, n_ep=300, warm=True, veh_head="argmax",
         sig_head="argmax", sig_mode="continuous", tag=""):
    rl = RunLog("e28-intersection")
    env = IntersectionBatch(veh_head=veh_head, sig_head=sig_head, sig_mode=sig_mode)
    print("world: vehicle leaf %s, signal leaf %s, signal %s" % (veh_head, sig_head, sig_mode))
    rl.config(env, dict(rounds=list(rounds), seed=seed, n_ep=n_ep))
    t0 = time.time()
    # the SMALL budget: two arms a round, half the CEM, 300 episodes a test.
    # Every stage warm-starts from the store at rank 0, and `fit` checkpoints
    # every round, so a run cut short resumes rather than restarts.
    cfg = dict(n_ep=n_ep, T=env.duration, seed=11, val_ep=800, mem_at=99,
               beta_at=1, steps_at=1, grow_arms=2, min_n=200, n_cover=6000,
               explore_ep=200, cover_ep=120, cem_iter=6, cem_K=48, grow_pool=40,
               mem_pool=20, mem_arms=2,
               # returns on the traffic reward are in the hundreds (e33); a
               # floor of 0.3 would admit gains the size of the noise
               min_gain=1.0)
    # MEMORY WHERE THE WORLD CAN REWARD IT. In event mode the light is heard
    # once, so the vehicle stage runs the blackboard search at the end of its
    # first round; the signal observes its queues every tick and gets none.
    veh_cfg = dict(cfg, mem_at=0 if sig_mode == "event" else 99)
    sig_cfg = dict(cfg, mem_at=99)

    def ref_line(vb, sb, label):
        m = heldout(env, vb if env.agent == "vehicle" else sb, n_ep=1000)
        print("   %-44s held-out G %7.2f +-%.2f" % (label, m["G"], m["ci"]))
        return m["G"]

    # ---- stage 1: vehicles under the fixed plan ------------------------------
    env.set_agent("vehicle")
    env.signal_bank, env.vehicle_bank = None, None
    print("stage 1 -- the vehicle tree, fixed-time signal", flush=True)
    print("   reference: hand-written follower", flush=True)
    g_ref_v = ref_line(env.default_vehicle_bank(), None, "hand-written red-stop follower")
    vb, log1, m1 = fit(env, list(env.names), rounds=rounds[0], warm=warm,
                       run_seed=seed, tag="e28-veh" + tag, cfg=veh_cfg, branch=0)
    report(env, vb, "discovered vehicle tree (stage 1): %.2f vs hand-written %.2f"
           % (m1["G"], g_ref_v))

    # ---- stage 2: the signal over the discovered vehicles ---------------------
    env.set_agent("signal")
    env.vehicle_bank, env.signal_bank = vb, None
    print("\nstage 2 -- the signal tree, discovered vehicles", flush=True)
    sb, log2, m2 = fit(env, list(env.names), rounds=rounds[1], warm=warm,
                       run_seed=seed, tag="e28-sig" + tag, cfg=sig_cfg, branch=0)
    report(env, sb, "discovered signal tree (stage 2): %.2f (fixed plan gave %.2f)"
           % (m2["G"], m1["G"]))

    # ---- stage 3: vehicles again, under the discovered signal ---------------
    env.set_agent("vehicle")
    env.signal_bank, env.vehicle_bank = sb, None
    print("\nstage 3 -- the vehicle tree again, under the discovered signal",
          flush=True)
    vb2, log3, m3 = fit(env, list(env.names), rounds=rounds[2], warm=True,
                        run_seed=seed + 1, tag="e28-veh" + tag, cfg=veh_cfg, branch=0)
    report(env, vb2, "vehicle tree (stage 3): %.2f" % m3["G"])

    from experiments.e33_traffic_reward import served_phases
    s_chk = env.sample_starts(300, np.random.default_rng(77))
    print("\nphases served (share of episodes): fixed plan %s | discovered signal %s"
          % (" ".join("%.2f" % x for x in served_phases(env, vb2, None, s_chk)),
             " ".join("%.2f" % x for x in served_phases(env, vb2, sb, s_chk))))
    print("\n%-40s %8s" % ("", "held-out G"))
    print("%-40s %8.2f" % ("cruise, fixed plan (e27)", -37.2))
    print("%-40s %8.2f" % ("hand-written follower, fixed plan", g_ref_v))
    print("%-40s %8.2f" % ("stage 1 vehicles, fixed plan", m1["G"]))
    print("%-40s %8.2f" % ("stage 2 + discovered signal", m2["G"]))
    print("%-40s %8.2f" % ("stage 3 vehicles refit", m3["G"]))
    print("[%.0fs]" % (time.time() - t0))
    rl.finish(dict(stage1=m1, stage2=m2, stage3=m3, ref_vehicle=g_ref_v),
              bt=emit(vb2, env.veh_names) + "\n\n" + emit(sb, env.sig_names))
    return vb2, sb


if __name__ == "__main__":
    r = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (2, 1, 1)
    kw = dict(a.split("=") for a in sys.argv[3:])
    main(rounds=r, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0, **kw)
