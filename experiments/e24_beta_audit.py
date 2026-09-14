r"""e24 -- is beta silent because there is nothing to buy, or because it is broken?

Beta search had accepted essentially nothing across every run since it was
built, and that pattern had been wrong before: the memory stage reported +0.00
on every candidate for four rounds and the cause turned out to be four
structural bugs, not a null result. So the stage is audited the way the memory
premise was -- by asking the WORLD first, with hand-written controllers, before
asking whether the search can find anything in it.

FOUR MEASUREMENTS, in the order that makes each one interpretable.

  1  IS THE WORLD SENSITIVE TO HYSTERESIS AT ALL? A hand-written flee rule that
     enters at `d_threat < 0.09`, swept over where it LEAVES. `leave = 0.09` is
     the memoryless controller exactly, which is what `beta = NOT(guard)` is, so
     the sweep is measured against a null the candidate class contains.

  2  CAN THE REPRESENTATION BUY IT? The discovered bank, each arm made sticky in
     turn with a beta over the same column. If a hand-written gain exists and
     the bank cannot express it, that is a bug in the representation.

  3  DOES THE SEARCH FIND IT? `search_beta` on the same bank, same seeds.

  4  IF NOT, WHY NOT? Switch rate, the turn angle AT a switch against the turn
     angle between switches, and the norm of the action averaged over a window.

WHAT CAME OUT, and it is three separate answers rather than one.

  UNMASKED, HAND-WRITTEN: hysteresis is worth +3.96 (11.10 -> 15.06 at
  `leave = 0.20`), with a clean unimodal profile either side. The world IS
  sensitive to beta, and the Schmitt trigger in `betasearch`'s docstring is real.

  MASKED: it is strictly harmful, -1.42 at 0.12 falling to -3.67 at 0.50. Under
  masking the agent must search for food it cannot see, so productive time is
  scarce and every extra tick committed to fleeing costs more than the collision
  it avoids. Every run of the stage so far had been on the masked world, so its
  silence there was correct.

  UNMASKED, ON THE DISCOVERED TREE: also harmful -- every arm, every
  threshold, every position, best of them +0.01 -- and `search_beta` accepts
  0 of 82, which agrees with measurement 2 rather than contradicting it. The
  search is not missing anything the representation can buy.

MEASUREMENT 4 RULED OUT THE EXPLANATION IT WAS BUILT TO TEST. The discovered
tree chatters -- it switches arm on 41% of live ticks, turning 137 degrees at a
switch against 0.1 between them, and its mean action over a 2-tick window has
norm 0.737 against 1.0 for a committed controller. So the Fallback IS being used
as a pulse-width modulator, alternating arms to synthesise a heading neither
affine law can emit, and latching would destroy that. But the hand-written
controller dithers just as much (35% of ticks, 0.750), and hysteresis is worth
+3.96 to IT. Dithering does not separate the two cases.

NOR DOES THE QUALITY OF THE ESCAPE LAW, which was the next guess and is wrong in
the interesting direction. Mixing a tangential component into the retreat makes
hysteresis matter MORE, not less: +12.94 at k=1 and +15.93 at k=0.5, against
+3.96 for the pure radial retreat. A better law does not remove the need to
commit.

  5  WHAT THE GUARD READS. Same escape law throughout, changing only the column
     the flee decision is guarded on:

         d_threat  < 0.09      6.93  ->  19.87 with hysteresis   +12.94
         t_capture < 6        22.82  ->  21.22 with hysteresis    -1.60

     HYSTERESIS IS A SUBSTITUTE FOR A PREDICTIVE OBSERVATION. A latch is how a
     controller remembers "I am in danger" when its guard only reports how FAR
     the threat is. `t_capture` already integrates relative speed and heading,
     so the temporal information the latch was carrying is in `z`, and
     committing on top of it only delays noticing the danger has passed. The
     predictive guard alone beats the distance guard plus its best hysteresis,
     22.82 against 19.87.

AND THAT IS WHY BETA IS SILENT. The search found the predictive columns on its
own and guards on them: `t_capture` is the second-ranked column in the learned
proposal weights (15 accepted of 4335) and the top guard of every masked tree
this project has emitted. A stage offering to remember what the controller can
already predict has nothing to sell. The world does reward beta -- +12.94 -- but
only to a controller that reads the world less well than this one does.

SO THE STAGE IS NOT BROKEN. Its silence is correct on both worlds for reasons
that are now measured rather than assumed. The audit also says something the
return alone does not: BOTH controllers emit a behaviour that is a time-average
their structure does not disclose. A reader of either tree would say "it flees";
the agent goes at an angle to the threat that appears in no arm. That is a cost
of measuring the readability goal only through return, and it is not fixed here.
"""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("experiments", 1)[0])
from btind.betasearch import churn, search_beta
from btind.envs.nest import NestWorld, OBS_NAMES
from btind.memory import MemBank, emit, mem_names
from btind.policies import evaluate
from btind.runlog import RunLog
from btind.structure import absorb_universal, score
from btind import store as ST

I = {n: i for i, n in enumerate(OBS_NAMES)}
BASE = dict(day_len=80.0, food_persistent=True)
MASK = dict(vision_r=0.30, night_vision_r=0.15, threat_vision_r=0.35)
_B = lambda o, n: o[:, [I["bear_%s_x" % n], I["bear_%s_y" % n]]]
_U = lambda u: u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)


class Flee:
    """Reactive; `off` > `on` latches the flee arm until d_threat passes `off`."""

    def __init__(self, on=0.09, off=None, k=0.0, col="d_threat"):
        self.on, self.off = on, (on if off is None else off)
        # `k` mixes a TANGENTIAL component into the escape; `col` is the
        # column the flee decision is guarded on.
        self.k, self.col = k, col

    def reset(self, n):
        self.latched = np.zeros(n, bool)

    def act(self, o):
        if not hasattr(self, "latched") or len(self.latched) != len(o):
            self.reset(len(o))
        d, seen = o[:, I[self.col]], o[:, I["threat_seen"]] > 0.5
        self.latched = (((d < self.on) & seen)
                        | (self.latched & (d < self.off) & seen))
        carry = o[:, [I["carrying"]]] > 0.5
        fs = o[:, [I["food_seen"]]] > 0.5
        u = np.where(carry, _B(o, "nest"),
                     np.where(fs, _B(o, "food"), -_B(o, "nest")))
        bt = _B(o, "threat")
        esc = _U(-bt + self.k * np.stack([-bt[:, 1], bt[:, 0]], 1))
        u[self.latched] = esc[self.latched]
        return _U(u)

    def arbitrate(self, Z):
        return self.latched.astype(int)


def _hand(env, pol, T=400, seeds=(11, 12, 13)):
    g = []
    for sd in seeds:
        env.seed_kernels(sd)
        g.append(evaluate(env, pol, n_ep=3000, T=T, seed=sd)["G"])
    g = np.concatenate(g)
    return float(g.mean()), float(1.96 * g.std() / np.sqrt(len(g)))


def _absorbed(env, pol_fn):
    """The stored bank with any universal guard folded into the default."""
    bank, _ = ST.best(env)
    bank["names"] = list(OBS_NAMES)
    rng = np.random.default_rng(3)
    env.seed_kernels(3)
    p = pol_fn(bank)
    obs = env.observe(env.sample_states(6000, rng))
    p.reset(len(obs))
    Z = p.z(obs, update=False)
    b, _, _ = absorb_universal(env, bank, pol_fn, Z, verbose=False)
    return b, Z


def _dither(env, pol, n=400, T=400, seed=5):
    """Switch rate, turn at a switch vs between, and |mean action| over a window."""
    rng = np.random.default_rng(seed)
    env.seed_kernels(seed)
    s = env.sample_starts(n, rng)
    pol.reset(n)
    alive = np.ones(n, bool)
    Uu, AL, AR = [], [], []
    for _ in range(T):
        o = env.observe(s)
        u = pol.act(o)
        AR.append(np.asarray(pol.arbitrate(pol.z(o) if hasattr(pol, "z")
                                           else o)).copy())
        Uu.append(u.copy())
        AL.append(alive.copy())
        s, _, d = env.step(s, u)
        alive &= ~d
        if not alive.any():
            break
    Uu, AL, AR = np.array(Uu), np.array(AL), np.array(AR)
    live = AL[1:] & AL[:-1]
    sw = (AR[1:] != AR[:-1]) & live
    cos = (Uu[1:] * Uu[:-1]).sum(-1)
    ang = lambda m: (float(np.degrees(np.arccos(np.clip(cos[m], -1, 1))).mean())
                     if m.any() else 0.0)
    win = {}
    for w in (2, 5, 15):
        m = [np.linalg.norm(Uu[t:t + w, AL[t:t + w].all(0)].mean(0), axis=-1)
             for t in range(0, len(Uu) - w) if AL[t:t + w].all(0).any()]
        win[w] = float(np.concatenate(m).mean()) if m else 1.0
    return dict(switch=100.0 * sw.sum() / max(live.sum(), 1), at=ang(sw),
                between=ang(live & ~sw), win=win)


def main():
    rl = RunLog("e24-beta-audit")
    n_obs = len(OBS_NAMES)
    pol_fn = lambda b: MemBank(b, n_obs)
    out = {}

    print("1  IS THE WORLD SENSITIVE TO HYSTERESIS? hand-written flee, "
          "enter at d_threat<0.09\n")
    for tag, kw in (("unmasked", BASE), ("masked", dict(BASE, **MASK))):
        env = NestWorld(**kw)
        base = None
        print("   %s" % tag)
        print("   %-38s %8s %7s %8s" % ("leave at", "G", "+-", "vs null"))
        for off in (0.09, 0.12, 0.15, 0.20, 0.30, 0.50):
            g, c = _hand(env, Flee(0.09, off))
            base = g if base is None else base
            lab = ("d_threat>0.090  (= NOT guard, the null)" if off == 0.09
                   else "d_threat>%.3f" % off)
            print("   %-38s %8.2f %7.2f %+8.2f" % (lab, g, c, g - base))
            out.setdefault("hand_%s" % tag, {})["%.2f" % off] = g
        print()

    env = NestWorld(**BASE)                       # the world that rewards it
    bank, Z = _absorbed(env, pol_fn)
    zn = mem_names(OBS_NAMES, bank.get("mem"))
    C = len(bank["clauses"])
    jd = zn.index("d_threat")
    cur = score(env, bank, pol_fn, 1200, 400, 90210)
    base = float(cur.mean())
    print(emit(bank, OBS_NAMES))
    print("\n   discovered unmasked bank: G %.2f (validation seed, T=400)\n" % base)

    print("2  CAN THE REPRESENTATION BUY IT? each arm sticky, beta over d_threat")
    ch = churn(env, bank, pol_fn)
    print("   %-3s %-24s %7s %7s %11s %8s"
          % ("arm", "guard", "share", "churn", "best beta", "delta"))
    for c in range(C):
        lab = " AND ".join("%s%s%.3f" % (zn[j], "<=" if n else ">", t)
                           for j, t, n in bank["clauses"][c])
        best = (-9e9, "-")
        for thr in (0.12, 0.15, 0.20, 0.30, 0.50):
            betas = list(bank.get("betas") or [None] * C)
            st = list(bank.get("sticky") or [False] * C)
            betas[c], st[c] = [[jd, float(thr), False]], True
            g = score(env, dict(bank, betas=betas, sticky=st), pol_fn,
                      1200, 400, 90210).mean()
            if g - base > best[0]:
                best = (float(g - base), "d_threat>%.2f" % thr)
        print("   %-3d %-24s %6.1f%% %7.1f %11s %+8.2f"
              % (c, lab[:24], 100 * ch[c]["share"], ch[c]["reentries"],
                 best[1], best[0]))
        out.setdefault("represent", {})[lab] = best

    print("\n3  DOES THE SEARCH FIND IT?")
    t0 = time.time()
    nb, blog, _ = search_beta(env, bank, zn, Z, pol_fn, cur_G=cur, n_ep=1200,
                              T=400, seed=90210, z=2.0)
    g = float(score(env, nb, pol_fn, 1200, 400, 90210).mean())
    print("   %d/%d candidates accepted, G %.2f -> %.2f (%+.2f) [%.0fs]"
          % (sum(b["accepted"] for b in blog), len(blog), base, g, g - base,
             time.time() - t0))
    out["search"] = dict(accepted=int(sum(b["accepted"] for b in blog)),
                         tried=len(blog), delta=g - base)

    print("\n4  IF NOT, WHY NOT? is the switching discontinuous, and is it work?")
    print("   %-26s %8s %9s %10s %7s %7s %7s"
          % ("controller", "switch%", "turn AT", "turn btwn", "|u|w2", "|u|w5",
             "|u|w15"))
    for nm, pol in (("hand-written reactive", Flee()),
                    ("discovered tree", pol_fn(bank))):
        d = _dither(env, pol)
        print("   %-26s %7.1f%% %8.1fd %9.1fd %7.3f %7.3f %7.3f"
              % (nm, d["switch"], d["at"], d["between"], d["win"][2],
                 d["win"][5], d["win"][15]))
        out.setdefault("dither", {})[nm] = d
    print("\n   |u|w is the norm of the action averaged over w ticks. A committed")
    print("   controller keeps it near 1; cancellation means the Fallback is")
    print("   alternating arms to synthesise a heading neither law can emit.")

    print()
    print("5  WHAT THE GUARD READS. same escape law, different flee column")
    print("   %-26s %10s %11s %9s"
          % ("flee guard", "no hyst.", "best hyst.", "gain"))
    for col, on, offs in (("d_threat", 0.09, (0.12, 0.15, 0.20, 0.30)),
                          ("t_capture", 6.0, (8.0, 12.0, 20.0, 35.0))):
        a, _ = _hand(env, Flee(on, None, k=1.0, col=col))
        best = max((_hand(env, Flee(on, o, k=1.0, col=col))[0], o)
                   for o in offs)
        print("   %-26s %10.2f %11.2f %+9.2f   (leave at %g)"
              % ("%s < %g" % (col, on), a, best[0], best[0] - a, best[1]))
        out.setdefault("guard", {})[col] = dict(plain=a, hyst=best[0],
                                                gain=best[0] - a,
                                                off=best[1])
    print()
    print("   A latch is how a controller remembers 'I am in danger' when")
    print("   its guard only reports how FAR the threat is. `t_capture`")
    print("   already integrates speed and heading, so the latch has nothing")
    print("   to carry -- and the search guards on it unprompted.")

    rl.finish(out)
    print("\nlogged to %s" % rl.dir)


if __name__ == "__main__":
    main()
