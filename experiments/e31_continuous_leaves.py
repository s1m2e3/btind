r"""e31 -- what a continuous leaf buys: affine acceleration, affine green time.

The arbitration is discrete; the leaf need not be. A vehicle arm may command
an acceleration affine in its own state instead of one of five levels, and a
signal arm a green time affine in its queues instead of EXTEND/SWITCH each
tick. Hand-written controllers of both kinds, on the same episodes, against
their discrete counterparts from e27 -- the e24 protocol, so that when the
search is asked to find continuous leaves we know what they are worth.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btind.envs.intersection import (A_MAX, ACTIONS, B_MAX, T_MAX, T_MIN,
                                     IntersectionBatch)
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, emit, mem_names

N_EP, SEED = 600, 11


def main(n_ep=N_EP):
    env = IntersectionBatch()
    N, SN = env.veh_names, env.sig_names
    ix, sx = N.index, SN.index
    dv = len(mem_names(N, None)) + 1
    ds = len(mem_names(SN, None)) + 1

    def P(k, d=dv, n=5):
        th = np.zeros((d, n))
        th[-1, k] = 1.0
        return th
    A = {a: P(i) for i, a in enumerate(ACTIONS)}

    def lin(d, **coef):
        th = np.zeros((d, 1))
        for k, v in coef.items():
            th[-1 if k == "bias" else (ix(k) if d == dv else sx(k)), 0] = v
        return th

    close = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 10.0, True]]
    closing = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 25.0, True],
               [ix("lead_dv"), 0.5, False]]
    red = lambda lo, hi: [[ix("green"), 0.5, True], [ix("green"), -0.5, False],
                          [ix("d_stop"), lo, False], [ix("d_stop"), hi, True]]
    lead = [[ix("has_lead"), 0.5, False], [ix("lead_gap"), 40.0, True]]

    V = {}
    V["discrete: flat follower + red stop (e27)"] = check_arms(dict(
        names=list(N), laws_on_z=True, head="argmax", actions=list(ACTIONS),
        clauses=[close, red(0, 6), red(0, 45), closing],
        laws=[A["BRAKE_HARD"], A["BRAKE_HARD"], A["BRAKE"], A["BRAKE"]],
        default=A["ACCEL_MAX"]))
    # continuous: IDM-flavoured affine follower, affine approach to a red
    V["continuous: affine follower + affine red stop"] = check_arms(dict(
        names=list(N), laws_on_z=True, head="scalar", u_range=(-B_MAX, A_MAX),
        clauses=[lead, red(0, 50)],
        laws=[lin(dv, lead_gap=0.25, lead_dv=-1.2, v=-0.1, bias=-2.0),   # a = .25 gap - 1.2 dv - .1 v - 2
              lin(dv, d_stop=0.12, v=-0.6, bias=-1.5)],                    # a = .12 d - .6 v - 1.5
        default=lin(dv, v=-0.4, bias=4.4)))                               # a = 4.4 - .4 v -> V0
    V["continuous: same, discrete red stop arms"] = check_arms(dict(
        V["continuous: affine follower + affine red stop"],
        clauses=[lead, red(0, 6), red(0, 45)],
        laws=[lin(dv, lead_gap=0.25, lead_dv=-1.2, v=-0.1, bias=-2.0),
              lin(dv, bias=-B_MAX), lin(dv, bias=-1.5)]))

    S = {"fixed plan": None}
    own_empty = lambda k: [[sx("ph%d" % k), 0.5, False], [sx("n%d" % k), 0.5, True]]
    S["discrete: switch when own approach empty (e27)"] = check_arms(dict(
        names=list(SN), laws_on_z=True, head="argmax", actions=["EXTEND", "SWITCH"],
        clauses=[own_empty(k) for k in range(4)], laws=[P(1, ds, 2)] * 4,
        default=P(0, ds, 2)))
    S["continuous: green = 6 + 2.5 s per queued car"] = check_arms(dict(
        names=list(SN), laws_on_z=True, head="duration", u_range=(T_MIN, T_MAX),
        clauses=[[[sx("ph%d" % k), 0.5, False]] for k in range(4)],
        laws=[lin(ds, **{"q%d" % k: 2.5, "bias": 6.0}) for k in range(4)],
        default=lin(ds, bias=15.0)))
    S["continuous: green = 6 + 2.5 per queued - 1 per other-side queued"] = check_arms(dict(
        S["continuous: green = 6 + 2.5 s per queued car"],
        laws=[lin(ds, **{"q%d" % k: 2.5, "q%d" % ((k + 2) % 4): -1.0, "bias": 6.0})
              for k in range(4)]))

    s = env.sample_starts(n_ep, np.random.default_rng(SEED))
    t0 = time.time()
    print("VEHICLE leaves under the fixed-time plan (%d episodes):" % n_ep)
    ref = None
    best = None
    for name, v in V.items():
        G = IF.run(env, v, None, s, env.duration)
        ref = G if ref is None else ref
        se = float((G - ref).std() / np.sqrt(len(G)))
        print("   %-58s G %8.2f  (%+6.2f se %.2f)" % (name, G.mean(), (G - ref).mean(), se))
        if best is None or G.mean() > best[1]:
            best = (name, G.mean(), v)
    print("\nSIGNAL leaves with the best vehicle tree (%s):" % best[0])
    ref = None
    for name, sg in S.items():
        G = IF.run(env, best[2], sg, s, env.duration)
        ref = G if ref is None else ref
        se = float((G - ref).std() / np.sqrt(len(G)))
        print("   %-58s G %8.2f  (%+6.2f se %.2f)" % (name, G.mean(), (G - ref).mean(), se))
    print("\n" + emit(V["continuous: affine follower + affine red stop"], N))
    print("[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_EP)
