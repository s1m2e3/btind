"""Fit a behaviour tree for ForageWorld and print it.

    python fit_bt.py                     defaults
    python fit_bt.py --seed 3            a different pipeline draw
    python fit_bt.py --runs 3            repeat, and report the spread that
                                         matters (run-to-run, not per-run CI)
    python fit_bt.py --add-arms          enable the proposal operator (off by
                                         default: 27 proposals, 27 rejected)
    python fit_bt.py --landscape-guards  let guards mention V_hat and leverage

The per-run confidence interval is about +-0.20 and the spread between
nominally identical runs is about +-0.35, because the food respawn draw lives in
a per-thread numba stream that no seed pins. One run is a sample, not a result.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from btind.envs.forage import ForageWorld, OBS_NAMES
from btind.pipeline import fit_bt, DEFAULTS
from btind.policies import CEMPolicy, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--dagger", type=int, default=DEFAULTS["n_dagger"])
    ap.add_argument("--polish", type=int, default=DEFAULTS["n_polish"])
    ap.add_argument("--add-arms", action="store_true")
    ap.add_argument("--landscape-guards", action="store_true")
    ap.add_argument("--no-drop", action="store_true")
    ap.add_argument("--oracle", action="store_true",
                    help="also score the replanning CEM policy (slow)")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    a = ap.parse_args()

    env = ForageWorld()
    runs = []
    for k in range(a.runs):
        if a.runs > 1:
            print("\n### run %d" % k)
        r = fit_bt(env, OBS_NAMES, seed=a.seed + k, n_dagger=a.dagger,
                   n_polish=a.polish, add_arms=a.add_arms,
                   landscape_guards=a.landscape_guards, drop=not a.no_drop)
        print("\n" + r["bt"])
        runs.append(r)

    if a.runs > 1:
        g1 = np.array([r["imitation"]["G"] for r in runs])
        g2 = np.array([r["final"]["G"] for r in runs])
        print("\n%-24s %8s %8s %8s" % ("", "mean", "spread", "meals"))
        print("%-24s %8.2f %8.2f %8.2f"
              % ("stage 1 (imitation)", g1.mean(), np.ptp(g1) / 2,
                 np.mean([r["imitation"]["eaten"] for r in runs])))
        print("%-24s %8.2f %8.2f %8.2f"
              % ("stage 2 (improvement)", g2.mean(), np.ptp(g2) / 2,
                 np.mean([r["final"]["eaten"] for r in runs])))
        print("%-24s %8.2f" % ("delta, paired", (g2 - g1).mean()))

    if a.oracle:
        env.seed_kernels(11)
        o = evaluate(env, CEMPolicy(env, np.random.default_rng(3)), n_ep=200,
                     T=200, seed=11, from_state=True)["G"].mean()
        print("\nCEM oracle (replans every tick, not deployable): %.2f" % o)

    if a.out:
        with open(a.out, "w") as fh:
            json.dump([{k: v for k, v in r.items()
                        if k not in ("vhat", "qhat", "bank")} for r in runs],
                      fh, indent=1, default=float)
        print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
