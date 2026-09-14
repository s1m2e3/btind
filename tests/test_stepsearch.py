"""The step and fail search: plumbing, and one planted discovery.

The planted case is the one effect e26 measured on highway: a latched
lane-change arm loses 2.5 return units for want of a fail clause on the lead
gap. `search_fails` is given the alphabet every guard is drawn from, planted
distractors included, and has to find a literal on `v1_dx` that repairs it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.highway_batch import ACTIONS, HighwayBatch
from btind.envs.highway_task import constant_bank, law_width
from btind.memory import MemBank, check_arms, mem_names
from btind.stepsearch import search_fails, search_steps, with_fail, with_step
from btind.structure import score

ENV = HighwayBatch()
N = ENV.names
ZN = mem_names(N, None)
D = law_width(N)
POL = lambda b: MemBank(b, len(N))


def _pref(k):
    th = np.zeros((D, 5))
    th[-1, k] = 1.0
    return th


def _ix(n):
    return N.index(n)


def _cover(n=3000):
    obs = ENV.observe(ENV.sample_states(n, np.random.default_rng(0)))
    return np.hstack([obs, np.zeros((n, 2))])


def test_with_step_and_with_fail_keep_arms_valid():
    b = dict(constant_bank(N, 3), clauses=[[[_ix("v1_dx"), 20.0, True]]],
             laws=[_pref(4)])
    b2 = with_step(b, 0, [[_ix("ego_vx"), 23.0, True]], _pref(0))
    check_arms(b2, "with_step")
    assert b2["sticky"] == [True] and len(b2["steps"][0]) == 1
    assert b.get("steps") is None                   # the incumbent is untouched
    b3 = with_fail(b2, 0, [[_ix("v1_dx"), 8.0, True]])
    check_arms(b3, "with_fail")
    assert b3["fails"][0] == [[_ix("v1_dx"), 8.0, True]]


def test_search_steps_runs_and_returns_a_valid_bank():
    b = dict(constant_bank(N, 3), actions=list(ACTIONS),
             clauses=[[[_ix("v1_dx"), 25.0, True], [_ix("v1_dx"), 0.0, False]]],
             laws=[_pref(4)], sticky=[True],
             betas=[[[_ix("v1_dx"), 30.0, False]]])
    out, log = search_steps(ENV, b, ZN, _cover(), POL, n_adv=3, screen_ep=60,
                            confirm_ep=200, n_confirm=2, T=40, seed=11,
                            rng=np.random.default_rng(0), verbose=False,
                            n_law_sample=4)
    check_arms(out, "search_steps")
    assert log and all("clause" in e and "accepted" in e for e in log)
    assert all(e["kind"] == "step" for e in log)


def test_search_fails_finds_the_planted_cut_in_repair():
    ix = _ix
    close = [[ix("v1_dx"), 25.0, True], [ix("v1_dx"), 0.0, False],
             [ix("v1_dvx"), 0.0, True]]
    b = dict(constant_bank(N, 3), actions=list(ACTIONS),
             clauses=[close + [[ix("ego_y"), 2.0, False]], close],
             laws=[_pref(0), _pref(4)], sticky=[True, False],
             betas=[[[ix("v1_dx"), 30.0, False]], None],
             steps=[[([[ix("ego_vy"), -0.3, True]], _pref(1))], None])
    cur = score(ENV, b, POL, 600, 40, 11)
    out, log = search_fails(ENV, b, ZN, _cover(), POL, cur_G=cur, n_try=24,
                            n_ep=600, T=40, seed=11, rng=np.random.default_rng(0),
                            verbose=False)
    acc = [e for e in log if e["accepted"]]
    assert acc, "no fail clause was accepted; best %+.2f" % max(e["delta"] for e in log)
    best = max(acc, key=lambda e: e["delta"])
    assert best["delta"] > 1.0, best
    cols = {ZN[l[0]] for l in out["fails"][0]}
    assert cols <= {"v1_dx", "v1_dvx"}, "fail clause reads %s" % cols
    assert not cols & {"t_norm", "noise"}


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
