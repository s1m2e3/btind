"""Event-mode V2I: the message is delivered once, and the kernel agrees.

The message must appear on exactly one tick per vehicle, carry a time to
change that counts down to the actual change under the fixed plan, and be
gone afterwards. A blackboard with the countdown flag must expose `left_<c>`
columns that the kernel and the numpy path compute identically.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import (ACTIONS, IntersectionBatch, NEAR_INT,
                                     SIG_HIDDEN)
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, mem_names
from btind.tick import trace_array

ENV = IntersectionBatch(n_max=24, T_end=60.0, sig_mode="event")
N = ENV.veh_names
IX = N.index


def _pref(k, d, n=5):
    th = np.zeros((d, n))
    th[-1, k] = 1.0
    return th


def cruise_bank(mem=None):
    d = len(mem_names(N, mem)) + 1
    return check_arms(dict(names=list(N), laws_on_z=True, head="argmax",
                           actions=list(ACTIONS), clauses=[], laws=[],
                           default=_pref(2, d), mem=mem), "cruise")


def test_message_arrives_once_and_counts_down():
    s = ENV.sample_starts(30, np.random.default_rng(3))
    b = cruise_bank()
    d = len(mem_names(N, None)) + 1
    seen = 0
    for q in range(ENV.N):
        dev = np.zeros((len(s), 4))
        dev[:, 3] = q
        tr = trace_array(len(s), ENV.duration, d)
        IF.run(ENV, b, None, s, ENV.duration, trace=tr, dev=dev)
        alive = tr[:, :, d - 1] > 0.5
        for i in range(len(s)):
            on = np.flatnonzero(alive[i])
            if not len(on):
                continue
            z = tr[i, on]
            heard = np.flatnonzero(z[:, IX("green")] > SIG_HIDDEN + 0.5)
            assert len(heard) <= 1, "message delivered %d times" % len(heard)
            if len(heard) == 1:
                seen += 1
                k = heard[0]
                assert 0.0 < z[k, IX("d_stop")] <= NEAR_INT
                assert z[k, IX("t_sig")] >= 0.0
                # every other tick the three columns are hidden
                rest = np.delete(np.arange(len(on)), k)
                assert np.all(z[rest, IX("green")] == SIG_HIDDEN)
                assert np.all(z[rest, IX("t_sig")] == SIG_HIDDEN)
    assert seen > 20


def test_event_kernel_matches_model_with_countdown_memory():
    mem = dict(cols=[IX("green"), IX("t_sig")],
               write=[[IX("green"), SIG_HIDDEN + 0.5, False]], clear=None,
               countdown=True)
    zn = mem_names(N, mem)
    assert zn[-2:] == ["left_green", "left_t_sig"]
    d = len(zn) + 1
    P = lambda k: _pref(k, d)
    zi = zn.index
    b = dict(cruise_bank(mem), default=P(4),
             clauses=[[[IX("has_lead"), 0.5, False], [IX("lead_gap"), 10.0, True]],
                      [[zi("have_mem"), 0.5, False], [zi("mem_green"), 0.5, True],
                       [zi("left_t_sig"), 0.0, False], [IX("d_stop"), 0.0, False]],
                      [[zi("have_mem"), 0.5, False], [zi("mem_green"), 0.5, False],
                       [zi("left_t_sig"), 6.0, True], [IX("d_stop"), 0.0, False]]],
             laws=[P(0), P(1), P(1)])
    check_arms(b, "event-mem")
    s = ENV.sample_starts(60, np.random.default_rng(5))
    gk = IF.run(ENV, b, None, s, ENV.duration)
    gp = ENV.python_rollout(b, None, s)
    assert np.abs(gk - gp).max() < 1e-9, np.abs(gk - gp).max()
    # and the remembered clock changed behaviour against the memoryless cruise
    assert np.abs(gk - IF.run(ENV, cruise_bank(), None, s, ENV.duration)).max() > 0


def test_continuous_mode_unchanged_by_the_event_machinery():
    env = IntersectionBatch(n_max=24, T_end=60.0)
    s = env.sample_starts(40, np.random.default_rng(7))
    b = cruise_bank()
    gk = IF.run(env, b, None, s, env.duration)
    gp = env.python_rollout(b, None, s)
    assert np.abs(gk - gp).max() < 1e-9


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
