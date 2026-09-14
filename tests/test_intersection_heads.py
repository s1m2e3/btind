"""Continuous leaves on the intersection: kernel and model agree to the bit.

A `scalar` vehicle bank commands an affine acceleration; a `duration` signal
bank commands an affine green time read once per phase. Both are held to the
numpy reference, alone and together, and each must change behaviour against
its discrete counterpart so the test is not passing on an inert head.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import (A_MAX, B_MAX, T_MAX, T_MIN, IntersectionBatch)
from btind.envs import intersection_fast as IF
from btind.memory import MemBank, check_arms, mem_names
from btind.policies import evaluate

ENV = IntersectionBatch(n_max=24, T_end=60.0, veh_head="scalar", sig_head="duration")
N, SN = ENV.veh_names, ENV.sig_names
IX, SX = N.index, SN.index


def scalar_follower():
    """An affine follower: accelerate toward V0, brake by gap and closing speed."""
    zn = mem_names(N, None)
    d = len(zn) + 1
    default = np.zeros((d, 1))
    default[IX("v"), 0] = -0.3
    default[-1, 0] = 3.3                      # a = 3.3 - 0.3 v  -> settles near V0
    brake = np.zeros((d, 1))
    brake[IX("lead_gap"), 0] = 0.15
    brake[IX("lead_dv"), 0] = -0.8
    brake[-1, 0] = -2.5                       # a = -2.5 + 0.15 gap - 0.8 dv
    red = np.zeros((d, 1))
    red[IX("d_stop"), 0] = 0.08
    red[IX("v"), 0] = -0.5
    red[-1, 0] = -2.0
    return check_arms(dict(
        names=list(N), laws_on_z=True, head="scalar", u_range=(-B_MAX, A_MAX),
        clauses=[[[IX("has_lead"), 0.5, False], [IX("lead_gap"), 30.0, True]],
                 [[IX("green"), 0.5, True], [IX("green"), -0.5, False],
                  [IX("d_stop"), 0.0, False], [IX("d_stop"), 50.0, True]]],
        laws=[brake, red], default=default), "scalar")


def duration_signal():
    """Green time affine in the phase's own queue: 8 s plus 3 s per queued car."""
    zn = mem_names(SN, None)
    d = len(zn) + 1
    laws = []
    for k in range(4):
        th = np.zeros((d, 1))
        th[SX("q%d" % k), 0] = 3.0
        th[-1, 0] = 8.0
        laws.append(th)
    default = np.zeros((d, 1))
    default[-1, 0] = 15.0
    return check_arms(dict(
        names=list(SN), laws_on_z=True, head="duration", u_range=(T_MIN, T_MAX),
        clauses=[[[SX("ph%d" % k), 0.5, False]] for k in range(4)],
        laws=laws, default=default), "duration")


def _starts(n=50, seed=9):
    return ENV.sample_starts(n, np.random.default_rng(seed))


def test_scalar_vehicles_exact_and_alive():
    s = _starts()
    vb = scalar_follower()
    gk = IF.run(ENV, vb, None, s, ENV.duration)
    gp = ENV.python_rollout(vb, None, s)
    assert np.abs(gk - gp).max() < 1e-9, np.abs(gk - gp).max()
    g_default = IF.run(ENV, ENV.default_vehicle_bank(), None, s, ENV.duration)
    assert np.abs(gk - g_default).max() > 0


def test_duration_signal_exact_and_alive():
    s = _starts()
    vb, sb = scalar_follower(), duration_signal()
    gk = IF.run(ENV, vb, sb, s, ENV.duration)
    gp = ENV.python_rollout(vb, sb, s)
    assert np.abs(gk - gp).max() < 1e-9, np.abs(gk - gp).max()
    assert np.abs(gk - IF.run(ENV, vb, None, s, ENV.duration)).max() > 0


def test_generic_interface_with_continuous_heads():
    s = _starts(30)
    vb, sb = scalar_follower(), duration_signal()
    ref = ENV.python_rollout(vb, sb, s)
    for agent, bank in (("vehicle", vb), ("signal", sb)):
        ENV.set_agent(agent)
        ENV.vehicle_bank = None if agent == "vehicle" else vb
        ENV.signal_bank = sb if agent == "vehicle" else None
        ENV._search_bank = bank
        keep = ENV.sample_starts
        ENV.sample_starts = lambda n, rng: s.copy()
        try:
            g = evaluate(ENV, MemBank(bank, len(ENV.names)), n_ep=len(s),
                         T=ENV.duration, seed=0)["G"]
        finally:
            ENV.sample_starts = keep
        assert np.abs(g - ref).max() < 1e-9, (agent, np.abs(g - ref).max())
    ENV.set_agent("vehicle")
    ENV.vehicle_bank = ENV.signal_bank = None


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
