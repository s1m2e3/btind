"""The fused intersection kernel must agree with the numpy model to the bit.

Both trees, the fixed-time plan, memory on the vehicle tree, steps and fails,
and a deviation of one vehicle and of the signal. The world is deterministic
once the arrival schedule is drawn, so agreement is exact or something is wrong.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import (ACTIONS, SIG_ACTIONS, IntersectionBatch)
from btind.envs import intersection_fast as IF
from btind.memory import check_arms, mem_names
from btind.tick import trace_array

ENV = IntersectionBatch(n_max=24, T_end=60.0)
N = ENV.names
SN = ENV.sig_names
IX = N.index


def _pref(k, d, n):
    th = np.zeros((d, n))
    th[-1, k] = 1.0
    return th


def veh_bank(mem=None, **kw):
    zn = mem_names(N, mem)
    d = len(zn) + 1
    P = lambda k: _pref(k, d, 5)
    b = dict(names=list(N), laws_on_z=True, head="argmax", actions=list(ACTIONS),
             clauses=[[[IX("has_lead"), 0.5, False], [IX("lead_gap"), 12.0, True]],
                      [[IX("green"), 0.5, True], [IX("d_stop"), 0.0, False],
                       [IX("d_stop"), 40.0, True]],
                      [[IX("near_int"), 0.5, False], [IX("d_conf"), 12.0, True]],
                      [[IX("has_lead"), 0.5, False], [IX("lead_gap"), 25.0, True],
                       [IX("lead_dv"), 0.0, False]]],
             laws=[P(0), P(1), P(1), P(1)], default=P(4), mem=mem)
    b.update(kw)
    return check_arms(b, "veh")


def sig_bank():
    d = len(mem_names(SN, None)) + 1
    P = lambda k: _pref(k, d, 2)
    return check_arms(dict(names=list(SN), laws_on_z=True, head="argmax",
                           actions=list(SIG_ACTIONS),
                           clauses=[[[SN.index("t_phase"), 20.0, False]],
                                    [[SN.index("q2"), 3.0, False], [SN.index("ph0"), 0.5, False]]],
                           laws=[P(1), P(1)], default=P(0)), "sig")


def _starts(n=60, seed=5):
    return ENV.sample_starts(n, np.random.default_rng(seed))


def _both(vb, sb, s, dev=None, py_dev=None):
    gk = IF.run(ENV, vb, sb, s, ENV.duration, dev=dev)
    gp = ENV.python_rollout(vb, sb, s, veh_dev=py_dev)
    return gp, gk


def test_fixed_plan_and_plain_tree_exact():
    s = _starts()
    gp, gk = _both(veh_bank(), None, s)
    assert np.abs(gp - gk).max() < 1e-9, np.abs(gp - gk).max()
    assert np.isfinite(gp).all() and gp.std() > 0


def test_signal_tree_exact():
    s = _starts()
    gp, gk = _both(veh_bank(), sig_bank(), s)
    assert np.abs(gp - gk).max() < 1e-9, np.abs(gp - gk).max()
    # and the signal tree changed something against the fixed plan
    assert np.abs(gk - IF.run(ENV, veh_bank(), None, s, ENV.duration)).max() > 0


def test_memory_steps_and_fails_exact():
    mem = dict(cols=[IX("lead_gap"), IX("v")],
               write=[[IX("green"), 0.5, True]], clear=None)
    zn = mem_names(N, mem)
    d = len(zn) + 1
    P = lambda k: _pref(k, d, 5)
    b = veh_bank(mem=mem)
    b["laws"] = [P(0), P(1), P(1), P(1)]
    b["default"] = P(4)
    b["sticky"] = [False, True, True, False]
    b["betas"] = [None, [[IX("green"), 0.5, False]], [[IX("d_conf"), 30.0, False]], None]
    b["steps"] = [None, [([[IX("v"), 1.0, True]], P(2))], None, None]
    b["fails"] = [None, None, [[IX("lead_gap"), 5.0, True]], None]
    check_arms(b, "mem-steps")
    s = _starts()
    gp, gk = _both(b, sig_bank(), s)
    assert np.abs(gp - gk).max() < 1e-9, np.abs(gp - gk).max()


def test_vehicle_and_signal_deviations_exact():
    s = _starts(40)
    rng = np.random.default_rng(1)
    dev = np.zeros((len(s), 4))
    dev[:, 0] = rng.integers(5, 60, len(s))
    dev[:, 1] = rng.choice([1, 4, 10], len(s))
    dev[:, 2] = rng.integers(0, 5, len(s))
    dev[:, 3] = rng.integers(0, ENV.N, len(s))          # one slot per episode
    vb = veh_bank()

    def py_dev(t, a):
        a = a.reshape(len(s), ENV.N).copy()
        m = (t >= dev[:, 0]) & (t < dev[:, 0] + dev[:, 1])
        rows = np.flatnonzero(m)
        a[rows, dev[rows, 3].astype(int)] = dev[rows, 2].astype(int)
        return a.reshape(-1)
    gp, gk = _both(vb, None, s, dev=dev, py_dev=py_dev)
    assert np.abs(gp - gk).max() < 1e-9, np.abs(gp - gk).max()
    g0 = IF.run(ENV, vb, None, s, ENV.duration)
    assert np.abs(gk - g0).max() > 0                     # the deviation did something
    # the signal as the deviating agent: overriding SWITCH/EXTEND
    dev_s = dev.copy()
    dev_s[:, 2] = rng.integers(0, 2, len(s))
    dev_s[:, 3] = -1
    tr = trace_array(len(s), ENV.duration, len(SN) + 3)
    gs = IF.run(ENV, vb, sig_bank(), s, ENV.duration, trace=tr, dev=dev_s)
    assert np.isfinite(gs).all()
    assert (tr[:, :, len(SN) + 2] > 0.5).all()          # signal traced every tick


def test_generic_interface_matches_reference_for_both_agents():
    """`observe`/`step` speak for the agent under search and drive the other
    from its fixed bank; run through `policies.evaluate` they must reproduce
    `python_rollout` and the kernel exactly."""
    from btind.memory import MemBank
    from btind.policies import evaluate
    s = _starts(40)
    vb, sb = veh_bank(), sig_bank()
    ref = ENV.python_rollout(vb, sb, s)
    for agent, bank, other in (("vehicle", vb, sb), ("signal", sb, vb)):
        ENV.set_agent(agent)
        if agent == "vehicle":
            ENV.signal_bank, ENV.vehicle_bank = sb, None
        else:
            ENV.signal_bank, ENV.vehicle_bank = None, vb
        pol = MemBank(bank, len(ENV.names))
        # evaluate() draws its own starts from the seed; feed ours by monkeypatch
        keep = ENV.sample_starts
        ENV.sample_starts = lambda n, rng: s.copy()
        try:
            g = evaluate(ENV, pol, n_ep=len(s), T=ENV.duration, seed=0)["G"]
        finally:
            ENV.sample_starts = keep
        assert np.abs(g - ref).max() < 1e-9, (agent, np.abs(g - ref).max())
        gk = IF.run(ENV, vb, sb, s, ENV.duration)
        assert np.abs(gk - ref).max() < 1e-9
    ENV.set_agent("vehicle")
    ENV.signal_bank = ENV.vehicle_bank = None


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
