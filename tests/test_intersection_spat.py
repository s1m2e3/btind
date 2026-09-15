"""The V2I message never lies, whatever controller runs the light.

For every message a car receives, the tick at which its light is next observed
to change must fall inside the [earliest, latest] window it was told -- exactly,
no tolerance -- under the fixed plan, a green-time (duration) tree, and a
per-tick EXTEND/SWITCH tree. And the fixed plan, being fully known, must send a
window of width zero. Then the fused kernel must reproduce the numpy model to
the bit in event mode under both tree controllers.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import (SIG_HIDDEN, T_MAX, T_MIN, IntersectionBatch,
                                     _kind_of)
from btind.envs import intersection_fast as IF
from btind.memory import MemBank, check_arms, mem_names


def _signal_banks(env):
    SN = env.sig_names
    sx = SN.index
    d = len(mem_names(SN, None)) + 1

    def pref(k):
        th = np.zeros((d, 2))
        th[-1, k] = 1.0
        return th
    argmax = check_arms(dict(
        names=list(SN), laws_on_z=True, head="argmax", actions=["EXTEND", "SWITCH"],
        clauses=[[[sx("t_phase"), 18.0, False]],
                 [[sx("ph0"), 0.5, False], [sx("n0"), 0.5, True]]],
        laws=[pref(1), pref(1)], default=pref(0)), "argmax-signal")
    laws = []
    for k in range(4):
        th = np.zeros((d, 1))
        th[sx("q%d" % k), 0] = 3.0
        th[-1, 0] = 7.3                       # not a multiple of dt, on purpose
        laws.append(th)
    default = np.zeros((d, 1))
    default[-1, 0] = 14.1
    duration = check_arms(dict(
        names=list(SN), laws_on_z=True, head="duration", u_range=(T_MIN, T_MAX),
        clauses=[[[sx("ph%d" % k), 0.5, False]] for k in range(4)],
        laws=laws, default=default), "duration-signal")
    return {"fixed": None, "duration": duration, "argmax": argmax}


def _messages_and_changes(env, sb, n_ep=60, seed=4):
    """Run the numpy model; return every message and the observed light history."""
    N = env.N
    names = env.veh_names
    vb = env.default_vehicle_bank()
    s = env.sample_starts(n_ep, np.random.default_rng(seed))
    env._heads(vb, sb)
    env._sig_kind_override = _kind_of(sb)
    pv = MemBank(vb, len(names))
    pv.reset(n_ep * N)
    ps = None
    if sb is not None:
        ps = MemBank(sb, len(env.sig_names))
        ps.reset(n_ep)
    hist, msgs = [], []
    try:
        for t in range(env.duration):
            X = s[:, 5 * N:5 * N + env.N_MISC]
            hist.append(env._green_now(X).copy())             # (n, M) at obs time
            o = env.observe_vehicles(s).reshape(n_ep, N, -1)
            act = s[:, 3 * N:4 * N] == 1.0
            heard = act & (o[:, :, names.index("green")] > SIG_HIDDEN + 0.5)
            for i, q in zip(*np.nonzero(heard)):
                msgs.append((i, int(s[i, q]), t, o[i, q, names.index("t_sig")],
                             o[i, q, names.index("t_sig_max")],
                             o[i, q, names.index("green")]))
            a_v = pv.act(o.reshape(n_ep * N, -1))
            a_s = None if ps is None else ps.act(env.observe_signal(s))
            s, _, _ = env.step_both(s, a_v, a_s)
    finally:
        env._sig_kind_override = None
    return msgs, np.array(hist)


def test_message_window_always_contains_the_change():
    env = IntersectionBatch(sig_mode="event")
    for kind, sb in _signal_banks(env).items():
        msgs, hist = _messages_and_changes(env, sb)
        assert len(msgs) > 200, (kind, len(msgs))
        checked, widths = 0, []
        for i, m, t0, tmin, tmax, told_green in msgs:
            status = hist[t0, i, m]
            assert bool(status) == (told_green > 0.5)
            later = np.flatnonzero(hist[t0 + 1:, i, m] != status)
            widths.append(tmax - tmin)
            assert 1.0 <= tmin <= tmax, (kind, tmin, tmax)
            if len(later):
                k = later[0] + 1
                assert tmin <= k <= tmax, (kind, "told [%g, %g], changed after %d"
                                           % (tmin, tmax, k))
                checked += 1
            else:
                assert tmax >= env.duration - t0, (kind, tmax, env.duration - t0)
        assert checked > 100, (kind, checked)
        if kind == "fixed":
            assert max(widths) == 0.0          # a known plan sends an exact time
        else:
            assert max(widths) > 0.0           # an actuated one cannot


def test_event_kernel_exact_under_tree_signals():
    env = IntersectionBatch(n_max=24, T_end=60.0, sig_mode="event")
    s = env.sample_starts(50, np.random.default_rng(8))
    vb = env.default_vehicle_bank()
    for kind, sb in _signal_banks(env).items():
        gk = IF.run(env, vb, sb, s, env.duration)
        gp = env.python_rollout(vb, sb, s)
        assert np.abs(gk - gp).max() < 1e-9, (kind, np.abs(gk - gp).max())


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
