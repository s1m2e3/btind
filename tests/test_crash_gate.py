"""`accept` rejects a candidate that crashes more than the incumbent on the same
episodes, whatever its return (`structure.CRASH_GATE`).

The incumbent's counts are planted rather than rolled: the gate is a comparison
of two totals, and what has to be true is that a winning return does not get a
candidate past a higher total, that the counts reach `accept` by the contents of
the returns array (a copy still finds them), and that turning the gate off
restores the return test alone.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import btind.structure as ST
from btind.envs.intersection import IntersectionBatch
from btind.memory import MemBank

N_EP, T, SEED = 8, 40, 5


def _world():
    return IntersectionBatch(n_max=24, T_end=20.0, veh_reward="car",
                             veh_head="scalar", sig_head="duration",
                             vph=(120.0, 40.0, 40.0)).set_agent("vehicle")


def _pol(b):
    return MemBank(b, len(b["names"]))


def _setup():
    env = _world()
    bank = env.default_vehicle_bank()
    bank["names"] = list(env.veh_names)
    g = ST.score(env, bank, _pol, N_EP, T, SEED)
    c = ST.crashes_of(g, N_EP, T, SEED)
    return env, bank, g, c


def test_score_records_crash_counts_by_content():
    _, _, g, c = _setup()
    assert c is not None and c.shape == g.shape and (c >= 0).all()
    assert ST.crashes_of(g.copy(), N_EP, T, SEED) is not None     # a copy finds them
    assert ST.crashes_of(g, N_EP, T, SEED + 1) is None             # other episodes do not


def test_a_better_return_does_not_buy_more_crashes():
    env, bank, g, c = _setup()
    # an incumbent far worse on return, and with one crash FEWER in total
    ref = g - 1e4
    fewer = c.copy()
    fewer[0] -= 1.0
    ST._remember_crashes(ref, N_EP, T, SEED, fewer)
    ok, d, _ = ST.accept(env, bank, _pol, ref, N_EP, T, SEED, z=1.0)
    assert d > 0 and not ok
    # the same crashes as the incumbent: the return test decides
    ST._remember_crashes(ref, N_EP, T, SEED, c)
    ok, _, _ = ST.accept(env, bank, _pol, ref, N_EP, T, SEED, z=1.0)
    assert ok


def test_the_gate_can_be_switched_off():
    env, bank, g, c = _setup()
    ref = g - 1e4
    fewer = c.copy()
    fewer[0] -= 1.0
    ST._remember_crashes(ref, N_EP, T, SEED, fewer)
    ST.CRASH_GATE = False
    try:
        ok, _, _ = ST.accept(env, bank, _pol, ref, N_EP, T, SEED, z=1.0)
    finally:
        ST.CRASH_GATE = True
    assert ok


def test_the_incumbent_survives_a_long_screen():
    """`grow` scores its incumbent, screens hundreds of candidates, then
    confirms: the bounded memo had evicted the incumbent by then and every step
    logged "crashes unchecked". Held by the caller, its counts stay found."""
    _, _, g, c = _setup()
    for i in range(ST._CRASH_MEMO_N + 100):             # a long screen
        ST._remember_crashes(np.full(N_EP, float(i)), N_EP, T, SEED, np.zeros(N_EP))
    assert ST.crashes_of(g, N_EP, T, SEED) is not None
    assert np.array_equal(ST.crashes_of(g, N_EP, T, SEED), c)
    assert ST.crashes_of(g, N_EP, T, SEED + 1) is None   # still keyed by episodes


def test_fewer_crashes_at_no_cost_is_accepted_and_skips_the_bar():
    """SAFER_ACCEPTS: strictly fewer crashes and a mean return delta >= 0
    passes `accept` and `clears` a stage's `min_gain` without a return gain."""
    env, bank, g, c = _setup()
    more = c + 1.0                              # the incumbent crashed more
    ref = g.copy()                              # and returned exactly the same
    ST._remember_crashes(ref, N_EP, T, SEED, more)
    ok, d, _ = ST.accept(env, bank, _pol, ref, N_EP, T, SEED, z=1.0)
    assert ok and d == 0.0
    assert ST.clears(ok, d, 100.0)              # the +100 grow bar does not apply
    # a loss of return is not bought back by a crash removed
    ref2 = g + 1.0
    ST._remember_crashes(ref2, N_EP, T, SEED, more)
    ok, d, _ = ST.accept(env, bank, _pol, ref2, N_EP, T, SEED, z=1.0)
    assert not ok and not ST.clears(ok, d, 100.0)
    # the same crashes: the bar applies as before
    ref3 = g - 5.0
    ST._remember_crashes(ref3, N_EP, T, SEED, c)
    ok, d, _ = ST.accept(env, bank, _pol, ref3, N_EP, T, SEED, z=1.0)
    assert ok and not ST.clears(ok, d, 100.0) and ST.clears(ok, d, 1.0)


if __name__ == "__main__":
    test_fewer_crashes_at_no_cost_is_accepted_and_skips_the_bar()
    test_the_incumbent_survives_a_long_screen()
    test_score_records_crash_counts_by_content()
    test_a_better_return_does_not_buy_more_crashes()
    test_the_gate_can_be_switched_off()
    print("ok")
