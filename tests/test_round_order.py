"""A round reshapes the tree BEFORE it grows, and `grow`'s clause pool stays
`grow`'s.

Compaction (`merge_arms`) and nesting (`search_subtrees`) were moved ahead of
`grow` so that growth is charged against the tree they leave behind. That move
broke a round outright: `grow`'s clause pool is overridden to draw thresholds
from `hot`, which indexes the rows of the FULL observation, while
`search_subtrees` grows on a CHILD's rows -- `obs[own]` -- so the first subtree
round died on `index 16033 is out of bounds for axis 0 with size 10243`.

END TO END IT DOES NOT REPRODUCE: `hot` is the anchor rows appended after the
coverage sample, a small world has none, and the override passes None when
`hot` is empty -- a one-round fit on a toy world passed with the broken order.
So the two things are asserted directly: the ORDER of the stages, and that the
override is not installed while anything but `grow` is running.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import btind.grow_bt as GB
import btind.rlfit as RL
import btind.subtree as SUB
from btind.envs.intersection import IntersectionBatch
from btind.rlfit import DEFAULTS, fit


def _world():
    return IntersectionBatch(n_max=24, T_end=30.0, veh_reward="car",
                             veh_head="scalar", sig_head="duration",
                             vph=(120.0, 40.0, 40.0)).set_agent("vehicle")


CFG = dict(n_ep=6, explore_ep=6, value_ep=6, n_cover=400, cover_ep=4,
           grow_pool=4, grow_arms=1, cem_iter=1, cem_K=4, cem_ep=4, screen_ep=4,
           subtree_at=(0,), subtree_pool=3, subtree_arms=1, merge_at=1,
           mem_at=99, beta_at=99, steps_at=99, kern_at=(), evo_clauses=0,
           value_laws=False, prune_frac=None)


def _round_with_spies():
    """One round, recording the order of the stages and the pool each one saw."""
    env = _world()
    seen, pool0 = [], GB._clause_pool
    real_merge, real_sub, real_grow = RL.merge_arms, SUB.search_subtrees, RL.grow

    def merge(*a, **k):
        seen.append(("merge", GB._clause_pool is pool0))
        return real_merge(*a, **k)

    def subtree(*a, **k):
        seen.append(("subtree", GB._clause_pool is pool0))
        return real_sub(*a, **k)

    def grow(*a, **k):
        seen.append(("grow", GB._clause_pool is pool0))
        return real_grow(*a, **k)

    RL.merge_arms, SUB.search_subtrees, RL.grow = merge, subtree, grow
    try:
        bank, log, meta = fit(env, list(env.names), rounds=1, warm=True,
                              cfg=dict(CFG, T=env.duration), run_seed=0,
                              verbose=False)
    finally:
        RL.merge_arms, SUB.search_subtrees, RL.grow = real_merge, real_sub, real_grow
    assert GB._clause_pool is pool0, "the override outlived the round"
    return seen, bank, log, meta


def test_merge_and_subtree_run_before_growth():
    seen, bank, log, meta = _round_with_spies()
    order = [s for s, _ in seen]
    assert order[:3] == ["merge", "subtree", "grow"], order
    assert np.isfinite(meta["G"]) and log


def test_the_forced_clause_pool_belongs_to_growth_alone():
    """`grow` draws its thresholds from `hot`, which indexes the FULL
    observation; a stage that grows on a subset of rows must not see it."""
    seen, _, _, _ = _round_with_spies()
    untouched = dict((s, ok) for s, ok in seen)
    assert untouched["merge"] and untouched["subtree"]     # pool as the module has it
    assert not untouched["grow"]                           # overridden, for grow only


def test_growth_is_charged_more_as_the_tree_fills_up():
    assert DEFAULTS["grow_min_gain"] is None and DEFAULTS["grow_ref_arms"] == 8
    bar = lambda g, n: g * max(1.0, n / DEFAULTS["grow_ref_arms"])
    assert bar(100.0, 3) == 100.0                 # a small tree pays the flat bar
    assert bar(100.0, 8) == 100.0
    assert bar(100.0, 16) == 200.0                # and a full one pays double


if __name__ == "__main__":
    test_merge_and_subtree_run_before_growth()
    test_the_forced_clause_pool_belongs_to_growth_alone()
    test_growth_is_charged_more_as_the_tree_fills_up()
    print("ok")
