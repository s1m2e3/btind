"""Putting two arms back together, and the cost of adding one at all.

Every structural operator either removes an arm or nests one, so a tree
accumulates pairs that are one rule split in two -- and neither half is
droppable, because between them they own the region. `merge_arms` is the move
that closes that gap; like `drop_arm` it is judged on absence of loss, never on
a gain, because a compaction that preserves behaviour gains exactly zero.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.memory import check_arms
from btind.structure import _law_gap, _loosen, merge_arms


def _bank(d=6):
    """Two arms that are one rule at two thresholds, plus an unrelated third."""
    law = lambda v: np.full((d, 2), float(v))
    return dict(clauses=[[[0, 0.8, False]],            # x0 > 0.8
                         [[0, 0.2, False]],            # x0 > 0.2, same rule
                         [[3, 0.5, True]]],            # x3 <= 0.5, unrelated
                laws=[law(1.0), law(1.02), law(-4.0)],
                default=np.zeros((d, 2)),
                betas=[None, None, [[1, 0.2, True]]],
                sticky=[False, False, True])


def test_loosen_keeps_what_both_test_and_admits_more():
    # same column, same direction: the threshold that admits more states wins
    assert _loosen([[0, 0.8, False]], [[0, 0.2, False]]) == [[0, 0.2, False]]
    assert _loosen([[0, 0.8, True]], [[0, 0.2, True]]) == [[0, 0.8, True]]
    # a literal only one guard has is dropped: keeping it would exclude the
    # states the other arm owns
    assert _loosen([[0, 0.8, False], [2, 1.0, False]],
                   [[0, 0.2, False]]) == [[0, 0.2, False]]
    # NOTHING IN COMMON IS NOT A MERGE. An empty conjunction matches every
    # state, which would make the arm a second default and everything below it
    # unreachable, so the pair is refused instead.
    assert _loosen([[0, 0.8, False]], [[3, 0.5, True]]) is None
    assert _loosen([[0, 0.8, False]], [[0, 0.2, True]]) is None


def test_law_gap_orders_the_pairs_by_how_alike_they_are():
    a, b = np.ones((4, 2)), np.ones((4, 2)) * 1.02
    assert _law_gap(a, a) == 0.0
    assert 0.0 < _law_gap(a, b) < 0.05
    assert _law_gap(a, -a) > 0.9
    assert not np.isfinite(_law_gap(a, np.ones((5, 2))))   # not comparable


class _Env:
    """A world where the two alike arms are interchangeable and the third is
    not: the score depends only on how many states get the third arm's law."""

    def __init__(self, cost):
        self.cost = cost

    def rollout(self, bank, n_ep):
        return np.full(n_ep, self.cost(bank))


def _scored(cost):
    """`merge_arms` with `accept`/`score` replaced by a deterministic oracle."""
    import btind.structure as ST
    real_accept, real_score = ST.accept, ST.score

    def fake_score(env, bank, pol_fn, n_ep, T, seed, *a, **k):
        return np.full(n_ep, cost(bank))

    def fake_accept(env, cand, pol_fn, cur, n_ep, T, seed, z, side="gain",
                    margin=0.0, **k):
        g = fake_score(env, cand, pol_fn, n_ep, T, seed)
        d = float(g.mean() - np.mean(cur))
        ok = d > -margin if side == "noninferior" else d > margin
        return ok, d, g
    ST.accept, ST.score = fake_accept, fake_score
    try:
        yield_ = merge_arms(None, _bank(), None, n_ep=8, T=4, verbose=False)
    finally:
        ST.accept, ST.score = real_accept, real_score
    return yield_


def test_a_harmless_merge_is_taken_and_keeps_the_bank_consistent():
    # the two alike arms have the same law, so merging them changes nothing
    bank, _, log = _scored(lambda b: 0.0)
    assert len(bank["clauses"]) == 2
    check_arms(bank, "merge")
    assert bank["clauses"][0] == [[0, 0.2, False]]      # loosened, not dropped
    assert bank["clauses"][1] == [[3, 0.5, True]]       # the unrelated arm stays
    assert bank["sticky"][1] is True and bank["betas"][1] == [[1, 0.2, True]]
    assert log[0]["accepted"] and log[0]["arms"] == (0, 1)


def test_a_merge_that_costs_the_tree_is_refused():
    # any tree with fewer than three arms scores worse: every merge loses
    bank, _, log = _scored(lambda b: -1000.0 if len(b["clauses"]) < 3 else 0.0)
    assert len(bank["clauses"]) == 3
    assert log and not any(e["accepted"] for e in log)


if __name__ == "__main__":
    test_loosen_keeps_what_both_test_and_admits_more()
    test_law_gap_orders_the_pairs_by_how_alike_they_are()
    test_a_harmless_merge_is_taken_and_keeps_the_bank_consistent()
    test_a_merge_that_costs_the_tree_is_refused()
    print("ok")
