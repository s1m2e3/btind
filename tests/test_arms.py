"""Every arm operator must move `betas` and `sticky` with the arm.

These lists are indexed by arm exactly like `clauses` and `laws`, and for most
of this project they were always empty, so an operator that rebuilt two of the
four was indistinguishable from a correct one. Terminations made them non-empty
and six operators turned out to be wrong at once -- and the symptom appeared
nowhere near the cause: `MemBank.arbitrate` broadcasts `sticky` into an array
sized by the clause count, so the traceback named whichever stage happened to
roll the next rollout, two stages and sometimes two rounds downstream.

The test is cheap and exhaustive because the invariant is simple: after any
structural move, all four lists have the same length and the flags still belong
to the arms they were set on.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.escape import _drop, _swap
from btind.landscape import materialise_default
from btind.memory import check_arms, insert_arm, reindex
from btind.structure import _insert


def _bank(n=4, d=6):
    return dict(clauses=[[[i, 0.5, False]] for i in range(n)],
                laws=[np.full((d, 2), float(i)) for i in range(n)],
                default=np.zeros((d, 2)),
                betas=[None, [[1, 0.2, True]], None, None],
                sticky=[False, True, False, True])


def test_insert_keeps_lists_aligned():
    for pos in range(5):
        b = insert_arm(_bank(), [[0, 0.1, False]], np.zeros((6, 2)), pos)
        check_arms(b, "insert@%d" % pos)
        assert len(b["clauses"]) == 5
        shift = lambda i: i + (1 if i >= pos else 0)
        assert b["sticky"][shift(1)] is True and b["sticky"][shift(3)] is True
        assert b["betas"][shift(1)] == [[1, 0.2, True]]


def test_reindex_permutes_and_filters():
    b = reindex(_bank(), [3, 2, 1, 0])
    check_arms(b, "reverse")
    assert b["sticky"] == [True, False, True, False]
    b = reindex(_bank(), [0, 2, 3])
    check_arms(b, "drop")
    assert b["sticky"] == [False, False, True] and b["betas"][2] is None


def test_kicks_keep_lists_aligned():
    rng = np.random.default_rng(0)
    for _ in range(50):
        for f in (_drop, _swap):
            out, _ = f(_bank(), rng)
            check_arms(out, f.__name__)


def test_structure_insert_and_materialise():
    check_arms(_insert(_bank(), [[2, 0.3, True]], np.zeros((6, 2)), 2), "_insert")
    check_arms(_insert(_bank(), [[2, 0.3, True]], np.zeros((6, 2)), -1), "_insert-1")
    b, made = materialise_default(_bank())
    if made:
        check_arms(b, "materialise_default")


def test_check_arms_actually_catches_it():
    try:
        check_arms(dict(_bank(), sticky=[False, True]), "bad")
    except ValueError:
        return
    raise AssertionError("check_arms passed 2 sticky flags for 4 arms")


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
