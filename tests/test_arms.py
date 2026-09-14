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
                sticky=[False, True, False, True],
                steps=[None, None, None, [([[2, 0.7, False]], np.ones((d, 2)))]],
                fails=[None, None, [[0, 0.9, False]], None])


def test_insert_keeps_lists_aligned():
    for pos in range(5):
        b = insert_arm(_bank(), [[0, 0.1, False]], np.zeros((6, 2)), pos)
        check_arms(b, "insert@%d" % pos)
        assert len(b["clauses"]) == 5
        shift = lambda i: i + (1 if i >= pos else 0)
        assert b["sticky"][shift(1)] is True and b["sticky"][shift(3)] is True
        assert b["betas"][shift(1)] == [[1, 0.2, True]]
        assert b["fails"][shift(2)] == [[0, 0.9, False]]
        assert b["steps"][shift(3)] is not None and b["steps"][pos] is None


def test_insert_with_steps_forces_sticky():
    b = insert_arm(_bank(), [[0, 0.1, False]], np.zeros((6, 2)), 1,
                   steps=[([[1, 0.5, True]], np.zeros((6, 2)))])
    check_arms(b, "insert-steps")
    assert b["sticky"][1] is True and len(b["steps"][1]) == 1
    # a bank that never had the lists gets them when an arm brings one
    plain = dict(clauses=[[[0, 0.5, False]]], laws=[np.zeros((6, 2))],
                 default=np.zeros((6, 2)))
    b = insert_arm(plain, [[1, 0.1, False]], np.zeros((6, 2)), 0,
                   fails=[[2, 0.3, True]])
    check_arms(b, "insert-fails-into-plain")
    assert b["fails"] == [[[2, 0.3, True]], None] and b.get("steps") is None


def test_reindex_permutes_and_filters():
    b = reindex(_bank(), [3, 2, 1, 0])
    check_arms(b, "reverse")
    assert b["sticky"] == [True, False, True, False]
    assert b["steps"][0] is not None and b["fails"][1] is not None
    b = reindex(_bank(), [0, 2, 3])
    check_arms(b, "drop")
    assert b["sticky"] == [False, False, True] and b["betas"][2] is None
    assert b["fails"][1] == [[0, 0.9, False]] and b["steps"][2] is not None


def test_check_arms_requires_sticky_for_steps():
    b = dict(_bank(), sticky=[False, True, False, False])
    try:
        check_arms(b, "unsticky-steps")
    except ValueError:
        return
    raise AssertionError("check_arms passed a multi-step arm that is not sticky")


def test_json_round_trip_keeps_steps_and_fails():
    from btind.runlog import bank_from_json, bank_json
    import json
    b = _bank()
    d = json.loads(json.dumps(bank_json(b, names=list("abcdef"[:4]))))
    r = bank_from_json(d)
    check_arms(r, "json")
    assert r["fails"] == b["fails"] and r["steps"][:3] == [None] * 3
    adv, th = r["steps"][3][0]
    assert adv == [[2, 0.7, False]] and np.array_equal(th, np.ones((6, 2)))


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
