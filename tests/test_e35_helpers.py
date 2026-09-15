"""The bookkeeping e35 relies on: the best pair is kept and a significantly worse
cycle is flagged for reversion; the kernel stage reports which loop produced the
points it kept; the episode budget covers every rung."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.e35_condition_cycle as E
from btind.kernsearch import source_summary


def _rows(g, se=5.0):
    return dict(train=dict(discovered=g, se=se))


def test_best_pair_is_kept_and_a_worse_cycle_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "BEST", str(tmp_path / "best.json"))
    st = {}
    assert E.update_best(st, {"v": 0}, None, _rows(-100.0), 0) is False
    assert st["best"]["cycle"] == 0
    # within noise: not reverted, best unchanged
    assert E.update_best(st, {"v": 1}, None, _rows(-105.0), 1) is False
    assert st["best"]["cycle"] == 0
    # better: becomes the best
    assert E.update_best(st, {"v": 2}, {"s": 2}, _rows(-80.0), 2) is False
    assert st["best"]["cycle"] == 2 and st["best"]["sig"] == {"s": 2}
    # more than two standard errors below: revert
    assert E.update_best(st, {"v": 3}, None, _rows(-110.0), 3) is True
    assert st["best"]["cycle"] == 2
    assert os.path.exists(E.BEST)


def test_source_summary_counts_per_loop():
    log = [dict(op="add", src="dev", accepted=True), dict(op="add", src="dev", accepted=False),
           dict(op="add", src="anchor", accepted=True), dict(op="cem", accepted=False),
           dict(op="prune", accepted=False)]
    assert source_summary(log) == "dev 2 confirmed/1 kept; anchor 1 confirmed/1 kept"
    assert source_summary([dict(op="prune")]) == "no proposal reached confirmation"


def test_every_rung_has_an_episode_scale():
    assert len(E.EP_SCALE) == len(E.RUNGS) and E.EP_SCALE[-1] == 1.0
    assert all(a >= b for a, b in zip(E.EP_SCALE, E.EP_SCALE[1:]))
