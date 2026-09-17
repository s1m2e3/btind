"""A re-run picks up the best tree, not wherever the last one stopped.

A round is kept whenever it is not SIGNIFICANTLY worse, so a run can drift
down and did: rounds 3 and 4 of a real run lost 10.6 and 14.9 against a best
two rounds behind them. Restarting from `st["sig"]` restarts from the drift.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments"))

import pytest

import e38_signal_only as E


@pytest.fixture
def best_file(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "BEST", str(tmp_path / "best.json"))
    return tmp_path / "best.json"


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def test_a_best_of_this_configuration_loads(best_file):
    _write(best_file, dict(round=3, train=-8700.0, se=68.0,
                           sig={"clauses": []}, config=E.config_key()))
    b = E.load_best()
    assert b is not None and b["round"] == 3


def test_a_best_from_another_configuration_is_refused(best_file):
    cfg = dict(E.config_key())
    cfg["prior"] = "const"
    _write(best_file, dict(round=3, train=-8700.0, se=68.0,
                           sig={"clauses": []}, config=cfg))
    assert E.load_best() is None


def test_a_best_with_no_provenance_is_refused(best_file):
    """Files predating the law class in the key carry no provenance, and
    loading one under a class that can carry slopes is the silent swap."""
    _write(best_file, dict(round=1, train=-8958.9, se=70.5, sig={"clauses": []}))
    assert E.load_best() is None


def test_the_config_key_reports_the_prior_actually_in_use():
    """It asserted prior='const' as a literal while the cfg used 'sparse'."""
    assert E.config_key()["prior"] == E.PRIOR
    assert E.config_key()["prior_k"] == E.PRIOR_K


def test_update_best_stamps_the_configuration(best_file, monkeypatch):
    st = dict(sig={"clauses": []}, best=None)
    rows = {E.MAIN_BAND: {"discovered": -8700.0, "se": 68.0}}
    E.update_best(st, rows, 4)
    assert st["best"]["config"] == E.config_key()
    assert E.load_best()["round"] == 4
