"""A coefficient that changes nothing is not a rule.

The fixed-k sparse prior keeps exactly `prior_k` slopes whether the evidence
supports ten or one, and the clip makes some of them inert by construction: a
law asking for +3.04 where the world allows +2.60 argues about a command that
never happens. Measured on the car's round-3 tree, 15 of its 60 slopes could be
zeroed with a paired delta of +0.00 +-0.00 over 300 episodes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.collect import design_matrix
from btind.structure import dead_slopes, fold_constants, prune_laws


def _rows(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, 200, n),      # metres
                            rng.normal(0, 1, n),
                            rng.uniform(0, 1, n)])


U = (-4.5, 2.6)


def test_a_saturated_law_has_no_live_slopes():
    X = design_matrix(_rows())
    th = np.zeros((4, 2))
    th[-1, 0], th[0, 0], th[1, 0], th[2, 0] = 9.0, 0.01, 0.3, 0.5
    assert dead_slopes(th, X, U, cols=(0,)) == [0, 1, 2]
    assert np.clip(X @ th[:, 0], *U).std() < 1e-12      # it IS a constant


def test_a_live_slope_is_kept_and_a_negligible_one_is_not():
    X = design_matrix(_rows())
    th = np.zeros((4, 2))
    th[1, 0], th[2, 0] = 1.0, 1e-9
    assert dead_slopes(th, X, U, cols=(0,)) == [2]


def test_only_the_sign_of_a_declaration_is_read():
    """Twice the logit is the same message, so a slope that never flips is dead."""
    X = design_matrix(_rows())
    never = np.zeros((4, 2))
    never[-1, 1], never[1, 1] = 5.0, 0.1              # swamped by the intercept
    assert dead_slopes(never, X, U, cols=(0, 1)) == [1]
    flips = np.zeros((4, 2))
    flips[1, 1] = 1.0
    assert dead_slopes(flips, X, U, cols=(0, 1)) == []


def test_a_slope_on_a_constant_column_folds_into_the_intercept():
    """It is a disguised intercept: dropping it would move the command by 1.2."""
    Z = _rows()
    Z[:, 1] = 3.0                                      # constant on these rows
    X = design_matrix(Z)
    th = np.zeros((4, 2))
    th[1, 0] = 0.4
    assert dead_slopes(th, X, U, cols=(0,)) == []      # NOT dead as written
    folded = fold_constants(th, X)
    assert folded[1, 0] == 0.0 and abs(folded[-1, 0] - 1.2) < 1e-12
    assert np.abs(np.clip(X @ th[:, 0], *U)
                  - np.clip(X @ folded[:, 0], *U)).max() < 1e-12


def test_pruning_at_frac_zero_cannot_change_the_command():
    """The property the whole pass rests on, on every row of every region."""
    rng = np.random.default_rng(3)
    Z = _rows(600, seed=1)
    bank = dict(clauses=[[[0, 100.0, False]], [[2, 0.5, True]]],
                laws=[rng.normal(0, 2, (4, 2)), rng.normal(0, 2, (4, 2))],
                default=rng.normal(0, 2, (4, 2)), head="pass", u_range=U,
                laws_on_z=True, names=["a", "b", "c"])
    pruned, n_drop = prune_laws(bank, Z, frac=0.0)
    X = design_matrix(Z)
    from btind.landscape import _match_cols
    done = np.zeros(len(Z), bool)
    for c, cl in enumerate(bank["clauses"] + [None]):
        m = (~done if cl is None else (_match_cols(cl, Z) & ~done))
        done |= m
        a = bank["default"] if cl is None else bank["laws"][c]
        b = pruned["default"] if cl is None else pruned["laws"][c]
        if not m.any():
            continue
        assert np.abs(np.clip(X[m] @ np.asarray(a)[:, 0], *U)
                      - np.clip(X[m] @ np.asarray(b)[:, 0], *U)).max() < 1e-12
        assert np.array_equal((X[m] @ np.asarray(a)[:, 1]) > 0,
                              (X[m] @ np.asarray(b)[:, 1]) > 0)


def test_a_bigger_frac_drops_at_least_as_much():
    rng = np.random.default_rng(5)
    Z = _rows(600, seed=2)
    bank = dict(clauses=[[[0, 100.0, False]]],
                laws=[rng.normal(0, 2, (4, 2))], default=rng.normal(0, 2, (4, 2)),
                head="pass", u_range=U, laws_on_z=True, names=["a", "b", "c"])
    counts = [prune_laws(bank, Z, frac=f)[1] for f in (0.0, 0.01, 0.1, 1.0)]
    assert counts == sorted(counts)
    assert counts[-1] == sum(int((np.abs(np.asarray(t)[:-1]).max(1) > 1e-12).sum())
                             for t in [bank["default"]] + bank["laws"])


def test_an_empty_region_prunes_nothing_and_does_not_raise():
    Z = _rows(200)
    bank = dict(clauses=[[[0, 1e9, False]]],              # matches no row
                laws=[np.ones((4, 2))], default=np.ones((4, 2)),
                head="pass", u_range=U, laws_on_z=True, names=["a", "b", "c"])
    pruned, n = prune_laws(bank, Z, frac=0.0)
    assert np.array_equal(pruned["laws"][0], bank["laws"][0])


def test_folding_is_off_by_default():
    """It is exact on the rows measured and not beyond them.

    Measured on the car's tree across two seeds and episode counts: without
    folding 15 slopes go for a paired +0.0000 +-0.0000; with it 21 go and the
    delta is -7.79 +-7.27 and -10.50 +-9.79, which the non-inferiority gate
    rejects. Six coefficients are not worth losing the move.
    """
    Z = _rows(600, seed=4)
    Z[:, 2] = 0.25                                     # constant on these rows
    rng = np.random.default_rng(7)
    bank = dict(clauses=[[[0, 100.0, False]]], laws=[rng.normal(0, 2, (4, 2))],
                default=rng.normal(0, 2, (4, 2)), head="pass", u_range=U,
                laws_on_z=True, names=["a", "b", "c"])
    plain = prune_laws(bank, Z, frac=0.0)[1]
    folded = prune_laws(bank, Z, frac=0.0, fold=True)[1]
    assert folded > plain
