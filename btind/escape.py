"""Getting out of a basin, on purpose.

EVERY OPERATOR IN THIS PROJECT ACCEPTS ONLY IMPROVEMENTS. That is what makes
the numbers trustworthy -- nothing enters the tree without clearing a paired
test against the incumbent -- and it is also why a warm-started run reproduces
itself exactly: measured, two masked runs resumed from the same stored bank
with the same seeds returned character-identical trees. A monotone search that
has run out of accepted moves is finished, and running it again finds nothing.

BASIN-HOPPING is the standard answer (Wales & Doye 1997): take a deliberate
step that is NOT an improvement, re-optimise from there, and keep the result
only if it beats where you started. The monotone machinery is untouched -- it
is what does the re-optimisation -- and the guarantee is preserved at the outer
level, because a hop that does not pay is discarded whole.

THE KICKS ARE STRUCTURAL, NOT PARAMETRIC. Jittering a threshold is what
`polish_thresholds` already does under a test, so a kick that only jitters is a
worse version of a stage that exists. What the monotone search cannot reach is
a tree with a DIFFERENT SET OF ARMS: dropping an arm that currently pays for
itself, or moving one past a neighbour it dominates, opens regions that no
single accepted move leads to, because the intermediate state is worse than
both ends. That is precisely the shape of a basin wall.

WHY REVERSION IS AT THE OUTER LEVEL. A hop is judged on the bank that comes
back after a full round of growth and refitting, not on the kicked bank, which
is worse by construction. Judging the kick itself would reject every hop.
"""
import numpy as np


def _drop(bank, rng):
    """Remove one arm that is currently earning its place."""
    n = len(bank["clauses"])
    if n < 2:
        return None, ""
    c = int(rng.integers(n))
    return dict(bank,
                clauses=[x for i, x in enumerate(bank["clauses"]) if i != c],
                laws=[x for i, x in enumerate(bank["laws"]) if i != c]), \
        "drop arm %d" % c


def _swap(bank, rng):
    """Exchange two adjacent arms, changing what each one is asked to cover."""
    n = len(bank["clauses"])
    if n < 2:
        return None, ""
    c = int(rng.integers(n - 1))
    cl, laws = list(bank["clauses"]), list(bank["laws"])
    cl[c], cl[c + 1] = cl[c + 1], cl[c]
    laws[c], laws[c + 1] = laws[c + 1], laws[c]
    return dict(bank, clauses=cl, laws=laws), "swap arms %d,%d" % (c, c + 1)


def _relax(bank, rng, Z, frac=0.35):
    """Widen one guard sharply, handing its arm a region it was not fitted for.

    Not a jitter: the step is a large fraction of the column's spread and in the
    loosening direction only, so the arm claims states the arms below currently
    own. The refit that follows is what decides whether that was worth doing.
    """
    n = len(bank["clauses"])
    if not n:
        return None, ""
    c = int(rng.integers(n))
    k = int(rng.integers(len(bank["clauses"][c])))
    j, thr, neg = bank["clauses"][c][k]
    if j >= Z.shape[1]:
        return None, ""
    span = float(np.subtract(*np.percentile(Z[:, j], [75, 25])) or 1.0)
    cl = [[l[:] for l in x] for x in bank["clauses"]]
    cl[c][k][1] = float(thr + (frac * span if neg else -frac * span))
    return dict(bank, clauses=cl), "relax arm %d lit %d" % (c, k)


KICKS = (_drop, _swap, _relax)


def kick(bank, Z, rng, n=1):
    """One deliberate non-improving move, applied without a test."""
    out, why = bank, []
    for _ in range(n):
        f = KICKS[int(rng.integers(len(KICKS)))]
        b, lab = (f(out, rng, Z) if f is _relax else f(out, rng))
        if b is not None:
            out, _ = b, why.append(lab)
    return out, "; ".join(why) or "no-op"
