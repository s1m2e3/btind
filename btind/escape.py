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

from .memory import reindex


def _drop(bank, rng):
    """Remove one arm that is currently earning its place."""
    n = len(bank["clauses"])
    if n < 2:
        return None, ""
    c = int(rng.integers(n))
    return reindex(bank, [i for i in range(n) if i != c]), "drop arm %d" % c


def _swap(bank, rng):
    """Exchange two adjacent arms, changing what each one is asked to cover."""
    n = len(bank["clauses"])
    if n < 2:
        return None, ""
    c = int(rng.integers(n - 1))
    order = list(range(n))
    order[c], order[c + 1] = order[c + 1], order[c]
    return reindex(bank, order), "swap arms %d,%d" % (c, c + 1)


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

# A KICK MUST NOT DESTROY A FITTED LAW. Every kick is meant to be recoverable by
# the monotone machinery that follows it, and two of the three are: `_swap` and
# `_relax` move arms and boundaries around but every law survives, so `grow`,
# `improve_laws` and `polish_thresholds` have something to work back from.
# `_drop` is different -- the law goes with the arm, and a law is the product of
# many rounds of CEM that mostly get rejected, so nothing downstream can rebuild
# one. Measured: a `_drop` on a four-arm tree took the run from 16.87 to -12.64
# and four further rounds recovered it to -4.92, against a best of 16.87 that
# only survived because the final revert put it back.
#
# So dropping is allowed only where an arm is a small part of the controller,
# and the caller is given a RECOVERY BUDGET: a hop that has not beaten the
# incumbent within it is abandoned and the next kick is proposed from the
# incumbent again, which is what basin-hopping actually prescribes. Continuing
# from the wreckage, as this did, is a random walk with a safety net.
MIN_ARMS_TO_DROP = 5


def kick(bank, Z, rng, n=1):
    """One deliberate non-improving move, applied without a test."""
    out, why = bank, []
    for _ in range(n):
        pool = [f for f in KICKS
                if f is not _drop or len(out["clauses"]) >= MIN_ARMS_TO_DROP]
        f = pool[int(rng.integers(len(pool)))]
        b, lab = (f(out, rng, Z) if f is _relax else f(out, rng))
        if b is not None:
            out, _ = b, why.append(lab)
    return out, "; ".join(why) or "no-op"
