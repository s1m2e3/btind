"""Threshold grids that do not spend themselves on a sentinel.

A masked observation column is pinned at a single value whenever the thing it
describes is out of sight, and that value dominates the column. Measured on the
masked NestWorld:

    d_food     1.50 ("not visible") on 70% of rows
    d_threat   1.50                 on 49% of rows

Quantiles over all rows then put most of the grid on one number:

    d_food, quantiles 0.05..0.95 over ALL rows
        [0.067  0.168  1.5  1.5  1.5  1.5  1.5]

Five of seven points say the same thing, and the informative range -- where the
food is actually close -- gets two. The write event worth +9.09 in the memory
search is `d_food <= 0.068`, which survives only as the single lowest point, and
the guard worth +5.87 is `d_threat <= 0.09`, which sits below anything a
uniform-coverage grid proposes at all.

So the grid is built TWICE: once over the column, and once over the column with
its modal value removed. The second pass is what reaches the range that matters,
and costs nothing when there is no sentinel -- a column with no dominant value
returns nearly the same points twice and the union deduplicates them.

This is not specific to masking. Any column with a floor, a cap, a default or a
"none" encoding has the same shape, and every one of them would otherwise hide
its own interesting range.
"""
import numpy as np


def modal_value(col, bins=400):
    """The most common value, found on a histogram so floats can share a bin."""
    lo, hi = float(np.min(col)), float(np.max(col))
    if hi - lo < 1e-12:
        return lo, 1.0
    h, edges = np.histogram(col, bins=bins, range=(lo, hi))
    i = int(np.argmax(h))
    width = (hi - lo) / bins
    # THE BIN CENTRE IS NOT A DATA VALUE, and using it as one silently disables
    # the whole mechanism: the sentinel 1.5 sat 0.0019 from its bin centre while
    # the exclusion tolerance was 0.0015, so nothing was excluded and the second
    # pass returned the first pass. Take the median of the values IN the bin,
    # which is a real observation.
    inb = col[(col >= edges[i]) & (col <= edges[i + 1])]
    centre = float(np.median(inb)) if inb.size else 0.5 * (edges[i] + edges[i + 1])
    share = float(np.mean(np.abs(col - centre) <= width))
    return centre, share


def grid(col, n_thr=9, lo_q=0.05, hi_q=0.95, mode_share=0.15, atol=None):
    """Candidate thresholds for one column, including its non-modal range.

    `mode_share` is how dominant a value has to be before the second pass is
    worth taking; below it the two passes agree and the union is the same grid.
    """
    col = np.asarray(col, float)
    qs = np.linspace(lo_q, hi_q, n_thr)
    out = list(np.quantile(col, qs))
    centre, share = modal_value(col)
    if share >= mode_share:
        # tolerance tied to the histogram resolution that FOUND the mode, not
        # to an unrelated fraction of the range
        w = atol if atol is not None else max(1e-9,
                                              (col.max() - col.min()) / 400.0)
        rest = col[np.abs(col - centre) > w]
        if rest.size > max(20, 0.02 * col.size):
            out += list(np.quantile(rest, qs))
    return np.unique(np.round(out, 6))


def literals(col, j, n_thr=9, lo=0.02, hi=0.98, name=None):
    """(clause, label) pairs for one column, both senses, useful fractions only."""
    out = []
    for thr in grid(col, n_thr):
        for neg in (False, True):
            frac = float((col <= thr).mean() if neg else (col > thr).mean())
            if lo < frac < hi:
                out.append(([[int(j), float(thr), bool(neg)]],
                            "%s%s%.3f" % (name if name is not None else j,
                                          "<=" if neg else ">", thr)))
    return out
