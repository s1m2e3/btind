"""What the search learns about its own proposals.

Until now a rejection taught nothing. The memory stage screened 10,752
candidates, rejected every one, and the next round proposed from exactly the
same uniform pool -- a search with no memory of what keeps failing, which is a
strange thing to find inside a project about learning memory.

WHAT IS TRACKED. Per observation column, how often a literal on it appeared in a
move that was ACCEPTED against how often it was merely tried. The weight is a
Laplace-smoothed success rate,

    w_j = (accepted_j + a) / (tried_j + a + b)

with the prior deliberately optimistic (a = 2, b = 6, so an untried column sits
at 0.25 against a typical measured rate near 0.02). A column that has never been
tried outranks one that has failed forty times, which keeps the pool exploring
rather than collapsing onto whatever worked first.

WHY A COLUMN AND NOT A LITERAL. A threshold's value depends on where the arms
above it already cut, so "d_threat at 0.09 failed" transfers badly to the next
tree. "Literals on d_threat tend to be accepted" transfers well -- it is a claim
about the task, not about one arrangement of it. Thresholds are refined by drift
and `polish_thresholds`; this only decides which axis to look along.

IT PERSISTS PER WORLD, so a run starts knowing what earlier runs found worth
proposing. That is the same argument as the bank store: the expensive part is
discovering where to look, and throwing it away each run is a choice.
"""
import json
import os

import numpy as np

from .runlog import RUNS, env_signature
from .store import _key


class Weights:
    """Per-column proposal weights, learned from accept/reject outcomes."""

    def __init__(self, n_cols, a=2.0, b=6.0):
        self.n = n_cols
        self.a, self.b = a, b
        self.acc = np.zeros(n_cols)
        self.tried = np.zeros(n_cols)
        # WHAT EACH PROPOSAL SOURCE ACTUALLY YIELDED, carried across rounds in
        # the same per-world file. The usefulness gate judged the critic by
        # three PROXIES for a task it is not asked to do -- ranking arbitrary
        # deviations -- and twice got it wrong: it inflated a dead critic's
        # lift, and it would have silenced one whose proposals are in fact the
        # best source in the search (critic-sourced points 47 kept of 217, 22%,
        # against deviations' 10% and failure anchors' 18%). Realised keep rate
        # is the thing the gate was proxying for.
        self.src = {}                       # src -> [confirmed, kept]

    def note_sources(self, log):
        """Tally a kernel search's log by proposal source."""
        for e in log or ():
            s = e.get("src")
            if s is None or "accepted" not in e:
                continue
            row = self.src.setdefault(str(s), [0.0, 0.0])
            row[0] += 1.0
            row[1] += 1.0 if e["accepted"] else 0.0

    def yield_of(self, s):
        """(keep rate, evidence) for a source; (None, 0) when never tried."""
        row = self.src.get(str(s))
        if not row or row[0] < 1:
            return None, 0.0
        return row[1] / row[0], row[0]

    def probs(self, cols=None):
        w = (self.acc + self.a) / (self.tried + self.a + self.b)
        if cols is not None:
            cols = [c for c in cols if 0 <= c < self.n]
            m = np.zeros(self.n, bool)
            m[list(cols)] = True
            w = np.where(m, w, 0.0)
        s = w.sum()
        return w / s if s > 0 else np.full(self.n, 1.0 / self.n)

    def update(self, clause, accepted):
        for lit in clause:
            j = int(lit[0])
            if 0 <= j < self.n:
                self.tried[j] += 1.0
                self.acc[j] += 1.0 if accepted else 0.0

    def update_many(self, log, key="clause"):
        """Absorb a search log: entries with a clause and an `accepted` flag."""
        for e in log:
            cl = e.get(key)
            if cl:
                self.update(cl, bool(e.get("accepted")))

    def top(self, names, k=6, min_tried=1):
        """Ranked columns that have actually been proposed.

        Reporting over every slot ranks the ones that were never tried, since
        they keep the optimistic prior forever -- after 54094 recorded tries the
        top eight were all unused padding at "accepted 0 / 0", which says
        nothing about anything. Sampling was unaffected (it masks to the real
        columns and renormalises); the report was the broken part.
        """
        p = self.probs()
        idx = [i for i in np.argsort(-p) if self.tried[i] >= min_tried][:k]
        return [(names[i], float(p[i]), int(self.acc[i]), int(self.tried[i]))
                for i in idx]

    # -- persistence -------------------------------------------------------
    def to_json(self):
        return dict(n=self.n, a=self.a, b=self.b, acc=self.acc.tolist(),
                    tried=self.tried.tolist(),
                    src={k: list(v) for k, v in self.src.items()})

    @classmethod
    def from_json(cls, d):
        w = cls(int(d["n"]), float(d.get("a", 2.0)), float(d.get("b", 6.0)))
        w.acc = np.asarray(d["acc"], float)
        w.tried = np.asarray(d["tried"], float)
        w.src = {k: [float(v[0]), float(v[1])] for k, v in (d.get("src") or {}).items()}
        return w


def _path(env, root=RUNS):
    return os.path.join(root, "store", _key(env) + ".weights.json")


def load(env, n_cols, root=RUNS):
    p = _path(env, root)
    if os.path.exists(p):
        try:
            w = Weights.from_json(json.load(open(p, encoding="utf-8")))
            if w.n == n_cols:
                return w
        except Exception:
            pass
    return Weights(n_cols)


def save(env, w, root=RUNS):
    p = _path(env, root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(w.to_json(), fh)
    return p


def rand_literal(rng, alpha, X, hot, weights=None, cols=None):
    """`evotm._rand_literal`, with the column drawn from learned weights.

    Everything else is unchanged: the threshold comes from the alphabet, or from
    a hot row when one is offered, and the sense is a coin flip.
    """
    feats = list(alpha.feats if cols is None else cols)
    if weights is None:
        j = int(rng.choice(feats))
    else:
        p = weights.probs(feats)
        j = int(rng.choice(np.arange(weights.n), p=p))
        if j not in feats:            # weights are wider than this layout
            j = int(rng.choice(feats))
    if hot is not None and len(hot) and rng.random() < 0.5:
        thr = float(X[hot[int(rng.integers(len(hot)))], j])
    else:
        t = alpha.thr[j]
        thr = (float(t[int(rng.integers(len(t)))]) if len(t)
               else float(rng.uniform(alpha.lo[j], alpha.hi[j])))
    return [j, thr, bool(rng.random() < 0.5)]
