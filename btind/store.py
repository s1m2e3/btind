"""Keep what was learned, and start from it next time.

Every run so far began from nothing. The clauses a search spent ten minutes
finding were written to `runs/<id>/bank.json`, printed once, and then the next
run enumerated the same alphabet from scratch. That is the expensive half of the
work being thrown away on purpose.

TWO THINGS ARE WORTH REUSING and they are reused differently:

    BANKS       a whole controller, as a WARM START. Resuming from the best bank
                for a world is not cheating -- it is the same controller, and
                every subsequent move still has to clear its own rollout.
    CLAUSES     the guards previous runs ACCEPTED, as PROPOSALS. These go into
                the candidate pool beside the random ones and win or lose on
                their own merits. A clause that was worth +2 last run is a good
                guess and nothing more, because the bank around it has changed.

WHY A CLAUSE IS ONLY EVER A PROPOSAL. A guard's value depends on what the arms
above it already claim, so importing an accepted clause into a different tree
imports a number that was measured against a distribution that no longer exists.
`rsfi` has always treated its evolved seeds this way; the store just widens the
source.

THE KEY IS THE WORLD, NOT THE RUN. Banks are indexed by the environment's full
parameter signature, so a controller grown on a masked world is never resumed on
an unmasked one -- they are different tasks that happen to share a class name.
"""
import hashlib
import json
import os

import numpy as np

from .runlog import RUNS, bank_json, bank_from_json, env_signature


def _key(env):
    """A key that is the same in every process.

    `hash()` on a string is salted per interpreter, so a key built from it
    changes on every run -- the store looked like it worked because the first
    warm-start test ran cold and warm inside ONE process. Across processes every
    lookup missed and every run silently started from nothing, which is the
    exact failure the store exists to prevent.
    """
    sig = env_signature(env)
    h = hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()
    return "%s_%s" % (sig["class"], h[:12])


def _path(env, root=RUNS):
    return os.path.join(root, "store", _key(env) + ".json")


def save(env, bank, metrics, names=None, tag="", root=RUNS):
    """Record a bank for this world, keeping the best by held-out return."""
    p = _path(env, root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    blob = {"world": env_signature(env), "banks": []}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            blob = json.load(fh)
    entry = dict(bank=bank_json(bank, names), G=float(metrics.get("G", 0.0)),
                 eaten=float(metrics.get("eaten", 0.0)), tag=tag,
                 n_arms=len(bank["clauses"]))
    blob["banks"].append(entry)
    blob["banks"].sort(key=lambda e: -e["G"])
    blob["banks"] = blob["banks"][:20]
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1)
    return p


def best(env, root=RUNS, min_G=None):
    """The highest-scoring stored bank for this world, or None."""
    p = _path(env, root)
    if not os.path.exists(p):
        return None, None
    with open(p, encoding="utf-8") as fh:
        blob = json.load(fh)
    if not blob["banks"]:
        return None, None
    e = blob["banks"][0]
    if min_G is not None and e["G"] < min_G:
        return None, None
    return bank_from_json(e["bank"]), e


def clauses(env, root=RUNS, max_clauses=40, min_G=None):
    """Every guard any stored bank for this world accepted, newest first.

    Deduplicated on (feature, sense) with the threshold rounded, because two
    runs that found the same cut at 0.101 and 0.104 have found one clause, and
    the drift operator will re-explore the neighbourhood anyway.
    """
    p = _path(env, root)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        blob = json.load(fh)
    seen, out = set(), []
    for e in blob["banks"]:
        if min_G is not None and e["G"] < min_G:
            continue
        for cl in e["bank"]["clauses"]:
            k = tuple(sorted((int(j), bool(n), round(float(t), 2))
                             for j, t, n in cl))
            if k in seen:
                continue
            seen.add(k)
            out.append([[int(j), float(t), bool(n)] for j, t, n in cl])
            if len(out) >= max_clauses:
                return out
    return out


def summary(root=RUNS):
    """What is on disk, for a glance before a run."""
    d = os.path.join(root, "store")
    if not os.path.isdir(d):
        return []
    rows = []
    for f in sorted(os.listdir(d)):
        with open(os.path.join(d, f), encoding="utf-8") as fh:
            blob = json.load(fh)
        w = blob["world"]
        rows.append(dict(key=f[:-5], n=len(blob["banks"]),
                         best=blob["banks"][0]["G"] if blob["banks"] else None,
                         masked=bool(w.get("vision_r", 0)),
                         gamma=w.get("gamma")))
    return rows
