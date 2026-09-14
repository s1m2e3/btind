"""Persist every run: the rules discovered, the moves tried, and the console.

Two problems this solves, both measured rather than anticipated.

  NOTHING WAS KEPT. Every result in this project so far lived in a terminal
  buffer and a summary JSON written by whichever experiment produced it. The
  emitted trees -- the actual output of the whole method -- were printed and
  lost, so comparing today's controller with last week's meant rerunning last
  week.

  STAGE 1 IS 80% OF THE WALL CLOCK AND NEVER CHANGES. Imitation takes ~300s on
  NestWorld and is a pure function of (env, config, seed); every stage-2
  experiment re-derived the same bank before touching the thing under test.
  Cached, a stage-2 iteration costs what stage 2 costs.

LAYOUT. One directory per run under `runs/`, plus one append-only index line so
a hundred runs can be scanned without opening a hundred files:

    runs/index.jsonl                one line per run: id, tag, metrics, tree
    runs/<id>/config.json           env signature + full config
    runs/<id>/bank.json             clauses, laws, betas, the BT text
    runs/<id>/moves.json            every accepted AND rejected move
    runs/<id>/log.txt               the console, verbatim
    runs/cache/<hash>.json          stage-1 banks, keyed by what produced them

REJECTED MOVES ARE KEPT DELIBERATELY. The accepted ones are in the tree; the
rejected ones are the only record of what the proposal machinery suggested, and
every diagnosis in this project so far -- the bearing splits that lost their
rollouts, the 27 add proposals that never fired -- came from reading them.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def env_signature(env):
    """Every numeric parameter of the world, so a cached bank cannot be reused
    across two worlds that merely share a class name."""
    out = {"class": type(env).__name__}
    for k, v in sorted(vars(env).items()):
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, bool, str)):
            out[k] = v
    return out


def bank_json(bank, names=None, bt=None):
    names = names or bank.get("names")
    return dict(
        clauses=[[[int(j), float(t), bool(n)] for j, t, n in cl]
                 for cl in bank["clauses"]],
        laws=[np.asarray(t).tolist() for t in bank["laws"]],
        default=np.asarray(bank["default"]).tolist(),
        betas=_jsonable(bank.get("betas")),
        # laws_on_z IS PART OF THE BANK, not a runtime detail: a law sized for
        # the augmented layout multiplied by design_matrix(obs) is either a
        # crash or, when the widths happen to match, a controller whose every
        # coefficient is attached to the wrong feature.
        laws_on_z=bool(bank.get("laws_on_z")),
        mem=_jsonable(bank.get("mem")), sticky=_jsonable(bank.get("sticky")),
        # steps carry laws, so they go through the same ndarray -> list path
        steps=_jsonable(bank.get("steps")), fails=_jsonable(bank.get("fails")),
        head=bank.get("head"), actions=bank.get("actions"),
        u_range=bank.get("u_range"),
        names=list(names) if names is not None else None, bt=bt)


def bank_from_json(d):
    steps = d.get("steps")
    if steps:
        steps = [([(adv, np.asarray(th, float)) for adv, th in s] if s else None)
                 for s in steps]
    out = dict(clauses=[[[int(j), float(t), bool(n)] for j, t, n in cl]
                        for cl in d["clauses"]],
               laws=[np.asarray(t, float) for t in d["laws"]],
               default=np.asarray(d["default"], float),
               betas=d.get("betas"), names=d.get("names"),
               laws_on_z=bool(d.get("laws_on_z")), mem=d.get("mem"),
               sticky=d.get("sticky"), steps=steps, fails=d.get("fails"))
    for k in ("head", "actions", "u_range"):
        if d.get(k) is not None:
            out[k] = tuple(d[k]) if k == "u_range" else d[k]
    return out


class Tee:
    """stdout to both the console and the run log. Restored on close."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout
        sys.stdout = self

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()

    def close(self):
        sys.stdout = self.out
        self.f.close()


class RunLog:
    """One run: a directory, a tee'd console, and an index line at the end."""

    def __init__(self, tag="run", root=RUNS, enabled=True):
        self.enabled = enabled
        if not enabled:
            return
        self.id = "%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), tag)
        self.dir = os.path.join(root, self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.root, self.tag, self.t0 = root, tag, time.time()
        self.tee = Tee(os.path.join(self.dir, "log.txt"))
        self.moves = []

    def _w(self, name, obj):
        if not self.enabled:
            return
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
            json.dump(_jsonable(obj), fh, indent=1)

    def config(self, env, cfg, extra=None):
        self._w("config.json", dict(env=env_signature(env), cfg=cfg,
                                    **(extra or {})))

    def move(self, stage, rec):
        """Record one attempted move, accepted or not."""
        if self.enabled:
            self.moves.append(dict(stage=stage, **_jsonable(rec)))

    def bank(self, bank, names=None, bt=None):
        self._w("bank.json", bank_json(bank, names, bt))

    def finish(self, metrics, bt=None):
        if not self.enabled:
            return
        self._w("moves.json", self.moves)
        self._w("metrics.json", metrics)
        line = dict(id=self.id, tag=self.tag, seconds=time.time() - self.t0,
                    **_jsonable(metrics))
        if bt:
            line["bt"] = bt
        with open(os.path.join(self.root, "index.jsonl"), "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        self.tee.close()


# ------------------------------------------------------------------- cache
STAGE1_MODULES = ("collect.py", "search.py", "valuesplit.py", "evotm.py",
                  "joint.py", "rollout_select.py", "dagger.py", "policies.py",
                  "grow.py", "fluctuation.py", "landscape.py")


def source_hash(pkg=os.path.join(ROOT, "btind"), modules=None, extra=""):
    """Hash of the modules the cached object actually depends on.

    THE CONFIG IS NOT THE ONLY INPUT. A cached bank is a function of the code
    that produced it as much as of the config, and the two can diverge silently:
    normalising the emitted action changed which arms `rsfi` accepts without
    touching a single config value, so every cached bank from before that commit
    was wrong in a way no key built from `cfg` could detect. Hashing the source
    Hashing the WHOLE package was the first version and it was too blunt in the
    other direction: editing a stage-2 module invalidated every stage-1 bank and
    re-ran 191s of imitation for a change that could not affect it. `modules`
    narrows the hash to what the cached object is a function of, and `extra`
    carries the source of the specific function doing the caching, so a change
    to `imitate` invalidates while a change elsewhere in the same file does not.
    """
    h = hashlib.sha1()
    files = ([os.path.join(pkg, m) for m in (modules or
              sorted(f for f in os.listdir(pkg) if f.endswith(".py")))]
             + [os.path.join(pkg, "envs", f)
                for f in sorted(os.listdir(os.path.join(pkg, "envs")))
                if f.endswith(".py")])
    for p in files:
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    h.update(extra.encode())
    return h.hexdigest()[:12]


def cache_key(env, cfg, seed, keys, src=None):
    """Hash of everything that can change the cached object, and nothing else.

    `keys` names the config entries stage 1 actually reads. Hashing the whole
    config would invalidate every cached bank whenever a stage-2 knob moved,
    which is the failure mode that makes people stop trusting caches -- so the
    code hash carries the risk that the config cannot see.
    """
    blob = json.dumps(dict(env=env_signature(env), seed=seed,
                           src=src or source_hash(),
                           cfg={k: _jsonable(cfg[k]) for k in sorted(keys)
                                if k in cfg}), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def cache_get(key, root=RUNS):
    p = os.path.join(root, "cache", key + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return bank_from_json(json.load(fh)["bank"])


def cache_put(key, bank, meta=None, root=RUNS):
    d = os.path.join(root, "cache")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, key + ".json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(dict(bank=bank_json(bank), meta=meta or {})), fh,
                  indent=1)
