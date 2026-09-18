"""Carry a `pass` bank over to a `scalar` head: keep column 0, drop the logit.

Column 0 of a `pass` law IS the scalar head's single output -- the world clips
both to the same range -- so the command the tree emits is unchanged. What goes
is the declaration, which nothing read.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, r"C:\Users\samil\Documents\BTs")

src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as fh:
    d = json.load(fh)
v = d["veh"]
assert v.get("head") == "pass", "not a pass bank: %r" % v.get("head")

col0 = lambda t: [[row[0]] for row in t]
v["laws"] = [col0(t) for t in v["laws"]]
v["default"] = col0(v["default"])
if v.get("steps"):
    v["steps"] = [None if s is None else [[adv, col0(th)] for adv, th in s]
                  for s in v["steps"]]


def kern0(k):
    if k is None:
        return None
    k = dict(k)
    k["Y"] = [[y[0]] for y in k["Y"]]
    return k


if v.get("kerns"):
    v["kerns"] = [None if s is None else [kern0(k) for k in s] for s in v["kerns"]]
v["kern_default"] = kern0(v.get("kern_default"))
v["head"] = "scalar"
v["actions"] = None
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(d, fh)

# prove the command is unchanged
from btind.runlog import bank_from_json
from btind.collect import design_matrix
from btind.memory import mem_names
from btind.landscape import _match_cols
import btind.kernlaw as KL
from experiments.e39_vehicle_only import world

env = world("vehicle")
old = bank_from_json(json.load(open(src, encoding="utf-8"))["veh"])
new = bank_from_json(json.load(open(dst, encoding="utf-8"))["veh"])
zn = mem_names(new["names"], new.get("mem"))
Z, _ = env.coverage_rows(new, n_ep=60, seed=0)
if Z.shape[1] < len(zn):
    Z = np.hstack([Z, np.zeros((len(Z), len(zn) - Z.shape[1]))])
X = design_matrix(Z)
worst = 0.0
done = np.zeros(len(Z), bool)
for c, cl in enumerate(old["clauses"]):
    m = _match_cols(cl, Z) & ~done
    done |= m
    if not m.any():
        continue
    a = KL.evaluate(np.asarray(old["laws"][c], float), KL.kern_of(old, c, 0),
                    X[m], bounds=env.u_range)[:, 0]
    bb = KL.evaluate(np.asarray(new["laws"][c], float), KL.kern_of(new, c, 0),
                     X[m], bounds=env.u_range)[:, 0]
    worst = max(worst, float(np.abs(a - bb).max()))
a = KL.evaluate(np.asarray(old["default"], float), None, X[~done],
                bounds=env.u_range)[:, 0]
bb = KL.evaluate(np.asarray(new["default"], float), None, X[~done],
                 bounds=env.u_range)[:, 0]
worst = max(worst, float(np.abs(a - bb).max()))
print("wrote %s" % dst)
print("  head %r -> %r, laws (d,2) -> (d,%d), %d arms preserved"
      % ("pass", new["head"], np.shape(new["default"])[1], len(new["clauses"])))
print("  worst command difference over %d rows: %.3e" % (len(Z), worst))
