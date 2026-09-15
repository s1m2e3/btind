"""Nested subtrees: factoring is exact, and growing inside a child finds structure.

The planted case: a vehicle tree whose only child is "near the intersection ->
accelerate", over an accelerate default -- a controller that runs every red
light. The red stop lives entirely inside that child's region, so growing
there, on its own rows, must recover a braking child under it; the emitted
tree must show it nested, and every new guard must carry the parent's literal
verbatim so the nesting is recoverable.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.memory import MemBank, check_arms, emit, mem_names
from btind.subtree import depth, factor, grow_subtree, owners


def _flat_order(nodes):
    out = []
    for n in nodes:
        out += [n[1]] if n[0] == "arm" else _flat_order(n[2])
    return out


def test_factor_recovers_nesting_and_keeps_order():
    A, B, C, D, E, F, X = ([[j, 0.5, False]] for j in range(7))
    clauses = [A + B + X, A + B, A + C, A, D, E + F, E]
    nodes = factor(clauses)
    assert _flat_order(nodes) == list(range(len(clauses)))
    assert [n[0] for n in nodes] == ["group", "arm", "group"]
    g = nodes[0]
    assert g[1] == A
    inner = g[2]
    assert inner[0][0] == "group" and inner[0][1] == B       # A -> B -> {X, default}
    assert [n[0] for n in inner[1:]] == ["arm", "arm"]
    assert inner[-1][2] == []                                  # subtree default
    assert depth(nodes) == 3
    assert depth(factor([A, D, E])) == 1


def test_emit_prints_the_subtree():
    names = ["x0", "x1", "x2"]
    d = len(mem_names(names, None)) + 1

    def pref(k):
        th = np.zeros((d, 2))
        th[-1, k] = 1.0
        return th
    b = check_arms(dict(names=names, laws_on_z=True, head="argmax",
                        actions=["GO", "STOP"],
                        clauses=[[[0, 0.5, False], [1, 0.2, True]],
                                 [[0, 0.5, False]], [[2, 0.0, False]]],
                        laws=[pref(1), pref(0), pref(1)], default=pref(0)), "e")
    text = emit(b, names)
    assert "Sequence[ x0>0.500 , Fallback ]" in text, text
    assert "subtree default" in text


def test_growing_inside_a_child_recovers_the_red_stop():
    from btind.envs.intersection import ACTIONS, IntersectionBatch
    from btind.structure import score
    env = IntersectionBatch(n_max=24, T_end=60.0)
    N = env.names
    zn = mem_names(N, None)
    d = len(zn) + 1

    def pref(k):
        th = np.zeros((d, 5))
        th[-1, k] = 1.0
        return th
    near = [[N.index("near_int"), 0.5, False]]
    # a follower above the near-intersection child: without it a red stop
    # only makes the cars behind drive into the stopped car (measured -10.3)
    close = [[N.index("has_lead"), 0.5, False], [N.index("lead_gap"), 10.0, True]]
    bank = check_arms(dict(names=list(N), laws_on_z=True, head="argmax",
                           actions=list(ACTIONS), clauses=[close, near],
                           laws=[pref(0), pref(4)], default=pref(4)), "planted")
    pol = lambda b: MemBank(b, len(N))
    obs, _ = env.coverage_rows(bank, n_ep=20, seed=0)
    Z = np.hstack([obs, np.zeros((len(obs), 2))])
    assert (owners(bank, Z) == 1).mean() > 0.2
    g0 = score(env, bank, pol, 300, env.duration, 11).mean()
    new, log = grow_subtree(env, bank, 1, N, zn, pol, obs, Z, max_arms=1,
                            pool=20, screen_ep=80, confirm_ep=300,
                            T=env.duration, seed=11,
                            rng=np.random.default_rng(0), min_rows=100,
                            cem_top=3, cem_iter=2, cem_K=12, verbose=False)
    g1 = score(env, new, pol, 300, env.duration, 11).mean()
    added = len(new["clauses"]) - 2
    assert added >= 1, "nothing grown; best %+.2f" % max(
        (e["delta"] for e in log if "delta" in e), default=float("nan"))
    assert g1 - g0 > 3.0, (g0, g1)
    assert new["clauses"][0] == close                       # the follower untouched
    for cl in new["clauses"][1:1 + added]:
        assert near[0] in cl, cl
    nodes = factor(new["clauses"])
    grp = [n for n in nodes if n[0] == "group"]
    assert grp and grp[0][1] == near, nodes
    assert any(n[0] == "arm" and n[2] == [] for n in grp[0][2])     # parent is default
    assert "Sequence[ near_int>0.500 , Fallback ]" in emit(new, N)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
