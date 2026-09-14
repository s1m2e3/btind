"""The fused highway kernel must agree with the numpy model it was derived from.

A kernel that silently disagrees with its reference is worse than no kernel,
because it is fast enough to be believed. This is the same discipline that held
MemBank to LandscapeBank at exactly 0.0 and the NestWorld kernel to its Python
path at exactly 0.0, and it has already earned its keep twice here: it caught
MOBIL scoring a candidate lane change against a road where the WHOLE traffic
stream had shifted with the car being scored, and it caught MOBIL being decided
simultaneously for every vehicle when highway-env decides them in order, so car
q+1 sees where car q has already moved. Both were faults in the numpy model and
both were worth about 25 return units.

WHAT IS PINNED. Constant-action banks and banks exercising the tree, the
blackboard and stickiness agree EXACTLY -- every episode, to the last bit.

THE DISCREPANCY THAT WAS OPEN, AND WHAT IT WAS. On a bank with random laws, 3
episodes in 400 used to diverge, always late, and two explanations had failed:
not an argmax tie, not physics drift. The cause was numba's cache. `cache=True`
is keyed on the kernel's own source file, so constants imported from
`highway_batch` (frozen at compile time) and, later, the arbitration inlined
from `tick.py` could change without the cached kernel being recompiled -- the
kernel then ran a slightly different world or a different tree from the numpy
model it was compared with. Measured after `tick._drop_stale_kernel_caches`
was added: 1200 random-law episodes over three seeds agree to the bit, worst
difference 1.8e-15. So the bound this test used to carry (98% exact, mean to
0.01) is gone and the random-law case is held to exact agreement like the rest.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.highway_batch import ACTIONS, HighwayBatch
from btind.envs.highway_fast import flatten, params, rollout
from btind.envs.highway_task import constant_bank
from btind.memory import MemBank, mem_names

ENV = HighwayBatch()
N = ENV.names
NOBS = len(N)


def _numpy_G(bank, s):
    pol = MemBank(bank, NOBS)
    pol.reset(len(s))
    ss, alive, G, disc = s.copy(), np.ones(len(s), bool), np.zeros(len(s)), 1.0
    for _ in range(ENV.duration):
        ss, r, done = ENV.step(ss, pol.act(ENV.observe(ss)))
        G += disc * r * alive
        disc *= ENV.gamma
        alive &= ~done
        if not alive.any():
            break
    return G


def _fused_G(bank, s):
    from btind.tick import no_dev, no_trace, tick_args, world_args
    f = flatten(bank, NOBS)
    G = np.empty(len(s))
    rollout(np.ascontiguousarray(s), params(ENV), *world_args(f), ENV.duration,
            G, no_trace(), no_dev(), *tick_args(f))
    return G


def _starts(n=300, seed=11):
    return ENV.sample_starts(n, np.random.default_rng(seed))


def test_constant_actions_agree_exactly():
    s = _starts()
    for i, name in enumerate(ACTIONS):
        gn, gf = _numpy_G(constant_bank(N, i), s), _fused_G(constant_bank(N, i), s)
        assert np.abs(gn - gf).max() < 1e-12, "%s: %.3e" % (name, np.abs(gn - gf).max())


def _tree_bank(rng, mem=None, sticky=None, betas=None):
    zn = mem_names(N, mem)
    d = len(zn) + 1
    cl = [[[N.index("v1_dx"), 20.0, True]], [[N.index("ego_vx"), 26.0, False]]]
    if mem:
        cl = cl + [[[zn.index("have_mem"), 0.5, False]]]
    b = dict(constant_bank(N, 1), clauses=cl,
             laws=[rng.normal(0, 0.3, (d, 5)) for _ in cl],
             default=rng.normal(0, 0.3, (d, 5)))
    if mem:
        b["mem"] = mem
    if sticky is not None:
        b["sticky"], b["betas"] = sticky, betas
    return b


def test_tree_and_blackboard_and_sticky_agree():
    """Exact, random laws included -- see the docstring for why that is now safe."""
    s = _starts()
    mem = dict(cols=[N.index("v1_dx"), N.index("v1_dvx")],
               write=[[N.index("v1_dx"), 15.0, True]], clear=None)
    cases = [("tree", _tree_bank(np.random.default_rng(0))),
             ("blackboard", _tree_bank(np.random.default_rng(0), mem=mem)),
             ("sticky+beta", _tree_bank(np.random.default_rng(0), mem=mem,
                                        sticky=[True, False, True],
                                        betas=[[[N.index("ego_vx"), 24.0, False]],
                                               None, None]))]
    for name, b in cases:
        gn, gf = _numpy_G(b, s), _fused_G(b, s)
        d = np.abs(gn - gf)
        assert d.max() < 1e-9, "%s: %d episodes differ, worst %.3e" % (
            name, int((d >= 1e-9).sum()), d.max())


def test_write_rules_agree_exactly():
    """Every shape of blackboard rule the memory search can propose."""
    s = _starts()
    rules = [("never", [[N.index("ego_x"), -1e9, True]]),
             ("always", [[N.index("ego_x"), -1e9, False]]),
             ("v1_dx<=15", [[N.index("v1_dx"), 15.0, True]]),
             ("v1_dx<=40", [[N.index("v1_dx"), 40.0, True]])]
    for label, wr in rules:
        mem = dict(cols=[N.index("v1_dx"), N.index("v1_dvx")], write=wr, clear=None)
        zn = mem_names(N, mem)
        d = len(zn) + 1
        rng = np.random.default_rng(0)
        b = dict(constant_bank(N, 1), mem=mem,
                 clauses=[[[N.index("v1_dx"), 20.0, True]],
                          [[zn.index("have_mem"), 0.5, False]]],
                 laws=[rng.normal(0, 0.3, (d, 5)), rng.normal(0, 0.3, (d, 5))],
                 default=rng.normal(0, 0.3, (d, 5)))
        gn, gf = _numpy_G(b, s), _fused_G(b, s)
        assert np.abs(gn - gf).max() < 1e-12, \
            "%s: %.3e" % (label, np.abs(gn - gf).max())


def test_kernel_refuses_a_vector_head():
    """A vector-head bank must fall to the Python path, not be scored as actions."""
    from btind.structure import _fast_score_highway
    b = dict(constant_bank(N, 1))
    b["head"] = "vector"
    assert _fast_score_highway(ENV, b, 8, 40, 11) is None


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
