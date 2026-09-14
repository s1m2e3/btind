"""Steps and status: the numpy reference and the compiled tick must agree.

`MemBank.arbitrate` (vectorised numpy) and `tick._tick` (numba, per row) are
two implementations of one rule, kept separate so that each tests the other.
This file holds them to bit-exact agreement three ways:

    1  on the arbitration ALONE, over random banks and random feature
       sequences, with no physics in the way -- so a disagreement points at the
       rule and nothing else;
    2  through the highway kernel, on a structured multi-step bank with a fail
       condition, where the constant-preference laws make the comparison exact;
    3  through the NestWorld kernel, vector head, on a multi-step bank.

And it checks the IDENTITY moves, which are what make the richer class safe to
search: an advance clause that never fires, and a fail clause that never
fires, must leave the return of every episode unchanged.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.memory import MemBank, check_arms
from btind.tick import _tick, flatten, tick_args


# ----------------------------------------------------------- 1  the rule alone
def _rand_clause(rng, d, k=None):
    k = k or int(rng.integers(1, 3))
    return [[int(rng.integers(d - 1)), float(rng.uniform(-1, 1)),
             bool(rng.random() < 0.5)] for _ in range(k)]


def _rand_bank(rng, d, C):
    steps, fails, betas, sticky = [], [], [], []
    for _ in range(C):
        st = bool(rng.random() < 0.6)
        K = int(rng.integers(1, 4)) if st else 1
        steps.append([(_rand_clause(rng, d), rng.normal(size=(d, 2)))
                      for _ in range(K - 1)] or None)
        fails.append(_rand_clause(rng, d) if rng.random() < 0.5 else None)
        betas.append(_rand_clause(rng, d) if rng.random() < 0.6 else None)
        sticky.append(st)
    return check_arms(dict(clauses=[_rand_clause(rng, d) for _ in range(C)],
                           laws=[rng.normal(size=(d, 2)) for _ in range(C)],
                           default=rng.normal(size=(d, 2)), steps=steps,
                           fails=fails, betas=betas, sticky=sticky,
                           laws_on_z=True), "rand")


def test_tick_matches_membank_on_random_banks():
    rng = np.random.default_rng(0)
    d = 6
    for trial in range(60):
        C = int(rng.integers(0, 5))
        bank = _rand_bank(rng, d, C)
        f = flatten(bank, d - 3)
        n, T = 40, 30
        pol = MemBank(bank, d - 3)
        pol.reset(n)
        latch = np.full(n, -1)
        step = np.zeros(n, int)
        for t in range(T):
            Z = rng.uniform(-1, 1, (n, d))
            Z[:, -1] = 1.0
            a_ref = pol.arbitrate(Z)
            k_ref = pol._k
            for i in range(n):
                law, latch[i], step[i] = _tick(Z[i], int(latch[i]),
                                                int(step[i]), *tick_args(f))
                # law index -> (arm, step) as MemBank reports it
                arm = int(np.searchsorted(f["law_start"], law, side="right") - 1)
                k = law - f["law_start"][arm]
                if arm == C:
                    arm, k = -1, 0
                assert arm == a_ref[i] and k == k_ref[i], (
                    "trial %d tick %d row %d: kernel (%d,%d) vs numpy (%d,%d)"
                    % (trial, t, i, arm, k, a_ref[i], k_ref[i]))
            assert np.array_equal(latch, pol.latch)
            assert np.array_equal(step, pol.step)


def test_tick_semantics_by_hand():
    """A two-step arm walked through fail, advance, success and preemption."""
    d = 4                                   # [x0, x1, x2, 1]
    b = dict(clauses=[[[0, 0.5, False]],          # arm 0: x0 > 0.5 (higher)
                      [[1, 0.5, False]]],         # arm 1: x1 > 0.5, two steps
             laws=[np.zeros((d, 2)), np.zeros((d, 2))], default=np.zeros((d, 2)),
             steps=[None, [([[2, 0.5, False]], np.ones((d, 2)))]],
             betas=[None, [[1, -0.5, True]]],     # success: x1 <= -0.5
             fails=[None, [[2, -0.5, True]]],     # fail:    x2 <= -0.5
             sticky=[False, True], laws_on_z=True)
    pol = MemBank(b, 1)
    pol.reset(1)

    def run(x0, x1, x2):
        a = int(pol.arbitrate(np.array([[x0, x1, x2, 1.0]]))[0])
        return a, int(pol._k[0]), int(pol.latch[0]), int(pol.step[0])

    assert run(0, 1, 0) == (1, 0, 1, 0)      # enter arm 1 at step 0, latched
    assert run(0, 0, 0) == (1, 0, 1, 0)      # guard gone, still running (sticky)
    assert run(0, 0, 1) == (1, 1, 1, 1)      # x2 > 0.5: advance to step 1
    assert run(0, 0, 0) == (1, 1, 1, 1)      # keep step 1
    assert run(1, 0, 0) == (0, 0, -1, 0)     # arm 0 preempts; not sticky, no latch
    assert run(0, 1, 0) == (1, 0, 1, 0)      # re-enter arm 1 from step 0
    assert run(0, 0, 1) == (1, 1, 1, 1)      # advance again
    assert run(0, 0, -1) == (-1, 0, -1, 0)   # FAIL: released, excluded -> default
    assert run(0, 1, 0) == (1, 0, 1, 0)      # enter again
    assert run(0, 0, 1) == (1, 1, 1, 1)
    assert run(0, -1, 0) == (-1, 0, -1, 0)   # SUCCESS on last step -> default
    assert run(0, 1, 0) == (1, 0, 1, 0)      # enter again
    assert run(0, -1, 0) == (-1, 0, -1, 0)   # SUCCESS on step 0 too: beta is global

    # fail must hand the tick to an arm BELOW the failing one
    b2 = dict(b, clauses=[[[1, 0.5, False]], [[2, -0.9, False]]],
              steps=[[([[2, 0.5, False]], np.ones((d, 2)))], None],
              betas=[[[1, -0.5, True]], None], fails=[[[2, -0.5, True]], None],
              sticky=[True, False])
    pol = MemBank(b2, 1)
    pol.reset(1)
    assert run(0, 1, 0) == (0, 0, 0, 0)
    assert run(0, 1, -0.7) == (1, 0, -1, 0)  # arm 0 fails; arm 1 takes it


# ------------------------------------------------- 2  through the highway kernel
def _highway():
    from btind.envs.highway_batch import HighwayBatch
    from btind.envs.highway_task import constant_bank, law_width
    env = HighwayBatch()
    N = env.names
    d = law_width(N)

    def pref(k):
        th = np.zeros((d, 5))
        th[-1, k] = 1.0
        return th
    return env, N, constant_bank, pref


def _hw_G_numpy(env, bank, s):
    pol = MemBank(bank, len(env.names))
    pol.reset(len(s))
    ss, alive, G, disc = s.copy(), np.ones(len(s), bool), np.zeros(len(s)), 1.0
    for _ in range(env.duration):
        ss, r, done = env.step(ss, pol.act(env.observe(ss)))
        G += disc * r * alive
        disc *= env.gamma
        alive &= ~done
        if not alive.any():
            break
    return G


def _hw_G_fused(env, bank, s):
    from btind.structure import _fast_score_highway
    G = _fast_score_highway(env, bank, len(s), env.duration, 11)
    assert G is not None, "kernel refused a bank it should run"
    return G


def test_highway_multistep_with_fail_is_exact_and_not_a_no_op():
    env, N, constant_bank, pref = _highway()
    ix = N.index
    base = constant_bank(N, 3)                       # default FASTER
    one = dict(base, clauses=[[[ix("v1_dx"), 20.0, True]],
                              [[ix("ego_vx"), 26.0, False]]],
               laws=[pref(4), pref(1)], sticky=[True, False],
               betas=[[[ix("v1_dx"), 30.0, False]], None])
    multi = dict(one, steps=[[([[ix("ego_vx"), 23.0, True]], pref(0))], None],
                 fails=[[[ix("v1_dx"), 8.0, True]], None])
    check_arms(multi, "multi")
    s = env.sample_starts(300, np.random.default_rng(11))
    for name, b in (("one-step", one), ("multi-step+fail", multi)):
        gn, gf = _hw_G_numpy(env, b, s), _hw_G_fused(env, b, s)
        assert np.abs(gn - gf).max() < 1e-12, \
            "%s: %.3e" % (name, np.abs(gn - gf).max())
    # the steps must have changed SOMETHING, or the test proves nothing
    assert np.abs(_hw_G_fused(env, one, s) - _hw_G_fused(env, multi, s)).max() > 0


def test_identity_moves_leave_return_unchanged():
    env, N, constant_bank, pref = _highway()
    ix = N.index
    one = dict(constant_bank(N, 3), clauses=[[[ix("v1_dx"), 20.0, True]]],
               laws=[pref(4)], sticky=[True],
               betas=[[[ix("v1_dx"), 30.0, False]]])
    never = [[ix("ego_x"), -1e9, True]]             # ego_x <= -1e9: never true
    with_steps = dict(one, steps=[[(never, pref(0))]])
    with_fail = dict(one, fails=[never])
    s = env.sample_starts(300, np.random.default_rng(11))
    g0 = _hw_G_fused(env, one, s)
    assert np.array_equal(g0, _hw_G_fused(env, with_steps, s))
    assert np.array_equal(g0, _hw_G_fused(env, with_fail, s))
    assert np.array_equal(g0, _hw_G_numpy(env, with_steps, s))
    assert np.array_equal(g0, _hw_G_numpy(env, with_fail, s))


# ----------------------------------------------- 3  through the NestWorld kernel
def test_nest_multistep_is_exact():
    from btind.envs.nest import NestWorld, OBS_NAMES
    from btind.memory import mem_names
    from btind.policies import evaluate
    from btind.structure import _fast_score
    # PERSISTENT FOOD, because exactness needs a world with no random draw:
    # with teleporting food the respawn site comes from numba's per-thread RNG
    # stream, which the parallel kernel and the batched Python step consume in
    # different orders, so the two paths run different episodes and 30 return
    # units of "disagreement" is the world, not the tree.
    env = NestWorld(food_persistent=True)
    zn = mem_names(OBS_NAMES, None)
    d = len(zn) + 1
    ix = OBS_NAMES.index

    def to(a, b, sgn=1.0):
        th = np.zeros((d, 2))
        th[ix(a), 0] = sgn
        th[ix(b), 1] = sgn
        return th
    bank = dict(names=list(OBS_NAMES), laws_on_z=True, head="vector",
                clauses=[[[ix("t_capture"), 8.0, True]],
                         [[ix("carrying"), 0.5, False]]],
                laws=[to("bear_threat_x", "bear_threat_y", -1.0),
                      to("bear_nest_x", "bear_nest_y")],
                default=to("bear_food_x", "bear_food_y"),
                sticky=[True, True],
                steps=[[([[ix("d_threat"), 0.25, False]],
                         to("bear_food_x", "bear_food_y"))], None],
                betas=[[[ix("t_capture"), 20.0, False]],
                       [[ix("carrying"), 0.5, True]]],
                fails=[None, [[ix("t_capture"), 6.0, True]]])
    check_arms(bank, "nest")
    n_ep, T, seed = 200, 150, 11
    gf = _fast_score(env, bank, n_ep, T, seed)
    assert gf is not None
    env.seed_kernels(seed)
    gn = evaluate(env, MemBank(bank, len(OBS_NAMES)), n_ep=n_ep, T=T,
                  seed=seed)["G"]
    assert np.abs(gn - gf).max() < 1e-9, np.abs(gn - gf).max()


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
