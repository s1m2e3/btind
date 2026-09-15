"""The kernel-interpolation law: exact at its anchors, the affine law far from
them and with none, carried by every arm operator, and bit-identical between the
numpy reference and the compiled intersection kernel for both agents and heads.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind import kernlaw as KL
from btind.collect import design_matrix
from btind.envs.intersection import ACTIONS, IntersectionBatch
from btind.envs import intersection_fast as IF
from btind.memory import MemBank, check_arms, emit, insert_arm, mem_names, reindex
from btind.runlog import bank_from_json, bank_json

from tests.test_intersection_equiv import sig_bank, veh_bank
from tests.test_intersection_heads import duration_signal, scalar_follower


def test_exact_at_anchors_and_affine_far_away():
    rng = np.random.default_rng(0)
    d, nA = 7, 3
    th = rng.normal(size=(d, nA))
    kern = KL.make([1, 4], rng.normal(size=(4, 2)), rng.normal(size=(4, nA)), [0.7, 1.3])
    Z = rng.normal(size=(4, d - 1))
    Z[:, 1], Z[:, 4] = kern["X"][:, 0], kern["X"][:, 1]       # other columns arbitrary
    out = KL.evaluate(th, kern, design_matrix(Z))
    assert np.abs(out - kern["Y"]).max() < 1e-4
    far = rng.normal(size=(5, d - 1))
    far[:, 1] += 80.0
    Xd = design_matrix(far)
    assert np.abs(KL.evaluate(th, kern, Xd) - Xd @ th).max() < 1e-12
    assert np.array_equal(KL.evaluate(th, None, Xd), Xd @ th)


def _veh_kern(b, n_out):
    """Anchors on (v, d_stop): stopped close to the line -> brake hard; slow and
    far -> accelerate. Enough to change what cars do."""
    ix = b["names"].index
    if n_out == 1:
        Y = [[-4.0], [2.0], [0.5]]
    else:
        Y = np.zeros((3, n_out))
        Y[0, ACTIONS.index("BRAKE_HARD")] = 3.0
        Y[1, ACTIONS.index("ACCEL_MAX")] = 3.0
        Y[2, ACTIONS.index("HOLD")] = 3.0
    return KL.make([ix("v"), ix("d_stop")], [[4.0, 8.0], [3.0, 70.0], [11.0, 30.0]],
                   Y, [3.0, 15.0])


def _with_kernels(b, n_out):
    k = _veh_kern(b, n_out)
    b = KL.with_kern(b, -1, 0, k)
    return check_arms(KL.with_kern(b, 0, 0, k), "kern")


def _sig_kern(b):
    sx = b["names"].index
    Y = [[40.0], [6.0]] if b.get("head") == "duration" else [[3.0, 0.0], [0.0, 3.0]]
    return KL.make([sx("q0"), sx("q2")], [[8.0, 0.0], [0.0, 8.0]], Y, [3.0, 3.0])


def test_kernel_intersection_kernel_matches_reference_both_heads():
    for env, vb, sb in (
            (IntersectionBatch(n_max=24, T_end=60.0), veh_bank(), sig_bank()),
            (IntersectionBatch(n_max=24, T_end=60.0, veh_head="scalar",
                               sig_head="duration"), scalar_follower(), duration_signal())):
        n_out_v = np.asarray(vb["default"]).shape[1]
        vk = _with_kernels(vb, n_out_v)
        sk = check_arms(KL.with_kern(KL.with_kern(sb, -1, 0, _sig_kern(sb)), 0, 0,
                                     _sig_kern(sb)), "sig kern")
        s = env.sample_starts(40, np.random.default_rng(4))
        for v, g in ((vk, None), (vk, sk), (vb, sk)):
            gk = IF.run(env, v, g, s, env.duration)
            gp = env.python_rollout(v, g, s)
            assert np.abs(gk - gp).max() < 1e-9, (vb.get("head"), np.abs(gk - gp).max())
        # the kernels change behaviour, so the agreement is not on inert laws
        assert np.abs(IF.run(env, vk, None, s, env.duration)
                      - IF.run(env, vb, None, s, env.duration)).max() > 0
        assert np.abs(IF.run(env, vb, sk, s, env.duration)
                      - IF.run(env, vb, sb, s, env.duration)).max() > 0


def test_kernels_ride_with_their_arms_and_survive_json():
    b = _with_kernels(veh_bank(), 5)
    k0 = KL.kern_of(b, 0, 0)
    moved = check_arms(reindex(b, [1, 0, 2, 3]), "reindex")
    assert KL.kern_of(moved, 1, 0) is k0 and KL.kern_of(moved, 0, 0) is None
    ins = check_arms(insert_arm(b, [[0, 1.0, False]], b["default"], 0), "insert")
    assert KL.kern_of(ins, 1, 0) is k0 and KL.kern_of(ins, 0, 0) is None
    back = bank_from_json(bank_json(b))
    assert np.allclose(KL.kern_of(back, 0, 0)["X"], k0["X"])
    assert np.allclose(back["kern_default"]["Y"], b["kern_default"]["Y"])
    env = IntersectionBatch(n_max=24, T_end=60.0)
    s = env.sample_starts(20, np.random.default_rng(1))
    assert np.array_equal(IF.run(env, back, None, s, env.duration),
                          IF.run(env, b, None, s, env.duration))
    assert "K[(v=" in emit(b, b["names"])


if __name__ == "__main__":
    test_exact_at_anchors_and_affine_far_away()
    test_kernel_intersection_kernel_matches_reference_both_heads()
    test_kernels_ride_with_their_arms_and_survive_json()
    print("ok")
