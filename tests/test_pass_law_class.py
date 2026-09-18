"""The `pass` head is a continuous command that carries a declaration.

Every one of these is a bug that was live: the head was falling through to the
argmax branch of four different decisions, each of which assumed a head is
either a one-column command or a preference vector.
"""
import numpy as np
import pytest

from btind.collect import design_matrix
from btind.envs.intersection import IntersectionBatch
from btind.kernlaw import bounds_of, constrain
from btind.lawcem import cem_law
from btind.lawsearch import discrete_primitives, pass_primitives
from btind.runlog import bank_json, bank_from_json


def _world():
    env = IntersectionBatch(n_max=40, T_end=200, veh_reward="car",
                            veh_head="pass", sig_head="duration")
    return env.set_agent("vehicle")


def _Z(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, 200, n),      # a distance, in metres
                            rng.normal(0, 1, n),
                            np.ones(n)])                 # a column that is flat


NAMES = ["near_d", "dv", "flat"]


def test_command_terms_do_not_saturate():
    """The old vocabulary's unit coefficient on a 0-200 m column IS the clip."""
    lo, hi = -4.5, 2.6
    Z, d = _Z(), len(NAMES) + 1
    X = design_matrix(Z)

    def clipped(pool, col=0):
        out = []
        for th in pool.values():
            a = X @ np.asarray(th, float)[:, col]
            out.append(float(((a <= lo) | (a >= hi)).mean()))
        return np.array(out)

    old = discrete_primitives(NAMES, 2, d, rng=np.random.default_rng(0))
    cmd_old = {k: v for k, v in old.items() if np.abs(np.asarray(v)[:, 0]).any()}
    new = pass_primitives(NAMES, d, lo, hi, Z=Z, rng=np.random.default_rng(0))
    # the five constants across [lo, hi] include the endpoints, and a constant
    # AT a bound is the honest null a proportional term has to beat, not a
    # saturating law -- `grow_bt` keeps them whole for that reason
    cmd_new = {k: v for k, v in new.items()
               if not k.startswith(("decl:", "const["))}

    # the bug, reproduced: a unit coefficient on a 0-200 m column is the clip
    # everywhere except the sliver of rows below 2.6 m
    assert (clipped(cmd_old) > 0.95).any()
    assert (clipped(cmd_new) < 0.5).all()


def test_the_two_outputs_move_one_at_a_time():
    """A command candidate keeps the parent's declaration, and vice versa."""
    Z, d = _Z(), len(NAMES) + 1
    X = design_matrix(Z)
    par = np.zeros((d, 2))
    par[-1] = [0.3, -0.7]
    P = pass_primitives(NAMES, d, -4.5, 2.6, parent=par, Z=Z,
                        rng=np.random.default_rng(0))
    assert any(k.startswith("decl:") for k in P)
    for k, th in P.items():
        v = X @ np.asarray(th, float)
        if k.startswith("decl:"):
            assert np.allclose(v[:, 0], 0.3), k
        else:
            assert np.allclose(v[:, 1], -0.7), k


def test_a_flat_column_gates_nothing():
    Z, d = _Z(), len(NAMES) + 1
    P = pass_primitives(NAMES, d, -4.5, 2.6, Z=Z, n_decl=None,
                        rng=np.random.default_rng(0))
    assert not any("flat" in k for k in P)


def test_the_bank_carries_the_world_range():
    """`bounds_of` reads the BANK and has no env to fall back on."""
    env = _world()
    bank = dict(clauses=[], laws=[], default=np.zeros((2, 2)), head="pass",
                u_range=tuple(env.u_range))
    assert bounds_of(bank) == (-4.5, 2.6)
    assert env.actions is None                # a pass head has no action set


def test_prior_k_survives_a_round_trip():
    """It was not serialised, so a resume silently fell to `constrain`'s 4."""
    bank = dict(clauses=[], laws=[], default=np.zeros((5, 2)),
                names=["a", "b", "c", "d"], head="pass", u_range=(-4.5, 2.6),
                prior="sparse", prior_k=10, laws_on_z=True)
    back = bank_from_json(bank_json(bank))
    assert back["prior_k"] == 10
    assert back["u_range"] == (-4.5, 2.6)


def test_cem_returns_a_law_in_the_class():
    """It scores every candidate constrained and returned the dense mean."""
    env = _world()
    d = len(env.names) + 3
    bank = dict(clauses=[], laws=[], default=np.zeros((d, 2)),
                names=list(env.names), head="pass", laws_on_z=True,
                u_range=tuple(env.u_range), prior="sparse", prior_k=3,
                actions=None)
    pol_fn = __import__("btind.memory", fromlist=["MemBank"]).MemBank
    th, _ = cem_law(env, bank, -1, lambda b: pol_fn(b, len(env.names)),
                    n_iter=1, K=6, n_ep=4, T=60, seed=3,
                    rng=np.random.default_rng(0))
    kept = int((np.abs(np.asarray(th)[:-1]).max(axis=1) > 1e-12).sum())
    assert kept <= 3
    assert np.array_equal(th, constrain(bank, th))
