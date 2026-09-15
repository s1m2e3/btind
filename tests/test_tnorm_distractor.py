"""`t_norm` must be a distractor: no observed column may decode elapsed time.

On a truncated episode elapsed time is a real cue -- it says whether a car can
still exit and how much delay is left to charge -- and the e28 search bought it
in both trees. This test measures decodability directly: a flexible regressor
predicts the tick from observed columns, trained on some episodes and scored on
OTHER episodes (a row split lets `noise` identify the episode and the model
memorise its phase, which a tree never gets to do).

Two positive controls keep it from passing vacuously: the raw fraction decodes
perfectly, and a phase computed from the observed `noise` -- the first attempted
fix -- still decodes most of it. Only a phase drawn independently and held in
unobserved state carries nothing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import IntersectionBatch
from btind.runlog import env_signature


def _rows(n_ep=1500, seed=0):
    env = IntersectionBatch(n_max=4, T_end=40.0)
    s = env.sample_starts(n_ep, np.random.default_rng(seed))
    N, out = env.N, []
    for _ in range(env.duration):
        o = env.observe_signal(s)
        X = s[:, 5 * N:5 * N + env.N_MISC]
        raw = X[:, 4] / env.duration
        leak = np.mod(raw + np.mod(np.abs(X[:, 5]) * 7.31, 1.0), 1.0)
        out.append(np.c_[np.arange(n_ep), o[:, env.sig_names.index("t_norm")],
                         o[:, env.sig_names.index("noise")], X[:, 4], raw, leak])
        s, _, _ = env.step_both(s, np.zeros(n_ep * N), None)
    return env, np.vstack(out)


def _heldout_r2(feats, y, train):
    from sklearn.ensemble import ExtraTreesRegressor
    m = ExtraTreesRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1,
                            random_state=0).fit(feats[train], y[train])
    p = m.predict(feats[~train])
    yt = y[~train]
    return 1.0 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()


def test_elapsed_time_is_not_decodable_from_observed_columns():
    env, R = _rows()
    ep, t_norm, noise, tick, raw, leak = R.T
    train = ep < ep.max() / 2
    # positive controls: the constructions that DID leak
    assert _heldout_r2(np.c_[raw], tick, train) > 0.95
    assert _heldout_r2(np.c_[leak, noise], tick, train) > 0.5
    # the world's column, alone and with `noise`
    assert abs(np.corrcoef(t_norm, tick)[0, 1]) < 0.02
    assert _heldout_r2(np.c_[t_norm], tick, train) < 0.02
    assert _heldout_r2(np.c_[t_norm, noise], tick, train) < 0.02


def test_observation_version_is_part_of_the_store_key():
    """A checkpoint from the leaky observation must never be resumed here."""
    sig = env_signature(IntersectionBatch(n_max=4, T_end=40.0))
    assert sig.get("obs_version") == IntersectionBatch.OBS_VERSION >= 2


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok  %s" % k)
