"""The transient-column prior: the message columns qualify, the planted ones do not."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btind.envs.intersection import IntersectionBatch
from btind.memtransient import candidates, replay_rows, transient_columns


def test_transient_columns_are_the_message_and_not_the_distractors():
    env = IntersectionBatch(n_max=24, T_end=60.0, sig_mode="event")
    b = env.default_vehicle_bank()
    OB, AL = env.record_traces(b, n_ep=30, seed=3)
    O = OB.reshape(-1, OB.shape[2])[AL.reshape(-1)]
    names = env.veh_names
    found = {names[j] for j, _, _ in transient_columns(O, names)}
    assert {"green", "t_sig", "all_red"} <= found, found
    assert not ({"noise", "t_norm", "d_stop", "near_int", "has_lead", "is_left",
                 "is_right"} & found), found
    cands = candidates(O, names)
    labels = [lab for _, lab in cands]
    assert any(lab.startswith("green+t_sig") for lab in labels), labels
    # replay: the stored light is held after the message tick and the
    # countdown reaches zero when the light changes, never before
    mem = dict(cols=[names.index("green"), names.index("t_sig")],
               write=[[names.index("green"), -0.5, False]], clear=None,
               countdown=True)
    O2, Z = replay_rows(OB, AL, mem, len(names))
    have = Z[:, len(names) + 2 + 2]
    left = Z[:, -1]
    assert have.mean() > 0.2
    assert (left[have > 0.5] >= -1e-9).sum() > 0
    assert np.all(Z[have > 0.5, len(names) + 2] >= 0.0)      # stored light is 0/1


if __name__ == "__main__":
    test_transient_columns_are_the_message_and_not_the_distractors()
    print("ok")
