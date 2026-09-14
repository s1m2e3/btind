"""Discovering a blackboard for a message: what is transient is what to remember.

`memsearch.discover` screens every (columns, write event) pair -- thousands --
with one hand-shaped law, `to_mem`, that means "walk to the remembered point".
That is a vector head's law and it found the food site on NestWorld. On a
discrete head there is no direction to walk in, and the value of a memory only
appears once a GUARD reads the remembered columns together with the ego state
("told red, and still time left, and near the line -> brake"), which is two or
three literals away from anything a screen of the bare rule can see.

THE PRIOR THAT MAKES IT AFFORDABLE names no task. A column that sits at one
value most of the time and leaves it occasionally is carrying a MESSAGE, and
the tick it leaves the sentinel is the message's arrival. `thresholds.modal_value`
already finds such sentinels -- it was built for the masked food distance,
which is 1.5 whenever the food is out of sight and informative otherwise, the
same shape as a V2I broadcast that reads -1 except on the tick it is delivered.
So the candidates here are:

    STORE   a transient column, or two adjacent transient columns
    WRITE   "the column is not at its sentinel" (one side or the other)
    CLOCK   with or without the countdown flag (`left_<c>` = stored - age)

a few dozen rather than thousands, and every one of them is then given what a
memory needs to show its worth: the grower runs on the widened layout with the
memory columns in its alphabet and is asked for arms that READ the memory. The
rollout prices the whole thing -- rule plus the arms that use it -- against the
memoryless tree, and the planted distractors are in the store pool like every
other column: `noise` is constant within an episode and has no sentinel, so it
does not even qualify, which is the right outcome for a column that carries
nothing.

WHAT IS COMMITTED: the form (latch named columns on an event; optionally count
down). WHAT IS DISCOVERED: which columns, which event, whether it is a clock,
and every guard that reads it. No column and no task word is named here.
"""
import numpy as np

from .grow_bt import grow
from .memory import mem_names, widen
from .memsearch import latched
from .structure import accept, score
from .thresholds import modal_value


def transient_columns(O, names, mode_share=0.5, min_other=0.005, min_distinct=2):
    """Columns dominated by one value that they occasionally leave.

    Returns [(j, sentinel, side)]: `side` is +1 when the informative values lie
    above the sentinel, -1 below, and both entries are returned when they lie
    on both sides. The planted `noise` has no mode; `t_norm` has none either.

    A MESSAGE CARRIES SOMETHING BEYOND ITS ARRIVAL. A binary flag -- near the
    box, has a leader, is a left turn -- sits at 0 most of the time and leaves
    it for a single value, and remembering "it was 1 once" is what the flag
    itself already says. The non-modal part must take at least `min_distinct`
    values, so a light (0 or 1) and a time (many) qualify and a flag does not.
    """
    out = []
    for j in range(O.shape[1]):
        col = O[:, j]
        if col.std() < 1e-12:
            continue
        centre, share = modal_value(col)
        if share < mode_share:
            continue
        w = max(1e-9, (col.max() - col.min()) / 400.0)
        other = col[np.abs(col - centre) > w]
        if len(np.unique(np.round(other, 6))) < min_distinct:
            continue
        above = float((col > centre + w).mean())
        below = float((col < centre - w).mean())
        if above >= min_other:
            out.append((j, float(centre), +1))
        if below >= min_other:
            out.append((j, float(centre), -1))
    return out


def candidates(O, names, mode_share=0.5, adjacent=True):
    """(mem dict without the countdown flag, label) for every transient store."""
    tr = transient_columns(O, names, mode_share)
    cols = sorted({j for j, _, _ in tr})
    rules = {}
    for j, centre, side in tr:
        w = max(1e-9, (O[:, j].max() - O[:, j].min()) / 400.0)
        write = [[j, centre + (w if side > 0 else -w), side < 0]]
        rules.setdefault(j, []).append((write, "%s%s%.3g" % (
            names[j], ">" if side > 0 else "<=", write[0][1])))
    out = []
    for j in cols:
        for write, wl in rules[j]:
            out.append((dict(cols=[j], write=write, clear=None),
                        "%s @ %s" % (names[j], wl)))
            if adjacent:
                for k in (j - 1, j + 1):
                    if 0 <= k < O.shape[1] and k in cols and k > j:
                        out.append((dict(cols=[j, k], write=write, clear=None),
                                    "%s+%s @ %s" % (names[j], names[k], wl)))
    return out


def _mem_only_pool(zn, n_obs):
    """A `_clause_pool` filter: keep candidate guards that read the memory."""
    mem_idx = set(range(n_obs + 2, len(zn)))

    def keep(cl):
        return any(l[0] in mem_idx for l in cl)
    return keep


def replay_rows(OB, AL, mem, n_obs, max_rows=40000, rng=None):
    """Observation rows and their FULL z layout under a memory rule, replayed
    offline along the recorded trajectories: [obs, V_hat, leverage, slots,
    have, age, (left_...)] -- the rows a grower can propose memory guards on."""
    rng = rng or np.random.default_rng(0)
    live = np.flatnonzero(AL.reshape(-1))
    if len(live) > max_rows:
        live = np.sort(rng.choice(live, max_rows, replace=False))
    O = OB.reshape(-1, OB.shape[2])[live]
    V, H, AG = latched(OB, list(mem["cols"]), mem["write"], with_age=True)
    V = V.reshape(-1, len(mem["cols"]))[live]
    H = H.reshape(-1)[live].astype(float)[:, None]
    AG = AG.reshape(-1)[live][:, None]
    parts = [O, np.zeros((len(O), 2)), V, H, AG]
    if mem.get("countdown"):
        parts.append(V - AG)
    return O, np.hstack(parts)


def discover_transient(env, bank, names, pol_fn, OB, AL, n_obs, screen_ep=120,
                       confirm_ep=600, T=400, seed=777, z=2.0, min_gain=0.5,
                       max_arms=2, rng=None, weights=None, verbose=True,
                       mode_share=0.5, pool=40):
    """Find a message-shaped blackboard and the arms that read it, by rollout.

    `OB`, `AL` are recorded trajectories of the agent under search. Every
    candidate rule, with and without the countdown, is installed on the
    widened bank, its memory columns are replayed along the trajectories, and
    the grower is run with those columns in the alphabet, keeping only guards
    that read the memory. The winner is the (rule, arms) whose return clears
    the memoryless tree's by `min_gain`; the tree without memory is the
    reference throughout, as e23 learned it must be.
    """
    import btind.grow_bt as GB
    rng = rng or np.random.default_rng(0)
    zn0 = mem_names(names, None)
    obs = OB.reshape(-1, OB.shape[2])[AL.reshape(-1)]
    cands = candidates(obs, names, mode_share)
    if verbose:
        print("    transient columns: %s"
              % (", ".join(sorted({names[c["cols"][0]] for c, _ in cands}))
                 or "none"), flush=True)
    if not cands:
        return bank, []
    base = score(env, bank, pol_fn, confirm_ep, T, seed)
    rows, log = [], []
    for mem0, label in cands:
        for countdown in (False, True):
            mem = dict(mem0, countdown=countdown)
            zn = mem_names(names, mem)
            b = dict(widen(bank, zn0, zn), mem=mem)
            O, Zm = replay_rows(OB, AL, mem, n_obs, rng=rng)
            keep = _mem_only_pool(zn, n_obs)
            _cp = GB._clause_pool
            GB._clause_pool = (lambda *a, _k=keep, **kw:
                               [c for c in _cp(*a, **kw) if _k(c)])
            try:
                b2, glog = grow(env, b, names, zn, pol_fn, O, Zm,
                                max_arms=max_arms, pool=pool, max_arity=4,
                                min_n=max(50, len(O) // 200), min_gain=min_gain,
                                screen_ep=screen_ep, confirm_ep=confirm_ep,
                                n_confirm=8, T=T, seed=seed, z=z, rng=rng,
                                verbose=False, weights=weights, structural=True)
            finally:
                GB._clause_pool = _cp
            if len(b2["clauses"]) == len(b["clauses"]):
                log.append(dict(label=label, countdown=countdown, delta=0.0,
                                accepted=False, note="no arm read the memory"))
                if verbose:
                    print("    %-40s %-9s no arm read it" %
                          (label, "clock" if countdown else "plain"), flush=True)
                continue
            ok, d, g = accept(env, b2, pol_fn, base, confirm_ep, T, seed, z)
            acc = bool(ok and d > min_gain)
            log.append(dict(label=label, countdown=countdown, delta=d,
                            accepted=acc, arms=len(b2["clauses"]) - len(b["clauses"])))
            rows.append((d, b2, label, countdown))
            if verbose:
                print("    %-40s %-9s %+7.2f  %s" %
                      (label, "clock" if countdown else "plain", d,
                       "accepted" if acc else "rejected"), flush=True)
    rows.sort(key=lambda r: -r[0])
    if rows and rows[0][0] > min_gain:
        d, b2, label, cd = rows[0]
        if verbose:
            print("    memory: %s%s  %+.2f" % (label, " (clock)" if cd else "", d),
                  flush=True)
        return b2, log
    return bank, log
