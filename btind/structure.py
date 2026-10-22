"""Structure moves: add an arm, drop an arm, re-order the Fallback.

`qwire` can move a threshold; it cannot change how many arms there are or which
one sees a state first. Both of those are decisions the pipeline currently makes
by accident -- `rsfi` buys arms greedily and never revisits one, so the arm count
is whatever the acceptance test happened to allow and the PRIORITY is the order
in which arms were bought. In a Fallback the order is semantics, not layout.

Three operators, one acceptance rule, and one asymmetry that matters:

  ADD       a proposed clause is inserted directly ABOVE the arm it specialises,
            because that is the only position where it claims the rows the
            proposal was scored on.
  DROP      accepted on NON-INFERIORITY, not improvement: the paired test has to
            fail to show a loss. Everywhere else the burden of proof is on the
            change, here it is on the arm -- an arm that cannot demonstrate it is
            earning its place is a literal a reader has to hold for nothing.
  REORDER   adjacent swaps only, and only where the two clauses actually overlap;
            a swap of disjoint clauses is a no-op that would still consume a
            rollout and, at z = 2, occasionally be "accepted" by noise.
"""
import sys
import time
import weakref
from collections import OrderedDict

import numpy as np

from .collect import design_matrix
from .memory import reindex

from .policies import evaluate


def score(env, bank, pol_fn, n_ep, T, seed):
    """Per-episode returns. Uses the fused kernel when the bank allows it.

    The fast path is bit-exact against the Python one (verified at 0 difference
    on plain, memory, and memory+sticky+beta banks) and 150-280x faster, which
    is what makes a search measured in thousands of rollouts affordable. It
    cannot run a bank whose guards read `V_hat` or `leverage` -- those need an
    xgboost predict and a critic solve per tick -- so those fall back silently.
    """
    st = {}
    g = _fast_score(env, bank, n_ep, T, seed, stats=st)
    if g is not None:
        if "rear" in st:
            _remember_crashes(g, n_ep, T, seed, st["rear"] + st["cross"])
        return g
    env.seed_kernels(seed)
    if hasattr(env, "_search_bank"):
        env._search_bank = bank          # the head its actions are in
    return evaluate(env, pol_fn(bank), n_ep=n_ep, T=T, seed=seed)["G"]


def kernel_for(env, bank):
    """(params, rollout) of the fused kernel this bank may run on, or None.

    One place decides whether a bank is allowed onto a kernel, so the scoring
    path and the exploration path cannot disagree about it. Refusals, each
    measured once the hard way: a head the kernel does not implement (the
    NestWorld kernel normalises, the highway one argmaxes -- letting the other
    head through returns a heading where an action index is expected, silently
    and 200x faster than the path that is correct); laws not yet on the z
    layout; and a GUARD that compares against V_hat or leverage, which need an
    xgboost predict per tick. A law's coefficients on those columns are fine:
    the kernel zeroes them exactly as MemBank does with no critic attached.
    """
    if not bank.get("laws_on_z"):
        return None
    from .memory import _guards_read_vq
    if _guards_read_vq(bank, len(bank["names"])):
        return None
    kind = type(env).__name__
    try:
        if kind == "IntersectionBatch":
            ok = {"vehicle": ("argmax", "scalar", "pass"),
                  "signal": ("argmax", "duration")}[env.agent]
            if bank.get("head") not in ok:
                return None
            from .envs import intersection_fast as IF
            return IF.params, IF.run
        if kind == "HighwayBatch":
            if bank.get("head") != "argmax":
                return None
            from .envs.highway_fast import params, rollout
            return params, rollout
        if kind == "NestWorld":
            if bank.get("head", "vector") != "vector":
                return None
            from .envs.nest_fast import params, rollout_mem
            return params, rollout_mem
    except Exception:
        return None
    return None


def fast_rollout(env, bank, s, T, trace=None, dev=None, stats=None):
    """Roll `bank` from the states `s` on the fused kernel. Returns G, or None.

    `trace` and `dev` are the kernel's optional per-tick record and per-episode
    deviation (see `tick.trace_array`, `explore.py`); None means neither.
    """
    k = kernel_for(env, bank)
    if k is None:
        return None
    if type(env).__name__ == "IntersectionBatch":
        # TWO TREES: the bank being scored is whichever agent the world says is
        # under search; the other is held fixed on the world object.
        from .envs import intersection_fast as IF
        # The objective is NOT named here: `reward=None` lets the kernel read
        # `env.reward_mode`, which is where `reward_as` puts the objective a
        # joint round accepts on. Naming it here too would be a second place
        # for the two paths to disagree.
        if env.agent == "signal":
            return IF.run(env, env.vehicle_bank, bank, s, T, trace, dev,
                          stats=stats)
        return IF.run(env, bank, env.signal_bank, s, T, trace, dev, stats=stats)
    from .tick import flatten, no_dev, no_trace, tick_args, world_args
    params, rollout = k
    f = flatten(bank, len(bank["names"]))
    # only the intersection kernel evaluates kernel laws; anywhere else a
    # kernel would be silently ignored, so refuse
    assert not f["has_kern"], "kernel laws run on the intersection kernel only"
    G = np.empty(len(s))
    rollout(np.ascontiguousarray(s), params(env), *world_args(f), T, G,
            no_trace() if trace is None else trace,
            no_dev() if dev is None else dev, *tick_args(f))
    return G


STARTS_CAP = 256 * 1024 ** 2            # bytes of start states kept per world

# EVERY LEARNING STEP ON ONE LINE. A step is one candidate priced against the
# incumbent on a batch and judged -- the unit the search actually advances by.
# Off by default so runs that do not ask for it read as before; a driver turns
# it on with `structure.LOG_STEPS = True`.
LOG_STEPS = False
_steps = [0]

# A CHANGE MAY NOT BUY RETURN WITH CRASHES. The return already prices a crash,
# but at a rate: 1000 an at-fault rear-end against a per-car return in the tens
# of thousands, so a move that gains throughput and adds a crash on the SAME
# episodes can clear the gain test -- and every one of them accepted is a tree
# that crashes more than the one it replaced. So on top of the return test,
# `accept` counts collisions (rear-ends + crossings, every car in the episode)
# on the very episodes both banks were scored on, and a candidate whose total
# exceeds the incumbent's by more than CRASH_TOL is rejected, whatever its return.
# Paired, so no statistics: same starts, same seeds, same count or fewer.
CRASH_GATE, CRASH_TOL = True, 0
# AND FEWER CRASHES COUNT ON THEIR OWN. With the gate alone a crash removed was
# worth only what the return charges for it, and every search stage then demands
# a return gain of `min_gain` on top -- e39's first gated round turned away
# candidates at 7->4 and 7->5 crashes because they gained +19 and +82 against a
# bar of +100. A candidate that crashes STRICTLY less on the same episodes and
# does not lose return on average (mean delta >= 0) is accepted and exempt from
# the bar: `accept` passes it, and `clears` is how a stage asks whether an
# accepted candidate may skip its `min_gain`. The demand-group veto still
# applies, so no condition pays for another here either.
SAFER_ACCEPTS = True
_last = dict(safer=False)


def clears(ok, d, bar):
    """Did the candidate `accept` just judged clear a stage's `min_gain` bar --
    by gaining more than it, or by crashing less at no cost (SAFER_ACCEPTS)?
    Call it right after that `accept`."""
    return bool(ok and (d > bar or _last["safer"]))
# per-episode crash counts of recent scores, keyed by what was scored: the
# returns array's CONTENTS and the (n_ep, T, seed) it was scored on -- so a
# caller's copy of an incumbent's returns still finds its counts
_crash_memo = OrderedDict()
_CRASH_MEMO_N = 512
# AND BY THE ARRAY ITSELF, for as long as it lives. The memo above is bounded,
# and `grow` scores its incumbent once and then screens hundreds of candidates
# (CEM included) before it confirms any: the incumbent's entry was evicted by
# then, and every grow step of e39's first gated round logged "crashes
# unchecked". An incumbent's returns are held by the caller for exactly as long
# as they are compared against, so tying the counts to that object's lifetime
# keeps them precisely as long as they can be asked for.
_crash_live = {}


def _forget_live(k, r):
    e = _crash_live.get(k)
    if e is not None and e[0] is r:
        del _crash_live[k]


def _crash_key(G, n_ep, T, seed):
    G = np.ascontiguousarray(G, np.float64)
    return (int(n_ep), None if T is None else int(T), int(seed), G.tobytes())


def _remember_crashes(G, n_ep, T, seed, crashes):
    c = np.asarray(crashes, float).copy()
    if isinstance(G, np.ndarray):
        i = id(G)
        _crash_live[i] = (weakref.ref(G, lambda r, i=i: _forget_live(i, r)),
                          (int(n_ep), None if T is None else int(T), int(seed)), c)
    k = _crash_key(G, n_ep, T, seed)
    _crash_memo[k] = c
    _crash_memo.move_to_end(k)
    while len(_crash_memo) > _CRASH_MEMO_N:
        _crash_memo.popitem(last=False)


def crashes_of(G, n_ep, T, seed):
    """Per-episode collision counts behind a returns array from `score`, or
    None when that array was not produced by a kernel score on these episodes."""
    e = _crash_live.get(id(G))
    if (e is not None and e[0]() is G
            and e[1] == (int(n_ep), None if T is None else int(T), int(seed))):
        return e[2]
    return _crash_memo.get(_crash_key(G, n_ep, T, seed))


def _executed(th, X, u_range, col):
    """What the WORLD does with a law's output column, not what the law says.

    Column 0 of a continuous head is clipped to the world's range before it is
    executed, and on a `pass` head column 1 is a logit whose SIGN is the whole
    message -- twice the logit is the same declaration. Comparing raw outputs
    would call two identical controllers different.
    """
    v = X @ np.asarray(th, float)[:, col]
    if col == 0 and u_range is not None:
        return np.clip(v, float(u_range[0]), float(u_range[1]))
    return (v > 0.0).astype(float)


def fold_constants(law, X, tol=1e-9):
    """A slope on a column the region never varies is a DISGUISED INTERCEPT.

    `0.4 * a column that is always 3.0` is the constant 1.2, and dropping the
    slope would change the command by exactly that. Folding it into the
    intercept instead is behaviour-preserving to the last bit and leaves a law
    that reads as what it is. It also lets the elimination below see the term
    for what it is: after folding it is zero, not "live on every row".
    """
    law = np.array(law, float)
    X = np.asarray(X, float)
    if not len(X) or law.shape[0] != X.shape[1]:
        return law
    for j in range(law.shape[0] - 1):
        if not np.abs(law[j]).max() > 1e-12:
            continue
        col = X[:, j]
        c = float(col[0])
        if float(np.abs(col - c).max()) <= tol:
            law[-1] = law[-1] + law[j] * c
            law[j] = 0.0
    return law


def dead_slopes(law, X, u_range, frac=0.0, cols=(0,)):
    """Slopes that cannot move what the world executes, by backward elimination.

    A COEFFICIENT IS NOT A RULE UNLESS IT CHANGES SOMETHING. Measured on the
    car's tree, 14 of its 50 command slopes could be zeroed without moving one
    row's command: the default arm carried ten and nine of them were inert,
    with `lead_gap` doing all the work on 6317 of its 6321 rows. That is the
    fixed-k prior filling its quota -- `constrain` keeps exactly `prior_k`
    slopes whether the evidence supports ten or one -- and the surplus reads
    as a rule the tree does not have.

    WHY NOT L1, which is the obvious answer. A penalty on the fit residual
    cannot see the clip, and the clip is precisely why these terms are dead:
    the law asks for +3.04 where the world allows +2.60, so its slopes argue
    about a command that never happens. L1 also only reaches the one law
    source that goes through a regression -- CEM is a black-box rollout search
    with nothing to attach a penalty to, and this test does not care where a
    law came from.

    `frac` is how much a term must MOVE to earn its place: 0 drops only what
    is exactly inert, which cannot change behaviour on these rows at all.
    Above 0 it also drops terms that are live on too few rows to be a rule --
    on the car, six of arm1's ten moved 13 rows or fewer out of 50303.

    Backward elimination rather than one pass, because at `frac` > 0 dropping
    a term changes what the others are worth. Each step removes the single
    least influential survivor while it is under the bar, which is at most
    k(k+1)/2 matrix-vector products for k slopes.
    """
    law = np.asarray(law, float)
    X = np.asarray(X, float)
    if not len(X) or law.shape[0] != X.shape[1]:
        return []
    keep = np.array(law, float)
    drop = []
    alive = [j for j in range(law.shape[0] - 1)
             if np.abs(law[j, list(cols)]).max() > 1e-12]
    while alive:
        base = [_executed(keep, X, u_range, c) for c in cols]
        best = None
        for j in alive:
            th = np.array(keep, float)
            th[j, list(cols)] = 0.0
            moved = max(float((np.abs(_executed(th, X, u_range, c) - b) > 1e-9).mean())
                        for c, b in zip(cols, base))
            if best is None or moved < best[0]:
                best = (moved, j)
        if best[0] > frac:
            break
        keep[best[1], list(cols)] = 0.0
        drop.append(int(best[1]))
        alive.remove(best[1])
    return sorted(drop)


def prune_laws(bank, Z, frac=0.0, match_fn=None, fold=False):
    """Every law in the bank, pruned on the rows that law actually owns.

    An arm's law is judged on the rows that REACH it -- the rows matching its
    clause that no arm above it already claimed -- because a term that matters
    only where the arm never fires does not matter.

    `Z` is in whatever layout the laws are sized for, which the caller knows
    and this does not: `laws_on_z` decides it and the rows come from the round
    that already built them.
    """
    from .landscape import _match_cols
    match_fn = match_fn or _match_cols
    u_range = bank.get("u_range")
    Xd = design_matrix(np.asarray(Z, float))
    n_out = np.shape(bank["default"])[1] if np.ndim(bank["default"]) > 1 else 1
    cols = tuple(range(n_out))
    done = np.zeros(len(Z), bool)
    laws, n_drop = [], 0
    for c, cl in enumerate(bank["clauses"]):
        m = match_fn(cl, Z) & ~done
        done |= m
        th, n = _prune_one(bank["laws"][c], Xd[m], u_range, frac, cols, fold)
        laws.append(th)
        n_drop += n
    dflt, n = _prune_one(bank["default"], Xd[~done], u_range, frac, cols, fold)
    return dict(bank, laws=laws, default=dflt), n_drop + n


def _prune_one(law, X, u_range, frac, cols, fold=False):
    """Drop what cannot move the command; optionally fold disguised intercepts.

    `fold` IS OFF BY DEFAULT and it is the one part of this that is not free.
    Dropping an inert term is robust because "the clip already swallowed it"
    holds well beyond the rows it was measured on; "this column is constant"
    often holds only BECAUSE the coverage sample missed the variation, and
    folding it into the intercept is then exact on those rows and wrong
    everywhere else. Measured on the car: without folding, 15 slopes go for a
    paired +0.00 +-0.00; with it, 21 go and the delta is -7.79 +-7.27, which
    the non-inferiority gate rejects -- so folding buys six coefficients and
    then loses the whole move.
    """
    th = np.array(law, float)
    if not len(X):
        return th, 0
    before = int((np.abs(th[:-1]).max(axis=1) > 1e-12).sum())
    if fold:
        th = fold_constants(th, X)
    for j in dead_slopes(th, X, u_range, frac, cols):
        th[j, :] = 0.0
    after = int((np.abs(th[:-1]).max(axis=1) > 1e-12).sum())
    return th, before - after


def distinct_laws(pool, Xd, u_range):
    """Keep one law per DISTINCT command on these rows.

    A law is a different controller only if it commands something different
    somewhere. Two that differ on a column the region never reads, or that both
    land outside the world's range and clip to the same bound, are one
    candidate -- and screening both costs a rollout each. Measured on the
    signal, 50 laws gave 17 distinct commands, 33 of them the constant 5.
    """
    lo, hi = float(u_range[0]), float(u_range[1])
    Xd = np.asarray(Xd, float)
    seen, out = set(), []
    for name, th in pool:
        a = np.asarray(th, float)
        if a.shape[0] != Xd.shape[1]:
            out.append((name, th))
            continue
        key = np.round(np.clip(Xd @ a, lo, hi), 6).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((name, th))
    return out


def dedupe_screened(rows, score_of=None):
    """Drop candidates whose screen score is bit-identical to one already kept.

    Two candidates that score EXACTLY the same mean over the SAME episodes are
    the same controller. `scalar_primitives` puts a term on every column, so on
    any column the arm never reads every coefficient gives the identical
    rollout, and the top of the screen fills up with copies. Measured on a real
    round: growing confirmed 20 candidates that were 6 distinct ones, and the
    step search's top eight were four. Exact float equality over a 20-episode
    mean does not happen by chance, so this is safe as an identity test.
    """
    score_of = score_of or (lambda r: r[0])
    seen, out = set(), []
    for r in rows:
        v = float(score_of(r))
        if v in seen:
            continue
        seen.add(v)
        out.append(r)
    return out


class ticker:
    """A heartbeat for a SCREENING loop.

    A screen calls `score`, not `accept`, so `LOG_STEPS` never sees it: a stage
    that screens hundreds of candidates prints nothing between the line that
    announces it and the line that reports its answer. `search_steps` screened
    800 of them in silence, which is indistinguishable from a hang, and that --
    not the cost -- is what a run looked like it was stuck in.

    Call the instance once per screened candidate, with the candidate's delta if
    there is one. It prints at most one line every `period` seconds, plus one
    when the loop ends, so the output stays readable however long the loop is.
    """
    __slots__ = ("label", "total", "verbose", "period", "n", "best", "t0", "last")

    def __init__(self, label, total, verbose=True, period=15.0):
        self.label, self.total = str(label), int(total)
        self.verbose, self.period = bool(verbose), float(period)
        self.n, self.best = 0, None
        self.t0 = self.last = time.time()

    def __call__(self, delta=None):
        self.n += 1
        if delta is not None and (self.best is None or delta > self.best):
            self.best = float(delta)
        now = time.time()
        done = self.n >= self.total
        if not self.verbose or (now - self.last < self.period and not done):
            return
        self.last = now
        el = now - self.t0
        print("      %-20s screened %5d/%-5d  best %s  %4.0fs%s"
              % (self.label, self.n, self.total,
                 "   --   " if self.best is None else "%+8.2f" % self.best,
                 el, "" if done else "  ~%.0fs left"
                 % (el * (self.total - self.n) / max(self.n, 1))), flush=True)


def _make_starts(env, n_ep, seed):
    rng = np.random.default_rng(seed)
    return (env.sample_starts(n_ep, rng) if hasattr(env, "sample_starts")
            else env.sample_states(n_ep, rng))


def starts(env, n_ep, seed):
    """The episode starts every scoring path uses, from one seed.

    MEMOISED PER WORLD, because a paired test is the whole point: every
    candidate in a round is scored on the SAME episodes, so the same
    (n_ep, seed) is rebuilt thousands of times and `sample_starts` loops over
    episodes in Python. Measured on the intersection at 112 slots: 13.5 ms of a
    33 ms call at 120 episodes, 212 ms of a 685 ms call at 3000 -- 31-40% of
    every scoring call at every size, for an array that is identical each time.

    The cache lives ON THE WORLD, not in a module dict: two rungs sample
    different demands from the same seed, so a key of (n_ep, seed) alone would
    hand one rung the other's traffic. It dies with the world, and is bounded
    because the 3000-episode arrays are 19 MB each.

    A COPY IS HANDED OUT. The kernel does not write into the array (checked),
    but callers own what they are given, and a copy is 0.1-6.9 ms against the
    13-212 ms it saves.
    """
    try:
        cache = env._starts_cache
    except AttributeError:
        try:
            cache = env._starts_cache = OrderedDict()
        except Exception:                   # a world that will not be written to
            return _make_starts(env, n_ep, seed)
    key = (n_ep, seed)
    s = cache.get(key)
    if s is None:
        s = cache[key] = _make_starts(env, n_ep, seed)
        while len(cache) > 1 and sum(a.nbytes for a in cache.values()) > STARTS_CAP:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    return s.copy()


def heldout(env, bank, seeds=(11, 12, 13), n_ep=1000, T=None):
    """Return on seeds no stage of the search ever uses, through the kernel.

    The kernel is held bit-exact to the Python path by the equivalence tests,
    so scoring here is the same measurement `pipeline.evaluate_bank` makes,
    minus the world-specific extras (pickups, survival) that not every world
    has. Returns dict(G, ci) or None when the bank cannot run on a kernel.
    """
    T = T if T is not None else getattr(env, "duration", 400)
    gs = []
    for sd in seeds:
        env.seed_kernels(sd)
        g = fast_rollout(env, bank, starts(env, n_ep, sd), T)
        if g is None:
            return None
        gs.append(g)
    g = np.concatenate(gs)
    return dict(G=float(g.mean()), ci=float(1.96 * g.std() / np.sqrt(len(g))))


def _fast_score(env, bank, n_ep, T, seed, stats=None):
    """Fused rollout, or None when this bank/world combination cannot use it."""
    if kernel_for(env, bank) is None:
        return None
    env.seed_kernels(seed)
    if stats is not None and type(env).__name__ == "IntersectionBatch":
        return fast_rollout(env, bank, starts(env, n_ep, seed), T, stats=stats)
    return fast_rollout(env, bank, starts(env, n_ep, seed), T)


def _fast_score_highway(env, bank, n_ep, T, seed):
    """Kept for callers that ask the highway kernel by name."""
    if type(env).__name__ != "HighwayBatch":
        return None
    return _fast_score(env, bank, n_ep, T, seed)


def accept(env, cand, pol_fn, ref_G, n_ep, T, seed, z=2.0, side="gain",
           margin=0.25):
    """Paired z-test. 'gain' demands improvement; 'noninferior' absence of loss.

    NON-INFERIORITY NEEDS ITS OWN MARGIN AND CANNOT BORROW z. Inverting the gain
    test gives `d > -z*se`, and with se ~ 0.55 on 1000 episodes that tolerates a
    loss of 1.1 return units -- measured, a drop accepted on exactly that basis
    cost 1.08 return to remove two literals. Statistical inconclusiveness is not
    equivalence. The margin says how much return a literal is worth: an arm may
    be dropped only if it is worth less than `margin`, AND the test must also
    fail to show a loss, so a noisy estimate cannot wave a real cost through.
    """
    g = score(env, cand, pol_fn, n_ep, T, seed)
    d = g - ref_G
    se = max(d.std() / np.sqrt(len(d)), 1e-12)
    ok = (d.mean() > z * se if side == "gain"
          else (d.mean() > -margin and d.mean() > -z * se))
    _steps[0] += 1
    _step_row = (_steps[0], sys._getframe(1).f_code.co_name, int(n_ep),
                 float(d.mean()), float(se), side)
    # the crash counts of both, on these same episodes (see CRASH_GATE)
    n_new = n_ref = None
    crash_txt = ""
    if CRASH_GATE:
        c_new = crashes_of(g, n_ep, T, seed)
        c_ref = crashes_of(ref_G, n_ep, T, seed)
        if c_new is not None and c_ref is not None and len(c_new) == len(c_ref):
            n_new, n_ref = float(c_new.sum()), float(c_ref.sum())
            crash_txt = "  crashes %d->%d" % (n_ref, n_new)
        elif c_new is not None:
            crash_txt = "  crashes unchecked (no count for the incumbent)"
    # FEWER CRASHES AT NO COST IS A GAIN (see SAFER_ACCEPTS): a candidate that
    # crashes strictly less on these episodes and does not lose return on
    # average passes the gain test without having to show a return gain
    safer = bool(SAFER_ACCEPTS and side == "gain" and n_new is not None
                 and n_new < n_ref and d.mean() >= 0.0)
    if safer:
        ok = True
    # NO CONDITION MAY PAY FOR ANOTHER. On a world with a condition
    # distribution an average gain can hide a significant loss in light or in
    # heavy traffic; a move is rejected if any demand group loses by more than
    # z standard errors of that group.
    groups = (env.condition_groups(starts(env, n_ep, seed))
              if ok and hasattr(env, "condition_groups") else None)
    worst = None
    if groups is not None:
        for gi in np.unique(groups):
            dg = d[groups == gi]
            if len(dg) > 5 and dg.mean() < -z * max(dg.std() / np.sqrt(len(dg)), 1e-12):
                ok, worst = False, int(gi)
                break
    # NOR MAY IT CRASH MORE, on these same episodes (see CRASH_GATE)
    crash_rej = False
    if ok and n_new is not None and n_new > n_ref + CRASH_TOL:
        ok, crash_rej = False, True
    _last["safer"] = bool(ok and safer)
    if LOG_STEPS:
        n, who, nep, dm, sem, sd = _step_row
        print("      step %5d  %-18s %4d ep  %+8.2f +-%5.2f  %-11s %s%s"
              % (n, who, nep, dm, sem, sd,
                 ("ACCEPT (fewer crashes)" if _last["safer"] else "ACCEPT") if ok else
                 ("reject: more crashes" if crash_rej else
                  "reject: demand group %d loses" % worst
                  if worst is not None else "reject"), crash_txt),
              flush=True)
    return bool(ok), float(d.mean()), g


def _insert(bank, clause, law, arm):
    """Above the arm it specialises; a specialisation of the default goes last."""
    from .memory import insert_arm
    return insert_arm(bank, clause, law,
                      len(bank["clauses"]) if arm < 0 else arm)


def add_arm(env, bank, proposals, fit_law, pol_fn, cur_G, n_try=6, n_ep=1000,
            T=200, seed=777, z=2.0, min_n=400, match_fn=None, Z=None,
            etas=(0.05, 0.2, 0.6)):
    """Try the best-ranked proposals; keep the first that wins its rollout.

    `fit_law(mask, parent_arm, eta)` supplies the new arm law: the PARENT law
    plus a gradient step of size eta on the rows the new clause claims. Never a
    fresh solve -- an arm that starts at its own argmax starts somewhere no
    rollout has approved, the -14.6 failure measured in e19.

    The eta grid is not optional. A new arm carrying its parent law CHANGES
    NOTHING, so at small eta the candidate is a no-op the paired test cannot
    distinguish from the incumbent, and the add move could never be accepted by
    construction. The grid is the same discipline as everywhere else here:
    direction from the critic, distance from measured return.
    """
    log = []
    for p in proposals[:n_try]:
        m = match_fn(p["clause"], Z)
        if int(m.sum()) < min_n:
            continue
        for eta in etas:
            cand = _insert(bank, p["clause"], fit_law(m, p["arm"], eta),
                           p["arm"])
            ok, d, g = accept(env, cand, pol_fn, cur_G, n_ep, T, seed, z)
            log.append(dict(kind="add", arm=p["arm"], col=p["col"],
                            gain=p["gain"], n=int(m.sum()), eta=float(eta),
                            delta=d, accepted=ok))
            if ok:
                return cand, g, log
    return bank, cur_G, log


def drop_arm(env, bank, pol_fn, cur_G, n_ep=1000, T=200, seed=777, z=2.0,
             margin=0.25):
    """Remove an arm only if it is worth less than `margin` return units."""
    log = []
    for c in range(len(bank["clauses"])):
        cand = reindex(bank, [i for i in range(len(bank["clauses"])) if i != c])
        ok, d, g = accept(env, cand, pol_fn, cur_G, n_ep, T, seed, z,
                          side="noninferior", margin=margin)
        log.append(dict(kind="drop", arm=c, delta=d, accepted=ok))
        if ok:
            return cand, g, log
    return bank, cur_G, log


def _loosen(a, b):
    """The weakest conjunction implied by both guards, or None if there is none.

    A literal is kept only where both guards test the SAME column the SAME way,
    and its threshold moves to whichever of the two admits more: the lower bound
    for `>`, the upper for `<=`. Literals one guard has and the other does not
    are dropped -- keeping one would exclude the states the other owns, which is
    the opposite of a merge. An empty result means the two share nothing, and a
    guard of no literals matches every state, so that pair is refused rather
    than turned into a second default (`absorb_universal` exists for arms that
    have already become one).
    """
    out = []
    for j, t, neg in a:
        for j2, t2, neg2 in b:
            if j2 == j and neg2 == neg:
                out.append([j, (max(t, t2) if neg else min(t, t2)), neg])
                break
    return out or None


def _law_gap(a, b):
    """Relative distance between two laws, or inf when they are not comparable."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return np.inf
    scale = np.abs(a).sum() + np.abs(b).sum()
    return float(np.abs(a - b).sum() / scale) if scale > 1e-12 else 0.0


def merge_arms(env, bank, pol_fn, cur_G=None, n_ep=400, T=400, seed=777, z=2.0,
               margin=0.25, max_pairs=8, verbose=True, names=None):
    """Two arms that do the same thing in neighbouring regions, made one.

    Every structural move in this file either REMOVES (`drop_arm`, `simplify`,
    `collapse_bottom`) or NESTS (`search_subtrees`); nothing put two arms back
    together, so a tree accumulates near-duplicates that no single move can
    remove -- neither arm is droppable, because between them they own the
    region. Measured on e39's round-8 car: `lead_dv>0.827 -> Accel(0.309)` and
    `lead_dv>0.000 -> Accel(-0.765)` are one rule split at two thresholds, and
    two arms share `rival_dt>3.640`.

    PAIRS ARE TRIED IN ORDER OF HOW ALIKE THEIR LAWS ARE, since that is what
    makes a merge plausible, and the guard is the loosening of the two. The test
    is `drop_arm`'s: a merge is kept when the tree is no worse for it, never
    because it is smaller. Compaction that preserves behaviour gains exactly
    zero by construction, so a gain test could never accept one; what it buys is
    that every later CEM sweep, threshold sweep and polish pass has one arm
    fewer to iterate over, and one arm less surface to overfit.
    """
    from .memory import reindex
    cl = bank["clauses"]
    n = len(cl)
    cur = (score(env, bank, pol_fn, n_ep, T, seed) if cur_G is None else cur_G)
    log = []
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            g = _loosen(cl[i], cl[j])
            if g is None:
                continue
            pairs.append((_law_gap(bank["laws"][i], bank["laws"][j]), i, j, g))
    pairs.sort(key=lambda p: p[0])
    for gap, i, j, g in pairs[:max_pairs]:
        if not np.isfinite(gap):
            break
        cand = reindex(bank, [k for k in range(n) if k != j])
        cls = [[l[:] for l in x] for x in cand["clauses"]]
        cls[i] = g                                  # j > i, so i keeps its index
        cand = dict(cand, clauses=cls)
        ok, d, gg = accept(env, cand, pol_fn, cur, n_ep, T, seed, z,
                           side="noninferior", margin=margin)
        log.append(dict(kind="merge", arms=(i, j), law_gap=gap, delta=d,
                        accepted=bool(ok)))
        if ok:
            if verbose:
                txt = " AND ".join("%s%s%.3f" % (names[c] if names else c,
                                                 "<=" if ng else ">", t)
                                   for c, t, ng in g)
                print("    merge: arms %d and %d -> %s (%+.2f, law gap %.2f)"
                      % (i, j, txt, d, gap), flush=True)
            return cand, gg, log
    return bank, cur, log


def reorder(env, bank, pol_fn, cur_G, match_fn, Z, n_ep=1000, T=200, seed=777,
            z=2.0, min_overlap=50):
    """Adjacent swaps, restricted to pairs that actually contend for rows."""
    log = []
    for c in range(len(bank["clauses"]) - 1):
        ov = int((match_fn(bank["clauses"][c], Z)
                  & match_fn(bank["clauses"][c + 1], Z)).sum())
        if ov < min_overlap:
            continue
        order = list(range(len(bank["clauses"])))
        order[c], order[c + 1] = order[c + 1], order[c]
        cand = reindex(bank, order)
        ok, d, g = accept(env, cand, pol_fn, cur_G, n_ep, T, seed, z)
        log.append(dict(kind="reorder", arm=c, overlap=ov, delta=d,
                        accepted=ok))
        if ok:
            return cand, g, log
    return bank, cur_G, log


def simplify(env, bank, pol_fn, cur_G=None, n_ep=400, T=400, seed=777, z=2.0,
             margin=0.25, verbose=True, names=None):
    """Drop literals the tree does not need, one at a time, while it can.

    A random conjunction that wins a rollout wins it for ONE of its literals;
    the rest came along. Measured output before this: `noise>-1.292 AND
    t_capture>9.979`, where the first matches 90% of states and is a planted
    distractor -- the arm is really `t_capture>9.979` wearing a passenger.

    The test is `drop_arm`'s, one level down: a literal goes only if removing it
    costs less than `margin` AND the paired test cannot show a loss. Statistical
    inconclusiveness is not equivalence, and on a deterministic world the test
    has no noise to be inconclusive about.
    """
    cur = (score(env, bank, pol_fn, n_ep, T, seed) if cur_G is None else cur_G)
    changed, log = True, []
    while changed:
        changed = False
        for c, cl in enumerate(bank["clauses"]):
            if len(cl) < 2:
                continue
            for k in range(len(cl)):
                cls = [[l[:] for l in x] for x in bank["clauses"]]
                del cls[c][k]
                cand = dict(bank, clauses=cls)
                ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed, z,
                                  side="noninferior", margin=margin)
                log.append(dict(arm=c, lit=k, delta=d, accepted=bool(ok)))
                if ok:
                    if verbose:
                        nm = (names[cl[k][0]] if names else cl[k][0])
                        print("    simplify: arm %d drops %s (%+.2f)"
                              % (c, nm, d), flush=True)
                    bank, cur, changed = cand, g, True
                    break
            if changed:
                break
    return bank, cur, log


def coverage(clause, Z):
    """Fraction of states an arm's guard matches, ignoring the arms above it."""
    from .landscape import _match_cols
    return float(_match_cols(clause, Z).mean())


def _kern0(bank, c):
    """Arm c's step-0 kernel, which goes with its law when it becomes the default."""
    from .kernlaw import kern_of
    return kern_of(bank, c, 0)


def absorb_universal(env, bank, pol_fn, Z, cur_G=None, n_ep=600, T=400,
                     seed=777, z=2.0, max_cover=0.98, names=None, verbose=True):
    """An arm that matches everything IS the default. Say so, and reclaim the tail.

    Measured on the stored masked bank: arm 6 was `is_night > -0.040` and
    `is_night` takes the values {0, 1}, so the guard matched 100% of states. The
    default was reached on 0.0% of ticks and every arm below arm 6 was dead
    code -- which is exactly where `discover` appends the memory arm. That is
    the whole explanation for "memory gains +0.00 on every run": the arm was
    never executed, so all 10752 candidates and all 48 refinement literals
    scored identically zero, and the search was ranking noise.

    Nothing here changes behaviour -- the universal arm's law becomes the
    default's law and the dead arms below it are removed -- so the move is
    priced with the paired test and kept only if it is non-inferior. What it
    changes is REACHABILITY: after it, an appended arm is once again above the
    default and can fire.
    """
    cur = (score(env, bank, pol_fn, n_ep, T, seed) if cur_G is None else cur_G)
    for c in range(len(bank["clauses"])):
        if coverage(bank["clauses"][c], Z) < max_cover:
            continue
        cand = dict(reindex(bank, list(range(c))), default=bank["laws"][c],
                    kern_default=_kern0(bank, c))
        # NON-INFERIORITY, because this is an equivalence move: the arm's law
        # becomes the default's law and only unreachable arms are removed, so
        # the honest expected delta is exactly 0.00 -- which a gain test rejects.
        ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed, z,
                          side="noninferior", margin=0.25)
        if verbose:
            nm = (names[bank["clauses"][c][0][0]] if names else c)
            print("    universal guard on arm %d (%s) matches %.0f%% -- "
                  "%s as default, %d dead arms below (%+.2f)"
                  % (c, nm, 100 * coverage(bank["clauses"][c], Z),
                     "absorbed" if ok else "NOT absorbed",
                     len(bank["clauses"]) - c - 1, d), flush=True)
        if ok:
            return cand, g, True
    return bank, cur, False


def collapse_bottom(env, bank, pol_fn, cur_G=None, n_ep=600, T=400, seed=777,
                    z=2.0, margin=0.25, names=None, verbose=True):
    """If the bottom arm's guard does nothing, its law IS the default.

    `absorb_universal` catches a guard that matches everything. This catches
    the commoner disguise: a guard matching MOST states whose accepted gain
    came from the law it carries, not from the split. Measured on the
    intersection signal search: `noise > -1.046`, a planted distractor
    matching 85% of episodes, was bought for +4.4 -- the CEM-tuned law on that
    arm was simply better than the default's, and the guard was a passenger.

    The move is an equivalence test, so it is priced for NON-INFERIORITY: drop
    the bottom arm, make its law the default's, and keep the simpler tree if
    the paired test cannot show a loss beyond `margin`. Only the bottom arm,
    because collapsing a higher arm hands its rows to the arms below and is
    a different tree, not the same one written shorter.
    """
    C = len(bank["clauses"])
    if not C:
        return bank, cur_G, False
    cur = (score(env, bank, pol_fn, n_ep, T, seed) if cur_G is None else cur_G)
    c = C - 1
    cand = dict(reindex(bank, list(range(c))), default=bank["laws"][c],
                kern_default=_kern0(bank, c))
    ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed, z,
                      side="noninferior", margin=margin)
    if verbose:
        nm = (names[bank["clauses"][c][0][0]] if names else c)
        print("    collapse: bottom arm %d (%s) as default -- %s (%+.2f)"
              % (c, nm, "accepted" if ok else "kept", d), flush=True)
    return (cand, g, True) if ok else (bank, cur, False)


def polish_thresholds(env, bank, pol_fn, Z, cur_G=None, n_ep=400, T=400,
                      seed=777, z=2.0, min_gain=0.1, steps=(0.04, 0.1, 0.25, 0.6, 1.2),
                      max_sweeps=4, names=None, verbose=True, max_cover=0.98,
                      min_cover=0.02):
    """Coordinate-wise line search on every threshold, decided by rollout.

    A threshold arrives wherever the random draw that proposed it happened to
    land, and nothing afterwards moves it: `drift` only re-proposes the MOST
    RECENTLY accepted arm during growth, and the boundary term that used to do
    this needs a critic, which on this world is at chance. So an arm can be the
    right region carved in the wrong place -- measured, the grower's flee arm
    lands at `t_capture<=12.06` where the hand-tuned equivalent is far tighter,
    and on `d_threat` the difference between 0.09 and 0.20 is +5.87 against
    -3.81.

    Steps are scaled by each column's interquartile range, so one setting works
    across columns whose units have nothing to do with each other -- a bearing
    in [-1,1] beside a time-to-capture in steps.

    `min_gain` is lower here than for structural moves: this is refinement of
    something already bought, not the purchase of a new arm, so a tenth of a
    return unit is worth keeping.
    """
    from .landscape import _match_cols
    iqr = {j: float(np.subtract(*np.percentile(Z[:, j], [75, 25])) or 1.0)
           for j in range(Z.shape[1])}
    cur = (score(env, bank, pol_fn, n_ep, T, seed) if cur_G is None else cur_G)
    log = []
    for sweep in range(max_sweeps):
        moved = False
        for c in range(len(bank["clauses"])):
            for k in range(len(bank["clauses"][c])):
                j, thr, neg = bank["clauses"][c][k]
                base_m = _match_cols(bank["clauses"][c], Z)
                best, best_d = None, min_gain
                for mult in steps:
                    for sgn in (-1.0, 1.0):
                        new = thr + sgn * mult * abs(iqr[j])
                        cl = [[l[:] for l in x] for x in bank["clauses"]]
                        cl[c][k][1] = float(new)
                        # A THRESHOLD MAY NOT SLIDE PAST THE END OF ITS COLUMN.
                        # `is_night` is {0,1} with an IQR of 1, so one step of
                        # 0.6 took `> 0.5` to `> -0.04` -- a guard matching every
                        # state. It was accepted, correctly, as non-inferior:
                        # the arm's law simply replaced the default's. But it
                        # silently made the default and every arm below it
                        # unreachable, and the memory search appends there.
                        # A THRESHOLD MAY NOT SLIDE PAST THE OTHER END
                        # EITHER. `ph3` is a phase one-hot: its middle half is
                        # a single value, so its IQR is 0, the fallback scale
                        # is 1.0 -- the column's whole range -- and a step of
                        # 1.2 took `ph3 > 0.0` to `ph3 > 1.2`, which matches
                        # NOTHING. Coverage 0 passed the upper bound, the arm
                        # went dead, and the rollout scored that as +274.34
                        # because deleting the arm helped. `drop_arm` then
                        # accepted at +0.00, confirming there was nothing left.
                        # The search bought an arm and unbought it in one round.
                        m = _match_cols(cl[c], Z)
                        cv = float(m.mean())
                        if cv > max_cover or cv < min_cover:
                            continue
                        # SAME PARTITION, SAME CONTROLLER. A binary column has
                        # one meaningful cut, so every step inside (0, 1) gives
                        # the identical arm and a paired test of a controller
                        # against itself: 31 of round 4's 58 threshold
                        # confirmations came back +0.00 +- 0.00.
                        if np.array_equal(m, base_m):
                            continue
                        cand = dict(bank, clauses=cl)
                        ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed,
                                          z)
                        log.append(dict(arm=c, lit=k, thr=float(new), delta=d,
                                        accepted=clears(ok, d, min_gain)))
                        if clears(ok, d, min_gain) and (best is None or d > best_d):
                            best, best_d, best_g, best_thr = cand, d, g, new
                if best is not None:
                    if verbose:
                        nm = names[j] if names else j
                        print("    threshold: arm %d %s %.3f -> %.3f (%+.2f)"
                              % (c, nm, thr, best_thr, best_d), flush=True)
                    bank, cur, moved = best, best_g, True
        if not moved:
            break
    return bank, cur, log
