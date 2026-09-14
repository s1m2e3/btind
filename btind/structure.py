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
import numpy as np

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
    g = _fast_score(env, bank, n_ep, T, seed)
    if g is not None:
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
            ok = {"vehicle": ("argmax", "scalar"),
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


def fast_rollout(env, bank, s, T, trace=None, dev=None):
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
        if env.agent == "signal":
            return IF.run(env, env.vehicle_bank, bank, s, T, trace, dev)
        return IF.run(env, bank, env.signal_bank, s, T, trace, dev)
    from .tick import flatten, no_dev, no_trace, tick_args, world_args
    params, rollout = k
    f = flatten(bank, len(bank["names"]))
    G = np.empty(len(s))
    rollout(np.ascontiguousarray(s), params(env), *world_args(f), T, G,
            no_trace() if trace is None else trace,
            no_dev() if dev is None else dev, *tick_args(f))
    return G


def starts(env, n_ep, seed):
    """The episode starts every scoring path uses, from one seed."""
    rng = np.random.default_rng(seed)
    return (env.sample_starts(n_ep, rng) if hasattr(env, "sample_starts")
            else env.sample_states(n_ep, rng))


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


def _fast_score(env, bank, n_ep, T, seed):
    """Fused rollout, or None when this bank/world combination cannot use it."""
    if kernel_for(env, bank) is None:
        return None
    env.seed_kernels(seed)
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
        cand = dict(reindex(bank, list(range(c))), default=bank["laws"][c])
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
    cand = dict(reindex(bank, list(range(c))), default=bank["laws"][c])
    ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed, z,
                      side="noninferior", margin=margin)
    if verbose:
        nm = (names[bank["clauses"][c][0][0]] if names else c)
        print("    collapse: bottom arm %d (%s) as default -- %s (%+.2f)"
              % (c, nm, "accepted" if ok else "kept", d), flush=True)
    return (cand, g, True) if ok else (bank, cur, False)


def polish_thresholds(env, bank, pol_fn, Z, cur_G=None, n_ep=400, T=400,
                      seed=777, z=2.0, min_gain=0.1, steps=(0.04, 0.1, 0.25, 0.6, 1.2),
                      max_sweeps=4, names=None, verbose=True, max_cover=0.98):
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
    iqr = {j: float(np.subtract(*np.percentile(Z[:, j], [75, 25])) or 1.0)
           for j in range(Z.shape[1])}
    cur = (score(env, bank, pol_fn, n_ep, T, seed) if cur_G is None else cur_G)
    log = []
    for sweep in range(max_sweeps):
        moved = False
        for c in range(len(bank["clauses"])):
            for k in range(len(bank["clauses"][c])):
                j, thr, neg = bank["clauses"][c][k]
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
                        if coverage(cl[c], Z) > max_cover:
                            continue
                        cand = dict(bank, clauses=cl)
                        ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed,
                                          z)
                        log.append(dict(arm=c, lit=k, thr=float(new), delta=d,
                                        accepted=bool(ok and d > min_gain)))
                        if ok and d > best_d:
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
