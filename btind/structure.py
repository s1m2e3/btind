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
    return evaluate(env, pol_fn(bank), n_ep=n_ep, T=T, seed=seed)["G"]


def _fast_score(env, bank, n_ep, T, seed):
    """Fused rollout, or None when this bank/world combination cannot use it."""
    try:
        from .envs.nest_fast import flatten_mem, params, rollout_mem, uses_vq
    except Exception:
        return None
    if type(env).__name__ != "NestWorld" or not bank.get("laws_on_z"):
        return None
    n_obs = len(bank["names"])
    # The kernel sets V_hat and leverage to zero, exactly as MemBank does when
    # no critic is attached. So the only real incompatibility is a GUARD that
    # compares against them -- a law's coefficients there multiply a zero.
    from .memory import _guards_read_vq
    if _guards_read_vq(bank, n_obs):
        return None
    env.seed_kernels(seed)
    s = (env.sample_starts(n_ep, np.random.default_rng(seed))
         if hasattr(env, "sample_starts")
         else env.sample_states(n_ep, np.random.default_rng(seed)))
    f = flatten_mem(bank, n_obs)
    G = np.empty(len(s))
    rollout_mem(np.ascontiguousarray(s), params(env), f["lit_col"], f["lit_thr"],
                f["lit_neg"], f["cl_start"], f["cl_len"], f["b_col"], f["b_thr"],
                f["b_neg"], f["b_start"], f["b_len"], f["sticky"],
                f["mem_cols"], f["w_col"], f["w_thr"], f["w_neg"], f["laws"],
                f["n_obs"], T, G)
    return G


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
    pos = len(bank["clauses"]) if arm < 0 else arm
    cl = [[l[:] for l in c] for c in bank["clauses"]]
    laws = list(bank["laws"])
    cl.insert(pos, [l[:] for l in clause])
    laws.insert(pos, law)
    return dict(bank, clauses=cl, laws=laws)


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
        cand = dict(bank,
                    clauses=[x for i, x in enumerate(bank["clauses"]) if i != c],
                    laws=[x for i, x in enumerate(bank["laws"]) if i != c])
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
        cl = [[l[:] for l in x] for x in bank["clauses"]]
        laws = list(bank["laws"])
        cl[c], cl[c + 1] = cl[c + 1], cl[c]
        laws[c], laws[c + 1] = laws[c + 1], laws[c]
        cand = dict(bank, clauses=cl, laws=laws)
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
