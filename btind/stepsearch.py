"""Discovering STEPS and FAILURE conditions: the Sequence and the status.

An arm today is `Sequence[guard, action]`, one action. A hand-written tree has
sequences of several actions and children that can fail, and both are now in
the representation (`tick.py`): an arm carries `steps` -- further
(advance_clause, law) pairs it moves through while latched -- and a `fails`
clause that releases it and hands the tick to the arms below. This file
proposes them, from the tree's own rollouts and nothing else, and the paired
test decides.

THE NULL IS INSIDE THE CLASS, which is what makes the search safe. A step whose
advance clause never fires, and a fail clause that never fires, leave every
episode's return unchanged (tests/test_steps.py holds the kernels to that), so
the incumbent is a member of the candidate set and the search can only fail to
find an improvement. That required one decision measured the hard way: `betas`
is the termination of the WHOLE arm, checked at every step. With it checked on
the last step only, a never-firing advance trapped the arm forever and the
identity move did not exist.

WHERE THE CANDIDATES COME FROM. Advance and fail literals are drawn from the
same alphabet as every guard -- `beta_candidates`, the arm's own columns first
and at finer resolution, then the rest, planted distractors included. The law of
a new step comes from the same vocabulary the grower uses: the arm's current law
(identity), the constant preferences and single-column scores on a discrete
head, the structural directions on a vector head, and perturbations of the
parent. Nothing is named by hand and no expert labels anything.

WHAT THE HIGHWAY WORLD SAID ABOUT THIS (experiments/e26). Steps run -- a
hand-written two-step arm reached its second step in 68% of episodes -- and the
fail clause was worth +2.5 to +3.0 INSIDE the family of overtaking sequences.
But no sequence, hysteresis or fail variant beat the flat reactive "brake when
the lead is close" tree: the meta-actions are already macro-actions the
low-level controller completes, and the world punishes commitment. So on that
world this search should accept fail clauses where they repair a committed arm
and little else, and its silence on steps there is the world, not the operator.
The world where sequences are load-bearing by construction is the next one.

ONE MOVE PER CALL, the most-run arm first, as `search_beta` does: a step changes
which states every arm below sees, so two at once would be priced against each
other's stale distribution.
"""
import numpy as np

from .betasearch import beta_candidates, churn
from .lawsearch import discrete_primitives, structural_primitives
from .memory import check_arms
from .structure import accept, dedupe_screened, score, ticker
from .tick import law_of, n_steps_of


def with_step(bank, arm, adv, theta):
    """The bank with (adv, theta) appended as arm `arm`'s new last step.

    A stepped arm has to be sticky; an arm that was not is made so with no
    termination, so preemption and the new step's own future beta are what end
    it. That is a change of class the paired test prices like any other.
    """
    from .kernlaw import constrain
    C = len(bank["clauses"])
    steps = list(bank.get("steps") or [None] * C)
    steps[arm] = list(steps[arm] or []) + [([l[:] for l in adv],
                                            constrain(bank, theta))]
    sticky = list(bank.get("sticky") or [False] * C)
    sticky[arm] = True
    return check_arms(dict(bank, steps=steps, sticky=sticky), "with_step")


def with_fail(bank, arm, clause):
    C = len(bank["clauses"])
    fails = list(bank.get("fails") or [None] * C)
    fails[arm] = [l[:] for l in clause] if clause is not None else None
    return check_arms(dict(bank, fails=fails), "with_fail")


def _distinct_laws(pool, Z, u_range):
    """Keep one law per DISTINCT command on the data.

    A law is a different controller only if it commands something different
    somewhere. After clipping to the world's range most do not: on a real tree
    50 laws gave 17 distinct commands, 33 of them the constant 5. One matrix
    product settles it here, instead of the rollout discovering it 121 screened
    candidates later.
    """
    lo, hi = float(u_range[0]), float(u_range[1])
    Z = np.asarray(Z, float)
    seen, out = set(), []
    for name, th in pool:
        a = np.asarray(th, float)
        if a.shape[0] != Z.shape[1] + 1:
            out.append((name, th))
            continue
        u = np.clip(Z @ a[:-1] + a[-1], lo, hi)
        key = np.round(u, 6).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((name, th))
    return out


def _distinct_advances(cands, Z):
    """Keep one advance per DISTINCT set of rows it fires on."""
    from .landscape import _match_cols
    seen, out = set(), []
    for cl, label in cands:
        key = np.packbits(_match_cols(cl, Z)).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((cl, label))
    return out


def _law_pool(bank, zn, arm, rng, n_sample=40, n_perturb=4, sigma=0.4, Z=None):
    """Laws a new step may carry, from the same sources the grower uses."""
    parent = np.asarray(law_of(bank, arm, n_steps_of(bank, arm) - 1), float)
    d, n_out = parent.shape
    out = [("same", parent)]
    head = bank.get("head", "vector" if n_out == 2 else "argmax")
    if head in ("scalar", "duration"):
        from .lawsearch import scalar_primitives
        lo, hi = bank.get("u_range", (-1.0, 1.0))
        # SAMPLED, as the discrete branch below already was. The vocabulary is
        # four laws per column -- two scales, both signs -- so it grows with the
        # observation: widening the signal from 24 to 46 columns took it from
        # 102 to 190, and every law is a screening rollout for every advance.
        # The five constants are kept whole. They are the null a proportional
        # term has to beat, there are only five of them, and dropping one at
        # random would make the pool's floor depend on the draw.
        prim = list(scalar_primitives(zn, d, lo, hi, Z=Z).items())
        const = [p for p in prim if p[0].startswith("const[")]
        rest = [p for p in prim if not p[0].startswith("const[")]
        if len(rest) > n_sample:
            rest = [rest[i] for i in
                    rng.choice(len(rest), n_sample, replace=False)]
        out += const + rest
    elif head == "argmax":
        out += list(discrete_primitives(zn, n_out, d, rng=rng,
                                        n_sample=n_sample).items())
    else:
        out += list(structural_primitives(zn, d).items())
    for i in range(n_perturb):
        out.append(("rand%d" % i, parent + sigma * rng.standard_normal(parent.shape)))
    if Z is not None:
        out = _distinct_laws(out, Z, bank.get("u_range", (-1.0, 1.0)))
    return out


def _pick(cands, own_cols, n, rng, weights=None):
    """Every literal on the arm's own columns, then a draw from the rest.

    The hysteresis and the "advance once the manoeuvre has started" cases both
    live on the arm's own variables, so those are never sampled away; the
    remaining budget goes to the other columns, steered by the learned
    proposal weights when there are any.
    """
    own = [c for c in cands if c[0][0][0] in own_cols]
    rest = [c for c in cands if c[0][0][0] not in own_cols]
    k = max(0, n - len(own))
    if len(rest) > k:
        p = None
        if weights is not None:
            w = weights.probs([c[0][0][0] for c in rest])
            p = np.array([w[c[0][0][0]] for c in rest])
            p = p / p.sum() if p.sum() > 0 else None
        rest = [rest[i] for i in rng.choice(len(rest), k, replace=False, p=p)]
    return own + rest


def search_steps(env, bank, zn, Z, pol_fn, cur_G=None, arms=None, n_adv=16,
                 screen_ep=120, confirm_ep=600, n_confirm=8, T=400, seed=777,
                 z=2.0, min_gain=0.3, max_steps=3, rng=None, weights=None,
                 verbose=True, n_law_sample=40, n_law_keep=3):
    """Append one step to one arm, the (advance, law) pair that wins its rollout."""
    rng = rng or np.random.default_rng(0)
    C = len(bank["clauses"])
    if not C:
        return bank, []
    ch = churn(env, bank, pol_fn, T=T)
    order = (list(arms) if arms is not None else
             sorted(range(C), key=lambda c: -ch[c]["share"]))
    cur = score(env, bank, pol_fn, confirm_ep, T, seed) if cur_G is None else cur_G
    log = []
    for c in order:
        if ch[c]["share"] < 0.02 or n_steps_of(bank, c) >= max_steps:
            continue
        own = {l[0] for l in bank["clauses"][c]}
        # DISTINCT CANDIDATES ONLY, decided before any rollout is paid for.
        # Measured on a real tree: 46 advances x 50 laws = 2300 screens that
        # were 19 different controllers, 119 s to rank copies of each other.
        advs = _distinct_advances(
            _pick(beta_candidates(Z, zn, bank["clauses"][c]), own, n_adv,
                  rng, weights), Z)
        laws = _law_pool(bank, zn, c, rng, n_sample=n_law_sample, Z=Z)
        cheap = score(env, bank, pol_fn, screen_ep, T, seed)
        # SCREEN THE TWO DIMENSIONS SEPARATELY, not their product. Measured on a
        # real tree: 46 advances with the law held fixed gave ONE distinct
        # rollout, while 49 laws with the advance held fixed gave 18. An
        # appended step is the arm's LAST, so its advance says when to hand over
        # to a step that does not exist, and the product was 46 copies of the
        # law sweep -- 2254 screens, 52 distinct controllers, 120 s to rank
        # duplicates of each other. Sweeping the law first and the advance only
        # against the laws that survived costs 49 + 3 x 46 instead, and shrinks
        # the selection bias with it: the best of 187 noisy estimates sits far
        # closer to the truth than the best of 2254.
        tick = ticker("steps arm %d laws" % c, len(laws), verbose)
        lrows = []
        for lname, th in laws:
            g = score(env, with_step(bank, c, advs[0][0], th), pol_fn,
                      screen_ep, T, seed)
            dlt = float((g - cheap).mean())
            tick(dlt)
            lrows.append((dlt, lname, th))
        lrows.sort(key=lambda r: -r[0])
        lrows = dedupe_screened(lrows)[:max(1, n_law_keep)]
        tick = ticker("steps arm %d advances" % c, len(advs) * len(lrows),
                      verbose)
        rows = []
        for _, lname, th in lrows:
            for adv, alab in advs:
                g = score(env, with_step(bank, c, adv, th), pol_fn, screen_ep,
                          T, seed)
                dlt = float((g - cheap).mean())
                tick(dlt)
                rows.append((dlt, adv, alab, lname, th))
        rows.sort(key=lambda r: -r[0])
        rows = dedupe_screened(rows)
        for dlt, adv, alab, lname, th in rows[n_confirm:]:
            log.append(dict(kind="step", arm=c, clause=adv, adv=alab, law=lname,
                            screen=dlt, stage="screen", accepted=False))
        best, best_d = None, min_gain
        for dlt, adv, alab, lname, th in rows[:n_confirm]:
            cand = with_step(bank, c, adv, th)
            ok, d, g = accept(env, cand, pol_fn, cur, confirm_ep, T, seed, z)
            keep = bool(ok and d > min_gain)
            log.append(dict(kind="step", arm=c, clause=adv, adv=alab, law=lname,
                            screen=dlt, delta=d, accepted=keep))
            if keep and d > best_d:
                best, best_d, best_lab = cand, d, "%s -> %s" % (alab, lname)
        if verbose:
            print("    steps on arm %d (share %.0f%%, %d steps): %s"
                  % (c, 100 * ch[c]["share"], n_steps_of(bank, c),
                     ("then %s  %+.2f" % (best_lab, best_d)) if best
                     else "no step cleared %+.2f" % min_gain), flush=True)
        if best is not None:
            return best, log
    return bank, log


def search_fails(env, bank, zn, Z, pol_fn, cur_G=None, arms=None, n_try=24,
                 n_ep=600, screen_ep=None, n_confirm=6, T=400, seed=777, z=2.0,
                 min_gain=0.3, rng=None, weights=None, verbose=True):
    """Give one latched arm the fail clause that wins its rollout.

    Only sticky arms are offered one: a fail clause acts on a RUNNING arm, and a
    reactive arm is re-chosen every tick anyway, so on it the move is a no-op
    the paired test could not distinguish from the incumbent.
    """
    rng = rng or np.random.default_rng(0)
    C = len(bank["clauses"])
    if not C:
        return bank, []
    sticky = bank.get("sticky") or [False] * C
    fails = bank.get("fails") or [None] * C
    ch = churn(env, bank, pol_fn, T=T)
    order = (list(arms) if arms is not None else
             sorted(range(C), key=lambda c: -ch[c]["share"]))
    cur = score(env, bank, pol_fn, n_ep, T, seed) if cur_G is None else cur_G
    log = []
    for c in order:
        if not sticky[c] or fails[c] is not None or ch[c]["share"] < 0.02:
            continue
        own = {l[0] for l in bank["clauses"][c]}
        cands = _pick(beta_candidates(Z, zn, bank["clauses"][c]), own, n_try,
                      rng, weights)
        # SCREEN THEN CONFIRM, like every other stage. This was the one search
        # that priced its whole pool at the full episode count: measured over
        # five rounds it ran 236 confirmations and accepted none of them, every
        # delta between -300 and -500. Screening first costs a fifth as much
        # per candidate and still puts the survivors through the same test.
        sep = int(screen_ep or n_ep)
        if sep < n_ep and len(cands) > n_confirm:
            cheap = score(env, bank, pol_fn, sep, T, seed)
            ftick = ticker("fails arm %d" % c, len(cands), verbose)
            scr = []
            for cl, label in cands:
                g = score(env, with_fail(bank, c, cl), pol_fn, sep, T, seed)
                dlt = float((g - cheap).mean())
                ftick(dlt)
                scr.append((dlt, cl, label))
            scr.sort(key=lambda r: -r[0])
            cands = [(cl, label) for _, cl, label in
                     dedupe_screened(scr)[:n_confirm]]
        best, best_d = None, min_gain
        for cl, label in cands:
            cand = with_fail(bank, c, cl)
            ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed, z)
            keep = bool(ok and d > min_gain)
            log.append(dict(kind="fail", arm=c, clause=cl, fail=label, delta=d,
                            accepted=keep))
            if keep and d > best_d:
                best, best_d, best_lab = cand, d, label
        if verbose:
            print("    fail on arm %d (share %.0f%%): %s"
                  % (c, 100 * ch[c]["share"],
                     ("%s  %+.2f" % (best_lab, best_d)) if best
                     else "no fail clause cleared %+.2f" % min_gain), flush=True)
        if best is not None:
            return best, log
    return bank, log
