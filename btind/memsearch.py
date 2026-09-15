"""Discovering what to remember, and when to write it down.

THE SEARCH SPACE IS (what, when), AND THE SECOND HALF IS THE DANGEROUS ONE.
Measured on the masked NestWorld, with the SAME stored columns and the same
tree, only the write event differing:

    memory never written                      G  2.47
    write on `food_seen`                      G -0.07     worse than no memory
    write on `food_seen AND d_food <= 0.12`   G  9.68
    hand-written ceiling                      G 11.25

Latching at the moment of sighting stores a point up to `vision_r` = 0.30 away
from the thing being remembered, so the agent walks confidently back to the
wrong place. Fixing the event at "when you can see it" -- the obvious choice --
would have produced the conclusion that memory hurts.

NOTHING IS NAMED BY HAND. Every observation column is a candidate to store,
including the four planted distractors, and every write event is a literal drawn
from the same vocabulary the guards use. The task word "food" appears nowhere in
the search.

PROPOSE BY INFORMATION GAP, SELECT BY RETURN. The candidate space is thousands
of (columns, event) pairs and a rollout costs a third of a second, so it has to
be pruned by something cheap. The cheap thing is an information gap against a
PRIVILEGED reference: the planner sees the true state, so whatever it does that
an observation-only law cannot predict is exactly what the observation is
missing. Regress the planner's action on `z`, then on `z` plus a candidate's
latched columns; the rise in fit is what that candidate recovers.

ONE ROLLOUT SERVES EVERY CANDIDATE. The latched value of any (columns, event)
pair is a running "last value where the event fired", so a single recorded
trajectory answers the question for all of them offline. The expensive part is
labelling the recorded states with the planner, once.
"""
import time

import numpy as np

from .collect import design_matrix
from .memory import insert_arm, reindex
from .landscape import _match_cols
from .structure import accept, score

_EPS = 1e-12


# --------------------------------------------------------------- recording
def record(env, policy, rng, n_ep=400, T=400):
    """Trajectories of states and observations under the current controller."""
    s = (env.sample_starts(n_ep, rng) if hasattr(env, "sample_starts")
         else env.sample_states(n_ep, rng))
    if hasattr(policy, "reset"):
        policy.reset(n_ep)
    d = env.observe(s).shape[1]
    OB = np.zeros((n_ep, T, d))
    ST = np.zeros((n_ep, T, s.shape[1]))
    AL = np.zeros((n_ep, T), bool)
    alive = np.ones(n_ep, bool)
    for t in range(T):
        AL[:, t] = alive
        ST[:, t] = s
        o = env.observe(s)
        OB[:, t] = o
        s, _, done = env.step(s, policy.act(o))
        alive &= ~done
        if not alive.any():
            break
    return OB, ST, AL


def latched(OB, cols, write, zb=None, with_age=False):
    """Replay a write rule over recorded observations, offline.

    Returns (values (n,T,k), have (n,T)) as they WOULD have been had the rule
    been in force -- which is not the same as having run the controller with it,
    since the trajectory would have differed. That is exactly why this ranks
    candidates rather than deciding them.
    """
    n, T, _ = OB.shape
    Z = OB if zb is None else zb
    fire = np.stack([_match_cols(write, Z[:, t]) for t in range(T)], 1)
    vals = OB[:, :, cols]
    out = np.zeros_like(vals)
    have = np.zeros((n, T), bool)
    age = np.zeros((n, T))
    cur = np.zeros((n, len(cols)))
    hv = np.zeros(n, bool)
    ag = np.zeros(n)
    for t in range(T):
        f = fire[:, t]
        ag = np.where(hv, ag + 1, 0.0)
        cur[f] = vals[f, t]
        hv |= f
        ag[f] = 0.0
        out[:, t] = cur
        have[:, t] = hv
        age[:, t] = ag
    if with_age:
        return out, have, age
    return out, have


# ------------------------------------------------------------------ ranking
def _r2(X, Y, tr, te):
    """HELD-OUT multi-output R^2: fit on `tr`, score on `te`.

    In-sample R^2 cannot rank these candidates, because every candidate ADDS
    columns and in-sample fit never falls. Measured: `noise` is constant within
    an episode on this world, so latching it is a pure duplicate carrying no
    information -- and it still took rank 1 of 3774 on in-sample gain, paired
    with a real column, purely as regression capacity. Held-out scoring prices
    capacity at zero, which is what the distractor control is for.
    """
    coef, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
    res = ((Y[te] - X[te] @ coef) ** 2).sum()
    tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum()
    return float(1.0 - res / max(tot, _EPS))


def rank(OB, AL, U_star, cands, w=None, max_rows=40000, rng=None):
    """Information gain of each (cols, write) candidate over the plain features.

    The baseline is the observation itself, so a candidate is credited only with
    what it adds -- a latched column that merely duplicates a live one scores
    zero, which is the behaviour that keeps the ranking honest.
    """
    rng = rng or np.random.default_rng(0)
    live = AL.reshape(-1)
    idx = np.flatnonzero(live)
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    O = OB.reshape(-1, OB.shape[2])[idx]
    Y = U_star[idx]
    base = design_matrix(O)
    half = rng.random(len(idx)) < 0.5
    tr, te = np.flatnonzero(half), np.flatnonzero(~half)
    r0 = _r2(base, Y, tr, te)
    out = []
    for cols, write, label in cands:
        v, h = latched(OB, list(cols), write)
        V = v.reshape(-1, len(cols))[idx]
        H = h.reshape(-1)[idx].astype(float)[:, None]
        gain = _r2(np.hstack([base, V, H]), Y, tr, te) - r0
        out.append(dict(cols=list(cols), write=write, label=label,
                        gain=float(gain), wrote=float(h.reshape(-1)[idx].mean())))
    return sorted(out, key=lambda d: -d["gain"]), r0


def candidates(names, OB, AL, n_thr=9, pair_adjacent=True, max_write=None):
    """Every column as a store; every literal as a write event.

    Pairs are formed only between ADJACENT columns. That is a structural prior
    about the observation layout, not about the task: a world that reports a
    vector reports its components side by side. Singletons cover the rest.
    """
    live = AL.reshape(-1)
    O = OB.reshape(-1, OB.shape[2])[live]
    d = O.shape[1]
    # THE GRID HAS TO REACH THE TAILS. Measured: the write event worth +9.7
    # return is `d_food <= 0.12`, which sits at quantile 0.13 of that column --
    # outside a 0.2-0.8 grid, so the winning candidate was not in the pool at
    # all. Masked columns make this worse: `d_food` is pinned at its 1.5
    # sentinel for 63% of steps, so the upper half of any quantile grid is one
    # repeated value and the informative range lives entirely in the tail.
    from .thresholds import literals
    writes = []
    for j in range(d):
        writes += literals(O[:, j], j, n_thr=n_thr, lo=0.03, hi=0.97,
                           name=names[j])
    if max_write:
        writes = writes[:max_write]
    stores = [((j,), names[j]) for j in range(d)]
    if pair_adjacent:
        stores += [((j, j + 1), "%s+%s" % (names[j], names[j + 1]))
                   for j in range(d - 1)]
    return [(c, w, "%s @ %s" % (cn, wn)) for c, cn in stores for w, wn in writes]


# ------------------------------------------------------------------- search
def search_memory(env, bank, names, pol_fn, n_obs, U_star, OB, AL,
                  n_try=24, n_ep=600, T=400, seed=777, z=2.0, verbose=True,
                  rng=None):
    """Rank offline, then rollout the best few. The rollout decides.

    A candidate is installed as a blackboard rule and NOTHING ELSE changes --
    no arm is added to use it. That is deliberate for the first pass: it asks
    whether merely HAVING the memory helps, which is a weaker question than the
    one we care about, so a null result here is not yet evidence against memory.
    """
    cands = candidates(names, OB, AL)
    ranked, r0 = rank(OB, AL, U_star, cands, rng=rng)
    if verbose:
        print("    baseline R2(planner action | obs) = %.3f, %d candidates"
              % (r0, len(cands)))
        for c in ranked[:5]:
            print("      %-44s gain %+.4f  (writes on %.0f%% of steps)"
                  % (c["label"], c["gain"], 100 * c["wrote"]))
    cur = score(env, bank, pol_fn, n_ep, T, seed)
    best, best_d, log = None, 0.0, []
    for c in ranked[:n_try]:
        cand = dict(bank, mem=dict(cols=c["cols"], write=c["write"],
                                   clear=None))
        ok, d, g = accept(env, cand, pol_fn, cur, n_ep, T, seed, z)
        log.append(dict(label=c["label"], gain=c["gain"], delta=d,
                        accepted=bool(ok)))
        if ok and d > best_d:
            best, best_d = cand, d
    if verbose:
        print("    rollout: %d/%d cleared the test%s"
              % (sum(l["accepted"] for l in log), len(log),
                 ("; best %s %+.2f" % (max(log, key=lambda l: l["delta"])["label"],
                                       best_d)) if best else ""))
    return (best or bank), log, ranked


# ------------------------------------------------ the second half of a guard
def z_rows(OB, AL, mem, n_obs, max_rows=60000, rng=None):
    """Recorded observations laid out in a bank's FULL column order.

    The memory columns are replayed offline by `latched`, the same way the
    ranking stage builds them, so a guard alphabet over the blackboard costs no
    extra rollout. `V_hat` and `leverage` are present as zeros and are never
    offered: a guard that reads them forces the Python path and costs 60x.
    """
    rng = rng or np.random.default_rng(0)
    live = np.flatnonzero(AL.reshape(-1))
    if len(live) > max_rows:
        live = rng.choice(live, max_rows, replace=False)
    O = OB.reshape(-1, OB.shape[2])[live]
    cols = list(mem["cols"])
    V, H, AG = latched(OB, cols, mem["write"], with_age=True)
    V = V.reshape(-1, len(cols))[live]
    H = H.reshape(-1)[live].astype(float)[:, None]
    AG = AG.reshape(-1)[live][:, None]
    parts = [O, np.zeros((len(O), 2)), V, H, AG]
    if mem.get("countdown"):
        parts.append(V - AG)
    Z = np.hstack(parts)
    # every memory column is offered as a guard, the age included; V_hat and
    # leverage are not, and the intercept is not a column
    offered = list(range(n_obs)) + list(range(n_obs + 2, Z.shape[1]))
    return Z, offered


def refine_guard(env, bank, arm, zn, pol_fn_for, Z, offered, n_thr=7, n_try=48,
                 screen_ep=120, confirm_ep=600, n_keep=6, T=400, seed=777,
                 z=2.0, weights=None, rng=None, verbose=True, positions=None,
                 screen_pos=0, min_cover=0.005):
    """Conjoin a SECOND literal onto one arm's guard, chosen by rollout.

    THE MEMORY ARM CANNOT WORK AS A SINGLETON. `have_mem > 0.5` is true forever
    once the blackboard has been written, so the arm is either at the top, where
    it overrides everything the tree already does well, or at the bottom, where
    it inherits whatever no other arm claimed. Measured: every one of 10752
    candidates scored at or below zero, and the accepted gain was +0.00 on every
    run. The arm needs to say WHEN the remembered value is the thing to act on,
    and that condition -- in this world, "and you cannot see it now" -- is a
    second literal, not a different memory rule.

    Writing that conjunct by hand would be the task knowledge this project is
    trying not to spend, so it is searched from the same alphabet as any other
    guard, over the same columns, with the same paired test deciding.

    POSITION IS RE-SEARCHED with the narrowed guard, because the reason the arm
    was pushed to the bottom was that it claimed too much. A guard that fires on
    8% of states can sit above arms that a guard firing on 90% had to yield to.
    """
    from .thresholds import literals
    rng = rng or np.random.default_rng(0)
    cl0 = bank["clauses"][arm]
    cands = []
    for j in offered:
        cands += literals(Z[:, j], j, n_thr=n_thr, lo=0.03, hi=0.97,
                          name=zn[j] if j < len(zn) else str(j))
    cands = [(c, lb) for c, lb in cands if c[0][0] not in [l[0] for l in cl0]]
    if weights is not None and len(cands) > n_try:
        w = weights.probs([c[0][0] for c, _ in cands])
        p = np.array([w[c[0][0]] for c, _ in cands])
        p = p / p.sum() if p.sum() > 0 else None
        cands = [cands[i] for i in rng.choice(len(cands), n_try, replace=False,
                                              p=p)]
    elif len(cands) > n_try:
        cands = [cands[i] for i in rng.choice(len(cands), n_try, replace=False)]

    n_arm = len(bank["clauses"])

    def drop():
        """The same tree with this arm removed -- the honest reference."""
        return reindex(bank, [i for i in range(n_arm) if i != arm])

    def place(lit, pos):
        """This arm moved to `pos`, with `lit` conjoined onto its guard."""
        g = [l[:] for l in bank["clauses"][arm]] + ([list(lit)] if lit else [])
        per = {k: (bank.get(k) or [None] * n_arm)[arm]
               for k in ("betas", "sticky", "steps", "fails", "kerns")}
        return insert_arm(drop(), g, bank["laws"][arm], pos, beta=per["betas"],
                          sticky=bool(per["sticky"]), steps=per["steps"],
                          fails=per["fails"], kerns=per["kerns"])

    pol = pol_fn_for(zn)
    # THE REFERENCE IS THE TREE WITHOUT THE ARM, not the tree with its
    # unrefined guard. Scored against the bare guard, the winning move is a
    # conjunct that is never true: measured, `slack_food <= -45.377` scored
    # +15.14 that way and +0.00 against no memory at all, because all it did
    # was switch the arm off. Against the arm-free tree that move is worth
    # exactly zero and the only way to win is to make the arm useful.
    ref = drop()
    base = score(env, ref, pol, screen_ep, T, seed)
    rows = []
    # SCREEN AT THE TOP, NOT WHERE THE ARM SITS. Scoring a conjunction at the
    # bottom of the tree asks what it is worth on the states nothing above
    # claimed, and if the arms above claim nearly all of them every literal
    # ties -- measured, 48 literals all scored exactly +0.00 in under a second,
    # which is the arm not running rather than the literals not helping. At the
    # top the narrowed guard gets first refusal, so the screen measures the
    # literal; `pos_list` then prices it back into the tree.
    # SCREENED JOINTLY OVER LITERAL AND POSITION, not literal first. A literal
    # is only good SOMEWHERE: the conjunct that makes the arm right at depth 6
    # can be the worst of the 48 at the top, so ranking at one position and
    # position-searching the survivors drops the winner before it is ever
    # placed. The rollouts are cheap enough after the fused kernel that the
    # product is affordable -- 48 x 8 screens run in a few seconds -- and no
    # staging is worth a systematically missed candidate.
    pos_list = (list(range(len(bank["clauses"]))) if positions is None
                else list(positions))
    placed = []
    from .landscape import _match_cols
    cover0 = _match_cols(cl0, Z)
    for cl, label in cands:
        # A CONJUNCT THAT EMPTIES THE ARM IS NOT A REFINEMENT. It scores zero
        # against the arm-free reference by construction, so it cannot win, but
        # screening it is wasted budget -- and on a noisy 120-episode screen a
        # tie can still float to the top.
        if float((cover0 & _match_cols([cl[0]], Z)).mean()) < min_cover:
            continue
        for pos in pos_list:
            g = score(env, place(cl[0], pos), pol, screen_ep, T, seed)
            placed.append((float((g - base).mean()), cl[0], pos, label))
    placed.sort(key=lambda r: -r[0])
    if verbose:
        print("    guard refinement on arm %d: %d (literal, position) pairs "
              "screened, best %s @%d %+.2f"
              % (arm, len(placed), placed[0][3], placed[0][2], placed[0][0])
              if placed else "    guard refinement: nothing to screen",
              flush=True)

    cur = score(env, ref, pol, confirm_ep, T, seed)
    best, best_d, log = None, 0.0, []
    for d0, lit, pos, label in placed[:n_keep]:
        cand = place(lit, pos)
        ok, d, _ = accept(env, cand, pol, cur, confirm_ep, T, seed, z)
        log.append(dict(lit=label, pos=pos, screen=d0, delta=d,
                        accepted=bool(ok)))
        if ok and d > best_d:
            best, best_d, best_lab = cand, d, "%s @%d" % (label, pos)
    if verbose:
        print("    confirmed %d/%d; %s"
              % (sum(l["accepted"] for l in log), len(log),
                 ("AND %s %+.2f" % (best_lab, best_d)) if best
                 else "no conjunct beats dropping the arm"), flush=True)
    # WHEN NOTHING WINS, RETURN THE ARM-FREE TREE. That is what the reference
    # was, and keeping an arm that no guard could make worth its place is how a
    # -15.14 regression survives a stage that reported "nothing accepted".
    return (best or ref), log


# ------------------------------------------------- joint discovery (no proxy)
def discover(env, bank, names, pol_fn_for, n_obs, OB, AL, n_thr=7,
             screen_ep=120, confirm_ep=600, n_confirm=25, T=400, seed=777,
             z=2.0, verbose=True, n_refine=3, weights=None, rng=None):
    """Search (what to store, when to write) jointly with an arm that uses it.

    TWO THINGS THIS LEARNED THE HARD WAY.

    A MEMORY RULE ALONE IS A NO-OP. Nothing reads the columns, so every rollout
    ties and nothing is ever accepted. The unit of search has to be the rule
    PLUS an arm guarded on `have_mem` whose law walks to the latched point.

    AND THE RANKING PROXY DOES NOT WORK. Scoring candidates by how much a
    latched column improves prediction of the privileged planner's action put
    the known-good store at rank 6 of 8214, tied with `noise`, with 5818
    candidates scoring positive -- no discrimination at all. A latched value
    correlated with hidden state predicts the planner better whether or not it
    helps the controller. What replaced it is not a better proxy but a cheaper
    version of the real thing: a short rollout screens the grid, a long one
    confirms the survivors.
    """
    from .lawsearch import memory_primitives
    from .memory import mem_names, widen

    zn0 = mem_names(names, None)
    cands = [c for c in candidates(names, OB, AL, n_thr=n_thr) if len(c[0]) == 2]
    if verbose:
        print("    memory grid: %d joint candidates" % len(cands), flush=True)

    def with_mem(mem, pos=None, extra=None):
        """Install the rule plus an arm that reads it, AT A CHOSEN POSITION.

        Appending was the default and it is wrong: in a Fallback the last arm
        sees only what nothing above it claimed, so on a six-arm tree the memory
        arm got an arbitrary remainder and every one of 10752 candidates scored
        at or below zero. Position is semantics here, not layout -- the same
        mistake that made 27 add-arm proposals look identical to the incumbent.
        """
        zn = mem_names(names, mem)
        b = widen(bank, zn0, zn)
        idx = {n: i for i, n in enumerate(zn)}
        prim = memory_primitives(zn, mem, names)
        cl = [[l[:] for l in c] for c in bank["clauses"]]
        laws = list(b["laws"])
        p = len(cl) if pos is None else max(0, min(pos, len(cl)))
        g = ([[idx["have_mem"], 0.5, False]]
             + ([list(extra)] if extra is not None else []))
        return dict(insert_arm(dict(b, clauses=cl, laws=laws), g,
                               prim["to_mem"], p), mem=mem), zn

    base_cheap = score(env, bank, pol_fn_for(zn0), screen_ep, T, seed)
    # Screen at the TOP and at the BOTTOM: the top guarantees the arm actually
    # fires so its rule can be judged, the bottom is where it belongs if the
    # arms above already handle everything it would.
    positions = (0, len(bank["clauses"]))

    # THE BARE GUARD ANTI-RANKS THE RIGHT ANSWER. `have_mem > 0.5` is true
    # forever after the first write, so an arm carrying a store that genuinely
    # works claims far more than it should and SCORES WORSE than one carrying a
    # store that does nothing. Measured on the absorbed masked bank, best
    # position each:
    #
    #     store                      bare have_mem      AND NOT(write)
    #     pos_x + pos_y                    -14.20              +8.53
    #     t_norm + noise (planted)         -13.53              -6.58
    #     bear_food_x + bear_food_y        -14.54              -9.75
    #
    # Bare, the three are indistinguishable and all negative, so ranking on it
    # is ranking noise -- which is what 8832 candidates screening between -0.09
    # and 0.00 actually was. The refinement stage only sees the top few and so
    # never reached the winner.
    #
    # THE CONJUNCT IS `NOT write`, and that is a structural prior about the FORM
    # of a memory rule -- write the value down when the condition holds, act on
    # what was written when it no longer does -- of exactly the same kind as
    # "adjacent columns form a vector". It names no column and no task: it is
    # read off whichever write event the candidate happens to propose. The
    # refinement stage afterwards is free to replace it with any literal in the
    # alphabet, and does; this only has to make the screen see the candidate.
    #
    # `None` stays in the set, so this can never screen worse than before.
    t0, rows = time.time(), []
    for i, (cols, write, label) in enumerate(cands):
        mem = dict(cols=list(cols), write=write, clear=None)
        conj = [None] + [[int(l[0]), float(l[1]), not bool(l[2])]
                         for l in write]
        best = None
        for pos in positions:
            for cj in conj:
                b, zn = with_mem(mem, pos, cj)
                g = score(env, b, pol_fn_for(zn), screen_ep, T, seed)
                d = float((g - base_cheap).mean())
                if best is None or d > best[0]:
                    best = (d, pos)
        rows.append((best[0], mem, "%s @%d" % (label, best[1])))
        if verbose and (i + 1) % 600 == 0:
            print("      screened %d/%d  best %+.2f  [%.0fs]"
                  % (i + 1, len(cands), max(r[0] for r in rows),
                     time.time() - t0), flush=True)
    rows.sort(key=lambda r: -r[0])

    base_full = score(env, bank, pol_fn_for(zn0), confirm_ep, T, seed)
    best, best_d, log, best_label = None, 0.0, [], None
    for d_screen, mem, label in rows[:n_confirm]:
        pos = int(label.rsplit("@", 1)[1])
        b, zn = with_mem(mem, pos)
        ok, dl, _ = accept(env, b, pol_fn_for(zn), base_full, confirm_ep, T,
                           seed, z)
        log.append(dict(label=label, screen=d_screen, delta=dl,
                        accepted=bool(ok)))
        if ok and dl > best_d:
            best, best_d, best_label = b, dl, label
    if verbose:
        print("    confirmed %d/%d; %s"
              % (sum(l["accepted"] for l in log), len(log),
                 ("winner %s %+.2f" % (best_label, best_d)) if best
                  else "nothing cleared the test"), flush=True)

    # THE SINGLETON GUARD IS THE HANDICAP, not the memory rule, so the best
    # SCREENED candidates are refined whether or not they were accepted -- a
    # rule that cannot pay for itself while its arm claims every state after the
    # first write may well pay once the arm says when to use it.
    for d_screen, mem, label in rows[:n_refine]:
        pos = int(label.rsplit("@", 1)[1])
        # Refine from the guard the SCREEN used, so the stage starts where the
        # ranking left off rather than from the bare guard it already rejected.
        b, zn = with_mem(mem, pos, [int(mem["write"][0][0]),
                                    float(mem["write"][0][1]),
                                    not bool(mem["write"][0][2])])
        Z, offered = z_rows(OB, AL, mem, n_obs, rng=rng)
        b2, rlog = refine_guard(env, b, pos, zn, pol_fn_for, Z, offered,
                                screen_ep=screen_ep, confirm_ep=confirm_ep,
                                T=T, seed=seed, z=z, weights=weights, rng=rng,
                                verbose=verbose)
        log += [dict(l, label="%s + %s" % (label, l["lit"])) for l in rlog]
        if b2 is b:
            continue
        ok, dl, _ = accept(env, b2, pol_fn_for(zn), base_full, confirm_ep, T,
                           seed, z)
        if verbose:
            print("    refined %-34s vs no memory: %+.2f  %s"
                  % (label, dl, "accepted" if ok else "rejected"), flush=True)
        if ok and dl > best_d:
            best, best_d, best_label = b2, dl, label + " (refined)"
    if verbose and best is not None:
        print("    memory: %s  %+.2f" % (best_label, best_d), flush=True)
    return (best or bank), log
