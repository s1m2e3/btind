r"""Learning where a leaf's inducing points go, and what each one says -- by rollout.

A kernel law (`kernlaw.py`) is a few rules "in state x_i do y_i" on top of the
affine law the leaf already had. This module finds them. No expert and no
gradient; three moves, each priced by the paired test like every other move:

    ADD      a point is proposed where exploration says the leaf is wrong, and
             kept if the tree with it clears `min_gain`
    TUNE     CEM over every point's location, target and the lengthscales
    PRUNE    a point whose removal costs less than `margin` goes

WHERE A POINT IS PROPOSED. `explore.deviations` pairs an episode with the same
episode in which one agent took a different action for k ticks. On this world
that difference is an exact advantage, so a deviation that paid is a triple the
law can be anchored on directly:

    location  x = z[cols] at the deviation tick
    target    discrete: the law's own preferences there, with the action that
              paid lifted above the rest; continuous: the command that paid

A learned critic (`intersection_critic.py`) proposes the same triples from far
more states than a batch of deviations touches; both go through the same test.

WHICH COLUMNS A KERNEL READS. The arm's own guard columns first -- the region is
already defined on them -- then the columns that best separate the deviations
that paid from the ones that did not, on the rows this law owns. A hint can be
given (`hint_cols`, what we are allowed to tell the model); it is offered first
and still has to earn its place through the rollouts. The planted columns are
never excluded, so the audit that catches them in guards catches them here.

LENGTHSCALES start at the spread of each column over the rows the law owns --
the range of the inputs, which is the other thing we are allowed to say.
"""
import time

import numpy as np

from . import explore as EX
from . import kernlaw as KL
from .collect import design_matrix
from .structure import accept, score
from .tick import n_steps_of


def flat_laws(bank):
    """(arm, step) per flat law index, in `tick.flatten` order; default last."""
    out = []
    for c in range(len(bank["clauses"])):
        out += [(c, k) for k in range(n_steps_of(bank, c))]
    return out + [(-1, 0)]


def _theta(bank, c, k):
    from .tick import law_of
    return np.asarray(law_of(bank, c, k), float)


def _guard_cols(bank, c):
    if c < 0:
        return []
    cols = [l[0] for l in bank["clauses"][c]]
    for adv, _ in (bank.get("steps") or [None] * (c + 1))[c] or []:
        cols += [l[0] for l in adv]
    return list(dict.fromkeys(cols))


def choose_cols(Z, paid, n_cols, prefer=(), hint=(), n_obs=None):
    """Columns for a new kernel: hints, then guard columns, then separation."""
    n_obs = Z.shape[1] if n_obs is None else n_obs
    sd = Z[:, :n_obs].std(0)
    live = [j for j in range(n_obs) if sd[j] > 1e-9]
    out = [j for j in list(hint) + list(prefer) if j in live]
    out = list(dict.fromkeys(out))[:n_cols]
    if len(out) < n_cols and paid.any() and (~paid).any():
        sep = np.abs(Z[paid][:, live].mean(0) - Z[:, live].mean(0)) / sd[live]
        for i in np.argsort(-sep):
            if len(out) >= n_cols:
                break
            if live[i] not in out:
                out.append(live[i])
    return out


def candidates(ex, bank, c, k, kern, head, n_prop, u_range=None):
    """Point proposals (X_row, Y_row, adv) for law (c, k) from paid deviations."""
    L = flat_laws(bank).index((c, k))
    own = ex["law"].astype(int) == L
    paid = own & (ex["adv"] > 0)
    if not paid.any():
        return []
    idx = np.flatnonzero(paid)
    idx = idx[np.argsort(-ex["adv"][idx])]
    th = _theta(bank, c, k)
    out = []
    for i in idx:
        z = ex["z0"][i:i + 1]
        x = z[0, kern["cols"]]
        # a point that sits on top of one already there says nothing new
        if len(kern["X"]) and (np.abs(kern["X"] - x) / kern["ls"]).sum(1).min() < 0.5:
            continue
        near = [o for o in out
                if (np.abs(o[0] - x) / kern["ls"]).sum() < 0.5 and o[2] != float(ex["adv"][i])]
        if near:
            continue
        u0 = KL.evaluate(th, kern if KL.n_points(kern) else None, design_matrix(z))[0]
        if head == "argmax":
            a = int(ex["a"][i, 0])
            y = u0.copy()
            y[a] = u0.max() + 1.0 + 0.25 * (u0.max() - u0.min())
            ys = [y]
        else:
            # THE COMMAND THAT PAID IS A DIRECTION, NOT A DOSE. A deviation is
            # drawn from five levels across the range, so the one that paid is
            # often too much: measured, braking at -4.5 on a red near the line
            # lost 3.3 while -2.7 gained 15.5. Points along the way from the
            # law's current command to the paying one are all proposed.
            a = float(ex["a"][i, 0])
            ys = [np.array([u0[0] + f * (a - u0[0])]) for f in (1.0, 0.75, 0.5, 0.25)]
        for y in ys:
            out.append((x, y, float(ex["adv"][i])))
        if len(out) >= n_prop:
            break
    return [(kern, x, y, a) for x, y, a in out]


def _add(kern, x, y):
    return KL.make(kern["cols"], np.vstack([kern["X"], x[None]]),
                   np.vstack([kern["Y"], y[None]]), kern["ls"])


def add_points(env, bank, c, k, cands, kern, pol_fn, cur, cfg, verbose=False,
               regen=None):
    """Greedy: screen every candidate, confirm the best few, keep the best.

    A candidate is (kernel it joins, x, y, adv). While the law has no points the
    candidates may sit on DIFFERENT column sets -- which columns a kernel reads
    is decided here, by rollout, not by the statistic that proposed them. After
    the first point is kept, `regen(kern)` re-proposes on the chosen columns.
    """
    log = []
    for _ in range(cfg["max_points"]):
        if not cands:
            break
        cur_cheap = score(env, bank, pol_fn, cfg["screen_ep"], cfg["T"], cfg["seed"])
        rows = []
        for j, (kb, x, y, adv) in enumerate(cands):
            b = KL.with_kern(bank, c, k, _add(kb, x, y))
            g = score(env, b, pol_fn, cfg["screen_ep"], cfg["T"], cfg["seed"])
            rows.append(((g - cur_cheap).mean(), j))
        rows.sort(key=lambda r: -r[0])
        best = None
        for d_screen, j in rows[:cfg["n_confirm"]]:
            kb, x, y, _ = cands[j]
            kc = _add(kb, x, y)
            cand = KL.with_kern(bank, c, k, kc)
            ok, d, g = accept(env, cand, pol_fn, cur, cfg["n_ep"], cfg["T"],
                              cfg["seed"], cfg["z"])
            log.append(dict(op="add", arm=c, step=k, screen=float(d_screen),
                            delta=d, accepted=bool(ok and d > cfg["min_gain"])))
            if ok and d > cfg["min_gain"] and (best is None or d > best[0]):
                best = (d, j, kc, cand, g)
        if best is None:
            break
        d, j, kern, bank, cur = best
        cands = (regen(kern) if regen is not None else
                 [cd for i, cd in enumerate(cands) if i != j])
        if verbose:
            print("      + point %d on %s: %s  %+.2f"
                  % (KL.n_points(kern), _where(c, k), _pt(kern, -1, cfg), d),
                  flush=True)
    return bank, kern, cur, log


def cem_kernel(env, bank, c, k, kern, pol_fn, cur, cfg, rng, scale, y_sigma,
               verbose=False):
    """CEM over locations (in column-spread units), targets and log lengthscales."""
    M, D = kern["X"].shape
    nA = kern["Y"].shape[1]
    mu = np.concatenate([(kern["X"] / scale).ravel(), kern["Y"].ravel() / y_sigma,
                         np.log(kern["ls"])])
    sig = np.concatenate([np.full(M * D, 0.25), np.full(M * nA, 0.5), np.full(D, 0.3)])

    def unpack(p):
        X = p[:M * D].reshape(M, D) * scale
        Y = p[M * D:M * D + M * nA].reshape(M, nA) * y_sigma
        return KL.make(kern["cols"], X, Y, np.exp(p[M * D + M * nA:]))
    n_el = max(3, cfg["cem_K"] // 4)
    for it in range(cfg["cem_iter"]):
        P = mu[None] + sig[None] * rng.standard_normal((cfg["cem_K"], len(mu)))
        P[0] = mu
        g = np.array([score(env, KL.with_kern(bank, c, k, unpack(p)), pol_fn,
                            cfg["screen_ep"], cfg["T"], cfg["seed"]).mean() for p in P])
        el = np.argsort(-g)[:n_el]
        mu = P[el].mean(0)
        sig = np.maximum(P[el].std(0), 0.03)
    kc = unpack(mu)
    cand = KL.with_kern(bank, c, k, kc)
    ok, d, g = accept(env, cand, pol_fn, cur, cfg["n_ep"], cfg["T"], cfg["seed"], cfg["z"])
    keep = bool(ok and d > cfg["min_gain"])
    if verbose:
        print("      tune %d points on %s: %+.2f %s"
              % (M, _where(c, k), d, "accepted" if keep else "rejected"), flush=True)
    rec = dict(op="cem", arm=c, step=k, delta=d, accepted=keep)
    return (cand, kc, g, rec) if keep else (bank, kern, cur, rec)


def prune_points(env, bank, c, k, kern, pol_fn, cur, cfg, verbose=False):
    """Drop every point the tree does not measurably need, one at a time."""
    log = []
    i = KL.n_points(kern) - 1
    while i >= 0 and KL.n_points(kern):
        keep = [j for j in range(KL.n_points(kern)) if j != i]
        kc = (KL.make(kern["cols"], kern["X"][keep], kern["Y"][keep], kern["ls"])
              if keep else None)
        cand = KL.with_kern(bank, c, k, kc)
        ok, d, g = accept(env, cand, pol_fn, cur, cfg["n_ep"], cfg["T"], cfg["seed"],
                          cfg["z"], side="noninferior", margin=cfg["prune_margin"])
        log.append(dict(op="prune", arm=c, step=k, delta=d, accepted=ok))
        if ok:
            if verbose:
                print("      - point %d on %s pruned (%+.2f)" % (i, _where(c, k), d),
                      flush=True)
            bank, cur = cand, g
            kern = kc if kc is not None else dict(kern, X=kern["X"][:0], Y=kern["Y"][:0])
        i -= 1
    return bank, kern, cur, log


def _where(c, k):
    return "default" if c < 0 else ("arm %d" % c + (" step %d" % k if k else ""))


def _pt(kern, i, cfg):
    names = cfg.get("zn")
    x = ", ".join("%s=%.3g" % (names[j] if names else "z%d" % j, v)
                  for j, v in zip(kern["cols"], kern["X"][i]))
    y = kern["Y"][i]
    acts = cfg.get("actions")
    what = (acts[int(np.argmax(y))] if acts and len(y) > 1 else "%.3g" % y[0])
    return "(%s) -> %s" % (x, what)


KDEFAULTS = dict(n_laws=2, max_points=3, n_cols=2, col_pool=5, n_prop=24, n_confirm=4,
                 dev_ep=800, ks=(1, 3, 8), screen_ep=120, cem_iter=3, cem_K=24,
                 prune_margin=0.25, min_share=0.05, hint_cols=())


def search_kernels(env, bank, names, zn, pol_fn, cur, T, seed, z=2.0, min_gain=0.3,
                   n_ep=300, rng=None, verbose=True, critic=None, **kw):
    """Grow, tune and prune inducing points on the laws that carry the most rows.

    `critic`, when given, is called as critic(bank, c, k, kern, head) and returns
    extra (x, y, adv) proposals for that law, ranked with the deviations'.
    """
    cfg = dict(KDEFAULTS, **kw)
    cfg.update(T=T, seed=seed, z=z, min_gain=min_gain, n_ep=n_ep, zn=zn,
               actions=bank.get("actions"))
    rng = rng or np.random.default_rng(seed)
    head = bank.get("head")
    t0 = time.time()
    ex = EX.deviations(env, bank, n_ep=cfg["dev_ep"], T=T, seed=seed + 5, rng=rng,
                       ks=cfg["ks"])
    if ex is None or not len(ex["adv"]):
        return bank, cur, []
    laws = flat_laws(bank)
    share = np.bincount(ex["law"].astype(int), minlength=len(laws)) / len(ex["law"])
    order = [int(i) for i in np.argsort(-share) if share[i] >= cfg["min_share"]]
    Zall = ex["z0"]
    log = []
    for L in order[:cfg["n_laws"]]:
        c, k = laws[L]
        own = ex["law"].astype(int) == L
        paid = ex["adv"][own] > 0
        kern = KL.kern_of(bank, c, k)
        n_out = np.asarray(bank["default"]).shape[1]

        def empty(cols):
            ls = np.maximum(Zall[own][:, cols].std(0), 1e-3)
            return KL.make(cols, np.zeros((0, len(cols))), np.zeros((0, n_out)), ls)

        def propose(kb, n):
            cs = candidates(ex, bank, c, k, kb, head, n)
            if critic is not None:
                cs = cs + [(kb,) + tuple(t) for t in critic(bank, c, k, kb, head)]
            return sorted(cs, key=lambda t: -t[3])
        if KL.n_points(kern):
            cands = propose(kern, cfg["n_prop"])
            sets = [kern["cols"]]
        else:
            # WHICH COLUMNS: every set of `n_cols` from a short pool -- hints,
            # the arm's guard columns, then the columns that best separate the
            # deviations that paid -- each proposing its own points, all
            # screened together
            from itertools import combinations
            pool = choose_cols(Zall[own], paid, cfg["col_pool"], _guard_cols(bank, c),
                               cfg["hint_cols"], n_obs=len(names))
            sets = [list(s) for s in combinations(pool, min(cfg["n_cols"], len(pool)))]
            per = max(2 if head == "argmax" else 8, cfg["n_prop"] // max(1, len(sets)))
            cands = [cd for s in sets for cd in propose(empty(s), per)]
        if verbose:
            print("    kernel on %s (%.0f%% of deviations, %d paid): %d column sets "
                  "from {%s}, %d proposals"
                  % (_where(c, k), 100 * share[L], int(paid.sum()), len(sets),
                     ",".join(sorted({zn[j] for s in sets for j in s})), len(cands)),
                  flush=True)
        bank, kern, cur, alog = add_points(env, bank, c, k, cands, kern, pol_fn, cur,
                                           cfg, verbose,
                                           regen=lambda kb: propose(kb, cfg["n_prop"]))
        if verbose and KL.n_points(kern):
            print("      columns kept: %s" % ",".join(zn[j] for j in kern["cols"]),
                  flush=True)
        log += alog
        if KL.n_points(kern):
            scale = np.maximum(Zall[own][:, kern["cols"]].std(0), 1e-3)
            y_sig = (1.0 if head == "argmax" else
                     0.25 * float(np.subtract(*reversed(bank.get("u_range")
                                                        or env.u_range))))
            bank, kern, cur, rec = cem_kernel(env, bank, c, k, kern, pol_fn, cur, cfg,
                                              rng, scale, y_sig, verbose)
            log.append(rec)
            bank, kern, cur, plog = prune_points(env, bank, c, k, kern, pol_fn, cur,
                                                 cfg, verbose)
            log += plog
    if verbose:
        n_pts = sum(KL.n_points(KL.kern_of(bank, c, k)) for c, k in flat_laws(bank))
        print("    kernels: %d inducing points in the tree [%.0fs]"
              % (n_pts, time.time() - t0), flush=True)
    return bank, cur, log
