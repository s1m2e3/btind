"""Nested subtrees: a child of the Fallback that holds its own Fallback.

Until now every tree here was one level deep. A child could carry a Sequence
of several leaves, a termination and a failure condition, but not a subtree of
its own guarded children. This file adds that, in the form that keeps every
kernel and every existing move exact:

    HIERARCHY AS SOURCE, FLAT AS TARGET.

        Fallback                                   Fallback (as run)
        |-- Sequence[ G , Fallback ]               |-- G AND g1 -> L1
        |   |-- Sequence[ g1 , L1 ]      ==        |-- G AND g2 -> L2
        |   |-- Sequence[ g2 , L2 ]                |-- G        -> D
        |   \\-- D                                  |-- ...
        |-- ...

The two are the same controller, tick for tick: the first matching child wins
at every level, and a flat arm `G AND g1` matches exactly when the subtree's
condition holds and its first child's does. So the bank stays flat, the kernels
never learn about nesting, and the nested tree is RECOVERED for reading by
factoring (`factor`): a contiguous run of children sharing literals exactly is
a subtree under those literals, recursively.

ONE SEMANTIC CHOICE, stated: a latched descendant keeps running until its own
termination even after its subtree's condition stops holding -- a condition
checked on entry, as in a memory Sequence. That is what the flat form does,
and it is a standard behaviour-tree semantics, not an approximation of one.

DISCOVERY (`grow_subtree`). The operator picks an existing child and grows new
children INSIDE it:

    rows       only the coverage rows that child actually owns
    alphabet   threshold grids fitted on those rows alone -- a whole-world
               alphabet spends its grid on regions the child never sees, which
               is why region-specific cuts are not found at the root
    start law  the child's own law, which becomes the subtree's default
    guards     the child's guard, verbatim, conjoined with each candidate
    placement  inside the child's block, above its default

and the paired rollout test accepts or rejects, as for every other move. The
null is inside the class: a subtree with no children is the child it started
from.
"""
import numpy as np

from .grow_bt import grow
from .landscape import _match_cols


# ------------------------------------------------------------------ factoring
def _key(l):
    return (int(l[0]), float(l[1]), bool(l[2]))


def factor(clauses, idx=None):
    """Recover the nested structure of a flat Fallback, recursively.

    Returns a list of nodes in priority order:
        ("arm", i, local_literals)            a child; local_literals are the
                                              guard minus its ancestors'
        ("group", shared_literals, children)  a subtree under shared_literals

    A group is a maximal contiguous run of two or more children whose guards
    share at least one literal exactly. Inside it the shared literals are
    removed and the residual guards are factored again. A child whose residual
    is empty is the subtree's default.
    """
    if idx is None:
        idx = list(range(len(clauses)))
        lits = [[list(l) for l in clauses[i]] for i in idx]
    else:
        lits = clauses
    sets = [{_key(l) for l in cl} for cl in lits]
    out, i = [], 0
    while i < len(idx):
        common, j = sets[i], i
        while j + 1 < len(idx) and common & sets[j + 1]:
            common = common & sets[j + 1]
            j += 1
        if j > i and common:
            shared = [l for l in lits[i] if _key(l) in common]
            sub = [[l for l in lits[k] if _key(l) not in common]
                   for k in range(i, j + 1)]
            out.append(("group", shared, factor(sub, idx[i:j + 1])))
        else:
            out.append(("arm", idx[i], lits[i]))
        i = j + 1
    return out


def depth(nodes):
    """Nesting depth of a factored tree: 1 for a flat Fallback."""
    if not nodes:
        return 1
    return 1 + max((depth(n[2]) for n in nodes if n[0] == "group"), default=0)


def has_subtree(bank):
    return depth(factor(bank["clauses"])) > 1


def emit_tree(bank, names, arm_text, conj, default_text):
    """Nested text of a flat bank. `arm_text(c, local_literals)` renders one
    child; `conj` renders a literal list; `default_text` the root default."""
    out = ["Fallback"]
    latched_inside = [False]

    def walk(nodes, indent):
        for k, node in enumerate(nodes):
            last = k == len(nodes) - 1 and indent != ""
            branch = ("\\-- " if last else "|-- ")
            if node[0] == "arm":
                c, local = node[1], node[2]
                text = arm_text(c, local)
                out.append(indent + branch + text)
            else:
                shared, kids = node[1], node[2]
                if any((bank.get("sticky") or [False] * len(bank["clauses"]))[n[1]]
                       for n in kids if n[0] == "arm"):
                    latched_inside[0] = True
                has_default = any(n[0] == "arm" and not n[2] for n in kids)
                out.append(indent + branch + "Sequence[ %s , Fallback%s ]"
                           % (conj(shared), "" if has_default
                              else "   # may fail to the next sibling"))
                walk(kids, indent + ("    " if last else "|   "))
    walk(factor(bank["clauses"]), "")
    out.append("\\-- %s          # totality guard" % default_text)
    if latched_inside[0]:
        out.append("# a subtree's condition is checked on entry: a running child "
                   "finishes on its own termination")
    return out


# ------------------------------------------------------------------ discovery
def owners(bank, Z):
    """Which child each row goes to, reactively (-1 = the root default)."""
    out = np.full(len(Z), -1, int)
    for c in range(len(bank["clauses"]) - 1, -1, -1):
        out[_match_cols(bank["clauses"][c], Z)] = c
    return out


def _locator(parent_clause, parent_law):
    """`where` for the grower: screen and place new children inside the block,
    directly above the subtree's default, wherever insertions have moved it."""
    pkeys = sorted(_key(l) for l in parent_clause)

    def where(bank):
        cl = bank["clauses"]
        d = next(i for i in range(len(cl))
                 if sorted(_key(l) for l in cl[i]) == pkeys
                 and np.array_equal(np.asarray(bank["laws"][i]),
                                    np.asarray(parent_law)))
        start = d
        while start > 0 and set(pkeys) <= {_key(l) for l in cl[start - 1]}:
            start -= 1
        return d, list(range(start, d + 1))
    return where


def grow_subtree(env, bank, arm, names, zn, pol_fn, obs, Z, max_arms=2,
                 pool=30, max_arity=2, min_gain=0.3, screen_ep=120,
                 confirm_ep=600, T=400, seed=777, z=2.0, rng=None,
                 weights=None, verbose=True, min_rows=200, cem_top=6,
                 cem_iter=3, cem_K=24, law_sample=10, n_perturb=4):
    """Grow children inside child `arm`, turning it into a subtree."""
    rng = rng or np.random.default_rng(0)
    own = np.flatnonzero(owners(bank, Z) == arm)
    if len(own) < min_rows:
        return bank, []
    sticky = bank.get("sticky") or [False] * len(bank["clauses"])
    steps = bank.get("steps") or [None] * len(bank["clauses"])
    if sticky[arm] or steps[arm]:
        # a latched or multi-step child wraps its own leaves; children beside
        # it would not be inside that latch, so it is not offered as a parent
        return bank, []
    parent_clause = [list(l) for l in bank["clauses"][arm]]
    parent_law = np.asarray(bank["laws"][arm])
    # SWEEP THE REGION, DON'T SAMPLE IT. A child's region is small, and a
    # random pool of a few dozen conjunctions over every column measurably
    # never proposed the light, the speed or the distance to the line inside
    # the near-intersection child -- 584 screens over six columns. Every single
    # threshold test on the region's own alphabet goes in as a seed instead,
    # screened with a light law set (constants, a few single-column scores, a
    # few perturbations of the parent), and CEM refines the best of them.
    from .thresholds import literals
    Zr = Z[own]
    n_obs = len(names)
    readable = [j for j in range(Zr.shape[1]) if not (n_obs <= j < n_obs + 2)]
    seeds = []
    for j in readable:
        seeds += [cl for cl, _ in literals(Zr[:, j], j, n_thr=5, lo=0.03, hi=0.97)]
    new, log = grow(env, bank, names, zn, pol_fn, obs[own], Zr,
                    max_arms=max_arms, pool=pool, max_arity=max_arity,
                    seed_clauses=seeds, law_sample=law_sample,
                    n_perturb=n_perturb,
                    min_n=max(20, len(own) // 50), min_gain=min_gain,
                    screen_ep=screen_ep, confirm_ep=confirm_ep, n_confirm=6,
                    T=T, seed=seed, z=z, rng=rng, verbose=False,
                    cem_top=cem_top, cem_iter=cem_iter, cem_K=cem_K,
                    weights=weights, prefix=parent_clause,
                    parent_law=parent_law,
                    where=_locator(parent_clause, parent_law))
    for e in log:
        e["kind"] = "subtree"
        e["parent"] = int(arm)
    return new, log


def search_subtrees(env, bank, names, zn, pol_fn, obs, Z, max_depth=3,
                    verbose=True, **kw):
    """Offer each child, most-owned first, as the parent of a new subtree.

    One subtree per call, as for steps and terminations: a subtree changes
    which rows every child below it owns, so two at once would be priced
    against each other's stale distribution.
    """
    C = len(bank["clauses"])
    if not C or depth(factor(bank["clauses"])) >= max_depth:
        return bank, []
    own = owners(bank, Z)
    order = sorted(range(C), key=lambda c: -(own == c).sum())
    log = []
    for c in order:
        share = float((own == c).mean())
        if share < 0.05:
            continue
        before = len(bank["clauses"])
        new, lg = grow_subtree(env, bank, c, names, zn, pol_fn, obs, Z,
                               verbose=verbose, **kw)
        log += lg
        added = len(new["clauses"]) - before
        if verbose:
            best = max((e["delta"] for e in lg if "delta" in e), default=None)
            print("    subtree under child %d (owns %.0f%%): %s"
                  % (c, 100 * share, ("%d child%s grown" % (added, "ren" if added > 1 else ""))
                     if added else ("nothing cleared the test (best %+.2f)" % best
                                    if best is not None else "no region to grow in")),
                  flush=True)
        if added:
            return new, log
    return bank, log
