r"""The kernel-interpolation law: a leaf that is a handful of rules, "in state x_i
do y_i", interpolated -- with the affine law it replaces kept as its prior mean.

THE LAW, per leaf, on the augmented vector z (intercept last):

    phi   = z[cols]                                   the leaf's own inputs
    k_i   = exp( - sum_d |phi_d - X_id| / ls_d )      ARD Matern-1/2
    m(z)  = theta^T z                                 the prior mean
    mbar_i = m( z with z[cols] := X_i )               the prior at anchor i
    u(z)  = m(z) + k(phi)^T K^-1 ( y - mbar(z) )      K_ij = k(X_i, X_j) + jitter

which is `sumo_test/utils.py:gp_posterior` with two differences: the anchors X
and targets y are LEARNED (here, by rollout), and the prior mean is the affine
law the leaf already had, evaluated with the query's other coordinates. Then:

    at phi = X_i          u = y_i exactly, whatever the other columns say
    far from every X_i    u -> theta^T z, the law the leaf started from

A discrete head reads u as one preference per action and takes the argmax; a
continuous head reads u[0] and the world clips it to its bounds.

WHY THE AFFINE LAW STAYS AS THE PRIOR. Every bank in the store, every operator
and every kernel already speaks theta, and a law with no anchors is exactly the
affine law bit for bit -- so the richer class contains the incumbent, the same
safety property every other addition here was built on. The search grows
anchors onto a leaf; it never has to translate a tree into a new language. When
a leaf is born from its parent (grow, subtree) theta starts as the parent's law,
so far from its anchors a leaf does what its ancestor did.

WHERE A KERNEL LIVES. `bank["kerns"][c]` is None or a list aligned with the
arm's steps (entry k is step k's kernel, None = plain affine); the default's is
`bank["kern_default"]`. `kerns` is a per-arm list like `betas`, so every arm
operator moves it with its arm.

A CONTINUOUS LEAF IS BOUNDED. For an acceleration in [-B_MAX, A_MAX] or a green
time in [T_MIN, T_MAX] the posterior above is clipped to the range at the
output, and every target is kept inside it -- so every command the leaf can
produce respects the bounds, the leaf is exact at each anchor, and between
anchors the posterior's overshoot (the Matern kernel's K^-1 has negative
weights) is cut at the limit. No action set is defined: how hard to brake or
accelerate is a target the search sets.

WHY NOT A LOGIT SCALE. Interpolating logit((u - lo) / (hi - lo)) and mapping back
through a sigmoid is also bounded, and it was tried first: with the prior at a
limit (full acceleration, logit +6.9) a braking target of -2.7 (logit -1.1) was
swamped, so a car 10 m from its anchor still accelerated at +2.2 m/s^2 and a
red stop on (green, d_stop) gained +0.1 where the same point in command units
gained +12.7. The blend has to happen in the command's own units.

A CONSTANT PRIOR (`bank["prior"] = "const"`) keeps only the law's intercept, so
every state dependence of a leaf is one of its points -- "by default do b; near
x_i do y_i" -- and nothing hides in a dense affine prior. `constrain` enforces
it wherever a law is created or tuned.

NUMERICS. The reference evaluation accumulates in the same order as the compiled
one -- over anchors, then dimensions, one at a time -- so an argmax tie between
two actions whose targets agree is broken identically by both.
"""
import numpy as np

JITTER = 1e-6


def bounds_of(bank):
    """(lo, hi) for a continuous leaf with a declared range, else None."""
    if bank.get("head") in ("scalar", "duration") and bank.get("u_range") is not None:
        lo, hi = bank["u_range"]
        return float(lo), float(hi)
    return None


def constrain(bank, theta):
    """A law as the bank's prior allows: with a constant prior, the intercept only."""
    th = np.array(theta, float)
    if bank.get("prior") == "const":
        th[:-1] = 0.0
    return th


def make(cols, X, Y, ls):
    """A kernel on columns `cols`: anchors X (M, D), targets Y (M, n_out), ls (D)."""
    cols = [int(c) for c in cols]
    X = np.asarray(X, float).reshape(-1, len(cols))
    Y = np.asarray(Y, float)
    Y = Y.reshape(len(X), Y.shape[-1] if Y.size == 0 else -1)
    ls = np.maximum(np.asarray(ls, float).reshape(len(cols)), 1e-6)
    return dict(cols=cols, X=X, Y=Y, ls=ls)


def n_points(kern):
    return 0 if kern is None else int(len(kern["X"]))


def kern_of(bank, c, k=0):
    """The kernel of arm `c` at step `k`; `c < 0` is the default. None = affine."""
    if c < 0:
        return bank.get("kern_default")
    ks = bank.get("kerns")
    if not ks or ks[c] is None or k >= len(ks[c]):
        return None
    return ks[c][k]


def with_kern(bank, c, k, kern):
    """The same bank with arm `c`'s step-`k` kernel replaced (c < 0: default)."""
    if c < 0:
        return dict(bank, kern_default=kern)
    C = len(bank["clauses"])
    ks = list(bank.get("kerns") or [None] * C)
    cur = list(ks[c] or [])
    cur += [None] * (k + 1 - len(cur))
    cur[k] = kern
    ks[c] = cur if any(x is not None for x in cur) else None
    return dict(bank, kerns=ks)


COND_MAX = 1e8        # the diagonal is raised until K is at least this well conditioned


def ainv(kern):
    """K^-1 over the anchors. Jitter on the diagonal, raised tenfold at a time
    while K is ill-conditioned (two anchors nearly on top of each other): the
    law then interpolates them jointly instead of amplifying their difference."""
    X, ls = kern["X"], kern["ls"]
    M = len(X)
    K = np.empty((M, M))
    for i in range(M):
        for j in range(M):
            s = 0.0
            for d in range(X.shape[1]):
                s += abs(X[i, d] - X[j, d]) / ls[d]
            K[i, j] = np.exp(-s)
    jit = JITTER
    while True:
        Kj = K + jit * np.eye(M)
        if np.linalg.cond(Kj) < COND_MAX or jit > 1.0:
            return np.linalg.inv(Kj)
        jit *= 10.0


def evaluate(theta, kern, Xd, A=None, bounds=None):
    """(n, n_out) law outputs on design rows Xd (intercept last). With `bounds`
    (a continuous leaf) a kernel law's output is clipped to [lo, hi]."""
    th = np.asarray(theta, float)
    m = Xd @ th
    if kern is None or not n_points(kern):
        return m
    cols, X, Y, ls = kern["cols"], kern["X"], kern["Y"], kern["ls"]
    A = ainv(kern) if A is None else A
    n, M, D = len(Xd), len(X), len(cols)
    kv = np.empty((n, M))
    for i in range(M):
        s = np.zeros(n)
        for d in range(D):
            s = s + np.abs(Xd[:, cols[d]] - X[i, d]) / ls[d]
        kv[:, i] = np.exp(-s)
    out = m.copy()
    for i in range(M):
        w = np.zeros(n)
        for j in range(M):
            w = w + A[i, j] * kv[:, j]
        delta = np.zeros((n, th.shape[1]))
        for d in range(D):
            delta = delta + th[cols[d]][None, :] * (X[i, d] - Xd[:, cols[d]])[:, None]
        out = out + w[:, None] * (Y[i][None, :] - m - delta)
    if bounds is not None:
        out = np.minimum(np.maximum(out, bounds[0]), bounds[1])
    return out


# ---------------------------------------------------------------- flattening
KERN_KEYS = ("k_start", "k_D", "k_cols", "k_X", "k_Y", "k_ls", "k_A", "k_Astart",
             "k_lo", "k_hi")


def pack(kerns, n_out, bounds=None):
    """Per FLAT law index (the order `tick.flatten` builds), the arrays a kernel
    reads. A law with no kernel has k_start[L+1] == k_start[L]."""
    L = len(kerns)
    Dmax = max([len(k["cols"]) for k in kerns if n_points(k)] + [1])
    k_start = np.zeros(L + 1, np.int64)
    k_D = np.zeros(L, np.int64)
    k_cols = np.zeros((L, Dmax), np.int64)
    k_ls = np.ones((L, Dmax))
    k_Astart = np.zeros(L + 1, np.int64)
    Xs, Ys, As = [], [], []
    for l, k in enumerate(kerns):
        M = n_points(k)
        k_start[l + 1] = k_start[l] + M
        k_Astart[l + 1] = k_Astart[l] + M * M
        if not M:
            continue
        D = len(k["cols"])
        k_D[l] = D
        k_cols[l, :D] = k["cols"]
        k_ls[l, :D] = k["ls"]
        Xp = np.zeros((M, Dmax))
        Xp[:, :D] = k["X"]
        Xs.append(Xp)
        Ys.append(np.asarray(k["Y"], float).reshape(M, n_out))
        As.append(ainv(k).reshape(-1))
    return dict(
        k_start=k_start, k_D=k_D, k_cols=k_cols, k_ls=k_ls, k_Astart=k_Astart,
        k_X=np.ascontiguousarray(np.vstack(Xs) if Xs else np.zeros((1, Dmax))),
        k_Y=np.ascontiguousarray(np.vstack(Ys) if Ys else np.zeros((1, n_out))),
        k_A=np.ascontiguousarray(np.concatenate(As) if As else np.zeros(1)),
        # a continuous leaf's range; hi <= lo means an unbounded head
        k_lo=np.array([bounds[0] if bounds else 0.0]),
        k_hi=np.array([bounds[1] if bounds else 0.0]))


def kern_args(f):
    return tuple(f[k] for k in KERN_KEYS)


try:
    from numba import njit

    @njit(cache=True, inline="always")
    def law_out(z, law, laws, k_start, k_D, k_cols, k_X, k_Y, k_ls, k_A,
                k_Astart, k_lo, k_hi, out, kv):
        """Write flat law `law`'s outputs on z into `out` (n_out,). `kv` is a
        scratch buffer at least as long as the largest anchor count."""
        d = laws.shape[1]
        nA = laws.shape[2]
        for a in range(nA):
            s = 0.0
            for q in range(d):
                s += z[q] * laws[law, q, a]
            out[a] = s
        M = k_start[law + 1] - k_start[law]
        if M == 0:
            return
        D = k_D[law]
        b0 = k_start[law]
        for i in range(M):
            s = 0.0
            for dd in range(D):
                s += abs(z[k_cols[law, dd]] - k_X[b0 + i, dd]) / k_ls[law, dd]
            kv[i] = np.exp(-s)
        a0 = k_Astart[law]
        for a in range(nA):
            m = out[a]
            acc = m
            for i in range(M):
                w = 0.0
                for j in range(M):
                    w = w + k_A[a0 + i * M + j] * kv[j]
                delta = 0.0
                for dd in range(D):
                    c = k_cols[law, dd]
                    delta = delta + laws[law, c, a] * (k_X[b0 + i, dd] - z[c])
                acc = acc + w * (k_Y[b0 + i, a] - m - delta)
            out[a] = acc
        if k_hi[0] > k_lo[0]:
            out[0] = min(max(out[0], k_lo[0]), k_hi[0])
except ImportError:            # the numpy reference needs no numba
    law_out = None


# ------------------------------------------------------------------ reading
def to_json(kern):
    if kern is None:
        return None
    return dict(cols=list(kern["cols"]), X=np.asarray(kern["X"]).tolist(),
                Y=np.asarray(kern["Y"]).tolist(), ls=np.asarray(kern["ls"]).tolist())


def from_json(d):
    return None if d is None else make(d["cols"], d["X"], d["Y"], d["ls"])


def label(kern, zn, head=None, actions=None, u_range=None):
    """One rule per anchor: `(v=0.0, d_stop=5.1) -> BRAKE_HARD`."""
    if not n_points(kern):
        return ""
    rules = []
    for i in range(n_points(kern)):
        where = ", ".join("%s=%.3g" % (zn[c] if c < len(zn) else "z%d" % c, x)
                          for c, x in zip(kern["cols"], kern["X"][i]))
        y = np.asarray(kern["Y"][i])
        if head == "argmax":
            k = int(np.argmax(y))
            what = actions[k] if actions and k < len(actions) else "a%d" % k
        else:
            v = float(y[0])
            if u_range is not None:
                v = float(np.clip(v, *u_range))
            what = "%.3g" % v
        rules.append("(%s) -> %s" % (where, what))
    return "K[%s]" % "; ".join(rules)
