"""A controller with state: a latched arm, and a blackboard.

Everything before this file is a function of the current observation. This one
carries two things between ticks, which is what a behaviour tree's temporal
primitives actually are:

    l_t   LATCH        which arm is running. Decorator / memory-Sequence.
    b_t   BLACKBOARD   values latched off the observation on an event, held
                       until overwritten or cleared.

WHAT EVERY NODE SEES. The blackboard is GLOBAL, as it is in any behaviour tree:
guards, terminations and laws all read the same augmented vector

    z_t = [ obs_t , V_hat , leverage , b_t , have_t ]

so a guard can test `have_target > 0.5` and partition on whether there IS a
memory, a law can steer by `b - pos` because both are columns and the output is
normalised, and beta can mention either. Restricting any of them to `obs` would
be a handicap with no argument behind it.

THE ARBITRATION, and the one comparison that is the whole of it:

    f_t = first clause that matches z_t          (-1 if none: the default arm)
    a_t = f_t      if f_t >= 0 and f_t < l_t     PREEMPT: something higher fires
        = l_t      elif l_t >= 0 and not beta_{l_t}(z_t)   still running
        = f_t      otherwise                     finished, or nothing latched

`f_t < l_t` is what keeps the Fallback REACTIVE. Without it a latched arm could
not be interrupted, and an agent committed to "return to nest" would walk into a
bear rather than reconsider.

NESTING IS THE SAFETY PROPERTY. With `sticky` all false and no write rule, the
arbitration collapses to `a_t = f_t` and the blackboard columns are constant --
exactly the memoryless controller, bit for bit. So the memoryless policy class
is contained in this one, and an honest acceptance test can only fail to find an
improvement; it cannot lose ground. `test_equivalence` in this module checks
that rather than asserting it.

WHAT IS COMMITTED AND WHAT IS DISCOVERED. The FORM of the write rule is a
commitment: a slot holds values latched from named columns when an event fires.
WHICH columns and WHICH event are discovered from the same literal vocabulary as
every guard, with the planted distractors left in the pool. The fully general
alternative, b_{t+1} = f(b_t, x_t) with f free, is a recurrent cell -- more
discoverable and no longer a tree anyone can read.
"""
import numpy as np

from .collect import design_matrix
from .landscape import _match_cols, leverage


def base_names(names):
    """Columns every controller has, before any blackboard slot is added."""
    return list(names) + ["V_hat", "leverage"]


def mem_names(names, mem):
    """Full column layout for a bank. Memory columns are APPENDED, never
    inserted, so adding a slot cannot invalidate a clause that already exists."""
    out = base_names(names)
    if mem:
        out += ["mem_%s" % names[j] for j in mem["cols"]] + ["have_mem"]
    return out


def _guards_read_vq(bank, n_obs):
    """Does any GUARD, termination or write rule COMPARE against V_hat/leverage?

    Separate from whether a law has coefficients there: a comparison needs the
    real value, while a coefficient only needs the column, which is zero
    whenever no critic is attached. Conflating the two made a randomly
    initialised law -- nonzero on every column, including these two -- look like
    it required a critic, which silently pushed every rollout of a policy search
    off the fused kernel and onto the Python path, 63x slower.
    """
    cols = {n_obs, n_obs + 1}
    for c in list(bank.get("clauses") or []) + list(
            filter(None, bank.get("betas") or [])):
        if any(l[0] in cols for l in c):
            return True
    m = bank.get("mem")
    if m and any(l[0] in cols for l in m["write"] + (m.get("clear") or [])):
        return True
    return False


def _reads_vq(bank, n_obs):
    """Does any clause, termination, write rule or LAW mention V_hat/leverage?"""
    cols = {n_obs, n_obs + 1}
    for c in list(bank.get("clauses") or []) + list(
            filter(None, bank.get("betas") or [])):
        if any(l[0] in cols for l in c):
            return True
    m = bank.get("mem")
    if m and any(l[0] in cols for l in m["write"] + (m.get("clear") or [])):
        return True
    if bank.get("laws_on_z"):
        for th in list(bank.get("laws") or []) + [bank.get("default")]:
            if th is not None and np.abs(np.asarray(th)[list(cols)]).max() > 0:
                return True
    return False


class MemBank:
    """Executable bank with an optional latch and an optional blackboard.

    bank keys beyond the memoryless ones:
        betas    list, one per clause; None means "no termination" (the arm runs
                 until preempted or until its own guard is re-evaluated)
        sticky   list of bool, one per clause; all False = memoryless
        mem      None, or dict(cols=[j...], write=clause, clear=clause|None)
    """

    def __init__(self, bank, n_obs, vhat=None, qhat=None):
        self.b, self.n_obs, self.v, self.q = bank, n_obs, vhat, qhat
        # PAY FOR V_hat AND leverage ONLY WHEN SOMETHING READS THEM. They cost
        # an xgboost predict plus a critic solve and an eigendecomposition PER
        # TICK, and a rollout is 400 ticks: measured, computing them
        # unconditionally took a 400-episode evaluation from under a second to
        # 7.8s, which is the difference between a search that finishes and one
        # that times out. Almost no emitted bank mentions them.
        self.need_vq = _reads_vq(bank, n_obs)
        self.mem = bank.get("mem")
        self.betas = bank.get("betas") or [None] * len(bank["clauses"])
        self.sticky = bank.get("sticky") or [False] * len(bank["clauses"])
        self.latch = None
        self.slots = None

    # -- state ---------------------------------------------------------------
    def reset(self, n):
        self.latch = np.full(n, -1, dtype=int)
        k = len(self.mem["cols"]) if self.mem else 0
        self.slots = np.zeros((n, k))
        self.have = np.zeros(n, bool)

    def set_state(self, latch, slots, have):
        """Restore a recorded controller state, for probing and for replay."""
        self.latch = np.asarray(latch, int).copy()
        self.slots = np.asarray(slots, float).copy()
        self.have = np.asarray(have, bool).copy()

    def state(self):
        return (self.latch.copy(), self.slots.copy(), self.have.copy())

    def _ensure(self, n):
        if self.latch is None or len(self.latch) != n:
            self.reset(n)

    # -- features ------------------------------------------------------------
    def _base(self, obs):
        n = len(obs)
        if not self.need_vq:
            return np.hstack([obs, np.zeros((n, 2))])
        v = (self.v.predict(obs) if self.v is not None else np.zeros(n))
        lam = (leverage(self.q, obs) if self.q is not None else np.zeros(n))
        return np.hstack([obs, v[:, None], lam[:, None]])

    def z(self, obs, update=True):
        """The augmented vector, after applying the write rule for this tick.

        The write happens BEFORE the action: seeing the food and remembering it
        are the same tick, so the law can already steer by the fresh value.
        """
        self._ensure(len(obs))
        zb = self._base(obs)
        if not self.mem:
            return zb
        if update:
            fire = _match_cols(self.mem["write"], zb)
            if fire.any():
                self.slots[fire] = obs[np.ix_(fire, self.mem["cols"])]
                self.have[fire] = True
            clr = self.mem.get("clear")
            if clr:
                c = _match_cols(clr, zb)
                self.have[c] = False
                self.slots[c] = 0.0
        return np.hstack([zb, self.slots, self.have[:, None].astype(float)])

    # -- arbitration ---------------------------------------------------------
    def arbitrate(self, Z):
        """Returns the active arm per row and updates the latch."""
        n, C = len(Z), len(self.b["clauses"])
        M = (np.stack([_match_cols(c, Z) for c in self.b["clauses"]], 1)
             if C else np.zeros((n, 0), bool))
        f = np.full(n, -1, dtype=int)
        for c in range(C - 1, -1, -1):
            f[M[:, c]] = c

        lat = self.latch
        live = lat >= 0
        # beta is evaluated only for rows whose latched arm has one
        fired = np.zeros(n, bool)
        for c in range(C):
            if self.betas[c] is None:
                continue
            rows = live & (lat == c)
            if rows.any():
                fired[rows] = _match_cols(self.betas[c], Z)[rows]

        preempt = (f >= 0) & live & (f < lat)
        keep = live & ~preempt & ~fired
        a = np.where(keep, lat, f)

        st = np.zeros(C + 1, bool)
        st[:C] = self.sticky
        self.latch = np.where(st[a], a, -1)      # st[-1] is the default arm slot
        return a

    # -- action --------------------------------------------------------------
    def scores(self, obs):
        """The active arm's law applied to the features -- before any head.

        Every head reads the same affine quantity; they differ only in what
        they do with it. Splitting it out means the arbitration, the memory and
        the laws are shared by both heads rather than duplicated per head.
        """
        Z = self.z(obs)
        a = self.arbitrate(Z)
        Xd = design_matrix(Z) if self.b.get("laws_on_z") else design_matrix(obs)
        out = Xd @ self.b["default"]
        for c in np.unique(a[a >= 0]):
            m = a == c
            out[m] = Xd[m] @ self.b["laws"][c]
        return out

    def preferences(self, obs, temp=1.0):
        """Softmax of the scores: the FUZZY reading of a discrete-head law.

        `argmax` is the defuzzification the environment actually receives; this
        is the membership it is taken from, and it is what a differentiable
        fitter would work on if one is ever wanted. Nothing in the search reads
        it -- CEM scores theta by rollout and never needs a gradient, which is
        why the discrete head costs this project nothing.
        """
        s = self.scores(obs) / max(temp, 1e-9)
        e = np.exp(s - s.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def act(self, obs):
        out = self.scores(obs)
        # TWO HEADS, ONE LAW CLASS. `vector` normalises the score into a unit
        # heading, which is what a holonomic 2-D agent takes. `argmax` reads the
        # same score as a PREFERENCE over a discrete action set and returns the
        # index of the winner -- a linear scoring rule per region, which makes
        # the effective policy piecewise constant on polyhedra nested inside the
        # partition the tree already carves.
        if self.b.get("head") == "argmax":
            return np.argmax(out, axis=1)
        nrm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(nrm, 1e-9)


def relayout(theta, old_zn, new_zn):
    """Move a law onto a wider feature layout without changing what it does.

    A law is indexed by COLUMN POSITION, and `design_matrix` puts the intercept
    last -- so widening the layout shifts the intercept and silently rebinds
    every coefficient. Mapping by NAME and zero-filling the new columns is the
    only rewrite that provably leaves behaviour unchanged, which matters because
    this runs between two measured stages and any drift would be scored as if
    the search had caused it.
    """
    old_i = {n: i for i, n in enumerate(list(old_zn) + ["<intercept>"])}
    out = np.zeros((len(new_zn) + 1, theta.shape[1]))
    for i, n in enumerate(list(new_zn) + ["<intercept>"]):
        if n in old_i:
            out[i] = theta[old_i[n]]
    return out


# --------------------------------------------------- arms and their per-arm state
# `betas` and `sticky` are indexed BY ARM, exactly like `clauses` and `laws`,
# and every operator that adds, removes or reorders an arm has to move all four
# together. Most of them did not: `drop_arm`, `reorder`, both `_insert`s,
# `materialise_default` and the memory search each rebuilt `clauses` and `laws`
# and left the other two behind.
#
# THE FAILURE IS NOT LOCAL, which is why it survived so long. A bank with a
# mismatched `sticky` runs perfectly until `arbitrate` tries to broadcast it
# into an array sized by the clause count, so the traceback points at whatever
# stage happened to roll a rollout next -- observed twice, from `failure_states`
# several rounds after the stage that actually caused it, once after a kick and
# once after a beta was accepted and a later `simplify` dropped an arm. Before
# terminations existed both lists were empty and nothing could go wrong; they
# are the new thing, so every arm operator now goes through here.
PER_ARM = ("betas", "sticky")
_PER_ARM_FILL = {"betas": None, "sticky": False}


def reindex(bank, order):
    """The same bank with its arms permuted or filtered by `order`."""
    out = dict(bank, clauses=[bank["clauses"][i] for i in order],
               laws=[bank["laws"][i] for i in order])
    for k in PER_ARM:
        if bank.get(k) is not None:
            v = list(bank[k])
            out[k] = [v[i] if i < len(v) else _PER_ARM_FILL[k] for i in order]
    return out


def insert_arm(bank, clause, law, pos, beta=None, sticky=False):
    """Add an arm at `pos`, extending every per-arm list in step."""
    cl = [[l[:] for l in c] for c in bank["clauses"]]
    laws = list(bank["laws"])
    pos = max(0, min(pos, len(cl)))
    cl.insert(pos, [l[:] for l in clause])
    laws.insert(pos, law)
    out = dict(bank, clauses=cl, laws=laws)
    for k, x in (("betas", beta), ("sticky", sticky)):
        if bank.get(k) is not None:
            v = list(bank[k])
            v += [_PER_ARM_FILL[k]] * (len(cl) - 1 - len(v))
            v.insert(pos, x)
            out[k] = v
    return out


def check_arms(bank, where=""):
    """Raise where the misalignment HAPPENS, not several rounds downstream."""
    n = len(bank["clauses"])
    if len(bank["laws"]) != n:
        raise ValueError("%s: %d laws for %d arms" % (where, len(bank["laws"]), n))
    for k in PER_ARM:
        if bank.get(k) is not None and len(bank[k]) != n:
            raise ValueError("%s: %d %s for %d arms" % (where, len(bank[k]), k, n))
    return bank


def widen(bank, old_zn, new_zn):
    """Relayout every law in a bank. Guard literals are unaffected: memory
    columns are APPENDED, so existing column indices keep their meaning."""
    return dict(bank,
                laws=[relayout(t, old_zn, new_zn) for t in bank["laws"]],
                default=relayout(bank["default"], old_zn, new_zn),
                laws_on_z=True)


def without_memory(bank, names):
    """The same tree with the blackboard removed -- the no-memory control.

    Dropping `mem` also drops columns, so every law has to be relaid out or it
    is multiplied by a design matrix three columns narrower than it expects.
    Arms whose guards read a memory column go too: they can no longer be
    evaluated, and leaving them in would silently change which arm claims what.
    """
    zn_old = mem_names(names, bank.get("mem"))
    zn_new = mem_names(names, None)
    keep = [i for i, c in enumerate(bank["clauses"])
            if not any(l[0] >= len(zn_new) for l in c)]
    out = dict(bank, mem=None,
               clauses=[[l[:] for l in bank["clauses"][i]] for i in keep],
               laws=[relayout(bank["laws"][i], zn_old, zn_new) for i in keep],
               default=relayout(bank["default"], zn_old, zn_new),
               betas=([bank["betas"][i] for i in keep]
                      if bank.get("betas") else None),
               sticky=([bank["sticky"][i] for i in keep]
                       if bank.get("sticky") else None))
    return out


# ------------------------------------------------------------------ checks
def test_equivalence(env, bank, n_obs, n_ep=300, T=200, seed=11):
    """A MemBank with no stickiness and no blackboard must equal LandscapeBank.

    The nesting claim is the reason the whole extension is safe to try, so it is
    measured rather than asserted: same states, same actions, every tick.
    """
    from .landscape import LandscapeBank
    plain = LandscapeBank(bank, None, None, n_obs)
    mem = MemBank(dict(bank, betas=None, sticky=None, mem=None), n_obs)
    rng = np.random.default_rng(seed)
    s = (env.sample_starts(n_ep, rng) if hasattr(env, "sample_starts")
         else env.sample_states(n_ep, rng))
    mem.reset(n_ep)
    worst = 0.0
    for t in range(T):
        o = env.observe(s)
        worst = max(worst, float(np.abs(plain.act(o) - mem.act(o)).max()))
        s, _, done = env.step(s, plain.act(o))
        if done.all():
            break
    return worst


def emit(bank, names):
    """The tree as text, including the temporal nodes when a bank has them."""
    zn = mem_names(names, bank.get("mem"))
    betas = bank.get("betas") or [None] * len(bank["clauses"])
    sticky = bank.get("sticky") or [False] * len(bank["clauses"])
    lit = lambda l: "%s%s%.3f" % (zn[l[0]], "<=" if l[2] else ">", l[1])
    out = []
    if bank.get("mem"):
        m = bank["mem"]
        out.append("Blackboard: mem <- [%s]  when  %s%s"
                   % (", ".join(names[j] for j in m["cols"]),
                      " AND ".join(lit(l) for l in m["write"]),
                      ("   cleared when " + " AND ".join(lit(l)
                       for l in m["clear"])) if m.get("clear") else ""))
    out.append("Fallback")
    for c, cl in enumerate(bank["clauses"]):
        guard = " AND ".join(lit(l) for l in cl)
        body = "Sequence[ %s , Action(u = K x + b) ]" % guard
        if sticky[c]:
            b = (" AND ".join(lit(l) for l in betas[c]) if betas[c]
                 else "never")
            body = "KeepRunningUntilFailure( %s )   beta: %s" % (body, b)
        out.append("|-- " + body)
    out.append("\\-- Action(default)          # totality guard")
    return "\n".join(out)
