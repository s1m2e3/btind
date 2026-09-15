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

STEPS AND STATUS, added on top of that rule and documented in `tick.py`, which
is the single compiled copy both kernels run. An arm may carry `steps` -- further
(advance_clause, law) pairs it moves through while latched -- and a `fails`
clause that releases it AND excludes it from this tick's Fallback, so the arms
below it get the state: a child returning FAILURE. `betas` remains the
termination of the arm, now checked at every step. The implementation here is
the vectorised numpy REFERENCE; the kernels are the other implementation, and
the tests compare them.

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
        # `mem_age` is HOW LONG AGO the slot was written, in ticks. A guard on
        # it is a remembered clock: "told green 12 ticks ago" is a condition a
        # tree can act on after the message itself is gone. Zero until the
        # first write, like `have_mem`.
        out += (["mem_%s" % names[j] for j in mem["cols"]]
                + ["have_mem", "mem_age"])
        if mem.get("countdown"):
            # `left_<c>` = mem_<c> - mem_age: what is left of a remembered
            # DURATION, in ticks. A message "green for 24 ticks" becomes a
            # column that reaches zero when the light changes, so a guard can
            # read the remembered clock as one number. Whether a stored column
            # is a duration is not known here; the flag is a candidate the
            # memory search prices like any other, and on a stored bearing the
            # column is harmless junk the rollout ignores.
            out += ["left_%s" % names[j] for j in mem["cols"]]
    return out


def all_clauses(bank):
    """Every clause a bank evaluates: guards, terminations, fail and advance
    conditions, and the blackboard's write and clear rules. None entries skipped."""
    out = list(bank.get("clauses") or [])
    out += [c for c in (bank.get("betas") or []) if c]
    out += [c for c in (bank.get("fails") or []) if c]
    for st in (bank.get("steps") or []):
        out += [adv for adv, _ in (st or [])]
    m = bank.get("mem")
    if m:
        out.append(m["write"])
        if m.get("clear"):
            out.append(m["clear"])
    return out


def all_laws(bank):
    """Every law a bank can act with: each arm's steps, then the default."""
    out = []
    for c, th in enumerate(bank.get("laws") or []):
        out.append(th)
        st = bank.get("steps")
        if st and st[c]:
            out += [t for _, t in st[c]]
    if bank.get("default") is not None:
        out.append(bank["default"])
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
    return any(l[0] in cols for c in all_clauses(bank) for l in c)


def _reads_vq(bank, n_obs):
    """Does any clause, termination, write rule or LAW mention V_hat/leverage?"""
    if _guards_read_vq(bank, n_obs):
        return True
    cols = [n_obs, n_obs + 1]
    if bank.get("laws_on_z"):
        for th in all_laws(bank):
            if np.abs(np.asarray(th)[cols]).max() > 0:
                return True
    return False


class MemBank:
    """Executable bank with an optional latch and an optional blackboard.

    bank keys beyond the memoryless ones:
        betas    list, one per clause; None means "no termination" (the arm runs
                 until preempted or until its own guard is re-evaluated). On a
                 multi-step arm it terminates the whole arm at any step.
        sticky   list of bool, one per clause; all False = memoryless
        steps    list, one per clause; None or [(advance_clause, law), ...] for
                 steps 1..K-1 -- a Sequence of actions inside the arm
        fails    list, one per clause; None or a clause that releases the arm
                 and hands this tick to the arms BELOW it (FAILURE status)
        kerns    list, one per clause; None or a list of per-step kernels
                 (`kernlaw.py`), each None or the anchors that refine that
                 step's affine law; `kern_default` is the default's
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
        C = len(bank["clauses"])
        self.betas = bank.get("betas") or [None] * C
        self.sticky = bank.get("sticky") or [False] * C
        self.steps = bank.get("steps") or [None] * C
        self.fails = bank.get("fails") or [None] * C
        self.n_steps = [1 + len(s or []) for s in self.steps]
        self.latch = None
        self.step = None
        self.slots = None
        self._k = None            # the step each row acted with on the last tick
        self._ainv = {}           # K^-1 per (arm, step), computed once per bank

    # -- state ---------------------------------------------------------------
    def reset(self, n):
        self.latch = np.full(n, -1, dtype=int)
        self.step = np.zeros(n, dtype=int)
        k = len(self.mem["cols"]) if self.mem else 0
        self.slots = np.zeros((n, k))
        self.have = np.zeros(n, bool)
        self.age = np.zeros(n, int)

    def set_state(self, latch, slots, have, step=None, age=None):
        """Restore a recorded controller state, for probing and for replay."""
        self.latch = np.asarray(latch, int).copy()
        self.slots = np.asarray(slots, float).copy()
        self.have = np.asarray(have, bool).copy()
        self.step = (np.zeros(len(self.latch), int) if step is None
                     else np.asarray(step, int).copy())
        self.age = (np.zeros(len(self.latch), int) if age is None
                    else np.asarray(age, int).copy())

    def state(self):
        return (self.latch.copy(), self.slots.copy(), self.have.copy(),
                self.step.copy(), self.age.copy())

    def law_of(self, c, k=0):
        """The law arm `c` acts with at step `k`; `c < 0` is the default."""
        if c < 0:
            return self.b["default"]
        return self.b["laws"][c] if k == 0 else self.steps[c][k - 1][1]

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
            # the age counts ticks SINCE the write: 0 on the write tick, and
            # it only runs once something has been written
            self.age = np.where(self.have, self.age + 1, 0)
            if fire.any():
                self.slots[fire] = obs[np.ix_(fire, self.mem["cols"])]
                self.have[fire] = True
                self.age[fire] = 0
            clr = self.mem.get("clear")
            if clr:
                c = _match_cols(clr, zb)
                self.have[c] = False
                self.slots[c] = 0.0
                self.age[c] = 0
        cols = [zb, self.slots, self.have[:, None].astype(float),
                self.age[:, None].astype(float)]
        if self.mem.get("countdown"):
            cols.append(self.slots - self.age[:, None])
        return np.hstack(cols)

    # -- arbitration ---------------------------------------------------------
    def arbitrate(self, Z):
        """Returns the active arm per row (-1 = default) and updates latch and step.

        The tick, in the order `tick.py` fixes: fail, match (skipping the arm
        that failed), preempt, advance, success, keep. Vectorised over rows;
        the kernels run the same rule per row, compiled, and the tests hold the
        two to bit-exact agreement.
        """
        self._ensure(len(Z))
        n, C = len(Z), len(self.b["clauses"])
        lat, stp = self.latch.copy(), self.step.copy()
        live = lat >= 0

        # 1  FAIL: the latched arm's fail clause releases it and excludes it.
        failed = np.zeros(n, bool)
        for c in range(C):
            if self.fails[c] is None:
                continue
            rows = live & (lat == c)
            if rows.any():
                failed[rows] = _match_cols(self.fails[c], Z)[rows]
        excl = np.where(failed, lat, -1)
        lat[failed], stp[failed] = -1, 0
        live = lat >= 0

        # 2  MATCH, first clause wins, the failed arm skipped on its own row.
        M = (np.stack([_match_cols(c, Z) for c in self.b["clauses"]], 1)
             if C else np.zeros((n, 0), bool))
        if failed.any():
            M[np.flatnonzero(failed), excl[failed]] = False
        f = np.full(n, C, dtype=int)
        for c in range(C - 1, -1, -1):
            f[M[:, c]] = c

        # 3  PREEMPT: rows where f < lat simply take f. The rest of the latched
        #    rows are the "zone" where the arm may advance, finish or keep.
        zone = live & (f >= lat)
        adv = np.zeros(n, bool)
        succ = np.zeros(n, bool)
        for c in range(C):
            rows = zone & (lat == c)
            if not rows.any():
                continue
            # 4  SUCCESS: the arm's beta releases it, at whatever step it is on.
            if self.betas[c] is not None:
                succ[rows] = _match_cols(self.betas[c], Z)[rows]
            # 5  ADVANCE: step k -> k+1 on the tick step k+1's clause fires.
            for k in range(self.n_steps[c] - 1):
                r = rows & ~succ & (stp == k)
                if r.any():
                    adv[r] = _match_cols(self.steps[c][k][0], Z)[r]

        # 6  KEEP everything in the zone that did not finish.
        keep = zone & ~succ
        a = f.copy()
        a[keep] = lat[keep]
        k_new = np.zeros(n, dtype=int)
        k_new[keep] = stp[keep] + adv[keep]

        st = np.zeros(C + 1, bool)
        st[:C] = self.sticky
        latched = st[a]                           # st[C] is the default slot
        self.latch = np.where(latched, a, -1)
        self.step = np.where(latched, k_new, 0)
        self._k = k_new
        return np.where(a == C, -1, a)

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
        out = self._law_out(-1, 0, Xd)
        for c in np.unique(a[a >= 0]):
            for k in np.unique(self._k[a == c]):
                m = (a == c) & (self._k == k)
                out[m] = self._law_out(int(c), int(k), Xd[m])
        return out

    def _law_out(self, c, k, Xd):
        """The law of arm `c` step `k` on design rows: affine, plus its kernel."""
        from . import kernlaw as KL
        kern = KL.kern_of(self.b, c, k)
        if not KL.n_points(kern):
            return Xd @ self.law_of(c, k)
        if (c, k) not in self._ainv:
            self._ainv[(c, k)] = KL.ainv(kern)
        return KL.evaluate(self.law_of(c, k), kern, Xd, self._ainv[(c, k)])

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
        if self.b.get("head") in ("scalar", "duration"):
            # A CONTINUOUS LEAF: the affine score IS the command -- an
            # acceleration, a green time -- and the WORLD clips it to its
            # physical range, because the range is the world's to know. The
            # arbitration above the leaf is as discrete as ever.
            return out[:, 0]
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
PER_ARM = ("betas", "sticky", "steps", "fails", "kerns")
_PER_ARM_FILL = {"betas": None, "sticky": False, "steps": None, "fails": None,
                 "kerns": None}


def reindex(bank, order):
    """The same bank with its arms permuted or filtered by `order`."""
    out = dict(bank, clauses=[bank["clauses"][i] for i in order],
               laws=[bank["laws"][i] for i in order])
    for k in PER_ARM:
        if bank.get(k) is not None:
            v = list(bank[k])
            out[k] = [v[i] if i < len(v) else _PER_ARM_FILL[k] for i in order]
    return out


def insert_arm(bank, clause, law, pos, beta=None, sticky=False, steps=None,
               fails=None, kerns=None):
    """Add an arm at `pos`, extending every per-arm list in step.

    An arm with `steps` is made sticky whether or not the caller said so: a
    step index lives inside the latch and is meaningless without one.
    """
    cl = [[l[:] for l in c] for c in bank["clauses"]]
    laws = list(bank["laws"])
    pos = max(0, min(pos, len(cl)))
    cl.insert(pos, [l[:] for l in clause])
    laws.insert(pos, law)
    out = dict(bank, clauses=cl, laws=laws)
    sticky = bool(sticky or steps)
    given = dict(betas=beta, sticky=sticky, steps=steps, fails=fails, kerns=kerns)
    for k in PER_ARM:
        if bank.get(k) is not None or given[k] not in (None, False):
            v = list(bank.get(k) or [])
            v += [_PER_ARM_FILL[k]] * (len(cl) - 1 - len(v))
            v.insert(pos, given[k])
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
    st = bank.get("steps")
    if st:
        sticky = bank.get("sticky") or [False] * n
        for c in range(n):
            if st[c] and not sticky[c]:
                raise ValueError("%s: arm %d has %d steps but is not sticky"
                                 % (where, c, 1 + len(st[c])))
    return bank


def _relayout_steps(steps, old_zn, new_zn):
    if not steps:
        return steps
    return [([(adv, relayout(th, old_zn, new_zn)) for adv, th in s]
             if s else None) for s in steps]


def widen(bank, old_zn, new_zn):
    """Relayout every law in a bank. Guard literals are unaffected: memory
    columns are APPENDED, so existing column indices keep their meaning."""
    return dict(bank,
                laws=[relayout(t, old_zn, new_zn) for t in bank["laws"]],
                default=relayout(bank["default"], old_zn, new_zn),
                steps=_relayout_steps(bank.get("steps"), old_zn, new_zn),
                laws_on_z=True)


def upgrade_layout(bank, names):
    """Bring a stored bank onto the current column layout, by name.

    A bank saved before `mem_age` existed has laws one column narrower than
    `mem_names` now says; multiplying them by today's design matrix would
    either raise or bind every coefficient past the slots to the wrong column.
    The old layout is known -- names, V_hat, leverage, the slots, have_mem --
    so the laws are relaid out by name and the new column is zero-filled.
    """
    if not bank.get("mem") or not bank.get("laws_on_z"):
        return bank
    want = len(mem_names(names, bank["mem"])) + 1
    have = np.asarray(bank["default"]).shape[0]
    if have == want:
        return bank
    old_zn = (base_names(names)
              + ["mem_%s" % names[j] for j in bank["mem"]["cols"]] + ["have_mem"])
    if len(old_zn) + 1 != have:
        raise ValueError("stored bank has %d law rows; neither the current "
                         "layout (%d) nor the pre-age one (%d)"
                         % (have, want, len(old_zn) + 1))
    return widen(bank, old_zn, mem_names(names, bank["mem"]))


def without_memory(bank, names):
    """The same tree with the blackboard removed -- the no-memory control.

    Dropping `mem` also drops columns, so every law has to be relaid out or it
    is multiplied by a design matrix three columns narrower than it expects.
    Arms whose guards read a memory column go too: they can no longer be
    evaluated, and leaving them in would silently change which arm claims what.
    A termination, fail or advance clause that read a memory column is dropped
    from the kept arm for the same reason.
    """
    zn_old = mem_names(names, bank.get("mem"))
    zn_new = mem_names(names, None)
    w = len(zn_new)
    fits = lambda cl: cl is not None and not any(l[0] >= w for l in cl)
    keep = [i for i, c in enumerate(bank["clauses"]) if fits(c)]
    out = reindex(bank, keep)
    out = widen(out, zn_old, zn_new)
    out["mem"] = None
    out["clauses"] = [[l[:] for l in c] for c in out["clauses"]]
    if out.get("betas"):
        out["betas"] = [b if fits(b) else None for b in out["betas"]]
    if out.get("fails"):
        out["fails"] = [b if fits(b) else None for b in out["fails"]]
    if out.get("steps"):
        out["steps"] = [([(adv, th) for adv, th in s if fits(adv)] or None)
                        if s else None for s in out["steps"]]
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


def law_label(theta, bank):
    """A readable name for a law where one exists, else the generic form.

    On an argmax head a law that is a bias on one action IS that action, and
    saying so is the difference between `Action(u = K x + b)` and `LANE_LEFT`.
    """
    th = np.asarray(theta)
    if bank.get("head") in ("scalar", "duration"):
        body, bias = th[:-1, 0], th[-1, 0]
        nz = np.flatnonzero(np.abs(body) > 1e-9)
        if not len(nz):
            return "%s(%.3g)" % ("Accel" if bank.get("head") == "scalar"
                                 else "Green", bias)
        return "%s(theta z)" % ("Accel" if bank.get("head") == "scalar" else "Green")
    if bank.get("head") == "argmax":
        body, bias = th[:-1], th[-1]
        if not np.any(body) and np.count_nonzero(bias) == 1:
            k = int(np.argmax(bias))
            acts = bank.get("actions")
            return acts[k] if acts and k < len(acts) else "always[%d]" % k
        return "Prefer(argmax theta z)"
    return "Action(u = K x + b)"


def emit(bank, names):
    """The tree as text, including the temporal nodes when a bank has them."""
    C = len(bank["clauses"])
    zn = mem_names(names, bank.get("mem"))
    betas = bank.get("betas") or [None] * C
    sticky = bank.get("sticky") or [False] * C
    steps = bank.get("steps") or [None] * C
    fails = bank.get("fails") or [None] * C
    lit = lambda l: "%s%s%.3f" % (zn[l[0]], "<=" if l[2] else ">", l[1])
    conj = lambda cl: " AND ".join(lit(l) for l in cl)
    out = []
    if bank.get("mem"):
        m = bank["mem"]
        out.append("Blackboard: mem <- [%s]  when  %s%s"
                   % (", ".join(names[j] for j in m["cols"]), conj(m["write"]),
                      ("   cleared when " + conj(m["clear"]))
                      if m.get("clear") else ""))
    from . import kernlaw as KL
    klab = lambda c, k: KL.label(KL.kern_of(bank, c, k), zn, bank.get("head"),
                                 bank.get("actions"), bank.get("u_range"))
    def with_k(text, c, k):
        lab = klab(c, k)
        return text + (" + " + lab if lab else "")
    def arm_text(c, guard):
        acts = [with_k(law_label(bank["laws"][c], bank), c, 0)]
        for i, (adv, th) in enumerate(steps[c] or []):
            acts[-1] += " until " + conj(adv)
            acts.append(with_k(law_label(th, bank), c, i + 1))
        # an arm whose guard is entirely its subtree's is that subtree's default
        body = ("Sequence[ %s , %s ]" % (conj(guard), " , ".join(acts)) if guard
                else " , ".join(acts) + "          # subtree default")
        if sticky[c]:
            b = conj(betas[c]) if betas[c] else "never"
            body = "KeepRunningUntilFailure( %s )   beta: %s" % (body, b)
        if fails[c]:
            body += "   fail: " + conj(fails[c])
        return body
    default_text = with_k(law_label(bank["default"], bank)
                          .replace("Action(u = K x + b)", "Action(default)"), -1, 0)
    # NESTED WHEN THE BANK HAS SUBTREES: contiguous children sharing literals
    # are printed as a Sequence over a Fallback of their own (`subtree.py`).
    # The flat bank is what runs; this is an exact rewrite of it.
    from .subtree import emit_tree
    out += emit_tree(bank, names, arm_text, conj, default_text)
    return "\n".join(out)
