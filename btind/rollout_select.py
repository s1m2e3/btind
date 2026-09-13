"""Arms selected by measured return. No proxy anywhere in the loop.

FOUR PROXIES HAVE NOW FAILED ON THIS TASK, all for one reason.

    sup-LM instability      e06/e07: the most detectable split (`energy`) was a
                            planner artifact; banning it IMPROVED return by 1.4
    M-weighted value loss   e14: rgfi beat the bank by 35 points out of sample
                            and lost by 8 return units, eating 1.95 meals
                            instead of 3.20
    gap-weighted loss       e14: swapping the weighting recovered 0.4 meals and
                            left the gap intact
    one-step TD advantage   the validation above: ranks policies BACKWARDS
                            (random -0.87 > global -1.05 > bank -1.25 > the good
                            bank -1.58, whose return is +10.25)

Each is a LOCAL quantity -- curvature in the action, leverage at a state, a
regression residual, a one-step residual -- and this task's return is made of
accumulation over ~35 steps. Every one of them prices not-making-a-mistake-now
and none of them prices reward acquired later, which is why the method that
optimises them hardest becomes the most cautious and the least hungry.

Return itself is not expensive here. 200 episodes of 100 steps is ~0.05s, so a
candidate arm can simply be TRIED: build the bank it would produce, run it, keep
it if the measured return goes up.

PROPOSE CHEAPLY, SELECT EXPENSIVELY. Rollouts give no per-clause credit -- a
trajectory scores the whole controller -- so there is nothing to hill-climb on
per literal, and a full GA with rollout fitness would cost pop x generations
rollouts per arm. Instead the loss-based machinery is demoted to a PROPOSAL
distribution, where being wrong is harmless, and every accepted decision is made
by measured return. A bad proposal simply loses its rollout.

TWO THINGS THAT MAKE A NOISY OBJECTIVE USABLE.

  COMMON RANDOM NUMBERS. Every candidate within a fit is evaluated on the same
  episode seed, so candidates are compared on identical starting states and the
  environment's own variance cancels out of the comparison instead of being
  added to it.

  THE SELECTION SEED IS NOT THE EVALUATION SEED. Selection runs on `sel_seed`;
  the number that gets reported comes from a different seed and more episodes.
  Otherwise this would do exactly what every overfitting story in this project
  has done, and pick the arms that happen to suit the test episodes.

AND A PAIRED TEST, BECAUSE THE FIRST VERSION OVERFITTED IMMEDIATELY. Accepting
the argmax of ~80 noisy rollout estimates is a winner's curse: the first run
selected `pos_y>0.600` -- a declared distractor -- on a measured +0.43, and it
scored -1.46 held out against the loss-based bank's -0.48. Because every
candidate is rolled out from identical starting states, the per-episode
differences are PAIRED, and the variance of the difference is far smaller than
the variance of either arm. An arm is now accepted only when that paired
difference clears `z` standard errors, so noise cannot buy a literal.
"""
import numpy as np

from .collect import design_matrix
from .evotm import Alphabet, _match, _rand_literal, dedupe_literals
from .policies import evaluate
from .valuesplit import fit_value_law


class _Bank:
    """A partial bank, executable while it is still being built."""

    def __init__(self, clauses, laws, default, Xd_fn):
        self.clauses, self.laws, self.default = clauses, laws, default
        self.Xd_fn = Xd_fn

    def act(self, obs):
        Xd = self.Xd_fn(obs)
        out = Xd @ self.default
        done = np.zeros(len(obs), bool)
        for cl, th in zip(self.clauses, self.laws):
            m = _match(cl, obs) & ~done
            if m.any():
                out[m] = Xd[m] @ th
                done |= m
        n = np.linalg.norm(out, axis=1, keepdims=True)
        # A DIRECTION, NOT A VECTOR. Least squares shrinks the magnitude of
        # its prediction toward the mean, and a shrunk action is a slower
        # agent: measured on NestWorld, clipping to the disk instead of
        # normalising cost 5.0 return units on an identical bank.
        return out / np.maximum(n, 1e-9)


def rsfi(env, obs, U, M, split_vars, names, n_arms=6, pool=60, refine=20,
         max_arity=3, min_n=400, n_ep=300, T=200, sel_seed=777, seed=0,
         seed_clauses=None, sig_thr_frac=0.06, z=2.0, verbose=True):
    """Greedily add the arm that most improves MEASURED return; stop when none do.

    `seed_clauses` are free proposals from a loss-fitted bank. They are a
    proposal distribution only: each still has to win its rollout.
    """
    rng = np.random.default_rng(seed)
    alpha = Alphabet().fit(obs, split_vars)
    sig = {j: sig_thr_frac * (alpha.hi[j] - alpha.lo[j]) for j in split_vars}
    Xd = design_matrix(obs)
    default = fit_value_law(Xd, U, M)

    def score(clauses, laws):
        """Per-episode returns, not the mean -- the paired test needs the vector."""
        pol = _Bank(clauses, laws, default, design_matrix)
        return evaluate(env, pol, n_ep=n_ep, T=T, seed=sel_seed)["G"]

    def rand_clause(hot=None):
        return dedupe_literals([_rand_literal(rng, alpha, obs, hot)
                                for _ in range(1 + int(rng.integers(max_arity)))])

    def drift(cl):
        cl = [l[:] for l in cl]
        i = int(rng.integers(len(cl)))
        j = cl[i][0]
        cl[i][1] = float(np.clip(cl[i][1] + rng.normal(0, sig[j]),
                                 alpha.lo[j], alpha.hi[j]))
        if rng.random() < 0.15:
            cl[i][2] = not cl[i][2]
        if rng.random() < 0.20 and len(cl) < max_arity:
            cl.append(_rand_literal(rng, alpha, obs, None))
        elif rng.random() < 0.15 and len(cl) > 1:
            del cl[int(rng.integers(len(cl)))]
        return dedupe_literals(cl)

    clauses, laws = [], []
    claimed = np.zeros(len(obs), bool)
    cur = score(clauses, laws)
    best = float(cur.mean())
    trace = [best]
    if verbose:
        print("  [arm 0] default law alone: G %.2f" % best)

    for k in range(n_arms):
        cands = [rand_clause() for _ in range(pool)]
        if seed_clauses:
            cands += [c for c in seed_clauses]
        if clauses:
            cands += [drift(clauses[-1]) for _ in range(6)]

        def try_all(cs, ref):
            """Best candidate whose PAIRED gain over `ref` clears z std errors."""
            gain, win = 0.0, None
            for cl in cs:
                m = _match(cl, obs) & ~claimed
                if int(m.sum()) < min_n:
                    continue
                law = fit_value_law(Xd[m], U[m], M[m])
                arr = score(clauses + [cl], laws + [law])
                d = arr - ref
                se = d.std() / np.sqrt(len(d))
                if d.mean() > gain and d.mean() > z * max(se, 1e-12):
                    gain, win = float(d.mean()), (cl, m, law, arr)
            return gain, win

        gain, win = try_all(cands, cur)
        if win is None:
            if verbose:
                print("  [arm %d] nothing beat the noise floor -- stopping"
                      % (k + 1))
            break
        # local refinement: drift the winner, keep it only if it wins again
        g2, w2 = try_all([drift(win[0]) for _ in range(refine)], win[3])
        if w2 is not None:
            gain, win = gain + g2, w2

        cl, win_m, win_law, cur = win
        clauses.append(cl)
        laws.append(win_law)
        claimed |= win_m
        best = float(cur.mean())
        trace.append(best)
        win = (cl, win_m)
        if verbose:
            print("  [arm %d] G %.2f  (+%.2f)  claims %d rows  %s"
                  % (k + 1, best, best - trace[-2], int(win_m.sum()),
                     " AND ".join("%s%s%.3f" % (names[j], "<=" if n else ">", t)
                                  for j, t, n in cl)))

    return dict(clauses=clauses, laws=laws, default=default, names=names,
                G_select=best, trace=np.array(trace),
                n_literals=sum(len(c) for c in clauses))
