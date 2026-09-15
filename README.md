# btind — inducing behaviour trees by measured return

A behaviour tree is a readable controller: a `Fallback` of guarded `Sequence`s,
each guard a condition on the observation and each action a control law -- an
affine law refined by a few learned inducing points ("in state x do y").
This repository asks whether such a tree can be **discovered** rather than
written — the partition, the laws, and (latterly) the memory.

The bank is not translated into a tree at the end. It **is** one: `clauses[c]` is
a Condition node, `laws[c]` the Action beside it, and the list order is the
Fallback's child order.

## The method

Two stages, and they are different algorithms.

**Stage 1 — imitation.** A cross-entropy planner (`search.py`) fires K
piecewise-constant action sequences through the simulator per state and returns
`u* = argmax_a Q*(s,a)` with the action curvature `M = -H_a Q*`. DAgger
re-labels the states the controller actually visits. This produces a competent
controller on a fully observed world and, as it turns out, a degenerate one on a
partially observed world — the planner is *privileged*, and no observable guard
can explain its labels.

**Stage 2 — policy improvement.** The planner is dropped. `V^pi` by fitted value
iteration, `Q^pi` by probing the action space, then the region-restricted
deterministic policy gradient on the laws and a boundary term on the guards.
Each mechanism runs on its own cadence (`loop.py`): gradients every round, the
discrete law library and structure moves every third, memory once.

**Stage 3 — discovery from zero on the intersection.** No planner and no
expert. Guards, steps, terminations, failure clauses, nested subtrees and
blackboards are grown by rollout (`rlfit.py`). Each leaf is a kernel-interpolation
law whose inducing points, targets, lengthscales and input columns are learned
(`kernlaw.py`, `kernsearch.py`), proposed where exploratory deviations paid and
where a critic trained on those exact advantages predicts they would
(`intersection_critic.py`). The vehicle and signal trees are trained in
alternating stages against populations of each other, over per-episode
operating conditions (100–600 veh/h per approach, turning shares, plan greens),
with a demand curriculum and per-condition acceptance (e35).

**Every move is a proposal.** Nothing is applied because a criterion liked it —
a paired rollout test accepts or rejects it.

## The recurring result

Seven local criteria have now been tried as selection rules and every one has
improved while measured return fell: sup-LM instability, the M-weighted value
loss, gap-weighted loss, value-regime loss, one-step TD advantage, the fitted
control law itself (better label-match, ten return units worse), and an
information-gap ranking for memory (the known-good candidate at rank 6 of 8214,
tied with a planted distractor). What has never mis-ranked is a short rollout.

## Layout

    btind/envs/      ForageWorld, NestWorld, a fused model of highway-env, and
                     IntersectionBatch: a signalised intersection on SUMO's
                     geometry where one shared tree drives every vehicle and a
                     second tree runs the signal (signal visible near the box,
                     or delivered once as a V2I message; optional occlusion;
                     discrete or continuous leaves per agent)
    btind/           search, collect, valuesplit, evotm, rollout_select   stage 1
                     critic, qwire, landscape, structure, lawsearch       stage 2
                     memory, memsearch, betasearch, grow_bt, loop         temporal
                     tick        the ONE compiled arbitration both kernels run:
                                 fallback, latch, steps, success, failure
                     stepsearch  proposes a Sequence's further steps and a
                                 child's FAILURE condition, by rollout
                     explore     counterfactual deviations: an exact advantage
                                 for one action or one manoeuvre, paired
                     memtransient  a message-shaped blackboard for a discrete
                                 head: a transient column, its arrival as the
                                 event, a countdown, and the arms that read it
                     rlfit       the pure-RL loop, no expert anywhere
                     subtree     nested subtrees: grown inside a child, compiled
                                 flat, factored back for display
                     kernlaw     the kernel-interpolation leaf (affine prior
                                 mean + learned inducing points), numpy and numba
                     kernsearch  proposes, tunes and prunes inducing points
                     intersection_critic  V-hat and A-hat for the intersection,
                                 scored on held-out data, proposing points
    experiments/     e18-e35, each with its findings in the docstring
                     e26-e27: highway and the reactive intersection do NOT
                     reward sequences (progress is observable); e29, e32: the
                     event-mode and occluded intersections DO reward memory
                     (+36 and +6.5 hand-written); e28, e30: discovery from zero
                     e33: a traffic reward that ranks stop-all < crash/starve <
                     fixed plan < actuated; e34: one-arm growth cannot reach a
                     follower at full demand (each piece alone loses), a per-car
                     reward and a demand ladder can (team -95 vs hand-written
                     -223, no crashes); e35: kernel leaves, critic, operating
                     conditions and the vehicle/signal stage cycle
    data/, figs/     measured results
    fit_bt.py        entry point

## Running

Everything runs in a conda environment (`bt-induction`), never the system Python:

    conda activate bt-induction
    python -m pytest -q tests
    python experiments/e35_condition_cycle.py cycles=6   # resumable

Note: import `numba` before `xgboost` — the reverse order breaks llvmlite's DLL
load on Windows.

## Honest status

On the intersection, the vehicle tree climbed a demand ladder from zero to a
team score of -95 against -223 for a hand-written follower, with a two-step
Sequence, a termination, a fail clause and a nested subtree -- but one of its
arms read the fixed plan's 30 s green, which is why training now spans
operating conditions and partner populations (e35, in progress). The learned
critic ranks advantages usefully at medium and heavy demand and poorly in light
traffic; it only proposes.

On ForageWorld the emitted tree scores ~21 against a replanning CEM baseline of
~13. On NestWorld with partial observability the pipeline does **not** yet work:
imitation from a privileged planner emits a single arm matching 100% of states,
and growing a tree from empty by rollout does little better. A hand-written
blackboard controller scores 11.25 there against a reactive 2.47, so the 8.78
that memory is worth has been measured but not yet won by the search.
