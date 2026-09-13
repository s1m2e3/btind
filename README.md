# btind — inducing behaviour trees by measured return

A behaviour tree is a readable controller: a `Fallback` of guarded `Sequence`s,
each guard a condition on the observation and each action an affine control law.
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

    btind/envs/      ForageWorld and NestWorld, batched numba kernels
    btind/           search, collect, valuesplit, evotm, rollout_select   stage 1
                     critic, qwire, landscape, structure, lawsearch       stage 2
                     memory, memsearch, betasearch, grow_bt, loop         temporal
    experiments/     e18-e22, each with its findings in the docstring
    data/, figs/     measured results
    fit_bt.py        entry point

## Running

    pip install -r requirements.txt
    python fit_bt.py --runs 2

Note: import `numba` before `xgboost` — the reverse order breaks llvmlite's DLL
load on Windows.

## Honest status

On ForageWorld the emitted tree scores ~21 against a replanning CEM baseline of
~13. On NestWorld with partial observability the pipeline does **not** yet work:
imitation from a privileged planner emits a single arm matching 100% of states,
and growing a tree from empty by rollout does little better. A hand-written
blackboard controller scores 11.25 there against a reactive 2.47, so the 8.78
that memory is worth has been measured but not yet won by the search.
