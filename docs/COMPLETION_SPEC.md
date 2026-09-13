# Completion specification: exact-to-learning hub-location pipeline

## Purpose and evidence boundary

This is an implementation specification, not an implementation claim. It is
based on the current repository, PR #19, and the client alignment document
`client_model_alignment_response_fa.pdf`, pp. 2-5. That document confirms the
single-allocation diagonal convention, the one-route interpretation, the
finite-scenario beta convention, and the intended DLHr pipeline.

**Blocking citation gap:** neither supplied research paper is present in the
accessible repository/workspace. Therefore this specification does not invent
paper section, theorem, or equation numbers. Before implementing Benders or
CBS, attach both PDFs (or stable citations with the relevant pages), then
replace the two citation placeholders below:

| Required source | Needed evidence |
|---|---|
| Paper A (risk/Benders) | exact model, risk equations, decomposition and cut equations |
| Paper B (DLHr/CBS) | CBS state/transition/acceptance rules, DL-CBS score use, ranking-target equation |

The client alignment document is cited only as `Client alignment, p. N`; it is
not a substitute for either paper.

## 1. Current-state review

| Pipeline stage | Current status | Reuse decision / gap |
|---|---|---|
| Network matrices | Validated CSV/XLSX loader; CAB25/AP100/AP150/AP200 raw files | Reuse unchanged. |
| Scenarios | `generate_flow_scenarios`: seeded `default_rng`, node multipliers, Poisson draws, zero diagonal, normalized flows, 100 explicit equal weights | Reuse unchanged. It is already tested. |
| Route costs | `evaluate_scenario_costs` uses `d[i,k] + alpha*d[k,m] + d[m,j]` | Reuse as the sole candidate-cost oracle. |
| Risk | General weighted conditional beta-mean and equal-probability `K=ceil(beta*S)` helper | Reuse, but introduce one policy selector so every exact/heuristic evaluator chooses the same article or generalized mode. See §2. |
| Exact solve | Binary-route MILP in `model.py`, deliberately limited to six nodes | Keep as tiny regression oracle only. It is not a CAB/AP exact method. |
| Training data | Synthetic gravity demand, MDS adaptation, three weighted graphs, 39 normalized features | Reuse feature contract. Current n=25 labels are heuristic, hence not ground truth. |
| DLHr | CPU PyTorch three-view aggregation, GRU, jumping knowledge, pairwise loss/checkpoints | Reuse architecture and metrics. Replace label provenance before calling scores ground truth. |
| GVNS / DL-GVNS | Seeded feasible candidates, shared evaluator and neighborhoods; DL scores guide start/order | Reuse unchanged. |
| Benders | Absent | Implement as a new exact module after Paper A citation review. |
| CBS / DL-CBS | Absent | Do not implement until Paper B’s CBS algorithm is available. |
| Reporting | JSON/CSV/PNG helpers and ignored `outputs/` | Reuse schemas; add method/bound/provenance fields in §8. |

Important current inconsistency: the exact model applies the equal-probability
ceiling rule, while `evaluate_risk_objective` always calls the generalized
weighted evaluator. Existing planned values (`beta*S` integral for `S=100`)
coincide, but arbitrary non-integral beta does not. Resolve this before exact
comparisons; never compare two methods evaluated under different tail rules.

## 2. Mathematical contract

### 2.1 Sets, data, and single allocation

Let `N` be nodes, `S` scenarios, `A=N×N` OD pairs, and `p` the hub count.
Scenario demand is `w^s_ij`, probability is `q_s`, and

\[
  c_{ijkm}=d_{ik}+\alpha d_{km}+d_{mj}.
\]

Use one binary assignment variable:

\[
z_{ik}=1 \iff i\text{ is assigned to hub }k.
\]

The diagonal is the opening decision, `y_k=z_kk`; do not add an independent
hub variable. The valid single-allocation master constraints are

\[
\sum_k z_{kk}=p,\qquad
\sum_k z_{ik}=1\quad\forall i,\qquad
z_{ik}\le z_{kk}\quad\forall i,k,\qquad z_{ik}\in\{0,1\}.
\]

They imply that every open hub assigns to itself. This is the convention
requested in Client alignment, p. 2, item 1.

For the tiny explicit formulation only, use binary
`X^s_ijkm=1` when OD `(i,j)` uses assigned hubs `(k,m)` in scenario `s`, with

\[
\sum_{k,m}X^s_{ijkm}=1,
\quad X^s_{ijkm}\le z_{ik},\quad X^s_{ijkm}\le z_{jm},\quad
X^s_{ijkm}\ge z_{ik}+z_{jm}-1.
\]

This is a binary route-selection adaptation, not a claim that Paper A’s flow
variable has the same domain; Client alignment, pp. 2-3, items 2-3 records
that distinction. The scenario index is redundant for first-stage assignments
but retained in the explicit model for comparison.

### 2.2 Conditional beta-mean policy

For arbitrary explicit probabilities, retain

\[
 R_q(C)=\min_{\eta}\left\{\eta+\frac1\beta
 \sum_{s\in S}q_s(C_s-\eta)_+\right\},\quad 0<\beta\le1.
\]

For the documented 100 equal-probability scenarios, use the article-visible
finite form

\[
K=\lceil\beta |S|\rceil,\qquad
R_K(C)=\text{mean of the largest }K\text{ values of }C_s
=\min_{\eta}\left\{\eta+\frac1K\sum_s(C_s-\eta)_+\right\}.
\]

When `beta*|S|` is integral, `q_s/beta=1/K`; otherwise `R_q` and `R_K` are
different valid conventions. A run must record `risk_mode` as either
`equal_probability_ceiling` or `weighted_probability_tail`. The first is the
default for this project’s 100 scenarios. This is the client-requested
clarification in Client alignment, p. 3, item 4; Paper A equation citation is
pending the missing PDF.

## 3. Valid exact Benders design

### 3.1 Master problem

The master contains `z`, one lower-bound scenario cost `theta_s >= 0`, tail
threshold `eta >= 0`, and excess `xi_s >= 0`. It has the single-allocation
constraints in §2.1 and

\[
 \xi_s\ge\theta_s-\eta\quad\forall s.
\]

For equal scenarios minimize `eta + (1/K) sum_s xi_s`; otherwise minimize
`eta + sum_s(q_s/beta) xi_s`. Initially this is a relaxation because
`theta_s` has no route-cost cuts.

### 3.2 Scenario LP subproblem and dual

For a fixed feasible assignment `z̄`, scenario `s` has a continuous coupling
LP. This is exact because both assignment margins are one-hot:

\[
\begin{aligned}
Q_s(z̄)=\min_{r\ge0}\;&\sum_{i,j,k,m}w^s_{ij}c_{ijkm}r^s_{ijkm}\\
\text{s.t. }&\sum_m r^s_{ijkm}=z̄_{ik} &&\forall i,j,k,\\
&\sum_k r^s_{ijkm}=z̄_{jm} &&\forall i,j,m.
\end{aligned}
\]

At integral `z̄`, exactly the assigned `(k,m)` has value one for each `(i,j)`;
thus this LP equals the route-cost evaluator and needs no binary route variable.
Its free dual variables `u^s_ijk` and `v^s_ijm` solve

\[
\begin{aligned}
\max_{u,v}\;&\sum_{i,j,k}u^s_{ijk}z̄_{ik}+
\sum_{i,j,m}v^s_{ijm}z̄_{jm}\\
\text{s.t. }&u^s_{ijk}+v^s_{ijm}\le w^s_{ij}c_{ijkm}
&&\forall i,j,k,m.
\end{aligned}
\]

For an optimal dual solution, add the valid optimality cut

\[
\theta_s\ge\sum_{i,j,k}u^s_{ijk}z_{ik}+
\sum_{i,j,m}v^s_{ijm}z_{jm}.
\]

The stated subproblem is always feasible whenever the master satisfies the
assignment equations, so **no mathematical feasibility cuts are required**.
A numerical LP failure is an implementation error, not a reason to add an
unsupported cut. This derivation is a direct multiple-to-single allocation
adaptation: both margins use the one selected origin/destination hub. Validate
it against Paper A before code is written.

### 3.3 Bounds, convergence, and scale control

At each Benders iteration:

1. Solve the master MIP to optimality; its objective is the lower bound `LB`.
2. Evaluate all scenario LPs, preferably in deterministic scenario order or
   parallel with deterministic cut sorting; compute true `C_s` and true risk.
3. Update the incumbent upper bound `UB` if true risk improves, then add every
   violated scenario cut (`theta_s < Q_s(z̄)-tolerance`).
4. Stop only when `UB-LB <= max(abs_tol, rel_tol*max(1,abs(UB)))` **and** the
   master and all LPs have optimal statuses.

The literal LP contains `O(|S||N|^4)` coefficients, so it is a correctness
design, not permission to claim CAB/AP scalability. Implement sparse positive
OD storage, scenario-by-scenario LPs, dual-cut caching, and a measured node/
memory guard. Exact claims are permitted only after the guard, every master,
and every subproblem finish optimally. CAB25 may still be time-limited; AP100+
must default to heuristic methods until benchmarking proves otherwise.

If a time limit occurs, record the incumbent `UB`, any solver-certified `LB`,
gap if available, elapsed time, and status `time_limited`; never set
`proven_optimal=true`. CBC master bounds must be treated as unavailable unless
the API exposes them reliably and tests confirm their meaning.

## 4. Ground-truth ranking data

For a fixed synthetic instance and target grid `G` of `(p,alpha,beta,seed)`,
run the exact Benders solver and obtain each optimum hub set `H_g^*`. The
canonical frequency target is

\[
score_i=\frac1{|G|}\sum_{g\in G}\mathbf1[i\in H_g^*].
\]

If multiple optimal hub sets exist, choose one policy and record it:

- **exhaustive-optima mode:** enumerate all distinct optimal hub sets by adding
  a hub-set no-good constraint `sum_{k in H} z_kk <= p-1`, rerunning Benders
  under the proven optimum value, and continuing until the next exact lower
  bound exceeds that value; average membership over all enumerated optima.
- **canonical-optimum mode:** use deterministic solver tie breaking and label
  the target `canonical_exact`, not an all-optima frequency.

Only fully converged Benders labels are `ground_truth_exact`. Time-limited
Benders labels are `bounded_incumbent`; heuristic labels remain
`approximate_heuristic`. Current `training_data.py` uses a single normalized
base-demand scenario and, for n=25, heuristic labels; it must not be relabelled
as ground truth without this replacement. Preserve metadata for risk mode,
scenario seed/count, grid, statuses, bounds, gap, tie policy, and every hub
set used.

## 5. DLHr and graph contract

Retain the current validated contract: MDS-derived 2D coordinates for CAB/AP,
weighted production graph, weighted attraction graph, symmetric inverse-distance
spatial graph, and 39 normalized features. Retain the compact three-view
row-normalized aggregation, GRU updates, jumping-knowledge concatenation, one
score/node, pairwise loss, CPU checkpoints, and seeded splits.

Replace only the training target provenance with §4 data. Keep CAB/AP test
instances out of synthetic ranker training. Report train/validation loss,
pairwise accuracy, top-p recall, checkpoint hash, and target provenance.

## 6. CBS and DL-CBS: blocked specification

The repository has no CBS implementation, and Paper B is unavailable. Its
exact state representation, construction rule, neighborhood/acceptance logic,
stopping rule, and DL-score integration cannot be reconstructed safely.

When Paper B is supplied, the implementation specification must quote its
section/equation/algorithm and define, line-by-line:

1. the feasible CBS solution state and initialization;
2. each candidate-generation and repair rule preserving exactly `p` hubs and
   self-assigned open hubs;
3. objective evaluation exclusively through the shared 100-scenario evaluator;
4. acceptance, termination, seed use, and evaluation/time budget;
5. DL-CBS’s only permitted differences (ranker initialization/order/tie break)
   and a test proving it does not consume baseline CBS’s final solution.

Until then, expose no `cbs` CLI option and publish no CBS/DL-CBS comparison.

## 7. Fair experiment protocol

For each dataset/configuration/seed, materialize one scenario bundle once and
pass the same immutable flows and probabilities to every method. Hold constant:
dataset, `p`, `alpha`, `beta`, risk mode, scenario count/seed, search seed,
and method-comparable evaluation or wall-clock budget.

| Method | Permitted scope | Result label |
|---|---|---|
| Tiny binary MILP | regression oracle only | exact only after solver proof |
| Benders | guarded small instances; possibly time-limited CAB25 | exact / time-limited |
| GVNS | CAB25/AP100/AP150/AP200 | heuristic, never optimal |
| DL-GVNS | same as GVNS, fixed synthetic-only checkpoint | heuristic, never optimal |
| CBS / DL-CBS | unavailable pending Paper B | not implemented |

Do not charge synthetic ranker training time to a single DL heuristic run;
report it separately with checkpoint identity. For paired baseline/DL runs,
use identical evaluation budget and call counts. Compare objectives only when
risk mode and scenarios match. Report mean, standard deviation, seed-level
win/tie/loss, objective improvement, runtime, evaluations, and failure rate.

## 8. Required implementation outputs and tests

New exact-result records need: method, exact-status, `proven_optimal`, `LB`,
`UB`, gap, iteration/cut count, LP solve count, scenario/risk provenance,
hubs, assignments, objective, runtime, seed, and error. Preserve complete
solutions in JSON and compact rows in CSV. Generated artifacts stay ignored.

Minimum tests:

1. Existing scenario invariants: exactly 100, zero diagonal, normalized flows,
   probabilities sum to one, same seed repeats.
2. Risk: `K(100,.5)=50`, `K(100,.1)=10`, integral equal-weight equivalence,
   non-integral ceiling distinction, and unequal-weight generalized behavior.
3. Formulation: diagonal opening, one assignment, assignment only to open hubs,
   self-assigned open hubs, and route-cost equality with the evaluator.
4. Benders: each LP primal equals direct scenario cost; dual objective equals
   primal; each cut is valid at random feasible assignments; master `LB<=UB`;
   convergence matches brute force on tiny instances; no feasibility cuts are
   requested for a feasible master; time-limit status never claims optimality.
5. Labels: exact-frequency calculation, no-good alternative-optimum handling,
   metadata completeness, and rejection of approximate labels in an
   `ground_truth_exact` dataset.
6. DLHr: existing deterministic forward/loss/checkpoint tests plus target
   provenance and held-out synthetic ranking checks.
7. GVNS/CBS family: candidate feasibility after every move, deterministic same
   seed/budget, non-worsening local search, score-guided ordering only for DL
   variants, and shared paired scenario identity.
8. Smoke: tiny exact Benders versus brute force; CAB25 100-scenario bounded
   Benders status; CAB25/AP100/AP150/AP200 bounded GVNS/DL-GVNS; CBS tests only
   after Paper B is attached.

## Acceptance criteria

The repository is complete only when every run identifies its method class and
proof status correctly, all exact labels have a certified convergence record,
all heuristics remain labelled heuristic, every cross-method comparison shares
its stochastic input and risk mode, raw datasets remain unchanged, and Paper A
and Paper B citations have been added at the locations marked above.
