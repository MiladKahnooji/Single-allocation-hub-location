# Completion specification: exact-to-learning hub-location pipeline

This specification is grounded in the supplied papers: Paper A, *A conditional
β-mean approach to risk-averse stochastic multiple allocation hub location
problems* (Transportation Research Part E 158, 2022, `74.pdf`), Paper B,
*Machine learning augmented approaches for hub location problems* (Computers &
Operations Research 154, 2023, `1-s2.0-S0305054823000527-main (2).pdf`), and
the client alignment response (pp. 2–5). Page numbers are printed article pages.

## 1. Evidence and reusable implementation

Paper A §3.2 (pp. 5–7) defines the stochastic risk model (Eqs. 12–20 and
30–37); §4–4.1 (pp. 7–9) defines branch-and-Benders-cut, the slave/dual
formulation (Eqs. 38–55), and Algorithms 1–2. Paper B §3.2 (pp. 5–6) gives
the single-allocation USApHMP (Eqs. 1–7); §4.1–4.2 (pp. 6–8) gives graph
construction Eqs. 8–10 and features Eqs. 11–17. Appendix A.1 (pp. 20–21)
specifies CBS node ranking, potential hub set and clustering (Algorithm 3).

The validated loader, seeded 100-scenario generator, route evaluator, risk
functions, tiny binary MILP, synthetic MDS/graphs/features, DLHr and GVNS
implementations are reusable. Benders and CBS are currently absent and must be
added without changing those contracts or raw data.

## 2. Single-allocation mathematical contract

For nodes `N`, scenarios `S`, OD pairs `(i,j)`, demand `w^s_ij`, probability
`q_s`, distance `d`, and discount `alpha`,
`c_ijkm = d[i,k] + alpha*d[k,m] + d[m,j]`.
Use binary `z_ik` with the standard Paper B convention `z_kk=1` iff `k` is a
hub:

```
sum_k z_kk = p;       sum_k z_ik = 1 (every i);       z_ik <= z_kk;       z_ik in {0,1}.
```

These constraints imply open hubs assign to themselves; no separate
`assign[i][i] == hub[i]` constraint is needed (client alignment, p. 2).
For the tiny article-visible MILP retain binary route selection
`X[s,i,j,k,m]`, enforce `sum(k,m) X[s,i,j,k,m] = 1`, and link it to both
assignments with the usual upper/product inequalities. This is the explicit
single-allocation adaptation of Paper A Eq. 14; the scenario index is retained
for comparison although first-stage assignments are common to all scenarios.
Paper A's `x` is a nonnegative continuous flow fraction (Eq. 16), whereas this
project's `X` is deliberately binary route selection (client alignment, pp. 2–3).

## 3. Conditional beta-mean

For explicit probabilities use Paper A Eq. 32:

```
R(C) = min_eta eta + (1/beta) * sum_s q_s * max(C_s - eta, 0),  0 < beta <= 1.
```

For the project's equal probabilities, Paper A Eqs. 18–20 use
`K = ceil(beta*S)` and the mean of the largest `K` scenario costs, equivalently
`eta + sum_s excess_s/K`. If `beta*S` is integral, `q_s/beta = 1/K`; this is
the `(1-beta)`-CVaR relationship in Proposition 1, Eqs. 30–37 (pp. 6–7).
Unequal probabilities remain supported by the weighted form and are recorded as
a distinct risk mode.

## 4. Iterative Benders design

The master contains binary `z`, lower-bound scenario costs `theta_s`, `eta`,
and `excess_s >= theta_s-eta`. Its objective is `eta + sum(excess)/K` for
equal scenarios (or `eta + sum(q_s*excess_s/beta)` otherwise), with the
single-allocation constraints above.

For an incumbent `zbar`, each scenario LP is the exact coupling relaxation:

```
min sum_ij,k,m w^s_ij*c_ijkm*r_ijkm
sum_m r_ijkm = zbar_ik;  sum_k r_ijkm = zbar_jm;  r >= 0.
```

Its free dual potentials `u_ijk,v_ijm` maximize
`sum_ij,k u_ijk*zbar_ik + sum_ij,m v_ijm*zbar_jm` subject to
`u_ijk+v_ijm <= w^s_ij*c_ijkm`. Every optimal dual solution yields the valid
cut `theta_s >= sum u_ijk*z_ik + sum v_ijm*z_jm`. This is the necessary
multiple-to-single adaptation: both margins select one assigned origin and
destination hub. Feasibility cuts are unnecessary because the assignment
equations make each LP feasible (Paper A §4.1, Eqs. 38–49).

At each iteration solve the master, solve all scenario LPs, update `UB` with
the shared evaluator/risk function, and add violated cuts. `LB` is the master
objective; stop only when `UB-LB <= max(abs_tol, rel_tol*max(1,|UB|))` and all
solves are optimal. Record iterations, cuts, LP statuses, bounds and gap. A
time/iteration limit produces `time_limited` and `proven_optimal=false`; never
claim optimality without solver proof. Use PuLP/CBC only. Sparse positive OD
storage and deterministic ordering are required; a node/memory guard may
decline impractical AP runs rather than silently approximate.

## 5. Ground-truth hub targets

For a target grid `G` of `(p, alpha, beta, scenario_seed)`, run converged
Benders and set `score_i = mean_g 1[i in H*_g]`. Canonical deterministic
optima are labelled `ground_truth_exact`; exhaustive alternatives require a
hub-set no-good cut and full re-certification. Time-limited Benders is
`bounded_incumbent`, and heuristic labels are `approximate_heuristic`, with
status, bounds, gap, seeds and grid stored in provenance. Only exact labels may
train a target set advertised as ground truth.

## 6. CBS and DL-CBS (Paper B)

Paper B Appendix A.1 first ranks nodes (`Imp1=(O_v+D_v)C_v` or `Imp9`, or DLHr
score), then chooses the top `2p` potential hubs `N^p`. Algorithm 3 builds
clusters using radius `PM = sum_{i in Np} min_{j in Np,j!=i} c_ij/(2p)` and
constructs `H`, isolated `H_S`, expanded `H'_S`, per-center `S_i`, and residual
`H_p`; the constrained original USApHMP is then solved. Implement the base CBS
with deterministic `Imp1` tie-breaking and DL-CBS with only the ranking/order
replaced by DLHr scores. Do not run baseline first or consume its solution.
Every state remains exactly `p` hubs, one assignment per node, and self-assigns
open hubs. Shared objective/scenarios and explicit method labels are mandatory.

## 7. Fair experiments, outputs and tests

For each paired method materialize one immutable 100-scenario bundle and hold
dataset, probabilities, `p`, `alpha`, `beta`, scenario/search seeds and budget
constant. Compare Benders (exact or time-limited), GVNS, CBS, DL-GVNS and
DL-CBS with objective, runtime, evaluations, status, hubs, assignments,
proven-optimal flag and ranker identity. Export detailed JSON, summary CSV,
aggregate means and win/tie/loss/improvement; ignore generated artifacts.

Required tests cover dual/primal equality and valid cuts, `LB<=UB`, convergence
against brute force, time-limit proof flags, exact target provenance, CBS
feasibility/determinism and DL-only guidance, paired scenario identity, and
existing risk/scenario/model/DLHr/GVNS regressions. Smoke tests are tiny exact,
bounded CAB25 Benders, and bounded CAB25/AP100/AP150/AP200 heuristic pairs;
never run the full paper grid. Exact means CBC proof, time-limited means a
certified bound plus incumbent, and heuristic means feasible but unproven.

## 8. Acceptance

No raw data, approved model/risk/scenario logic or unrelated files change. Every
result declares its method class and proof status; exact target claims have a
convergence record; all comparisons use identical stochastic inputs. Paper B's
CBS/DL-CBS rules are followed only to the extent stated above; CBS variants,
DL-CBS, Benders, and later experiments remain out of scope until implemented
and tested against this specification.
