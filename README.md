# Single-allocation hub location

This Python 3.12 project loads the supplied CAB/AP matrices, generates seeded
demand scenarios, and runs a risk-averse single-allocation hub model using an
exact MILP or a practical seeded heuristic.

## Install

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

## Data

The unchanged inputs belong in `data/raw/`:

- `wij_CAB25.xlsx` with `C-CAB25.csv`
- `wij_AP100.xlsx` with `C_AP100.csv`
- `wij_AP150.xlsx` with `C_AP150.csv`
- `wij_AP200.xlsx` with `C_AP200.csv`

## Run

One selected run:

```bash
salh-solve --dataset CAB25 --p 3 --alpha 0.5 --beta 0.5 --scenarios 100 --seed 11 --time-limit 10 --method heuristic
```

The committed four-dataset example batch:

```bash
salh-solve --config configs/example_runs.json
```

Write the batch results and plots under an ignored output directory:

```bash
salh-solve --config configs/example_runs.json --output-dir outputs/example
```

`p` is the number of hubs. `alpha` discounts inter-hub distance. `beta` is the
worst-scenario probability fraction, scenario count controls sampled demand
matrices, seed makes generation and heuristic search reproducible, and time
limit is the exact-solver or heuristic-search budget (not total loading time).
Method is `exact`, `heuristic`, or `auto`.

The final stochastic run uses 100 independently sampled, normalized demand
scenarios. Each has zero diagonal and explicit probability `0.01`; the seed
reproduces both the scenario matrices and their probability vector.

The probability-weighted conditional beta-mean is
`eta + sum(q_s * max(C_s - eta, 0)) / beta`, for `0 < beta <= 1`, and is
equivalent to `CVaR_(1-beta)`. Smaller beta values are more risk-averse;
`beta=1` is expected cost.

`exact` uses PuLP/CBC and is restricted to practical CAB25 runs of at most five
scenarios. A time-limited exact incumbent may not be proven optimal. `heuristic`
uses seeded local improvement and never claims optimality. `auto` applies the
exact practical limit and otherwise uses the heuristic.

With `--output-dir`, `summary.csv` contains one compact row per run,
`solutions.json` adds selected hubs and complete assignments, and
`objective_comparison.png` and `runtime_comparison.png` provide simple
comparisons. Failed runs remain in both data files. No geographic assignment
map is produced because coordinates are unavailable.

This is a coding implementation of the requested model and experiments, not a
publication-grade reproduction of every paper result. It does not run the full
parameter grid automatically.
