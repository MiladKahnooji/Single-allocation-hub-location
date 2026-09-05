# Single-allocation hub location

This repository provides validated matrix loading, reproducible demand scenarios,
and risk-averse exact or heuristic single-allocation runs. It does not
automatically execute an experiment grid.

## Setup and test

Use Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

## Input data

Place the supplied matrices in `data/raw/`. The currently expected dataset pairs
are:

- `data/raw/wij_CAB25.xlsx` and `data/raw/C-CAB25.csv`
- `data/raw/wij_AP100.xlsx` and `data/raw/C_AP100.csv`
- `data/raw/wij_AP150.xlsx` and `data/raw/C_AP150.csv`
- `data/raw/wij_AP200.xlsx` and `data/raw/C_AP200.csv`

`wij_*` is passed as the flow matrix and `C_*` as the distance matrix. Both files
must contain a single numeric, square matrix with finite, nonnegative entries and
the same dimensions. CSV and XLSX are both supported for either matrix.
The supplied XLSX flow files include a `0` through `n-1` column-index row; the
loader recognizes and removes that metadata row before validation.

```python
from pathlib import Path

from single_allocation_hub_location import load_matrix_pair

matrices = load_matrix_pair(
    flow_path=Path("data/raw/wij_AP100.xlsx"),
    distance_path=Path("data/raw/C_AP100.csv"),
)
print(matrices.size)
```

Generated results belong in `outputs/`; input files in `data/raw/` are not
modified by the loader.

## Generate demand scenarios

```python
from single_allocation_hub_location import generate_flow_scenarios, load_matrix

scenarios = generate_flow_scenarios(load_matrix("data/raw/wij_CAB25.xlsx"), seed=7)
print(scenarios.flows.shape, scenarios.probabilities.sum())
```

## Run selected configurations

The objective is the probability-weighted conditional beta-mean: the worst
`beta` fraction of scenario costs, equivalent to discrete `CVaR_(1-beta)`:
`min_eta eta + sum(q_s * max(C_s - eta, 0)) / beta`. Smaller positive `beta`
values are more risk-averse, while `beta=1` is the expected scenario cost.
Run one selected configuration:

```bash
salh-solve --dataset CAB25 --p 3 --alpha 0.5 --beta 0.8 --scenarios 2 --seed 7 --time-limit 10 --method auto
```

Or run a small JSON file containing `{"runs": [{...}, {...}]}`:

```bash
salh-solve --config runs.json
```

Exact time-limited runs report CBC's actual status and may be non-optimal.
Heuristic results always set `proven_optimal` to false. Results print to standard
output; the runner does not create result files.
