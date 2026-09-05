# Single-allocation hub location

This repository provides validated matrix loading, reproducible demand scenarios,
and a risk-averse single-allocation model for small instances and CAB25 smoke
solves. It does not include large-instance heuristics or experiment grids.

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

## Run CAB25

The objective is the probability-weighted upper-tail conditional beta-mean
(discrete CVaR): `min_eta eta + sum(q_s * max(C_s - eta, 0)) / (1 - beta)`.
Run a bounded solve with the open-source CBC solver:

```bash
salh-solve --p 3 --alpha 0.75 --beta 0.9 --scenarios 2 --seed 7 --time-limit 10
```

The JSON output includes the actual solver status and whether optimality was
proven within the time limit.
