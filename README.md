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

## Hub-ranking inputs

The paper-aligned input subset uses seeded bounded power-law populations and
symmetric gravity demand proportional to `P_i * P_j / distance_ij^2`.
Production and attraction graphs retain the largest outgoing or incoming arcs
until they cover fraction `p_w` of the respective total, and store the original
demand as edge weight: production row `i` stores selected `w_ij`, while
attraction row `i` stores selected incoming `w_ji` at column `j`. The spatial
graph selects each node's nearest
`ceil(p_c * (n - 1))` neighbors, takes the deterministic union of those directed
selections, and stores symmetric `1 / distance_ij` weights. Zero distances use
a small positive denominator floor.

Node features are combined produced/attracted demand, total distance, distance
to the demand-weighted center, and 36 equal-angle directional one-hot bins by
default. Each feature column uses safe min-max normalization; constant columns
become zero. Continuous hub-ranking targets are each node's hub-selection
frequency over a configurable `(p, alpha)` grid.

The thesis adaptations are a reduced configurable training volume instead of
the paper's large study, seeded heuristic target solutions for practical
25-node instances, and deterministic coordinate reconstruction for supplied
data. Tiny instances may use the existing exact solver for target generation.

Because the supplied CAB/AP matrices contain no geographic coordinates, their
2D coordinates are deterministically reconstructed from the distance matrix by
classical MDS after averaging negligible directional rounding differences.
These coordinates are a thesis adaptation for feature creation, not original
geographic coordinates.

## Lightweight hub ranker

The optional ranker uses direct CPU PyTorch without a graph framework:

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[ml,test]'
```

Each weighted graph receives a self-loop and row normalization. All three graph
views are aggregated at every layer, a compact GRU updates each node state, and
jumping-knowledge concatenation preserves the input projection and every layer
representation before decoding one score per node. Training uses only node
pairs with different `hub_scores` targets.

This Python entry point generates a small in-memory dataset, trains with early
stopping, saves/reloads a checkpoint, and ranks the supplied CAB25 nodes:

```python
from pathlib import Path
import numpy as np

from single_allocation_hub_location import build_hub_feature_data, load_matrix_pair
from single_allocation_hub_location.ranker import (
    RankerTrainingConfig,
    generate_ranker_dataset,
    load_ranker_checkpoint,
    rank_hubs,
    train_ranker,
)

config = RankerTrainingConfig(
    training_instance_count=8,
    validation_fraction=0.25,
    hidden_dimension=16,
    message_passing_layers=2,
    epochs=10,
    learning_rate=0.001,
    batch_size=4,
    patience=3,
    seed=7,
)
training_data = generate_ranker_dataset(config)
result = train_ranker(training_data, config, "outputs/ranker/checkpoint.pt")
model = load_ranker_checkpoint("outputs/ranker/checkpoint.pt")

matrices = load_matrix_pair(
    Path("data/raw/wij_CAB25.xlsx"), Path("data/raw/C-CAB25.csv")
)
cab25 = build_hub_feature_data(
    np.asarray(matrices.distance),
    np.asarray(matrices.flow),
    p=3,
    seed=7,
    time_limit=0.05,
)
ranking = rank_hubs(model, cab25)
print(result.pairwise_accuracy, result.top_p_recall)
print(ranking.ranked_nodes[:3], ranking.scores)
```

CPU-first defaults are 20 generated instances, 20% validation, hidden dimension
32, two message-passing layers, 50 epochs, learning rate `0.001`, batch size 4,
patience 8, and seed 0. Checkpoints and generated training artifacts are ignored
by Git. This is a lightweight thesis adaptation, not training at the paper's
11,000-instance scale or a reproduction of its reported model accuracy.
