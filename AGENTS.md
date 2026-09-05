# Repository instructions

## Goal
Build a runnable Python implementation of a risk-averse stochastic single-allocation hub location model using the provided CAB/AP datasets.

## Scope
- Deliver working code, tests, result files, and short usage documentation.
- Adapt the supplied multiple-allocation formulation to single allocation.
- Generate demand scenarios using the referenced conditional beta-mean paper.
- Support the requested p, alpha, and beta values through configuration.
- Use an exact solver only where practical and a simple reproducible heuristic for larger instances.

## Out of scope
- Writing a new academic paper.
- Reproducing every table or experiment from either paper.
- Retraining the full DLHr model or reproducing DL-CBS/DL-GVNS.
- Claiming proven optimality for large instances.
- Uploading client conversations, screenshots, or copyrighted article PDFs.

## Engineering rules
- Prefer Python 3.12 and a small dependency set.
- Keep input data unchanged and separate generated artifacts from source data.
- Use deterministic random seeds.
- Add focused tests for each implemented component.
- Keep commits and pull requests scoped to one issue.
- Do not merge pull requests or close issues without owner approval.
- Avoid unrelated refactors and unnecessary abstractions.
- Record assumptions briefly in code or README instead of long reports.
