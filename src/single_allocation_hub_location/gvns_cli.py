"""Command-line entry point for small fair GVNS/DL-GVNS comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from .gvns_experiments import run_small_grid, train_synthetic_ranker_checkpoint, write_gvns_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small paired GVNS/DL-GVNS grid")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranker-checkpoint")
    parser.add_argument("--train-ranker", action="store_true")
    parser.add_argument("--datasets", default="CAB25,AP100,AP150,AP200")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--max-evaluations", type=int, default=12)
    parser.add_argument("--time-limit", type=float)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    checkpoint = Path(args.ranker_checkpoint) if args.ranker_checkpoint else output_dir / "synthetic_ranker.pt"
    if args.train_ranker:
        train_synthetic_ranker_checkpoint(checkpoint, seed=int(args.seeds.split(",")[0]))
    if not checkpoint.is_file():
        parser.error("provide --ranker-checkpoint or use --train-ranker")
    records = run_small_grid(
        datasets=tuple(part.strip() for part in args.datasets.split(",") if part.strip()),
        seeds=tuple(int(part) for part in args.seeds.split(",") if part.strip()),
        p=args.p,
        alpha=args.alpha,
        beta=args.beta,
        max_iterations=args.max_iterations,
        max_evaluations=args.max_evaluations,
        time_limit=args.time_limit,
        ranker_checkpoint=checkpoint,
    )
    paths = write_gvns_outputs(records, output_dir)
    print("\n".join(f"{name}: {path}" for name, path in paths.items()))
