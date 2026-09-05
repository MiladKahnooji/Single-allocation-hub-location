"""Command-line experiment runner."""

from __future__ import annotations

import argparse
import json

from .experiments import RunConfig, load_run_configs, run_batch, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected hub-location configurations")
    parser.add_argument("--config", help="JSON file containing a runs list")
    parser.add_argument("--output-dir", help="write CSV, JSON, and PNG results here")
    parser.add_argument("--dataset", choices=("CAB25", "AP100", "AP150", "AP200"))
    parser.add_argument("--p", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--method", choices=("exact", "heuristic", "auto"), default="auto")
    args = parser.parse_args()

    if args.config:
        results = run_batch(load_run_configs(args.config))
        if args.output_dir:
            from .reporting import write_results

            write_results(results, args.output_dir)
        print(json.dumps([result.to_dict() for result in results]))
        return
    missing = [name for name in ("dataset", "p", "alpha", "beta") if getattr(args, name) is None]
    if missing:
        parser.error("single runs require --dataset, --p, --alpha, and --beta")
    result = run_experiment(
        RunConfig(
            dataset=args.dataset,
            p=args.p,
            alpha=args.alpha,
            beta=args.beta,
            scenario_count=args.scenarios,
            seed=args.seed,
            time_limit=args.time_limit,
            method=args.method,
        )
    )
    if args.output_dir:
        from .reporting import write_results

        write_results([result], args.output_dir)
    print(json.dumps(result.to_dict()))
