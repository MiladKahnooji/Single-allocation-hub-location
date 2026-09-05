import csv
import json
from pathlib import Path

from single_allocation_hub_location import RunResult
from single_allocation_hub_location.cli import main
from single_allocation_hub_location.reporting import write_results


def result(*, error: str | None = None) -> RunResult:
    return RunResult(
        dataset="CAB25",
        node_count=25 if error is None else None,
        method_requested="heuristic",
        method_used="heuristic" if error is None else None,
        status="Heuristic completed" if error is None else "error",
        proven_optimal=False,
        hubs=(1, 4) if error is None else (),
        assignments=(1, 1, 4) if error is None else (),
        objective=12.5 if error is None else None,
        p=2,
        alpha=0.5,
        beta=0.5,
        scenario_count=2,
        seed=7,
        time_limit=0.1,
        runtime=0.02,
        error=error,
    )


def test_exports_summary_solutions_failures_and_plots(tmp_path: Path) -> None:
    output_dir = tmp_path / "new" / "results"
    paths = write_results([result(), result(error="invalid run")], output_dir)

    with paths["summary"].open(newline="", encoding="utf-8") as summary_file:
        rows = list(csv.DictReader(summary_file))
    assert len(rows) == 2
    assert tuple(rows[0]) == (
        "dataset", "node_count", "method_requested", "method_used", "status",
        "proven_optimal", "objective", "p", "alpha", "beta", "scenario_count",
        "seed", "time_limit", "runtime", "error",
    )
    assert rows[1]["status"] == "error"
    assert rows[1]["error"] == "invalid run"

    solutions = json.loads(paths["solutions"].read_text(encoding="utf-8"))
    assert solutions[0]["hubs"] == [1, 4]
    assert solutions[0]["assignments"] == [1, 1, 4]
    assert solutions[1]["error"] == "invalid run"
    assert paths["objective_plot"].stat().st_size > 0
    assert paths["runtime_plot"].stat().st_size > 0


def test_end_to_end_cli_writes_new_output_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output_dir = tmp_path / "delivery"
    monkeypatch.setattr(
        "sys.argv",
        [
            "salh-solve", "--dataset", "CAB25", "--p", "2", "--alpha", "0.5",
            "--beta", "0.5", "--scenarios", "1", "--seed", "4",
            "--time-limit", "0.01", "--method", "heuristic",
            "--output-dir", str(output_dir),
        ],
    )

    main()

    assert output_dir.is_dir()
    files = {path.name: path for path in output_dir.iterdir()}
    assert set(files) == {
        "summary.csv", "solutions.json", "objective_comparison.png",
        "runtime_comparison.png",
    }
    assert all(path.stat().st_size > 0 for path in files.values())
    stdout_result = json.loads(capsys.readouterr().out)
    assert not stdout_result["proven_optimal"]
