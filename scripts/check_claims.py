"""Check result-derived numbers in README.md."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def require(text: str, value: str, document: str) -> None:
    if " ".join(value.split()) not in " ".join(text.split()):
        raise AssertionError(f"{document} is missing result-derived value: {value}")


def rmse(summary: dict) -> str:
    return f"{summary['test_rmse_mean']:.3f} +/- {summary['test_rmse_std']:.3f}"


def main() -> int:
    readme = (ROOT / "README.md").read_text()
    dense = load("task2_dense_baseline.json")
    scratch = load("task4_tt_scratch_per_rank_lr.json")
    forest = load("task7_random_forest.json")
    low_rank_scratch = load("task7_low_rank_scratch.json")

    require(readme, rmse(dense["summary"]), "README.md")
    require(readme, rmse(forest["summary"]), "README.md")
    for variant in scratch["variants"]:
        require(readme, rmse(variant["summary"]), "README.md")
        require(readme, f"{variant['compression']['tt_total_params']:,}", "README.md")
    for variant in low_rank_scratch["variants"]:
        summary = variant["summary"]
        compression = variant["compression"]
        row = (
            f"| low-rank matrix rank {variant['rank']}, from scratch | "
            f"{compression['low_rank_total_params']:,} | "
            f"{compression['model_compression']:.0f}x | {rmse(summary)} | "
            f"{summary['train_seconds_mean'] / 60:.1f} |"
        )
        require(readme, row, "README.md")

    require(readme, "TT has the lower mean in all four comparable-budget brackets", "README.md")
    print("README.md matches the committed results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
