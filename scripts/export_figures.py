"""Render every figure in the study to PNG files.

The notebook already carries each figure inline. These files are for looking at
them outside Jupyter, so they are derived artifacts and the directory is
gitignored.

Figures come from ``mat_tt.report.FIGURES``, the same helpers and committed
results the notebook uses, so exported files cannot drift from the notebook.

    figures/            notebook-size figures
  figures/runs/       one training curve per run, plus a contact sheet

Run the task scripts first, since each figure reads its own results file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import rc_context  # noqa: E402

from mat_tt.report import (  # noqa: E402
    FIGURES,
    RUN_CURVE_DIR,
    TRAINING_RESULTS,
    export_run_curves,
    load_payload,
    plot_all_run_curves,
    results_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "figures"
RUN_DIR = FIGURE_DIR / RUN_CURVE_DIR


def render(
    directory: Path,
    style: dict | None,
    dpi: int,
    figure_size: tuple[float, float] | None = None,
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    # Clear the previous numbered set first. Filenames carry the figure's
    # position, so inserting one renames everything after it and leaves the old
    # names behind, so clear the numbered set before rebuilding it.
    for extension in formats:
        for stale in directory.glob(f"[0-9][0-9]_*.{extension}"):
            stale.unlink()
    written = []
    for index, (name, build) in enumerate(FIGURES.items(), start=1):
        with rc_context(style or {}):
            figure = build()
            if figure_size is not None:
                figure.set_size_inches(*figure_size, forward=True)
                figure.tight_layout()
            for extension in formats:
                path = directory / f"{index:02d}_{name}.{extension}"
                figure.savefig(
                    path,
                    dpi=dpi,
                    bbox_inches="tight" if figure_size is None else None,
                    facecolor=figure.get_facecolor(),
                )
                written.append(path)
            plt.close(figure)
    return written


def render_run_curves() -> list[Path]:
    """Every training run's own curve, plus the one page that indexes them."""
    written = []
    for stem in TRAINING_RESULTS:
        written += export_run_curves(results_path(f"{stem}.json"), RUN_DIR)

    sheet = plot_all_run_curves([(stem, load_payload(stem)) for stem in TRAINING_RESULTS])
    path = RUN_DIR / "all_runs.png"
    sheet.savefig(path, dpi=110, bbox_inches="tight", facecolor=sheet.get_facecolor())
    plt.close(sheet)
    return written + [path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-runs", action="store_true", help="skip the per-run curves")
    args = parser.parse_args()

    try:
        for path in render(FIGURE_DIR, None, dpi=110):
            print(f"wrote {path.relative_to(REPO_ROOT)}  {path.stat().st_size // 1024} KB")
        if not args.no_runs:
            written = render_run_curves()
            print(f"wrote {len(written)} files to {RUN_DIR.relative_to(REPO_ROOT)}/")
    except FileNotFoundError as error:
        print(f"missing a results file: {error}. Run the scripts in scripts/ first.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
