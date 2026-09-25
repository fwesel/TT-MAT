"""Let a training script write its own runs' curves when it finishes.

The backend has to be chosen before ``mat_tt.report`` is imported, because that
module imports pyplot at module level. ``report`` cannot choose for itself: the
notebook sets an interactive backend first and a library that overrode it would
break inline figures. The training scripts are the opposite case, headless with
no display, so the choice is made here, once, instead of five times across
``scripts/``.

The import is deferred into the function so that a script which fails before it
finishes training never pays for matplotlib at all.
"""

from __future__ import annotations

from pathlib import Path


def save_run_curves(results_file: Path | str, prefix: str = "") -> list[Path]:
    """Write one training curve per run in the results file just written."""
    import matplotlib

    matplotlib.use("Agg")

    from mat_tt.report import export_run_curves

    return export_run_curves(results_file, prefix=prefix)
