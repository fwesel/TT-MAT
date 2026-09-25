"""Tables and figures for the notebook, built from saved run results.

Keeping these out of the notebook means the plotting code is linted and
testable, and the notebook cells stay short enough to read.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mat_tt.model import LinearEntry
from mat_tt.train import MEAN_PREDICTOR_RMSE, RunResult, load_results, mean_predictor_rmse


def _spread(values: list[float]) -> float:
    """Sample standard deviation, or 0 for a single value."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


PAPER_RMSE = 0.285
PAPER_RMSE_SPREAD = 0.022
"""MAT from scratch with a large hyperparameter search, quoted for orientation only."""

LOGS_PER_STANDARDIZED_UNIT = 2.0955
"""Multiply an RMSE in this study's units by this to get log S units.

MAT's CSV ships the ESOL labels z-scored, so every RMSE here is in units of the
dataset's own spread and cannot be compared to anything in the solubility
literature until it is converted back.

Standardization is a straight line with two unknowns, so two known points
recover it. Delaney's original file gives amygdalin at -0.77 log S, which MAT
stores as 1.0881, and fenfuram at -3.30, stored as -0.1193. Those give a
standard deviation of 2.0955 and a mean of -3.0501, and the mean and extremes
of the file then reproduce Delaney's published values to 0.01 without having
been used to fit them.

Only the standard deviation appears here. An RMSE is built from differences, so
a constant offset cancels and the mean plays no part.
"""

# Validated categorical slots 1 and 2 (light mode), plus surface and ink tokens.
# Checked with the dataviz palette validator: worst adjacent CVD Delta E 24.7,
# normal vision 33.6, both slots clear 3:1 against the surface.
PAGE = "#ffffff"
"""Behind the whole figure. The plot panel sits on it as a distinct block."""

SURFACE = "#f3f3f0"
"""The plot panel, and so the colour every mark is read against.

This is what the marker rings and the gaps between fills wear, which is why the
panel rather than the page carries the name: a ring is there to separate a mark
from what is immediately behind it.
"""

GRIDLINE = "#ffffff"
"""Gridlines are knocked out of the tinted panel rather than drawn on top of it,
so they separate without adding ink."""

AXIS_RULE = "#9b9a92"
"""The left and bottom rules and the tick marks. Dark enough to bound the panel,
light enough not to compete with a series."""

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_RECESSIVE = "#d7d7d3"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SERIES_3 = "#1baf7a"
"""Categorical slot 3. Was INK_PRIMARY, which the palette validator rejects as a
series colour: it fails the lightness band and the chroma floor because it reads
as black rather than as a hue. Ink is for references and text, not for identity.

Aqua warns at 2.74:1 against the surface, below the 3:1 the validator wants. The
skill's relief for that is visible labels or a table view, and every figure using
this slot carries a legend and has its values in a notebook table."""

DENSE_LABEL = "Dense MAT baseline"
"""How the uncompressed model is named in every figure that draws one run."""

LINE_WIDTH = 2.0
MARKER_AREA = 81
MARKER_SIZE = 10
OVERVIEW_MARKER_SIZE = 10
DIRECT_LABEL_SIZE = 10
SURFACE_RING = 2.0  # a 2px ring in the surface colour, so overlapping marks stay legible

FONT_STACK = ["DejaVu Sans"]

BASE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "demibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 14,
    "axes.labelsize": 10,
    "axes.labelpad": 8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": LINE_WIDTH,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "figure.facecolor": PAGE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": PAGE,
    "axes.axisbelow": True,
}
"""Applied to rcParams when this module is imported.

An import side effect, deliberately: every figure in the study comes from this
module, and applying the style here is what keeps them identical without each
call site remembering to.
"""

plt.rcParams.update(BASE_STYLE)


def parameter_table(inventory: list[LinearEntry], total_params: int) -> pd.DataFrame:
    """Parameters grouped by the role each matrix plays in MAT."""
    rows = {}
    for entry in inventory:
        row = rows.setdefault(entry.role, {"layers": 0, "shape": "", "params": 0})
        row["layers"] += 1
        row["shape"] = f"{entry.in_features} x {entry.out_features}"
        row["params"] += entry.params

    table = pd.DataFrame(rows).T.reset_index(names="role")
    table["share"] = (table["params"] / total_params).map(lambda v: f"{v:.1%}")
    return table.sort_values("params", ascending=False).reset_index(drop=True)


def seed_table(runs: list[RunResult]) -> pd.DataFrame:
    """One row per seed, plus mean and spread."""
    table = pd.DataFrame(
        [
            {
                "seed": str(run.seed),
                "val RMSE": round(run.val_rmse, 4),
                "test RMSE": round(run.test_rmse, 4),
                "best epoch": run.best_epoch,
                "epochs run": run.epochs_run,
                "minutes": round(run.train_seconds / 60, 1),
            }
            for run in runs
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "seed": "mean",
                "val RMSE": round(table["val RMSE"].mean(), 4),
                "test RMSE": round(table["test RMSE"].mean(), 4),
                "best epoch": "",
                "epochs run": "",
                "minutes": round(table["minutes"].sum(), 1),
            },
            {
                "seed": "spread (sd)",
                "val RMSE": round(table["val RMSE"].std(ddof=1), 4),
                "test RMSE": round(table["test RMSE"].std(ddof=1), 4),
                "best epoch": "",
                "epochs run": "",
                "minutes": "",
            },
        ]
    )
    return pd.concat([table, summary], ignore_index=True)


def reference_table(runs: list[RunResult] | None = None) -> pd.DataFrame:
    """What the baseline should be judged against.

    Pass ``runs`` to score each training split's mean on its corresponding test
    split, using the target scaler fitted during that run.
    """
    if runs:
        floors = [
            mean_predictor_rmse(run.test_targets, run.target_scaler.get("shift", 0.0))
            for run in runs
        ]
        floor_cell = f"{sum(floors) / len(floors):.3f} +/- {_spread(floors):.3f}"
        floor_note = f"training mean per seed; range {min(floors):.3f} to {max(floors):.3f}"
    else:
        floor_cell = f"{MEAN_PREDICTOR_RMSE:.3f}"
        floor_note = "nominal CSV-wide reference"

    return pd.DataFrame(
        [
            {
                "reference": "predict the training mean",
                "test RMSE": floor_cell,
                "note": floor_note,
            },
            {
                "reference": "MAT from scratch (paper)",
                "test RMSE": f"{PAPER_RMSE:.3f} +/- {PAPER_RMSE_SPREAD:.3f}",
                "note": "large hyperparameter search, orientation only",
            },
        ]
    )


REPORTED_VALUES = 11
"""Roughly how many labelled ticks a scale carries.

A value the reader has to interpolate is a value the figure did not report, and
these must be read without measuring between two labels. At five the accuracy
figure stepped in units of 0.05 and the whole result sat inside one gap. The
gridlines that come with the extra labels are knocked out of the panel rather
than drawn on it, so reporting more costs no extra ink.
"""


def _style_axes(ax, grid: str | None = "y") -> None:
    """A tinted plot panel, gridlines knocked out of it, ink-token text.

    The panel separates the plot from the page, so the reader sees where the
    data lives before reading anything, and it lets the gridlines be white
    rather than grey: they divide the space without adding a mark. Left and
    bottom rules bound the panel, short ticks sit under the labelled values, and
    the title sits left where a reader starts rather than centred.

    ``grid`` picks which gridlines a form wants: "y" for anything plotted
    against a vertical scale, "x" for horizontal bars and dot plots, "both" for
    a parity scatter, and None for diagrams that are not plots. A diagram keeps
    the page behind it, since a panel with no scale to bound is just a box.
    """
    ax.figure.patch.set_facecolor(PAGE)
    ax.set_facecolor(SURFACE if grid else PAGE)

    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(bool(grid) or side == "bottom")
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color(AXIS_RULE)

    for axis, wanted in ((ax.yaxis, grid in ("y", "both")), (ax.xaxis, grid in ("x", "both"))):
        if not wanted:
            continue
        ax.grid(axis=axis.axis_name, which="major", color=GRIDLINE, linewidth=1.1)
        # Axes whose ticks were set explicitly by the caller are left alone:
        # those carry meaning, like the TT ranks, and a minor tick between two
        # of them would imply a rank 3.
        if isinstance(axis.get_major_locator(), mticker.FixedLocator):
            continue
        if axis.get_scale() == "log":
            # Labelled within the decade, not just at it, or a curve living
            # between 0.2 and 0.5 gets one number on its whole scale. Plain
            # decimals rather than the default, which renders 0.4 as 4 x 10^-1
            # and makes a reader do arithmetic to read a chart. MaxNLocator
            # cannot be used here at all: it spaces ticks linearly and puts one
            # at 0.0, which is not a place a log axis can label.
            axis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
            axis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
            axis.set_minor_formatter(mticker.NullFormatter())
        else:
            axis.set_major_locator(
                mticker.MaxNLocator(nbins=REPORTED_VALUES, steps=[1, 2, 2.5, 5, 10])
            )
            axis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(axis=axis.axis_name, which="minor", color=GRIDLINE, linewidth=0.55, alpha=0.8)

    ax.tick_params(which="both", color=AXIS_RULE, length=3, width=0.8)
    ax.tick_params(which="both", labelcolor=INK_SECONDARY, pad=5)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK_PRIMARY)


def _takeaway(ax, text: str) -> None:
    """Keep plot titles with each figure rendering."""
    del ax, text


def _figure_takeaway(figure, text: str) -> None:
    """Keep plot titles with each figure rendering."""
    del figure, text


def plot_learning_curve(
    run: RunResult,
    ax=None,
    *,
    label: str = DENSE_LABEL,
    takeaway: str | None = None,
    log: bool = False,
) -> Figure:
    """Train and validation RMSE against epoch for one run.

    The defaults describe the dense baseline, which is the one learning curve
    the notebook shows. Any other run has to pass its own ``label`` and
    ``takeaway``, because a default that names one model would quietly mislabel
    every other one.

    ``log`` scales the y axis and lets the limits follow the data instead of
    using the fixed window below. The appendix set needs it: per-run maxima
    reach 3.6 while every run's decisive range is 0.2 to 0.5, so on a linear
    axis the part worth reading is a tenth of the panel.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))
    history = pd.DataFrame(run.history)

    ax.axhline(MEAN_PREDICTOR_RMSE, linestyle="--", linewidth=1, color=INK_SECONDARY)
    ax.text(
        0.07,
        MEAN_PREDICTOR_RMSE + 0.02,
        "nominal RMSE = 1 reference",
        transform=ax.get_yaxis_transform(),
        fontsize=9,
        color=INK_SECONDARY,
        va="bottom",
    )
    ax.plot(
        history["epoch"],
        history["train_rmse"],
        color=SERIES_1,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        label="train (dropout on)",
    )
    ax.plot(
        history["epoch"],
        history["val_rmse"],
        color=SERIES_2,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        label="validation",
    )
    # One selective direct label, on the epoch that was actually selected.
    ax.scatter(
        [run.best_epoch],
        [run.val_rmse],
        s=MARKER_AREA,
        color=SERIES_2,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
    )
    ax.annotate(
        f"best val {run.val_rmse:.3f}\nepoch {run.best_epoch}",
        xy=(run.best_epoch, run.val_rmse),
        xytext=(0, 58),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
        ha="center",
        linespacing=1.4,
        arrowprops={"arrowstyle": "-", "linewidth": 0.8, "color": INK_RECESSIVE, "shrinkB": 6},
    )

    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    title_label = "Dense MAT" if label == DENSE_LABEL else label
    ax.set_title(f"{title_label} training curve (seed {run.seed})")
    _takeaway(
        ax,
        takeaway
        or f"Best validation RMSE {run.val_rmse:.3f} at epoch {run.best_epoch}; "
        "early stopping restores this checkpoint",
    )
    if log:
        # Everything drawn stays in frame. A run that spiked to 3.6 and one that
        # never left 1.0 then read the same way, which is what makes the set
        # comparable at a glance.
        ax.set_yscale("log")
        drawn = pd.concat([history["train_rmse"], history["val_rmse"]])
        ax.set_ylim(drawn.min() * 0.85, max(drawn.max(), MEAN_PREDICTOR_RMSE) * 1.3)
        ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper right")
    else:
        # Truncated below, since RMSE never approaches zero here and the
        # interesting range is 0.2 to 0.5. The mean-predictor line stays in
        # frame as the anchor, which clips the first epoch or two of the train
        # curve.
        ax.set_ylim(0.15, 1.12)
        # Under the mean-predictor line, the only thing in the top band.
        ax.legend(
            frameon=False,
            labelcolor=INK_SECONDARY,
            loc="upper right",
            bbox_to_anchor=(1.0, 0.88),
        )
    _style_axes(ax)
    return ax.figure


def plot_parity(run: RunResult, ax=None) -> Figure:
    """Predicted against measured solubility on the test split.

    One series, so no legend box, the title names it.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    smiles = list(run.test_targets)
    measured = [run.test_targets[s] for s in smiles]
    predicted = [run.test_predictions[s] for s in smiles]

    limits = (min(measured + predicted) - 0.3, max(measured + predicted) + 0.3)
    ax.plot(limits, limits, linestyle="--", linewidth=1, color=INK_SECONDARY, zorder=1)
    ax.scatter(
        measured,
        predicted,
        s=MARKER_AREA,
        color=SERIES_1,
        alpha=0.75,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=2,
    )

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")
    ax.set_xlabel("measured log S")
    ax.set_ylabel("predicted")
    ax.set_title(f"Test set, seed {run.seed}, RMSE {run.test_rmse:.3f}, n = {len(smiles)}")
    # Against the split's own mean, not the nominal 1.0: the labels are
    # standardized over all of ESOL, so a single split's variance is only near 1.
    split_mean = sum(measured) / len(measured)
    residual = sum((m - p) ** 2 for m, p in zip(measured, predicted, strict=True))
    variance = sum((m - split_mean) ** 2 for m in measured)
    _takeaway(ax, f"The baseline explains {1 - residual / variance:.0%} of held-out variance")
    _style_axes(ax, grid="both")
    return ax.figure


def lr_sweep_table(sweep: dict) -> pd.DataFrame:
    """The learning rate sweep, with the chosen row marked."""
    table = pd.DataFrame(sweep["candidates"])
    table["lr"] = table["lr"].map(lambda v: f"{v:.0e}")
    table["val_rmse"] = table["val_rmse"].round(4)
    table["train_seconds"] = table["train_seconds"].round(0)
    table["chosen"] = ["yes" if row == f"{sweep['chosen_lr']:.0e}" else "" for row in table["lr"]]
    return table


def results_path(name: str) -> Path:
    """Path to a saved result file, resolved from the package rather than the CWD."""
    return Path(__file__).resolve().parents[2] / "results" / name


def tt_cost_table(payload: dict) -> pd.DataFrame:
    """What a TT-parameterized 256x256 matrix costs at each rank."""
    d = payload["d_model"]
    rows = []
    for entry in payload["cost_table"]:
        rows.append(
            {
                "TT rank": entry["rank"],
                "cores": f"{entry['tt_params']:,}",
                "dense": f"{entry['dense_params']:,}",
                "ratio": f"{entry['dense_params'] / entry['tt_params']:.0f}x",
                "all 12 matrices": f"{12 * entry['tt_params']:,}",
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["d_model"] = d
    return table


def tt_error_table(payload: dict) -> pd.DataFrame:
    """Mean reconstruction error per rank, with both controls beside it."""
    frame = pd.DataFrame(payload["rows"])
    rows = []
    for rank in payload["ranks"]:
        selected = frame[frame["tt_rank"] == rank]
        trained = selected[selected["source"] == "trained"]
        untrained = selected[selected["source"] == "untrained"]
        plain_rank = int(trained["plain_rank"].iloc[0])
        rows.append(
            {
                "TT rank": rank,
                "TT params": f"{int(trained['tt_params'].iloc[0]):,}",
                "TT error, trained": round(trained["tt_error"].mean(), 3),
                "TT retained norm": round(trained["tt_retained_norm"].mean(), 3),
                "TT error, untrained": round(untrained["tt_error"].mean(), 3),
                "matched plain rank": plain_rank if plain_rank else "too small",
                "plain SVD error": (
                    round(trained["plain_error"].mean(), 3) if plain_rank else "n/a"
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_tt_error(payload: dict, ax=None) -> Figure:
    """Reconstruction error against TT rank, with the two controls.

    Three series only, which is the all-pairs cap the palette carries for
    scatter and multi-line forms, so the 12 individual matrices are shown as
    faint background lines rather than as their own colors.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    frame = pd.DataFrame(payload["rows"])
    ranks = payload["ranks"]

    # Every individual trained matrix, recessive, to show the spread.
    trained = frame[frame["source"] == "trained"]
    for path in trained["path"].unique():
        series = trained[trained["path"] == path].sort_values("tt_rank")
        # A tint of the series they belong to, not grey: grey is now what the
        # gridlines wear, and recessive data reading as chrome is worse than
        # not drawing it.
        ax.plot(
            series["tt_rank"],
            series["tt_error"],
            color=SERIES_1,
            alpha=0.22,
            linewidth=0.9,
            zorder=2,
        )

    def mean_curve(source, column):
        subset = frame[frame["source"] == source]
        return [subset[subset["tt_rank"] == r][column].mean() for r in ranks]

    ax.plot(
        ranks,
        mean_curve("untrained", "tt_error"),
        color=SERIES_2,
        linewidth=6,
        alpha=0.45,
        solid_capstyle="round",
        label="TT, untrained weights (control)",
        zorder=2,
    )
    ax.plot(
        ranks,
        mean_curve("trained", "tt_error"),
        color=SERIES_1,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=SURFACE_RING,
        label="TT, trained weights",
        zorder=3,
    )

    plain = mean_curve("trained", "plain_error")
    ax.plot(
        ranks,
        plain,
        color=SERIES_3,
        linewidth=1.4,
        marker="s",
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=SURFACE_RING,
        label="truncated SVD, TT-matched parameter budget",
        zorder=2,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    # Zoomed to where the data sits. The full 0 to 1 range hid the three series
    # on top of each other, and the dashed line at 1.0 stays in frame so the
    # scale is unambiguous: everything visible here discards most of the matrix.
    # The floor follows the lowest curve rather than being fixed, since a fixed
    # 0.84 cropped the plain-SVD control off the bottom of the panel as soon as
    # that control did better.
    drawn = mean_curve("untrained", "tt_error") + mean_curve("trained", "tt_error") + plain
    ax.set_ylim(min(v for v in drawn if v == v) - 0.02, 1.03)
    ax.set_xlabel("TT rank")
    ax.set_ylabel("relative Frobenius residual")
    ax.set_title(
        f"TT-SVD poorly reconstructs trained {payload['d_model']}x{payload['d_model']} MAT weights"
    )
    # The control is the finding: if untrained weights compress as well as
    # trained ones, the decomposition is not finding anything the model learned.
    _takeaway(
        ax,
        f"At rank {ranks[-1]}, residual error is "
        f"{mean_curve('trained', 'tt_error')[-1]:.3f} trained versus "
        f"{mean_curve('untrained', 'tt_error')[-1]:.3f} untrained",
    )
    ax.axhline(1.0, linestyle=":", linewidth=1, color=INK_SECONDARY)
    ax.text(
        ranks[0] * 1.04,
        1.002,
        "1.0 = residual of the zero approximation",
        fontsize=9,
        color=INK_SECONDARY,
        va="bottom",
    )
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="lower left")
    _style_axes(ax)
    return ax.figure


def spectrum_table(payload: dict) -> pd.DataFrame:
    """Stable rank of the trained matrices against a random matrix of the same shape."""
    frame = pd.DataFrame(payload["rows"])
    trained = frame[frame["source"] == "trained"].drop_duplicates("path")
    reference = payload["random_matrix_reference"]
    d = payload["d_model"]
    return pd.DataFrame(
        [
            {
                "matrices": "MAT, trained",
                "stable rank": round(trained["stable_rank"].mean(), 1),
                "top 16 of 256 captures": f"{trained['energy_top_16'].mean():.1%}",
                "top 64 captures": f"{trained['energy_top_64'].mean():.1%}",
            },
            {
                "matrices": f"random {d}x{d}",
                "stable rank": round(reference["stable_rank"], 1),
                "top 16 of 256 captures": f"{reference['energy_top_16']:.1%}",
                "top 64 captures": f"{reference['energy_top_64']:.1%}",
            },
            {
                "matrices": "full rank would be",
                "stable rank": d,
                "top 16 of 256 captures": f"{16 / d:.1%}",
                "top 64 captures": f"{64 / d:.1%}",
            },
        ]
    )


ARM_LABELS = {
    "4b": "TT trained from scratch",
    "4a": "TT-SVD + fine-tuning",
    "low_rank_scratch": "Low-rank matrix factorization, trained from scratch",
    "low_rank_finetune": "Truncated SVD + fine-tuning",
}
"""How the two Task 4 arms are named in every table and figure."""


def _rmse_cell(summary: dict, key: str = "test") -> str:
    return f"{summary[f'{key}_rmse_mean']:.3f} +/- {summary[f'{key}_rmse_std']:.3f}"


def _payload_label(payload: dict) -> str:
    """An arm's name, or its own label when two protocols share an arm.

    Task 4b was run twice, once with a single learning rate for every rank and
    once with the rate swept per rank, so the arm alone does not identify a row.
    """
    display_label = payload.get("display_label", "")
    normalized = display_label.lower()
    if payload["arm"] in ("4a", "low_rank_scratch", "low_rank_finetune"):
        return ARM_LABELS[payload["arm"]]
    if "lr swept per rank" in normalized or payload.get("lr_per_rank"):
        return "TT trained from scratch, LR tuned per rank"
    if "one lr for all ranks" in normalized:
        return "TT trained from scratch, shared LR"
    return display_label or ARM_LABELS[payload["arm"]]


def tt_variant_table(payloads: list[dict], dense: dict | None = None) -> pd.DataFrame:
    """One row per TT arm and rank, with the dense baseline on top.

    This is most of what Task 6 asks to report: test RMSE with its spread over
    seeds, parameter counts for the tensorized layers and for the whole model,
    compression ratios, and training time.

    Task 6 asks for total and trainable counts at both levels, which is four
    numbers, but only two are distinct here: nothing is frozen at any point, so
    trainable equals total for the whole model and for the tensorized layers
    alike. The whole-model pair is carried as columns and the equality at layer
    level is stated in the notebook, rather than printing a column that is a copy
    of its neighbour twice over.
    """
    rows = []
    if dense is not None:
        summary = dense["summary"]
        rows.append(
            {
                "model": "dense baseline",
                "TT rank": "",
                "test RMSE": _rmse_cell(summary),
                "val RMSE": round(summary["val_rmse_mean"], 3),
                "tensorized layers": f"{789504:,}",
                "total params": f"{summary['total_params']:,}",
                "trainable": f"{summary['trainable_params']:,}",
                "compression": "1x",
                "fit min/seed": round(summary["train_seconds_mean"] / 60, 1),
                "pretrain + fit min/seed": round(summary["train_seconds_mean"] / 60, 1),
            }
        )

    for payload in payloads:
        for variant in payload["variants"]:
            summary, compression = variant["summary"], variant["compression"]
            fit_minutes = summary["train_seconds_mean"] / 60
            pretrain_minutes = (
                dense["summary"]["train_seconds_mean"] / 60
                if dense is not None and payload["arm"] == "4a"
                else 0.0
            )
            rows.append(
                {
                    "model": _payload_label(payload),
                    "TT rank": variant["rank"],
                    "test RMSE": _rmse_cell(summary),
                    "val RMSE": round(summary["val_rmse_mean"], 3),
                    "tensorized layers": f"{compression['tt_layer_params']:,}",
                    "total params": f"{compression['tt_total_params']:,}",
                    "trainable": f"{compression['tt_trainable_params']:,}",
                    "compression": f"{compression['model_compression']:.0f}x",
                    "fit min/seed": round(fit_minutes, 1),
                    "pretrain + fit min/seed": round(fit_minutes + pretrain_minutes, 1),
                }
            )
    return pd.DataFrame(rows)


def low_rank_variant_table(payloads: list[dict], dense: dict | None = None) -> pd.DataFrame:
    """One row per low-rank matrix-factorization arm and matrix rank."""
    rows = []
    dense_minutes = dense["summary"]["train_seconds_mean"] / 60 if dense else 0.0
    for payload in payloads:
        for variant in payload["variants"]:
            summary, compression = variant["summary"], variant["compression"]
            fit_minutes = summary["train_seconds_mean"] / 60
            pretrain_minutes = dense_minutes if payload["arm"] == "low_rank_finetune" else 0.0
            rows.append(
                {
                    "model": ARM_LABELS[payload["arm"]],
                    "matrix rank": variant["rank"],
                    "test RMSE": _rmse_cell(summary),
                    "val RMSE": round(summary["val_rmse_mean"], 3),
                    "factorized layers": f"{compression['low_rank_layer_params']:,}",
                    "total params": f"{compression['low_rank_total_params']:,}",
                    "trainable": f"{compression['low_rank_trainable_params']:,}",
                    "compression": f"{compression['model_compression']:.1f}x",
                    "fit min/seed": round(fit_minutes, 1),
                    "pretrain + fit min/seed": round(fit_minutes + pretrain_minutes, 1),
                }
            )
    return pd.DataFrame(rows)


def low_rank_budget_table(tt_payload: dict, low_rank_payload: dict) -> pd.DataFrame:
    """The uniform low-rank models immediately below and above TT ranks 4 and 8."""
    tt_by_rank = {variant["rank"]: variant for variant in tt_payload["variants"]}
    low_rank_by_rank = {variant["rank"]: variant for variant in low_rank_payload["variants"]}
    rows = []
    for low_rank_lower, tt_rank, low_rank_upper in ((1, 4, 2), (4, 8, 5)):
        lower = low_rank_by_rank[low_rank_lower]["compression"]["low_rank_total_params"]
        target = tt_by_rank[tt_rank]["compression"]["tt_total_params"]
        upper = low_rank_by_rank[low_rank_upper]["compression"]["low_rank_total_params"]
        rows.append(
            {
                "lower matrix rank": low_rank_lower,
                "lower params": f"{lower:,}",
                "TT rank": tt_rank,
                "TT params": f"{target:,}",
                "upper matrix rank": low_rank_upper,
                "upper params": f"{upper:,}",
                "lower difference": f"{(lower / target - 1):+.1%}",
                "upper difference": f"{(upper / target - 1):+.1%}",
            }
        )
    return pd.DataFrame(rows)


def init_table(payload: dict) -> pd.DataFrame:
    """What TT-SVD of the trained weights scores before any fine tuning.

    Arm 4a only. The before-training diagnostic is validation RMSE so the test
    split remains unread until model selection is complete.
    """
    rows = []
    for variant in payload["variants"]:
        summary = variant["summary"]
        rows.append(
            {
                "TT rank": variant["rank"],
                "val RMSE before fine tuning": round(summary["init_val_rmse_mean"], 3),
                "val RMSE after": round(summary["val_rmse_mean"], 3),
                "test RMSE after": round(summary["test_rmse_mean"], 3),
            }
        )
    return pd.DataFrame(rows)


def stability_table(payloads: list[dict]) -> pd.DataFrame:
    """Optimization behaviour per arm and rank, which Task 6 asks about by name.

    ``epochs to best`` against the 150-epoch cap separates a model that converged
    from one that was still improving when the budget ran out, and
    ``first epoch under 0.5`` is where an early plateau would show up.
    """
    rows = []
    for payload in payloads:
        for variant in payload["variants"]:
            histories = [pd.DataFrame(run["history"]) for run in variant["runs"]]
            crossings = []
            for history in histories:
                under = history[history["val_rmse"] < 0.5]
                crossings.append(int(under["epoch"].iloc[0]) if len(under) else -1)
            rows.append(
                {
                    "model": _payload_label(payload),
                    "TT rank": variant["rank"],
                    "epochs to best": variant["summary"]["best_epoch"],
                    "epochs run": variant["summary"]["epochs_run"],
                    "first epoch under 0.5 val": crossings,
                }
            )
    return pd.DataFrame(rows)


SERIES_STYLES = ((SERIES_1, "o"), (SERIES_2, "s"), (SERIES_3, "D"))
"""Three categorical slots, which is the palette's all-pairs cap for line forms."""

DODGE = 0.045
"""Sideways offset between series, as a fraction of x, so coincident points show."""


def plot_rank_curve(
    payloads: list[dict],
    dense: dict,
    ax=None,
    *,
    low_rank_payloads: list[dict] | None = None,
    title: str | None = None,
    takeaway: str | None = None,
) -> Figure:
    """Test RMSE against parameter count, with dense MAT as a reference band.

    Matrix controls reuse the training-path colors with dashed lines and hollow
    markers. All points use their actual parameter counts, without x offsets.
    Rank and compression are direct labels rather than a shared rank axis.
    """
    if len(payloads) > len(SERIES_STYLES):
        raise ValueError(
            f"{len(payloads)} series exceeds the {len(SERIES_STYLES)} validated slots. "
            f"Facet the figure or fold series together instead."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(8.2, 4.8) if low_rank_payloads else (6.5, 4.2))

    dense_mean = dense["summary"]["test_rmse_mean"]
    dense_spread = dense["summary"]["test_rmse_std"]

    ax.axhspan(
        dense_mean - dense_spread,
        dense_mean + dense_spread,
        color=INK_RECESSIVE,
        alpha=0.55,
        zorder=1,
    )
    ax.axhline(dense_mean, linestyle="--", linewidth=1, color=INK_SECONDARY, zorder=2)
    ax.text(
        0.02,
        0.02,
        f"dense MAT {dense_mean:.3f} +/- {dense_spread:.3f} seed SD",
        transform=ax.transAxes,
        fontsize=plt.rcParams["legend.fontsize"],
        color=INK_SECONDARY,
        va="bottom",
    )

    series = [
        (payload, color, marker, False)
        for payload, (color, marker) in zip(payloads, SERIES_STYLES, strict=False)
    ]
    for payload in low_rank_payloads or []:
        color, marker = {
            "low_rank_scratch": (SERIES_1, "^"),
            "low_rank_finetune": (SERIES_2, "v"),
        }[payload["arm"]]
        series.append((payload, color, marker, True))
    labels = {
        "4b": "TT, from scratch",
        "4a": "TT-SVD + fine-tuning",
        "low_rank_scratch": "low-rank matrix, from scratch",
        "low_rank_finetune": "truncated SVD + fine-tuning",
    }
    rank_bounds = {}
    for payload, color, marker, is_matrix in series:
        if low_rank_payloads and not is_matrix:
            color, marker = {"4b": (SERIES_1, "o"), "4a": (SERIES_2, "s")}[payload["arm"]]
        parameter_key = "low_rank_total_params" if is_matrix else "tt_total_params"
        variants = sorted(
            payload["variants"], key=lambda variant: variant["compression"][parameter_key]
        )
        parameters = [variant["compression"][parameter_key] for variant in variants]
        means = [variant["summary"]["test_rmse_mean"] for variant in variants]
        spreads = [variant["summary"]["test_rmse_std"] for variant in variants]
        label = labels[payload["arm"]] if low_rank_payloads else _payload_label(payload)
        ax.errorbar(
            parameters,
            means,
            yerr=spreads,
            color=color,
            linewidth=LINE_WIDTH,
            linestyle="--" if is_matrix else "-",
            marker=marker,
            markersize=MARKER_SIZE,
            markerfacecolor=SURFACE if is_matrix else color,
            markeredgecolor=color if is_matrix else SURFACE,
            markeredgewidth=SURFACE_RING,
            capsize=3,
            elinewidth=1,
            label=label,
            zorder=3 if is_matrix else 4,
        )
        for parameter_count, mean, spread, variant in zip(
            parameters, means, spreads, variants, strict=True
        ):
            key = (is_matrix, variant["rank"], parameter_count)
            bound = mean + spread if is_matrix else mean - spread
            previous, compression = rank_bounds.get(
                key, (bound, variant["compression"]["model_compression"])
            )
            rank_bounds[key] = (
                max(previous, bound) if is_matrix else min(previous, bound),
                compression,
            )

    for (is_matrix, rank, parameter_count), (bound, compression) in rank_bounds.items():
        ax.annotate(
            f"matrix r{rank}" if is_matrix else f"TT r{rank}\n{compression:.0f}x smaller",
            xy=(parameter_count, bound),
            xytext=(0, 8 if is_matrix else -9),
            textcoords="offset points",
            fontsize=plt.rcParams["xtick.labelsize"],
            color=INK_SECONDARY,
            ha="center",
            va="bottom" if is_matrix else "top",
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: _thousands(value)))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("total model parameters")
    ax.set_ylabel("test RMSE")
    ax.set_title(title or "Test RMSE by parameter count")
    rank_8_means = [
        variant["summary"]["test_rmse_mean"]
        for payload in payloads
        for variant in payload["variants"]
        if variant["rank"] == 8
    ]
    rank_8_takeaway = (
        "Rank 8 means are below dense; three seeds do not establish improvement or equivalence"
        if rank_8_means and all(value < dense_mean for value in rank_8_means)
        else "Three seeds do not establish improvement or equivalence"
    )
    _takeaway(
        ax,
        takeaway or rank_8_takeaway,
    )
    lower, upper = ax.get_ylim()
    ax.set_ylim(lower - 0.3 * (upper - lower), upper + 0.3 * (upper - lower))
    ax.margins(x=0.08)
    ax.legend(
        frameon=False,
        labelcolor=INK_SECONDARY,
        loc="upper right",
        ncol=2 if low_rank_payloads else 1,
    )
    _style_axes(ax)
    return ax.figure


SPEC_LABELS = {
    "attention": "attention only, 8 layers",
    "feed_forward": "feed forward only, 4 layers",
    "square": "the 12 square matrices",
    "embedding": "the embedding only",
    "square_embedding": "the 12 square matrices plus the embedding",
}

PROBE_DIRECT_LABELS = {
    "attention": "attention only",
    "square_embedding": "square + embedding",
}


def probe_table(probes: dict, scratch: dict) -> pd.DataFrame:
    """Varying which matrices are tensorized, at one rank, against the main target set.

    The reference row is the main from-scratch arm at the same rank, so the only
    thing differing across rows is the target set.
    """
    rank = probes["rank"]
    reference = next(v for v in scratch["variants"] if v["rank"] == rank)
    rows = [
        {
            "target set": SPEC_LABELS[scratch["spec"]],
            "layers": reference["compression"]["n_layers"],
            "total params": f"{reference['compression']['tt_total_params']:,}",
            "compression": f"{reference['compression']['model_compression']:.0f}x",
            "test RMSE": _rmse_cell(reference["summary"]),
        }
    ]
    for probe in probes["probes"]:
        rows.append(
            {
                "target set": SPEC_LABELS[probe["spec"]],
                "layers": probe["compression"]["n_layers"],
                "total params": f"{probe['compression']['tt_total_params']:,}",
                "compression": f"{probe['compression']['model_compression']:.0f}x",
                "test RMSE": _rmse_cell(probe["summary"]),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["rank"] = rank
    return table


def _variant_points(payloads: list[dict]) -> list[dict]:
    """Flatten TT payloads into plottable points, one per variant."""
    points = []
    for payload in payloads:
        for variant in payload["variants"]:
            points.append(
                {
                    "label": _payload_label(payload),
                    "rank": variant["rank"],
                    "params": variant["compression"]["tt_total_params"],
                    "compression": variant["compression"]["model_compression"],
                    "rmse": variant["summary"]["test_rmse_mean"],
                    "spread": variant["summary"]["test_rmse_std"],
                    "minutes": variant["summary"]["train_seconds_mean"] / 60,
                    "arm": payload["arm"],
                }
            )
    return points


def _low_rank_points(payloads: list[dict] | None) -> list[dict]:
    """Flatten low-rank matrix-factorization payloads into plottable points."""
    if not payloads:
        return []
    points = []
    for payload in payloads:
        for variant in payload["variants"]:
            points.append(
                {
                    "label": "low-rank matrix",
                    "rank": variant["rank"],
                    "params": variant["compression"]["low_rank_total_params"],
                    "compression": variant["compression"]["model_compression"],
                    "rmse": variant["summary"]["test_rmse_mean"],
                    "spread": variant["summary"]["test_rmse_std"],
                    "minutes": variant["summary"]["train_seconds_mean"] / 60,
                    "arm": payload["arm"],
                }
            )
    return points


def _low_rank_takeaway(tt_payloads: list[dict], low_rank_payloads: list[dict]) -> str:
    """Summarize the measured ordering without turning three seeds into a test."""
    tt_by_arm = {payload["arm"]: payload for payload in tt_payloads}
    low_rank_by_arm = {payload["arm"]: payload for payload in low_rank_payloads}
    comparisons = []
    overlaps = []
    for tt_arm, low_rank_arm in (("4b", "low_rank_scratch"), ("4a", "low_rank_finetune")):
        if tt_arm not in tt_by_arm or low_rank_arm not in low_rank_by_arm:
            continue
        tt_by_rank = {variant["rank"]: variant for variant in tt_by_arm[tt_arm]["variants"]}
        low_rank_by_rank = {
            variant["rank"]: variant for variant in low_rank_by_arm[low_rank_arm]["variants"]
        }
        for tt_rank, low_rank_ranks in ((4, (1, 2)), (8, (4, 5))):
            if tt_rank not in tt_by_rank or any(
                rank not in low_rank_by_rank for rank in low_rank_ranks
            ):
                continue
            tt_summary = tt_by_rank[tt_rank]["summary"]
            neighbours = [low_rank_by_rank[rank]["summary"] for rank in low_rank_ranks]
            comparisons.append(
                tt_summary["test_rmse_mean"]
                < min(summary["test_rmse_mean"] for summary in neighbours)
            )
            for summary in neighbours:
                overlaps.append(
                    tt_summary["test_rmse_mean"] - tt_summary["test_rmse_std"]
                    <= summary["test_rmse_mean"] + summary["test_rmse_std"]
                    and summary["test_rmse_mean"] - summary["test_rmse_std"]
                    <= tt_summary["test_rmse_mean"] + tt_summary["test_rmse_std"]
                )

    if comparisons and all(comparisons):
        ordering = (
            "TT means are lower in all four brackets"
            if len(comparisons) == 4
            else "TT means are lower in every measured bracket"
        )
    elif any(comparisons):
        ordering = "Mean ordering varies across the four brackets"
    else:
        ordering = "Low-rank matrix means are lower in all four brackets"
    uncertainty = (
        "most SD bars overlap"
        if overlaps and sum(overlaps) > len(overlaps) / 2
        else "SD overlap is mixed"
    )
    return f"{ordering}; {uncertainty} (n=3)"


PROBE_LABEL = "Rank 4 target-set probes"
"""The probes fold into one series. They are from-scratch TT runs too, and giving
each its own hue would exceed the palette's validated slots."""


def _probe_points(probes: dict | None) -> list[dict]:
    if not probes:
        return []
    return [
        {
            "label": PROBE_LABEL,
            "spec_label": SPEC_LABELS[probe["spec"]],
            "spec": probe["spec"],
            "rank": probe["rank"],
            "params": probe["compression"]["tt_total_params"],
            "compression": probe["compression"]["model_compression"],
            "rmse": probe["summary"]["test_rmse_mean"],
            "spread": probe["summary"]["test_rmse_std"],
            "minutes": probe["summary"]["train_seconds_mean"] / 60,
            "arm": "4b",
        }
        for probe in probes["probes"]
    ]


def _thousands(value: float) -> str:
    """15105 -> 15k, 799233 -> 799k. For axis ticks at the real values."""
    return f"{value / 1000:.0f}k"


def _parameter_axis(ax, params: list[int]) -> None:
    """A log parameter axis labelled at the measured parameter regions.

    Closely spaced low-parameter models are shown as an honest range rather than
    a row of colliding labels. The dedicated comparison keeps each actual value.
    """
    ax.set_xscale("log")
    clusters: list[list[int]] = []
    for value in sorted(set(params)):
        if (
            clusters
            and value < 100_000
            and clusters[-1][0] < 100_000
            and value / clusters[-1][0] <= 1.7
        ):
            clusters[-1].append(value)
        else:
            clusters.append([value])
    ticks = [math.sqrt(cluster[0] * cluster[-1]) for cluster in clusters]
    labels = [
        _thousands(cluster[0])
        if len(cluster) == 1
        else f"{cluster[0] / 1000:.0f}-{cluster[-1] / 1000:.0f}k"
        for cluster in clusters
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.minorticks_off()
    ax.set_xlabel("total parameters")


def _comparison_legend(
    ax,
    *,
    has_low_rank: bool,
    has_tt_fine_tuned: bool,
    has_low_rank_fine_tuned: bool,
    has_probes: bool,
    loc: str,
) -> None:
    """Use Figure 3's explicit model-and-training-path legend vocabulary."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=SERIES_1,
            markeredgecolor=SURFACE,
            markeredgewidth=SURFACE_RING,
            linestyle="none",
            label="TT, from scratch",
        )
    ]
    if has_tt_fine_tuned:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                markersize=MARKER_SIZE,
                markerfacecolor="none",
                markeredgecolor=SERIES_1,
                markeredgewidth=SURFACE_RING,
                linestyle="none",
                label="TT-SVD + fine-tuning",
            )
        )
    if has_low_rank:
        handles.append(
            Line2D(
                [],
                [],
                marker="^",
                markersize=MARKER_SIZE,
                markerfacecolor=SERIES_3,
                markeredgecolor=SURFACE,
                markeredgewidth=SURFACE_RING,
                linestyle="none",
                label="low-rank matrix, from scratch",
            )
        )
        if has_low_rank_fine_tuned:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="^",
                    markersize=MARKER_SIZE,
                    markerfacecolor="none",
                    markeredgecolor=SERIES_3,
                    markeredgewidth=SURFACE_RING,
                    linestyle="none",
                    label="truncated SVD + fine-tuning",
                )
            )
    if has_probes:
        handles.append(
            Line2D(
                [],
                [],
                marker="D",
                markersize=MARKER_SIZE,
                markerfacecolor=SERIES_2,
                markeredgecolor=SURFACE,
                markeredgewidth=SURFACE_RING,
                linestyle="none",
                label="selected TT targets, from scratch",
            )
        )
    ax.legend(handles=handles, frameon=False, labelcolor=INK_SECONDARY, loc=loc, ncol=2)


def plot_accuracy_against_size(
    payloads: list[dict],
    dense: dict,
    probes: dict | None = None,
    low_rank_payloads: list[dict] | None = None,
    forest: dict | None = None,
    ax=None,
) -> Figure:
    """Test RMSE against total parameter count, for every variant measured.

    The summary figure. Dense is a reference line with its seed spread as a band,
    anchored by a marker at its own parameter count so the reader can see how far
    left the TT models sit at the same height. The random forest is a line with no
    marker, because a forest's size is nodes and placing it on a parameter axis
    would invite a comparison that does not hold.

    Error bars are the spread across seeds. ``plot_paired_differences`` shows
    the same comparison with the split held identical, which turns out not to
    be any tighter.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 4.6))

    tt_points = _variant_points(payloads)
    low_rank_points = _low_rank_points(low_rank_payloads)
    probe_points = _probe_points(probes)
    points = tt_points + low_rank_points + probe_points
    dense_mean = dense["summary"]["test_rmse_mean"]
    dense_spread = dense["summary"]["test_rmse_std"]
    dense_params = dense["summary"]["total_params"]

    ax.set_xscale("log")
    ax.set_xlim(min(p["params"] for p in points) * 0.6, dense_params * 1.7)

    # Dense: band, line across the full width, and a marker at its own size.
    ax.axhspan(
        dense_mean - dense_spread,
        dense_mean + dense_spread,
        color=INK_RECESSIVE,
        alpha=0.5,
        zorder=1,
    )
    ax.axhline(dense_mean, linestyle="--", linewidth=1, color=INK_SECONDARY, zorder=2)
    ax.scatter(
        [dense_params],
        [dense_mean],
        s=MARKER_AREA + 30,
        color=INK_SECONDARY,
        marker="X",
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=4,
    )
    ax.annotate(
        f"dense MAT {dense_mean:.3f} +/- {dense_spread:.3f} SD\n{dense_params:,} params",
        xy=(dense_params, dense_mean),
        xytext=(-6, -34),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
        ha="right",
        linespacing=1.4,
    )

    if forest:
        forest_mean = forest["summary"]["test_rmse_mean"]
        ax.axhline(forest_mean, linestyle=":", linewidth=1.4, color=INK_SECONDARY, zorder=2)
        ax.text(
            math.sqrt(max(p["params"] for p in points) * dense_params),
            forest_mean + 0.004,
            f"random forest, 12 descriptors {forest_mean:.3f}",
            fontsize=9,
            color=INK_SECONDARY,
            va="bottom",
            ha="center",
        )

    if low_rank_payloads:
        families = (
            ("TT", tt_points, SERIES_1, "o"),
            ("low-rank matrix", low_rank_points, SERIES_3, "^"),
            (PROBE_LABEL, probe_points, SERIES_2, "D"),
        )
        for family, family_points, color, marker in families:
            for point_index, point in enumerate(family_points):
                fine_tuned = point["arm"] in ("4a", "low_rank_finetune")
                x_value = point["params"] * (1.025 if fine_tuned else 0.975)
                ax.errorbar(
                    [x_value],
                    [point["rmse"]],
                    yerr=[point["spread"]],
                    color=color,
                    elinewidth=1,
                    capsize=3,
                    marker=marker,
                    markersize=OVERVIEW_MARKER_SIZE,
                    markerfacecolor="none" if fine_tuned else color,
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    linestyle="none",
                    label=family if point_index == 0 else "_nolegend_",
                    zorder=3,
                )
                if fine_tuned:
                    continue
                if family == PROBE_LABEL:
                    direct_label = PROBE_DIRECT_LABELS.get(point["spec"], point["spec_label"])
                    if point["spec"] == "square_embedding":
                        direct_label = direct_label.replace(" + ", " +\n")
                    label_text = f"{direct_label}\nr{point['rank']}"
                    if point["spec"] == "square_embedding":
                        offset = (-10, 15)
                        horizontal = "right"
                    else:
                        offset = (10, 15)
                        horizontal = "left"
                else:
                    prefix = "TT r" if family == "TT" else "LR k"
                    label_text = f"{prefix}{point['rank']}"
                    if family == "TT":
                        offset = (0, -23)
                        horizontal = "center"
                    elif point["rank"] == 4:
                        offset = (-10, 15)
                        horizontal = "right"
                    elif point["rank"] == 5:
                        offset = (10, 15)
                        horizontal = "left"
                    else:
                        offset = (0, 15)
                        horizontal = "center"
                ax.annotate(
                    label_text,
                    xy=(x_value, point["rmse"]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=DIRECT_LABEL_SIZE,
                    color=color,
                    ha=horizontal,
                )
        n_groups = len(families)
    else:
        grouped: dict[str, list[dict]] = {}
        for point in points:
            grouped.setdefault(point["label"], []).append(point)
        if len(grouped) > len(SERIES_STYLES):
            raise ValueError(
                f"{len(grouped)} series exceeds the {len(SERIES_STYLES)} validated slots. "
                f"Facet the figure or fold series together instead."
            )

        center = (len(grouped) - 1) / 2
        for index, ((label, group), (color, marker)) in enumerate(
            zip(grouped.items(), SERIES_STYLES, strict=False)
        ):
            group = sorted(group, key=lambda p: p["params"])
            dodge = 1 + DODGE * (index - center)
            x_values = [p["params"] * dodge for p in group]
            ax.errorbar(
                x_values,
                [p["rmse"] for p in group],
                yerr=[p["spread"] for p in group],
                color=color,
                elinewidth=1,
                capsize=3,
                marker=marker,
                markersize=OVERVIEW_MARKER_SIZE,
                markeredgecolor=SURFACE,
                markeredgewidth=SURFACE_RING,
                linestyle="none",
                label="_nolegend_" if label == PROBE_LABEL else label,
                zorder=3,
            )
            for x_value, point in zip(x_values, group, strict=True):
                if label == PROBE_LABEL:
                    direct_label = PROBE_DIRECT_LABELS.get(point["spec"], point["spec_label"])
                    offset = (12, 16) if point["params"] < 50_000 else (-10, 18)
                    horizontal = "left" if point["params"] < 50_000 else "right"
                    ax.annotate(
                        f"{direct_label}\nrank {point['rank']}",
                        xy=(x_value, point["rmse"]),
                        xytext=offset,
                        textcoords="offset points",
                        fontsize=DIRECT_LABEL_SIZE,
                        color=color,
                        ha=horizontal,
                    )
                else:
                    ax.annotate(
                        f"r{point['rank']}",
                        xy=(x_value, point["rmse"]),
                        xytext=(0, -23 if index == 0 else 15),
                        textcoords="offset points",
                        fontsize=DIRECT_LABEL_SIZE,
                        color=color,
                        ha="center",
                    )
        n_groups = len(grouped)

    _parameter_axis(ax, [p["params"] for p in points] + [dense_params])
    ax.set_ylabel("test RMSE")
    ax.set_title("Test RMSE versus total model parameters")
    takeaway = (
        _low_rank_takeaway(payloads, low_rank_payloads)
        if low_rank_payloads
        else "Rank 8 means are below dense; three paired seeds do not establish improvement"
    )
    _takeaway(ax, takeaway)
    # Headroom for the legend, rather than letting it sit on the data. Three
    # long labels in the upper left covered the tallest error bar outright,
    # and moving them below the panel put them on the axis label.
    has_tt_fine_tuned = any(point["arm"] == "4a" for point in points)
    has_low_rank_fine_tuned = any(point["arm"] == "low_rank_finetune" for point in points)
    has_fine_tuned = has_tt_fine_tuned or has_low_rank_fine_tuned
    lower, upper = ax.get_ylim()
    ax.set_ylim(lower, upper + 0.12 * (n_groups + 2 * has_fine_tuned) * (upper - lower))
    _comparison_legend(
        ax,
        has_low_rank=bool(low_rank_points),
        has_tt_fine_tuned=has_tt_fine_tuned,
        has_low_rank_fine_tuned=has_low_rank_fine_tuned,
        has_probes=bool(probe_points),
        loc="upper left",
    )
    _style_axes(ax)
    ax.figure.tight_layout()
    return ax.figure


def plot_training_time(
    payloads: list[dict],
    dense: dict,
    probes: dict | None = None,
    low_rank_payloads: list[dict] | None = None,
    ax=None,
) -> Figure:
    """Minutes per seed against total parameters, the cost side of the trade.

    Shares the x axis of ``plot_accuracy_against_size``, so the two read as cost
    against benefit without ever putting two scales on one axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 3.6))

    dense_minutes = dense["summary"]["train_seconds_mean"] / 60
    dense_params = dense["summary"]["total_params"]
    tt_points = _variant_points(payloads)
    low_rank_points = _low_rank_points(low_rank_payloads)
    probe_points = _probe_points(probes)
    points = tt_points + low_rank_points + probe_points
    for point in points:
        point["pipeline_minutes"] = point["minutes"] + (
            dense_minutes if point["arm"] in ("4a", "low_rank_finetune") else 0.0
        )

    ax.axhline(dense_minutes, linestyle="--", linewidth=1, color=INK_SECONDARY, zorder=2)
    if low_rank_payloads:
        grouped = {
            "TT": (tt_points, SERIES_1, "o"),
            "low-rank matrix": (low_rank_points, SERIES_3, "^"),
            PROBE_LABEL: (probe_points, SERIES_2, "D"),
        }
    else:
        legacy_groups: dict[str, list[dict]] = {}
        for point in points:
            legacy_groups.setdefault(point["label"], []).append(point)
        if len(legacy_groups) > len(SERIES_STYLES):
            raise ValueError(
                f"{len(legacy_groups)} series exceeds the {len(SERIES_STYLES)} validated slots. "
                "Facet the figure or fold series together instead."
            )
        grouped = {
            label: (group, color, marker)
            for (label, group), (color, marker) in zip(
                legacy_groups.items(), SERIES_STYLES, strict=False
            )
        }
    for label, (group, color, marker) in grouped.items():
        fine_tuned = [point["arm"] in ("4a", "low_rank_finetune") for point in group]
        ax.scatter(
            [point["params"] for point in group],
            [point["pipeline_minutes"] for point in group],
            s=MARKER_AREA + 20,
            facecolors=["none" if is_fine_tuned else color for is_fine_tuned in fine_tuned],
            edgecolors=color,
            marker=marker,
            linewidth=1.5,
            zorder=3,
            label=label,
        )
        labelled = set()
        for point in group:
            label_key = point.get("spec", point["rank"])
            if label_key in labelled or point["arm"] in ("4a", "low_rank_finetune"):
                continue
            labelled.add(label_key)
            label_text = (
                PROBE_DIRECT_LABELS.get(point.get("spec"), point.get("spec_label", ""))
                if label == PROBE_LABEL
                else f"{'k' if label == 'low-rank matrix' else 'r'}{point['rank']}"
            )
            if point.get("spec") == "square_embedding":
                label_text = label_text.replace(" + ", " +\n")
            if label == PROBE_LABEL:
                if point.get("spec") == "square_embedding":
                    offset = (-8, 8)
                    horizontal = "right"
                else:
                    offset = (8, 13)
                    horizontal = "left"
            elif label == "low-rank matrix" and point["rank"] == 1:
                offset = (-7, -21)
                horizontal = "right"
            elif label == "low-rank matrix" and point["rank"] == 2:
                offset = (1, -21)
                horizontal = "center"
            elif label == "low-rank matrix" and point["rank"] == 4:
                offset = (-7, 14)
                horizontal = "right"
            elif label == "low-rank matrix" and point["rank"] == 5:
                offset = (8, -21)
                horizontal = "left"
            else:
                offset = (8, -19)
                horizontal = "left"
            ax.annotate(
                label_text,
                xy=(point["params"], point["pipeline_minutes"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=DIRECT_LABEL_SIZE,
                color=color,
                ha=horizontal,
            )
    ax.scatter(
        [dense_params],
        [dense_minutes],
        s=MARKER_AREA + 30,
        color=INK_SECONDARY,
        marker="X",
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=4,
        label="dense MAT",
    )

    _parameter_axis(ax, [p["params"] for p in points] + [dense_params])
    ax.set_xlim(min(p["params"] for p in points) * 0.6, dense_params * 1.7)
    ax.set_ylim(0, max(p["pipeline_minutes"] for p in points) * 1.25)
    ax.set_ylabel("training time (min)")
    ax.set_title("Training time versus total model parameters")
    _comparison_legend(
        ax,
        has_low_rank=bool(low_rank_points),
        has_tt_fine_tuned=any(point["arm"] == "4a" for point in points),
        has_low_rank_fine_tuned=any(point["arm"] == "low_rank_finetune" for point in points),
        has_probes=bool(probe_points),
        loc="upper right",
    )
    _style_axes(ax)
    ax.figure.tight_layout()
    return ax.figure


def plot_rank_learning_curves(payloads: list[dict], seed: int = 0, axes=None) -> Figure:
    """Validation RMSE against epoch, one panel per protocol, one line per rank.

    Small multiples rather than six lines in one frame, which would exceed the
    three categorical slots the palette validates. This is where Task 6's
    "early plateaus when training TT cores from scratch" is visible: under one
    shared learning rate, rank 8 stops improving within a handful of epochs,
    while the same rank under its own rate trains normally.
    """
    if axes is None:
        _, axes = plt.subplots(1, len(payloads), figsize=(10, 3.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    ranks = sorted({v["rank"] for p in payloads for v in p["variants"]})
    if len(ranks) > len(SERIES_STYLES):
        raise ValueError(f"{len(ranks)} ranks exceeds the {len(SERIES_STYLES)} validated slots")

    for ax, payload in zip(axes, payloads, strict=True):
        for variant in sorted(payload["variants"], key=lambda v: v["rank"]):
            run = next((r for r in variant["runs"] if r["seed"] == seed), None)
            if run is None:
                continue
            history = pd.DataFrame(run["history"])
            color, _ = SERIES_STYLES[ranks.index(variant["rank"])]
            ax.plot(
                history["epoch"],
                history["val_rmse"],
                color=color,
                linewidth=1.4,
                solid_capstyle="round",
                label=f"rank {variant['rank']}",
                zorder=3,
            )
            # One mark on the epoch that was actually selected.
            best = int(run["best_epoch"])
            ax.scatter(
                [best],
                [history.loc[history["epoch"] == best, "val_rmse"].iloc[0]],
                s=MARKER_AREA,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=4,
            )
        if payload.get("lr_per_rank"):
            rank_lr = payload["lr_per_rank"].get(str(ranks[-1]))
            panel_title = f"Per-rank LR (rank {ranks[-1]} = {rank_lr:.0e})"
        elif payload.get("lr"):
            panel_title = f"Shared LR = {payload['lr']:.0e}"
        else:
            panel_title = _payload_label(payload)
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("epoch")
        _style_axes(ax)

    axes[0].set_ylabel("validation RMSE")
    axes[0].set_ylim(0.25, 1.05)
    axes[0].legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper right")
    figure = axes[0].figure
    figure.suptitle(
        f"Rank is sensitive to learning rate (seed {seed}, rank {ranks[-1]})",
        x=0.125,
        y=0.99,
        ha="left",
        color=INK_PRIMARY,
        fontsize=plt.rcParams["axes.titlesize"],
        fontweight=plt.rcParams["axes.titleweight"],
    )
    figure.patch.set_facecolor(PAGE)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    shared_run = next(
        run
        for run in next(v for v in payloads[0]["variants"] if v["rank"] == ranks[-1])["runs"]
        if run["seed"] == seed
    )
    tuned_run = next(
        run
        for run in next(v for v in payloads[1]["variants"] if v["rank"] == ranks[-1])["runs"]
        if run["seed"] == seed
    )
    shared_history = pd.DataFrame(shared_run["history"])
    tuned_history = pd.DataFrame(tuned_run["history"])
    shared_best = shared_history.loc[
        shared_history["epoch"] == shared_run["best_epoch"], "val_rmse"
    ].iloc[0]
    tuned_best = tuned_history.loc[
        tuned_history["epoch"] == tuned_run["best_epoch"], "val_rmse"
    ].iloc[0]
    _figure_takeaway(
        figure,
        f"Rank {ranks[-1]} best validation RMSE improves from {shared_best:.3f} "
        f"to {tuned_best:.3f} after validation-based LR tuning",
    )
    return figure


def paired_differences(payloads: list[dict], dense: dict) -> pd.DataFrame:
    """Per-seed difference between each variant and dense on the identical split.

    Seeds fix the split, so a variant and the baseline are scored on exactly the
    same held-out molecules and the comparison is paired. The intent is that the
    split's difficulty cancels on subtraction.

    Cancelling requires the two models to move together across seeds. The table
    reports the paired differences directly so the small sample is visible.
    """
    dense_by_seed = {run.seed: run.test_rmse for run in dense["runs"]}
    rows = []
    for payload in payloads:
        for variant in payload["variants"]:
            for run in variant["runs"]:
                if run["seed"] not in dense_by_seed:
                    raise ValueError(f"no dense run for seed {run['seed']} to pair against")
                rows.append(
                    {
                        "model": _payload_label(payload),
                        "rank": variant["rank"],
                        "seed": run["seed"],
                        "difference": run["test_rmse"] - dense_by_seed[run["seed"]],
                    }
                )
    return pd.DataFrame(rows)


def paired_difference_table(payloads: list[dict], dense: dict) -> pd.DataFrame:
    """Mean and spread of the paired differences, beside the unpaired spread."""
    frame = paired_differences(payloads, dense)
    rows = []
    for (model, rank), group in frame.groupby(["model", "rank"], sort=False):
        differences = group["difference"].tolist()
        rows.append(
            {
                "model": model,
                "TT rank": rank,
                "mean difference from dense": round(sum(differences) / len(differences), 4),
                "spread of differences": round(_spread(differences), 4),
                "worst single seed": round(max(differences, key=abs), 4),
            }
        )
    return pd.DataFrame(rows)


def plot_paired_differences(payloads: list[dict], dense: dict, ax=None) -> Figure:
    """Each variant's per-seed difference from dense, with the mean.

    These dots measure what changes when dense is swapped for TT on the same
    molecules, which is the quantity the comparison is about. The band is the
    dense baseline's own seed spread, and it is the point of the figure: if
    pairing had removed the split's difficulty the differences would sit well
    inside it, and they do not.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    frame = paired_differences(payloads, dense)
    groups = list(frame.groupby(["model", "rank"], sort=False))
    labels = []

    # The noise pairing was supposed to remove, drawn to the same scale as what
    # is left after removing it. Taken from the runs rather than from a stored
    # summary, so this works on any set of runs handed to it.
    dense_spread = _spread([run.test_rmse for run in dense["runs"]])
    ax.axvspan(-dense_spread, dense_spread, color=INK_RECESSIVE, alpha=0.5, zorder=1)
    ax.axvline(0, linestyle="--", linewidth=1, color=INK_SECONDARY, zorder=2)
    for position, ((model, rank), group) in enumerate(reversed(groups)):
        differences = group["difference"].tolist()
        mean = sum(differences) / len(differences)
        color, marker = SERIES_STYLES[0] if "scratch" in model else SERIES_STYLES[1]
        ax.scatter(
            differences,
            [position] * len(differences),
            s=MARKER_AREA,
            color=color,
            alpha=0.55,
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            [mean],
            [position],
            s=MARKER_AREA + 50,
            color=color,
            marker=marker,
            edgecolor=SURFACE,
            linewidth=1.8,
            zorder=4,
        )
        short = "from scratch" if "scratch" in model else "fine tuned"
        labels.append(f"{short}, rank {rank}")

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_xlabel("test RMSE difference")
    ax.set_title("Scoring on identical molecules did not sharpen the comparison")
    means = [group["difference"].mean() for _, group in groups]
    _takeaway(
        ax,
        f"Means spread over {max(abs(m) for m in means):.3f} RMSE, no narrower than "
        f"the {dense_spread:.3f} seed noise the pairing was meant to remove",
    )
    ax.text(
        0.004,
        len(groups) - 0.45,
        "worse than dense",
        fontsize=9,
        color=INK_SECONDARY,
        va="top",
    )
    ax.text(
        -0.004,
        len(groups) - 0.45,
        "better",
        fontsize=9,
        color=INK_SECONDARY,
        va="top",
        ha="right",
    )
    _style_axes(ax, grid="x")
    ax.figure.tight_layout()
    return ax.figure


SETTING_SOURCES = {
    "d_model": "the case study names 128 or 256, and 256 is the one it tensorizes as 4x4x4x4",
    "n_layers": "the case study suggests 2 to 3",
    "n_heads": "MAT's default",
    "batch_size": "the case study suggests 16 to 32",
    "dropout": "MAT's default",
    "distance_matrix_kernel": "changed from MAT's default, see the note below",
    "max_epochs": "our choice, a cap early stopping rarely reaches",
    "patience": "our choice",
    "grad_clip": "MAT's default",
    "weight_decay": "our choice, none",
    "scheduler": "our choice, replacing MAT's Noam schedule at this smaller batch size",
    "seeds": "the case study suggests 3 if feasible",
    "n_modes": "the case study's suggested 4x4x4x4 for d_model 256",
    "ranks": "the case study suggests 2, 4, 8",
}
"""Where each fixed setting came from. Kept beside the table rather than in the
notebook so it is versioned with the code that reports it."""


def _sweep_row(name: str, chosen, candidates, note: str) -> dict:
    grid = ", ".join(f"{c:.0e}" for c in candidates)
    return {
        "setting": name,
        "value": f"{chosen:.0e}",
        "how it was decided": f"swept over {grid}; {note}",
    }


def protocol_table(scope: str = "training") -> pd.DataFrame:
    """Every setting, its value, and how it was decided.

    Split by scope so each half can appear where it bites: "training" covers the
    data, the model and the dense training budget, and "tt" covers the
    tensorization and the two TT learning rates. Keeping the TT rows out of the
    first table is what stops the notebook naming TT ranks before TT exists.

    Built from the config objects and the committed sweep files rather than
    typed out, so it cannot drift from what actually ran. The distinction that
    matters is between values taken on faith and values searched: only the
    learning rates were searched, each on seed 0 validation alone.
    """
    if scope not in ("training", "tt"):
        raise ValueError(f"unknown scope '{scope}', expected 'training' or 'tt'")

    from mat_tt.model import ModelConfig
    from mat_tt.train import TrainConfig

    model, budget = ModelConfig(), TrainConfig()
    if scope == "training":
        rows = [
            ("d_model", model.d_model, "d_model"),
            ("encoder layers", model.n_layers, "n_layers"),
            ("attention heads", model.n_heads, "n_heads"),
            ("dropout", model.dropout, "dropout"),
            ("distance kernel", model.distance_matrix_kernel, "distance_matrix_kernel"),
            ("batch size", budget.batch_size, "batch_size"),
            ("epoch cap", budget.max_epochs, "max_epochs"),
            ("early stopping patience", budget.patience, "patience"),
            ("gradient clip", budget.grad_clip, "grad_clip"),
            ("weight decay", budget.weight_decay, "weight_decay"),
            ("schedule", budget.scheduler, "scheduler"),
            ("seeds", "0, 1, 2", "seeds"),
        ]
    else:
        rows = [
            ("TT modes", "4 x 4 x 4 x 4", "n_modes"),
            ("TT ranks", "2, 4, 8", "ranks"),
        ]

    table = [
        {"setting": name, "value": str(value), "how it was decided": SETTING_SOURCES[key]}
        for name, value, key in rows
    ]

    if scope == "training":
        dense = json.loads(results_path("task2_lr_sweep.json").read_text())
        table.append(
            _sweep_row(
                "learning rate, dense",
                dense["chosen_lr"],
                [c["lr"] for c in dense["candidates"]],
                f"best on seed 0 validation at {dense['epochs']} epochs",
            )
        )
    else:
        tt = json.loads(results_path("task4_lr_sweep.json").read_text())
        for arm, payload in tt["arms"].items():
            table.append(
                _sweep_row(
                    f"learning rate, TT {arm}",
                    payload["chosen_lr"],
                    [c["lr"] for c in payload["candidates"]],
                    f"best on seed 0 validation at {tt['epochs']} epochs",
                )
            )
    return pd.DataFrame(table)


def plot_compression_targets(inventory, total_params: int, ax=None) -> Figure:
    """Which of MAT's linear layers are replaced, sized by parameter count.

    ``total_params`` is the whole model's count rather than the linear layers'
    sum, so the shares match the ones quoted elsewhere. The difference is the
    layer norms, which get their own segment rather than being silently dropped
    out of the denominator.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, 4.2))

    replaced = {"W_Q", "W_K", "W_V", "W_O", "feed_forward"}
    groups: dict[str, list] = {}
    for entry in inventory:
        groups.setdefault(entry.role, []).append(entry)

    order = sorted(groups, key=lambda role: -sum(e.params for e in groups[role]))
    total = total_params
    left = 0.0
    for role in order:
        params = sum(e.params for e in groups[role])
        width = params / total
        is_replaced = role in replaced
        ax.add_patch(
            plt.Rectangle(
                (left, 0),
                width,
                1,
                facecolor=SERIES_1 if is_replaced else INK_RECESSIVE,
                edgecolor=SURFACE,
                linewidth=2,
            )
        )
        if width > 0.06:
            ax.text(
                left + width / 2,
                0.5,
                f"{role}\n{len(groups[role])} layers\n{width:.1%}",
                fontsize=9.5,
                ha="center",
                va="center",
                color="white" if is_replaced else INK_SECONDARY,
                linespacing=1.6,
            )
        else:
            ax.annotate(
                f"{role}, {params:,} params",
                xy=(left + width / 2, 1),
                xytext=(0, 14 + 22 * (order.index(role) % 2)),
                textcoords="offset points",
                fontsize=9,
                color=INK_SECONDARY,
                ha="center",
                arrowprops={"arrowstyle": "-", "linewidth": 0.8, "color": INK_RECESSIVE},
            )
        left += width

    # Whatever the linear layers do not account for: layer norms, mostly.
    other = total - sum(e.params for entry_list in groups.values() for e in entry_list)
    if other > 0:
        ax.add_patch(
            plt.Rectangle(
                (left, 0),
                other / total,
                1,
                facecolor=INK_RECESSIVE,
                edgecolor=SURFACE,
                linewidth=2,
            )
        )
        ax.annotate(
            f"norms, {other:,} params",
            xy=(left + other / total / 2, 1),
            xytext=(0, 58),
            textcoords="offset points",
            fontsize=9,
            color=INK_SECONDARY,
            ha="center",
            arrowprops={"arrowstyle": "-", "linewidth": 0.8, "color": INK_RECESSIVE},
        )

    replaced_share = (
        sum(e.params for role in groups if role in replaced for e in groups[role]) / total
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.85)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the model's parameters")
    ax.set_title(f"Blue is replaced by TT: {replaced_share:.1%} of the model")
    tensorized = [e for role, entries in groups.items() if role in replaced for e in entries]
    shapes = {(e.in_features, e.out_features) for e in tensorized}
    shape = f"{next(iter(shapes))[0]}x{next(iter(shapes))[1]}" if len(shapes) == 1 else "square"
    _takeaway(ax, f"{len(tensorized)} matrices, every one of them {shape}, carry almost all of it")
    _style_axes(ax, grid=None)
    ax.figure.tight_layout()
    return ax.figure


def load_payload(name: str) -> dict:
    """Read one committed result file by stem."""
    return json.loads(results_path(f"{name}.json").read_text())


def _figure_dense_learning_curve() -> Figure:
    return plot_learning_curve(load_results(results_path("task2_dense_baseline.json"))["runs"][0])


def _figure_dense_parity() -> Figure:
    return plot_parity(load_results(results_path("task2_dense_baseline.json"))["runs"][0])


def _figure_tt_error() -> Figure:
    return plot_tt_error(load_payload("task3_weight_analysis"))


def _figure_rank_curve() -> Figure:
    # The shared-rate protocol is deliberately absent. It belongs in the table
    # and in plot_rank_learning_curves, where its rank 8 collapse is the point,
    # but drawn here as a peer series it read as "TT degrades with rank", which
    # is the opposite of the conclusion.
    return plot_rank_curve(
        [
            load_payload("task4_tt_scratch_per_rank_lr"),
            load_payload("task4_tt_finetune"),
        ],
        load_payload("task2_dense_baseline"),
        low_rank_payloads=[
            load_payload("task7_low_rank_scratch"),
            load_payload("task7_low_rank_finetune"),
        ],
    )


def _figure_accuracy_against_size() -> Figure:
    return plot_accuracy_against_size(
        [load_payload("task4_tt_scratch_per_rank_lr"), load_payload("task4_tt_finetune")],
        load_payload("task2_dense_baseline"),
        probes=load_payload("task4_probes"),
        low_rank_payloads=[
            load_payload("task7_low_rank_scratch"),
            load_payload("task7_low_rank_finetune"),
        ],
        forest=load_payload("task7_random_forest"),
    )


def _figure_rank_learning_curves() -> Figure:
    return plot_rank_learning_curves(
        [load_payload("task4_tt_scratch"), load_payload("task4_tt_scratch_per_rank_lr")]
    )


def _figure_paired_differences() -> Figure:
    return plot_paired_differences(
        [load_payload("task4_tt_scratch_per_rank_lr"), load_payload("task4_tt_finetune")],
        load_results(results_path("task2_dense_baseline.json")),
    )


def _figure_training_time() -> Figure:
    return plot_training_time(
        [load_payload("task4_tt_scratch_per_rank_lr"), load_payload("task4_tt_finetune")],
        load_payload("task2_dense_baseline"),
        probes=load_payload("task4_probes"),
        low_rank_payloads=[
            load_payload("task7_low_rank_scratch"),
            load_payload("task7_low_rank_finetune"),
        ],
    )


def _figure_compression_targets() -> Figure:
    from mat_tt.model import ModelConfig, build_model, count_parameters, linear_inventory

    model = build_model(ModelConfig())
    return plot_compression_targets(linear_inventory(model), count_parameters(model).total)


FIGURES: dict[str, Callable[[], Figure]] = {
    "dense_learning_curve": _figure_dense_learning_curve,
    "tt_reconstruction_error": _figure_tt_error,
    "tt_rank_curve": _figure_rank_curve,
    "accuracy_against_size": _figure_accuracy_against_size,
    "rank_learning_curves": _figure_rank_learning_curves,
    "training_time": _figure_training_time,
}
"""Primary figures in notebook order, each loading its own results.

``scripts/export_figures.py`` renders these to files. A figure added here is
exported without touching the script, and the notebook calls the same plotting
helpers on the same payloads, so the two cannot drift apart.
"""


RUN_CURVE_DIR = "runs"
"""Where the per-run appendix curves go, under the figures directory."""

PAYLOAD_LABELS = {"dense_baseline": DENSE_LABEL}
"""How a results file's own label is named in a figure, where the two differ.

The same job ``ARM_LABELS`` does for the TT arms and ``SPEC_LABELS`` for the
probe target sets. Anything absent falls back to its label with the underscores
taken out, which is what an experiment branch's own label needs."""

TRAINING_RESULTS = (
    "task2_dense_baseline",
    "task4_tt_scratch",
    "task4_tt_scratch_per_rank_lr",
    "task4_tt_finetune",
    "task7_low_rank_scratch",
    "task7_low_rank_finetune",
    "task4_probes",
)
"""The committed result files that contain trained runs, in notebook order.

``task7_random_forest`` is absent on purpose: a forest has no epochs, so there
is no curve to draw. ``enumerate_runs`` would skip it anyway, since it skips any
run without a history, but naming only the files that have one keeps the
appendix honest about what it covers.
"""


def enumerate_runs(payload: dict, stem: str):
    """Yield ``(slug, run, label)`` for every trained run in one results file.

    Three payload shapes are committed and all three are handled here: a bare
    ``runs`` list for the dense baseline, ``variants`` keyed by TT rank for the
    three TT arms, and ``probes`` keyed by target set for the probe runs. A run
    with no history is skipped, so this is safe to call on any results file.

    The slug carries the file stem, so two protocols that share a rank and a
    seed still land in separate files, and so a label invented by
    ``scripts/run_variant.py`` on an experiment branch works without a change
    here.
    """
    if "variants" in payload:
        groups = [
            (f"rank{v['rank']}", f"{_payload_label(payload)}, rank {v['rank']}", v["runs"])
            for v in payload["variants"]
        ]
    elif "probes" in payload:
        groups = [
            (
                f"{p['spec']}_rank{p['rank']}",
                f"{SPEC_LABELS[p['spec']]}, rank {p['rank']}",
                p["runs"],
            )
            for p in payload["probes"]
        ]
    else:
        raw = payload.get("display_label") or payload.get("label") or stem
        groups = [("", PAYLOAD_LABELS.get(raw, raw.replace("_", " ")), payload["runs"])]

    for suffix, label, runs in groups:
        for run in runs:
            run = run if isinstance(run, dict) else run.as_dict()
            if not run.get("history"):
                continue
            parts = [stem, suffix, f"seed{run['seed']}"]
            yield "_".join(p for p in parts if p), RunResult(**run), label


def run_curve_takeaway(run: RunResult) -> str:
    """The run's own record, so a single appendix PNG is self-describing.

    Every value comes from the run itself, learning rate included, so a curve
    cannot disagree with the tables built from the same file. It is also what
    explains the identical pairs in the set without a word of prose: the two
    from-scratch protocols share their rank 2 and rank 4 runs exactly, because
    the swept rate landed on the shared rate at those ranks.
    """
    lr = run.train_config.get("lr")
    # Reports what the run recorded and stays quiet about what it did not,
    # rather than printing a placeholder rate that could be read as real.
    rate = f"lr {lr:.0e}, " if lr else ""
    return (
        f"{rate}best val {run.val_rmse:.3f} at epoch {run.best_epoch} "
        f"of {run.epochs_run}, test {run.test_rmse:.3f}, "
        f"{run.train_seconds / 60:.1f} min"
    )


def plot_run_curve(run: RunResult, label: str, ax=None) -> Figure:
    """One appendix curve: log axis, the run's own record under the title."""
    return plot_learning_curve(run, ax=ax, label=label, takeaway=run_curve_takeaway(run), log=True)


def figures_path(*parts: str) -> Path:
    """Path inside the gitignored figures directory, resolved from the package.

    The counterpart of ``results_path``, so a caller does not have to know where
    the repository root is.
    """
    return Path(__file__).resolve().parents[2] / "figures" / Path(*parts)


def export_run_curves(
    results_file: Path | str, directory: Path | str | None = None, prefix: str = ""
) -> list[Path]:
    """Render every run in one results file to ``directory``.

    A writer in a module that otherwise only builds figures, because both
    callers need exactly this: ``scripts/export_figures.py`` rebuilds the whole
    appendix from the committed files, and each training script calls it on the
    file it has just written so a new run gets its curve without a separate
    step. Putting it here is what stops those two paths drifting.
    """
    results_file = Path(results_file)
    payload = json.loads(results_file.read_text())
    directory = Path(directory) if directory else figures_path(RUN_CURVE_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    written = []
    for slug, run, label in enumerate_runs(payload, results_file.stem):
        figure = plot_run_curve(run, label)
        path = directory / f"{prefix}{slug}.png"
        figure.savefig(path, dpi=110, bbox_inches="tight", facecolor=figure.get_facecolor())
        plt.close(figure)
        written.append(path)
    return written


def plot_all_run_curves(payloads: list[tuple[str, dict]], columns: int = 6) -> Figure:
    """Every run in the study as small multiples on one shared log axis.

    The page to hand over when someone asks whether the other seeds behaved.
    Deliberately not in ``FIGURES``: thirty-six panels belong in an appendix,
    not in a notebook built around eleven figures with one idea each.
    """
    entries = [
        (slug, run, label)
        for stem, payload in payloads
        for slug, run, label in enumerate_runs(payload, stem)
    ]
    rows = math.ceil(len(entries) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.5 * columns, 2.0 * rows), sharex=True, sharey=True
    )
    flat = np.atleast_1d(axes).ravel()

    for ax, (_slug, run, _label) in zip(flat, entries, strict=False):
        history = pd.DataFrame(run.history)
        ax.axhline(MEAN_PREDICTOR_RMSE, linestyle="--", linewidth=0.8, color=INK_SECONDARY)
        ax.plot(history["epoch"], history["train_rmse"], color=SERIES_1, linewidth=1.0)
        ax.plot(history["epoch"], history["val_rmse"], color=SERIES_2, linewidth=1.0)
        ax.scatter([run.best_epoch], [run.val_rmse], s=14, color=SERIES_2, zorder=3)
        # The slug, not the display label: this page is an index into the files
        # beside it, so the caption has to be the filename.
        ax.set_title("")
        ax.set_yscale("log")
        _style_axes(ax)

    for ax in flat[len(entries) :]:
        ax.set_visible(False)

    flat[0].set_ylim(0.15, 4.0)
    figure.patch.set_facecolor(PAGE)
    figure.tight_layout()
    # The header is placed against the grid rather than by _figure_takeaway,
    # whose offsets are fractions of the figure and open a hand's width of
    # blank page on something this tall.
    figure.subplots_adjust(top=1 - 0.85 / figure.get_size_inches()[1])
    return figure
