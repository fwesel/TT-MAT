"""Validation-based hyperparameter selection helpers."""

from __future__ import annotations


def learning_rate_choice(rows: list[dict]) -> dict:
    """Select the best validation row and identify a truncated grid."""
    if len(rows) < 3:
        raise ValueError("a learning-rate grid needs at least three candidates")
    ordered = sorted(rows, key=lambda row: row["lr"])
    rates = [row["lr"] for row in ordered]
    if len(set(rates)) != len(rates):
        raise ValueError("learning-rate candidates must be unique")

    chosen = min(ordered, key=lambda row: row["val_rmse"])
    index = ordered.index(chosen)
    edge = "lower" if index == 0 else "upper" if index == len(ordered) - 1 else None
    return {
        "chosen_lr": chosen["lr"],
        "chosen_at_grid_edge": edge is not None,
        "chosen_grid_edge": edge,
    }


def require_interior_choice(choice: dict, label: str) -> None:
    """Reject a truncated sweep before it can produce final runs."""
    edge = choice.get("chosen_grid_edge")
    if edge:
        raise RuntimeError(
            f"{label} selected {choice['chosen_lr']:.0e} at the {edge} edge of its grid; "
            "extend the grid before running final models"
        )
