"""Measure how well TT approximates MAT's trained weight matrices.

It needs no training: decompose the trained matrices, measure the reconstruction
error, and compare against two controls.

This is one read on the case study's premise, that molecular representations
carry structure a low rank tensor format can capture, and a narrow one. It covers
the 12 square matrices under the suggested 4x4x4x4 tensorization, so what it can
support is that this tensorization does not fit these weights. The embedding is
left out, and it is the matrix whose input axis indexes kinds of chemical
information, so it is where the premise is most directly testable. Task 4
measures it.

Control 1, truncated SVD at matched parameter count. If TT does not beat
low-rank matrix factorization for the same budget, the tensor structure is not earning its
place and the hypothesis is about parameter count rather than structure.

Control 2, the same measurement on an untrained model. If trained and random
matrices compress equally well, training did not create the structure and any
compressibility is a property of the shape, not of the chemistry.

Writes results/task3_weight_analysis.json for the notebook to render.
"""

import json
import logging
import sys

import torch

from mat_tt.analysis import (
    plain_low_rank_approximation,
    spectrum_summary,
    tt_approximation,
)
from mat_tt.model import ModelConfig, build_model, linear_inventory
from mat_tt.train import CHECKPOINT_DIR, RESULTS_DIR
from mat_tt.tt import factorize_dim, tt_parameter_count

RANKS = (2, 4, 8, 16)
N_MODES = 4
SEED = 0


def analyse(state_dict: dict, model: torch.nn.Module, label: str) -> list[dict]:
    """One row per (matrix, rank), with both controls attached."""
    rows = []
    square = [entry for entry in linear_inventory(model) if entry.is_square]

    for entry in square:
        weight = state_dict[f"{entry.path}.weight"]
        spectrum = spectrum_summary(weight)
        for rank in RANKS:
            error, retained_norm, cores = tt_approximation(weight, rank, n_modes=N_MODES)
            plain_error, plain_rank, plain_cost = plain_low_rank_approximation(weight, cores)
            rows.append(
                {
                    "source": label,
                    "path": entry.path,
                    "role": entry.role,
                    "layer": int(entry.path.split(".")[2]) if "layers" in entry.path else -1,
                    "tt_rank": rank,
                    "tt_params": cores,
                    "tt_error": error,
                    "tt_retained_norm": retained_norm,
                    "plain_rank": plain_rank,
                    "plain_params": plain_cost,
                    "plain_error": plain_error,
                    "dense_params": entry.weight_params,
                    "stable_rank": spectrum["stable_rank"],
                    "energy_top_16": spectrum["energy_top_16"],
                    "energy_top_64": spectrum["energy_top_64"],
                }
            )
    return rows


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    checkpoint_path = CHECKPOINT_DIR / f"dense_seed{SEED}.pt"
    if not checkpoint_path.is_file():
        print(f"no checkpoint at {checkpoint_path}, run scripts/run_task2.py first")
        return 1

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    print(
        f"checkpoint: seed {checkpoint['seed']}, epoch {checkpoint['best_epoch']}, "
        f"test RMSE {checkpoint['test_rmse']:.4f}"
    )

    trained = build_model(model_config)
    trained.load_state_dict(checkpoint["state_dict"])

    torch.manual_seed(1234)
    untrained = build_model(model_config)

    rows = analyse(checkpoint["state_dict"], trained, "trained")
    rows += analyse(untrained.state_dict(), untrained, "untrained")

    torch.manual_seed(0)
    d = model_config.d_model
    random_reference = spectrum_summary(torch.randn(d, d))

    modes = factorize_dim(d, N_MODES)
    payload = {
        "random_matrix_reference": random_reference,
        "seed": SEED,
        "d_model": d,
        "modes": list(modes),
        "n_modes": N_MODES,
        "ranks": list(RANKS),
        "checkpoint": {
            "best_epoch": checkpoint["best_epoch"],
            "val_rmse": checkpoint["val_rmse"],
            "test_rmse": checkpoint["test_rmse"],
        },
        "cost_table": [
            {
                "rank": rank,
                "tt_params": tt_parameter_count(modes, modes, rank),
                "dense_params": d * d,
            }
            for rank in RANKS
        ],
        "rows": rows,
    }
    path = RESULTS_DIR / "task3_weight_analysis.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {path} with {len(rows)} rows")

    trained_rows = [r for r in rows if r["source"] == "trained"]
    mean_stable = sum(r["stable_rank"] for r in trained_rows) / len(trained_rows)
    print(
        f"\nstable rank: trained matrices average {mean_stable:.1f}, "
        f"a random matrix of the same shape is {random_reference['stable_rank']:.1f} "
        f"(full rank {d})"
    )

    # Summary to stdout so the run is readable without opening the notebook.
    print(f"\nmean relative error over the 12 square matrices, {d}x{d} as {modes}")
    print(
        f"{'rank':>5} {'TT params':>10} {'TT trained':>11} {'TT untrained':>13} "
        f"{'plain rank':>11} {'plain trained':>14}"
    )
    for rank in RANKS:
        selected = [r for r in rows if r["tt_rank"] == rank]
        trained_rows = [r for r in selected if r["source"] == "trained"]
        untrained_rows = [r for r in selected if r["source"] == "untrained"]
        tt_t = sum(r["tt_error"] for r in trained_rows) / len(trained_rows)
        tt_u = sum(r["tt_error"] for r in untrained_rows) / len(untrained_rows)
        plain_rank = trained_rows[0]["plain_rank"]
        plain_values = [r["plain_error"] for r in trained_rows if plain_rank >= 1]
        plain = sum(plain_values) / len(plain_values) if plain_values else float("nan")
        print(
            f"{rank:>5} {trained_rows[0]['tt_params']:>10,} {tt_t:>11.3f} {tt_u:>13.3f} "
            f"{plain_rank:>11} {plain:>14.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
