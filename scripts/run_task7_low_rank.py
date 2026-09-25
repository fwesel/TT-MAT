"""Train low-rank matrix-factorization MAT controls at TT-bracketing parameter budgets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch import nn

from mat_tt.compress import low_rank_compression_report, low_rank_model
from mat_tt.data import (
    DedupResult,
    FeaturizedDataset,
    Splits,
    canonicalize_and_dedupe,
    featurize_molecules,
    make_splits,
)
from mat_tt.model import ModelConfig, build_model
from mat_tt.run_curves import save_run_curves
from mat_tt.train import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    TrainConfig,
    load_dense_checkpoint,
    run_seeds,
    summarize,
    train_one,
)
from mat_tt.tuning import learning_rate_choice, require_interior_choice

SPEC = "square"
RANKS = (1, 2, 4, 5)
SEEDS = (0, 1, 2)
ARMS = ("scratch", "finetune")
LR_CANDIDATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
SWEEP_EPOCHS = 60


def dense_checkpoint_path(seed: int) -> Path:
    return CHECKPOINT_DIR / f"dense_seed{seed}.pt"


def scratch_factory(rank: int):
    def factory(seed: int, splits: Splits) -> nn.Module:
        model, _ = low_rank_model(build_model(ModelConfig()), spec=SPEC, rank=rank)
        return model

    return factory


def finetune_factory(rank: int, features: FeaturizedDataset):
    def factory(seed: int, splits: Splits) -> nn.Module:
        dense = load_dense_checkpoint(dense_checkpoint_path(seed), features, splits)
        model, _ = low_rank_model(dense, spec=SPEC, rank=rank, mode="svd")
        return model

    return factory


def compression_at(rank: int) -> dict:
    dense = build_model(ModelConfig())
    factorized, paths = low_rank_model(dense, spec=SPEC, rank=rank)
    return low_rank_compression_report(factorized, dense, paths)


def sweep_lr(
    features: FeaturizedDataset,
    splits: Splits,
    candidates: tuple[float, ...],
    factory,
    arm: str,
    rank: int,
    sweep_epochs: int = SWEEP_EPOCHS,
) -> dict:
    """Select one learning rate from seed-0 validation for one rank and arm."""
    rows = []
    print(f"\nlearning-rate sweep, {arm}, matrix rank {rank}, {sweep_epochs} epochs")
    for learning_rate in candidates:
        torch.manual_seed(0)
        result = train_one(
            features,
            splits,
            model=factory(0, splits),
            train_config=TrainConfig(lr=learning_rate, max_epochs=sweep_epochs),
            seed=0,
            progress=False,
        )
        rows.append(
            {
                "lr": learning_rate,
                "val_rmse": result.val_rmse,
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "train_seconds": result.train_seconds,
                "init_val_rmse": result.init_val_rmse,
            }
        )
        print(f"  {learning_rate:.0e}: val {result.val_rmse:.4f} @ {result.best_epoch}")

    choice = learning_rate_choice(rows)
    print(f"chosen: {choice['chosen_lr']:.0e}")
    return {"arm": arm, "rank": rank, "candidates": rows, **choice}


def run_variant(
    arm: str,
    rank: int,
    learning_rate: float,
    seeds: tuple[int, ...],
    dedup: DedupResult,
    features: FeaturizedDataset,
    max_epochs: int = TrainConfig().max_epochs,
) -> dict:
    """Train one low-rank configuration through the shared multi-seed loop."""
    train_config = TrainConfig(lr=learning_rate, max_epochs=max_epochs)
    factory = scratch_factory(rank) if arm == "scratch" else finetune_factory(rank, features)
    results = run_seeds(
        dedup,
        features,
        train_config=train_config,
        seeds=seeds,
        progress=True,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_prefix=f"low_rank_{arm}_r{rank}",
        model_factory=factory,
    )
    return {
        "rank": rank,
        "lr": learning_rate,
        "compression": compression_at(rank),
        "summary": summarize(results),
        "runs": [result.as_dict() for result in results],
    }


def arm_payload(arm: str, variants: list[dict], seeds: tuple[int, ...]) -> dict:
    return {
        "label": f"low_rank_{arm}",
        "arm": f"low_rank_{arm}",
        "display_label": (
            "Low-rank matrix factorization, trained from scratch"
            if arm == "scratch"
            else "Truncated SVD + fine-tuning"
        ),
        "factorization": "ordinary_matrix_low_rank",
        "spec": SPEC,
        "ranks": [variant["rank"] for variant in variants],
        "lr_per_rank": {str(variant["rank"]): variant["lr"] for variant in variants},
        "seeds": list(seeds),
        "variants": variants,
    }


def write(payload: dict, name: str, *, export_curves: bool = False) -> Path:
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {path}")
    if export_curves:
        print(f"wrote {len(save_run_curves(path))} training curves")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--sweep-epochs", type=int, default=SWEEP_EPOCHS)
    parser.add_argument("--max-epochs", type=int, default=TrainConfig().max_epochs)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ranks, seeds = tuple(args.ranks), tuple(args.seeds)
    missing = [seed for seed in seeds if not dense_checkpoint_path(seed).is_file()]
    if "finetune" in args.arms and missing:
        print(f"no dense checkpoint for seeds {missing}; run scripts/run_task2.py first")
        return 1

    dedup = canonicalize_and_dedupe()
    features = featurize_molecules(dedup.clean_df["canonical_smiles"].tolist())
    splits_0 = make_splits(dedup.clean_df, seed=0)

    for rank in ranks:
        report = compression_at(rank)
        print(
            f"rank {rank}: {report['low_rank_total_params']:,} total params, "
            f"{report['model_compression']:.1f}x compression"
        )

    sweeps: dict[str, dict[str, dict]] = {arm: {} for arm in args.arms}
    for arm in args.arms:
        for rank in ranks:
            factory = (
                scratch_factory(rank) if arm == "scratch" else finetune_factory(rank, features)
            )
            result = sweep_lr(
                features,
                splits_0,
                LR_CANDIDATES,
                factory,
                arm,
                rank,
                args.sweep_epochs,
            )
            require_interior_choice(result, f"low rank {arm}, rank {rank}")
            sweeps[arm][str(rank)] = result
    write(
        {
            "factorization": "ordinary_matrix_low_rank",
            "seed": 0,
            "epochs": args.sweep_epochs,
            "by_arm": sweeps,
        },
        "task7_low_rank_lr_sweep.json",
    )

    started = time.perf_counter()
    for arm in args.arms:
        variants = []
        for rank in ranks:
            learning_rate = sweeps[arm][str(rank)]["chosen_lr"]
            print(f"\n{arm}, matrix rank {rank}, lr {learning_rate:.0e}, seeds {seeds}")
            variants.append(
                run_variant(
                    arm,
                    rank,
                    learning_rate,
                    seeds,
                    dedup,
                    features,
                    args.max_epochs,
                )
            )
        write(
            arm_payload(arm, variants, seeds),
            f"task7_low_rank_{arm}.json",
            export_curves=True,
        )
    print(f"total final-run wall clock: {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
