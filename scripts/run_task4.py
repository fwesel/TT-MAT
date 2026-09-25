"""Task 4: replace MAT's weight matrices with TT-parameterized equivalents.

Two arms, both requested by the case study, which says to train the TT model from
scratch or compress the trained dense model, or both.

- 4b, from scratch. Fresh TT cores, the unchanged TrainConfig budget, the same
  seeds and splits as the dense baseline. This is the budget-matched comparison.
- 4a, fine tune. TT-SVD of each seed's trained dense checkpoint, then training
  under the same budget. This arm has strictly more compute than the baseline,
  since it starts from a model that already trained, so it is reported as a
  separate arm rather than as a like-for-like.

The target is the 12 square 256x256 matrices, 789,504 of the model's 799,233
parameters. See mat_tt.compress for why, and run_task4_probes.py for the
attention-only and embedding variants.

Requires scripts/run_task2.py to have run, since arm 4a reads its checkpoints.
Writes results/task4_lr_sweep.json, results/task4_tt_scratch.json,
results/task4_tt_finetune.json and results/task4_compression.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch import nn

from mat_tt.compress import compression_report, tensorize_model
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
N_MODES = 4
RANKS = (2, 4, 8)
SEEDS = (0, 1, 2)

SWEEP_RANK = 4
SWEEP_EPOCHS = 60
SCRATCH_LR_CANDIDATES = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
"""TT cores are not the dense optimization problem, so the dense choice is not assumed.

The grid was widened twice, because the arm kept choosing the largest candidate
and a winner on the boundary means the arm may be under-tuned. It settles at
1e-2, which is ten times the dense choice of 1e-3.

That gap is a scale effect, not a gradient-size effect. Adam divides by the
gradient's own running magnitude, so a uniformly smaller gradient largely
cancels out. What does not cancel is the size of a step relative to the
parameters it moves, and the cores are large: measured at initialization, a core
has standard deviation 0.289 at rank 4 where the dense weight it replaces has
0.063, so 4.6 times larger. The same absolute step is a much smaller relative
one, hence the larger rate.

The same reasoning says higher ranks may want smaller rates, since spreading one
target variance over more cores makes each core smaller, 0.445 at rank 2 falling
to 0.231 at rank 8. scripts/run_task4_rank_lr.py sweeps per rank and rank 8
chooses 3e-3.
"""
FINETUNE_LR_CANDIDATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
"""Lower to start, since this arm begins from weights that already fit."""

SECONDS_PER_EPOCH = 4.0
"""Measured on this laptop. Used only for the estimate printed before the run."""


def dense_checkpoint_path(seed: int) -> Path:
    return CHECKPOINT_DIR / f"dense_seed{seed}.pt"


def load_dense_model(seed: int, features: FeaturizedDataset, splits: Splits) -> nn.Module:
    """The trained dense model for one seed, checked against its recorded score.

    The check is what makes arm 4a trustworthy. A checkpoint trained on a
    different seed's split would load without complaint and fine tune to a
    plausible-looking RMSE, with test molecules the dense model had already seen.
    Reproducing the recorded validation RMSE rules that out.
    """
    return load_dense_checkpoint(dense_checkpoint_path(seed), features, splits)


def scratch_factory(rank: int):
    """Arm 4b: fresh TT cores."""

    def factory(seed: int, splits: Splits) -> nn.Module:
        model, _ = tensorize_model(
            build_model(ModelConfig()), spec=SPEC, rank=rank, mode="scratch", n_modes=N_MODES
        )
        return model

    return factory


def finetune_factory(rank: int, features: FeaturizedDataset):
    """Arm 4a: TT-SVD of that seed's trained dense weights."""

    def factory(seed: int, splits: Splits) -> nn.Module:
        dense = load_dense_model(seed, features, splits)
        model, _ = tensorize_model(dense, spec=SPEC, rank=rank, mode="svd", n_modes=N_MODES)
        return model

    return factory


def compression_at(rank: int) -> dict:
    dense = build_model(ModelConfig())
    tt_model, paths = tensorize_model(dense, spec=SPEC, rank=rank, n_modes=N_MODES)
    return compression_report(tt_model, dense, paths)


def sweep_lr(
    features: FeaturizedDataset,
    splits: Splits,
    candidates: tuple[float, ...],
    factory,
    label: str,
    rank: int = SWEEP_RANK,
) -> dict:
    """Pick a learning rate on seed 0 validation at a reduced budget, as in Task 2.

    This script uses one learning rate per arm, chosen at rank 4 and reused across
    ranks. ``scripts/run_task4_rank_lr.py`` calls this per rank instead, which is
    why ``rank`` is a parameter rather than read off the module constant.

    Every row is rerun. Reusing a stored row could mix results from a different
    feature cache, target transform, dependency version, or training protocol.
    """
    print(f"\nlearning rate sweep, {label}, rank {rank}, seed 0, {SWEEP_EPOCHS} epochs")
    rows = []
    for lr in candidates:
        torch.manual_seed(0)
        result = train_one(
            features,
            splits,
            model=factory(0, splits),
            train_config=TrainConfig(lr=lr, max_epochs=SWEEP_EPOCHS),
            seed=0,
            progress=False,
        )
        rows.append(
            {
                "lr": lr,
                "val_rmse": result.val_rmse,
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "train_seconds": result.train_seconds,
                "init_val_rmse": result.init_val_rmse,
            }
        )
        print(f"  lr {lr:.0e}  val RMSE {result.val_rmse:.4f}  best epoch {result.best_epoch}")

    choice = learning_rate_choice(rows)
    edge_note = (
        f"  (at the {choice['chosen_grid_edge']} edge)" if choice["chosen_grid_edge"] else ""
    )
    print(f"chosen lr for {label}: {choice['chosen_lr']:.0e}{edge_note}")
    return {
        "label": label,
        "candidates": rows,
        **choice,
    }


def run_arm(
    arm: str,
    lr: float,
    ranks: tuple[int, ...],
    seeds: tuple[int, ...],
    dedup: DedupResult,
    features: FeaturizedDataset,
) -> dict:
    """One arm at every rank, through the shared training loop and budget."""
    train_config = TrainConfig(lr=lr)
    variants = []
    for rank in ranks:
        print(f"\n{arm}, TT rank {rank}, {len(seeds)} seeds, budget {train_config.as_dict()}")
        factory = scratch_factory(rank) if arm == "scratch" else finetune_factory(rank, features)
        results = run_seeds(
            dedup,
            features,
            train_config=train_config,
            seeds=seeds,
            progress=True,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_prefix=f"tt_{arm}_r{rank}",
            model_factory=factory,
        )
        summary = summarize(results)
        variants.append(
            {
                "rank": rank,
                "compression": compression_at(rank),
                "summary": summary,
                "runs": [r.as_dict() for r in results],
            }
        )
        print(
            f"  rank {rank}: test RMSE {summary['test_rmse_mean']:.4f} "
            f"+/- {summary['test_rmse_std']:.4f}, val {summary['val_rmse_mean']:.4f}, "
            f"before training {summary['init_val_rmse_mean']:.4f}"
        )

    return {
        "label": f"tt_{arm}",
        "arm": "4b" if arm == "scratch" else "4a",
        # Named explicitly because run_task4_rank_lr.py produces a second 4b
        # payload, so the arm alone no longer identifies a protocol.
        "display_label": (
            "4b, TT from scratch, one lr for all ranks"
            if arm == "scratch"
            else "4a, TT-SVD then fine tuned"
        ),
        "spec": SPEC,
        "n_modes": N_MODES,
        "lr": lr,
        "seeds": list(seeds),
        "train_config": train_config.as_dict(),
        "variants": variants,
    }


def write(payload: dict, name: str, *, export_curves: bool = True) -> Path:
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
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["scratch", "finetune"],
        default=["scratch", "finetune"],
        help="scratch is Task 4b, finetune is Task 4a",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ranks, seeds = tuple(args.ranks), tuple(args.seeds)

    missing = [s for s in seeds if not dense_checkpoint_path(s).is_file()]
    if missing:
        print(f"no dense checkpoint for seeds {missing}, run scripts/run_task2.py first")
        return 1

    n_runs = len(args.arms) * len(ranks) * len(seeds)
    candidate_counts = {
        "scratch": len(SCRATCH_LR_CANDIDATES),
        "finetune": len(FINETUNE_LR_CANDIDATES),
    }
    n_sweeps = sum(candidate_counts[arm] for arm in args.arms)
    budget = TrainConfig()
    estimate = (
        n_runs * budget.max_epochs * SECONDS_PER_EPOCH + n_sweeps * SWEEP_EPOCHS * SECONDS_PER_EPOCH
    )
    print(
        f"{n_runs} training runs plus {n_sweeps} sweep runs. At {SECONDS_PER_EPOCH:.0f} s per "
        f"epoch that is up to {estimate / 3600:.1f} h, less with early stopping."
    )

    dedup = canonicalize_and_dedupe()
    features = featurize_molecules(dedup.clean_df["canonical_smiles"].tolist())
    splits_0 = make_splits(dedup.clean_df, seed=0)

    for rank in ranks:
        report = compression_at(rank)
        print(
            f"rank {rank}: {report['tt_total_params']:,} params against "
            f"{report['dense_total_params']:,} ({report['model_compression']:.1f}x model, "
            f"{report['layer_compression']:.1f}x in the {report['n_layers']} tensorized layers)"
        )

    factories = {
        "scratch": (scratch_factory(SWEEP_RANK), SCRATCH_LR_CANDIDATES),
        "finetune": (finetune_factory(SWEEP_RANK, features), FINETUNE_LR_CANDIDATES),
    }

    sweeps = {}
    for arm in args.arms:
        factory, candidates = factories[arm]
        sweeps[arm] = sweep_lr(features, splits_0, candidates, factory, arm)
    write(
        {"rank": SWEEP_RANK, "epochs": SWEEP_EPOCHS, "seed": 0, "arms": sweeps},
        "task4_lr_sweep.json",
    )
    for arm, sweep in sweeps.items():
        require_interior_choice(sweep, f"{arm}, rank {SWEEP_RANK}")

    started = time.perf_counter()
    for arm in args.arms:
        payload = run_arm(arm, sweeps[arm]["chosen_lr"], ranks, seeds, dedup, features)
        write(payload, f"task4_tt_{arm}.json")

    write(
        {"spec": SPEC, "n_modes": N_MODES, "by_rank": {str(r): compression_at(r) for r in ranks}},
        "task4_compression.json",
        export_curves=False,
    )
    print(f"\ntotal wall clock for the final runs: {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
