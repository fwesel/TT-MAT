"""Run the Task 2 dense MAT baseline: learning rate sweep, then 3 seeds.

Writes results/task2_lr_sweep.json and results/task2_dense_baseline.json so the
notebook renders its table and figures without retraining.
"""

import json
import logging
import sys

from mat_tt.data import canonicalize_and_dedupe, featurize_molecules, make_splits
from mat_tt.model import ModelConfig, build_model, parameter_report
from mat_tt.run_curves import save_run_curves
from mat_tt.train import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    TrainConfig,
    run_seeds,
    save_results,
    summarize,
    train_one,
)
from mat_tt.tuning import learning_rate_choice, require_interior_choice

LR_CANDIDATES = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
SWEEP_EPOCHS = 60
SEEDS = (0, 1, 2)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dedup = canonicalize_and_dedupe()
    features = featurize_molecules(dedup.clean_df["canonical_smiles"].tolist())
    model_config = ModelConfig()

    report = parameter_report(build_model(model_config))
    (RESULTS_DIR / "task2_parameter_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"model: {report['total_params']:,} params, "
        f"{report['square_weight_fraction']:.1%} in "
        f"{report['n_square_linear_layers']} square matrices"
    )

    # Step 5: pick the learning rate on seed 0 validation only, at a reduced budget.
    print(f"\nlearning rate sweep, seed 0, {SWEEP_EPOCHS} epochs")
    splits_0 = make_splits(dedup.clean_df, seed=0)
    sweep = []
    for lr in LR_CANDIDATES:
        result = train_one(
            features,
            splits_0,
            model_config=model_config,
            train_config=TrainConfig(lr=lr, max_epochs=SWEEP_EPOCHS),
            seed=0,
            progress=False,
        )
        sweep.append(
            {
                "lr": lr,
                "val_rmse": result.val_rmse,
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "train_seconds": result.train_seconds,
            }
        )
        print(f"  lr {lr:.0e}  val RMSE {result.val_rmse:.4f}  best epoch {result.best_epoch}")

    choice = learning_rate_choice(sweep)
    best_lr = choice["chosen_lr"]
    (RESULTS_DIR / "task2_lr_sweep.json").write_text(
        json.dumps({"epochs": SWEEP_EPOCHS, "seed": 0, "candidates": sweep, **choice}, indent=1)
        + "\n"
    )
    print(f"chosen lr: {best_lr:.0e}")
    require_interior_choice(choice, "dense baseline")

    # Final runs at the full budget, one per seed.
    train_config = TrainConfig(lr=best_lr)
    print(f"\nfinal runs, {len(SEEDS)} seeds, budget {train_config.as_dict()}")
    results = run_seeds(
        dedup,
        features,
        model_config,
        train_config,
        seeds=SEEDS,
        progress=True,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    path = save_results(results, RESULTS_DIR / "task2_dense_baseline.json", label="dense_baseline")
    summary = summarize(results)
    print(f"\nwrote {path}")
    print(f"wrote {len(save_run_curves(path))} training curves")
    print(
        f"test RMSE {summary['test_rmse_mean']:.4f} +/- {summary['test_rmse_std']:.4f}"
        f"  (per seed: {[round(v, 4) for v in summary['test_rmse_per_seed']]})"
    )
    print(f"val  RMSE {summary['val_rmse_mean']:.4f} +/- {summary['val_rmse_std']:.4f}")
    print(f"mean predictor floor: {summary['mean_predictor_rmse']:.2f}")
    print(f"total wall clock: {summary['train_seconds_total'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
