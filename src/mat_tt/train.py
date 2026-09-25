"""Training loop, fixed budget, and multi-seed runner for the MAT baseline.

TrainConfig holds the shared training budget. Every variant in Tasks 4 and 5
reuses the same object unchanged, which is how "the same training budget" is
enforced rather than just asserted.

MAT's CSV labels were standardized over the full dataset. Training fits a second
affine transform on the training split so validation and test labels contribute
no statistics to optimization. Predictions are decoded to the CSV's standardized
units before scoring, which keeps RMSE comparable across seeds and to MAT's
reported 0.285.
"""

from __future__ import annotations

import json
import logging
import math
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mat_tt.data import DedupResult, FeaturizedDataset, Splits, assert_no_leakage, make_splits
from mat_tt.model import ModelConfig, build_model, count_parameters
from mat_tt.torch_data import make_loader

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
MEAN_PREDICTOR_RMSE = 1.0
"""Dataset-level floor. Exact over all of ESOL, approximate on any one split."""


def mean_predictor_rmse(targets: dict[str, float] | list[float], prediction: float = 0.0) -> float:
    """RMSE of a constant prediction on these targets.

    ``prediction`` defaults to the CSV-wide mean of zero for legacy results. New
    runs pass the training-split mean stored in their target scaler.
    """
    values = list(targets.values()) if isinstance(targets, dict) else list(targets)
    if not values:
        raise ValueError("no targets to compute a floor from")
    return math.sqrt(sum((value - prediction) ** 2 for value in values) / len(values))


@dataclass(frozen=True)
class TargetScaler:
    """An affine target transform fitted on the training split alone."""

    shift: float = 0.0
    scale: float = 1.0

    @classmethod
    def fit(cls, targets) -> TargetScaler:
        values = [float(value) for value in targets]
        if len(values) < 2:
            raise ValueError("need at least 2 targets to fit a scaler")
        scale = statistics.pstdev(values)
        if scale <= 0:
            raise ValueError("targets have no spread, cannot standardize")
        return cls(shift=statistics.fmean(values), scale=scale)

    @property
    def is_identity(self) -> bool:
        return self.shift == 0.0 and self.scale == 1.0

    def encode(self, values):
        return (values - self.shift) / self.scale

    def decode(self, values):
        return values * self.scale + self.shift

    def as_dict(self) -> dict:
        return asdict(self)


IDENTITY_SCALER = TargetScaler()


@dataclass(frozen=True)
class TrainConfig:
    """The shared training budget."""

    batch_size: int = 32
    max_epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    patience: int = 25
    scheduler: str = "cosine"
    device: str = "cpu"
    standardize_targets: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    """Everything one training run produced."""

    seed: int
    val_rmse: float
    test_rmse: float
    best_epoch: int
    epochs_run: int
    train_seconds: float
    total_params: int
    trainable_params: int
    init_val_rmse: float = float("nan")
    init_test_rmse: float | None = None
    history: list[dict] = field(default_factory=list)
    test_predictions: dict[str, float] = field(default_factory=dict)
    test_targets: dict[str, float] = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)
    train_config: dict = field(default_factory=dict)
    split_sizes: dict[str, int] = field(default_factory=dict)
    target_scaler: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return math.sqrt(float(nn.functional.mse_loss(predictions, targets)))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    scaler: TargetScaler = IDENTITY_SCALER,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Return RMSE in CSV units, plus predictions and targets keyed by SMILES."""
    model.eval()
    predictions, targets, smiles = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.node_features, batch.mask, batch.adjacency, batch.distance, None)
        predictions.append(out.cpu())
        targets.append(batch.y.cpu())
        smiles.extend(batch.smiles)

    predictions = scaler.decode(torch.cat(predictions).squeeze(-1))
    targets = torch.cat(targets).squeeze(-1)
    return (
        rmse(predictions, targets),
        dict(zip(smiles, predictions.tolist(), strict=True)),
        dict(zip(smiles, targets.tolist(), strict=True)),
    )


def load_dense_checkpoint(
    checkpoint_path: Path | str,
    features: FeaturizedDataset,
    splits: Splits,
    batch_size: int = 32,
    device: str = "cpu",
) -> nn.Module:
    """Load a dense checkpoint and verify it against its validation split."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = build_model(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    scaler = TargetScaler(**checkpoint.get("target_scaler", {}))
    validation_loader = make_loader(features, splits.val, batch_size)
    observed, _, _ = evaluate(model, validation_loader, device=device, scaler=scaler)
    expected = checkpoint["val_rmse"]
    if abs(observed - expected) > 1e-4:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} scores {observed:.6f} on the supplied validation "
            f"split but recorded {expected:.6f}; checkpoint and split do not match"
        )
    return model


def train_one(
    features: FeaturizedDataset,
    splits: Splits,
    model: nn.Module | None = None,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    seed: int = 0,
    progress: bool = True,
    checkpoint_path: Path | str | None = None,
) -> RunResult:
    """Train one model on one split and evaluate it.

    The epoch is selected by best validation RMSE, and the test set is read once
    at that checkpoint. It never informs any decision.

    Pass ``model`` to train an already-constructed model (Task 4 passes a
    TT-parameterized one), otherwise one is built from ``model_config``.

    ``checkpoint_path`` writes the selected (best validation) state dict to disk.
    Task 3 reads those weights to measure how well TT approximates trained
    matrices, and Task 4 needs them for the TT-SVD-then-fine-tune path.

    ``init_val_rmse`` is measured before the first optimizer step. For Task 4a it
    says where TT-SVD lands before fine tuning without reading the test set.
    """
    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()
    set_seed(seed)

    scaler = (
        TargetScaler.fit(splits.train["y"]) if train_config.standardize_targets else IDENTITY_SCALER
    )

    if model is None:
        model = build_model(model_config)
    model = model.to(train_config.device)
    counts = count_parameters(model)

    generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(
        features, splits.train, train_config.batch_size, shuffle=True, generator=generator
    )
    val_loader = make_loader(features, splits.val, train_config.batch_size)
    test_loader = make_loader(features, splits.test, train_config.batch_size)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.max_epochs)
        if train_config.scheduler == "cosine"
        else None
    )

    # Creating a DataLoader iterator draws a base seed from the global RNG. Keep
    # the diagnostic validation pass from changing the training trajectory.
    rng_state = torch.get_rng_state()
    init_val_rmse, _, _ = evaluate(model, val_loader, train_config.device, scaler)
    torch.set_rng_state(rng_state)

    best_val, best_epoch, best_state = math.inf, -1, None
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(train_config.max_epochs):
        model.train()
        squared_error, n_seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(train_config.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch.node_features, batch.mask, batch.adjacency, batch.distance, None)
            loss = nn.functional.mse_loss(out, scaler.encode(batch.y))
            loss.backward()
            if train_config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            squared_error += float(loss.detach()) * batch.y.shape[0]
            n_seen += batch.y.shape[0]

        if scheduler is not None:
            scheduler.step()

        # Measured in train mode, so dropout is active and this reads slightly
        # worse than a clean pass over the training set would.
        train_rmse = math.sqrt(squared_error / n_seen) * scaler.scale
        val_rmse, _, _ = evaluate(model, val_loader, train_config.device, scaler)
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})

        if val_rmse < best_val:
            best_val, best_epoch = val_rmse, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if progress and (epoch % 10 == 0 or epoch == train_config.max_epochs - 1):
            print(
                f"  seed {seed} epoch {epoch:3d}  train {train_rmse:.4f}  val {val_rmse:.4f}"
                f"  best {best_val:.4f} @ {best_epoch}"
            )

        if epoch - best_epoch >= train_config.patience:
            if progress:
                print(
                    f"  seed {seed} early stop at epoch {epoch}, "
                    f"no gain for {train_config.patience}"
                )
            break

    train_seconds = time.perf_counter() - started

    if best_state is not None:
        model.load_state_dict(best_state)
    test_rmse, predictions, targets = evaluate(model, test_loader, train_config.device, scaler)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "best_epoch": best_epoch,
                "val_rmse": best_val,
                "test_rmse": test_rmse,
                "model_config": model_config.as_dict(),
                "target_scaler": scaler.as_dict(),
            },
            checkpoint_path,
        )
        logger.info("Wrote checkpoint to '%s'", checkpoint_path)

    return RunResult(
        seed=seed,
        val_rmse=best_val,
        test_rmse=test_rmse,
        best_epoch=best_epoch,
        epochs_run=len(history),
        train_seconds=train_seconds,
        total_params=counts.total,
        trainable_params=counts.trainable,
        init_val_rmse=init_val_rmse,
        history=history,
        test_predictions=predictions,
        test_targets=targets,
        model_config=model_config.as_dict(),
        train_config=train_config.as_dict(),
        split_sizes={
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        target_scaler=scaler.as_dict(),
    )


def run_seeds(
    dedup: DedupResult,
    features: FeaturizedDataset,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    progress: bool = True,
    checkpoint_dir: Path | str | None = None,
    checkpoint_prefix: str = "dense",
    model_factory: Callable[[int, Splits], nn.Module] | None = None,
) -> list[RunResult]:
    """Train one model per seed, where a seed re-draws both the split and the init.

    Every variant is expected to reuse the same seeds, which keeps comparisons
    paired on identical splits.

    ``model_factory`` is called as ``factory(seed, splits)`` and returns the model
    to train. Task 4 passes one that builds a TT-parameterized model, either fresh
    or TT-SVD of that seed's dense checkpoint, so every variant goes through this
    same loop, the same ``TrainConfig`` budget and the same splits as the dense
    baseline rather than through a copy of it.
    """
    results = []
    for seed in seeds:
        splits = make_splits(dedup.clean_df, seed=seed)
        assert_no_leakage(splits)
        if progress:
            print(
                f"seed {seed}: train {len(splits.train)} "
                f"val {len(splits.val)} test {len(splits.test)}"
            )
        checkpoint_path = (
            Path(checkpoint_dir) / f"{checkpoint_prefix}_seed{seed}.pt"
            if checkpoint_dir is not None
            else None
        )
        model = None
        if model_factory is not None:
            # Seed before the factory so a from-scratch TT init is reproducible.
            # train_one seeds again, which puts the training loop's own randomness
            # in the same state whether the model came from a factory or not.
            set_seed(seed)
            model = model_factory(seed, splits)
        results.append(
            train_one(
                features=features,
                splits=splits,
                model=model,
                model_config=model_config,
                train_config=train_config,
                seed=seed,
                progress=progress,
                checkpoint_path=checkpoint_path,
            )
        )
    return results


def summarize(results: list[RunResult]) -> dict:
    """Mean and spread across seeds, plus totals for the results table."""
    test = [r.test_rmse for r in results]
    val = [r.val_rmse for r in results]
    spread = statistics.stdev if len(results) > 1 else (lambda _: 0.0)
    return {
        "n_seeds": len(results),
        "seeds": [r.seed for r in results],
        "test_rmse_mean": statistics.fmean(test),
        "test_rmse_std": spread(test),
        "val_rmse_mean": statistics.fmean(val),
        "val_rmse_std": spread(val),
        "test_rmse_per_seed": test,
        "val_rmse_per_seed": val,
        "total_params": results[0].total_params,
        "trainable_params": results[0].trainable_params,
        "train_seconds_mean": statistics.fmean([r.train_seconds for r in results]),
        "train_seconds_total": sum(r.train_seconds for r in results),
        "epochs_run": [r.epochs_run for r in results],
        "best_epoch": [r.best_epoch for r in results],
        "mean_predictor_rmse": MEAN_PREDICTOR_RMSE,
        # The floor of the test split actually reported, not the dataset-level one.
        "test_floor_per_seed": [
            mean_predictor_rmse(r.test_targets, r.target_scaler.get("shift", 0.0)) for r in results
        ],
        "test_floor_mean": statistics.fmean(
            mean_predictor_rmse(r.test_targets, r.target_scaler.get("shift", 0.0)) for r in results
        ),
        "init_val_rmse_mean": statistics.fmean([r.init_val_rmse for r in results]),
        "init_val_rmse_per_seed": [r.init_val_rmse for r in results],
    }


def save_results(results: list[RunResult], path: Path | str, label: str = "") -> Path:
    """Write runs plus their summary to JSON so the notebook does not have to retrain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "summary": summarize(results),
        "runs": [r.as_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def load_results(path: Path | str) -> dict:
    """Read back what save_results wrote. Runs come back as RunResult objects."""
    payload = json.loads(Path(path).read_text())
    payload["runs"] = [RunResult(**run) for run in payload["runs"]]
    return payload
