"""Train one configuration, dense or TT, and write it to results/<label>.json.

A parameterized runner so that varying the training budget does not mean another
near-copy of ``scripts/run_task4.py``. The stability experiments on the ``exp/*``
branches each invoke this with different flags and commit only their results.

It goes through the same ``mat_tt.train.run_seeds`` as every other result here,
so a run produced by this script is comparable to one produced by the task
scripts. The equivalence is checked rather than asserted: handing it the
configuration recorded for TT rank 8 reproduces that run's committed numbers, and
the result payload follows the shape the reporting helpers consume.

Examples, as the branches use it:

    # more seeds at the committed configuration
    uv run python scripts/run_variant.py --arch tt --ranks 8 --lr 3e-3 \\
        --seeds 3 4 5 --label exp_more_seeds_tt_r8

    # a tighter gradient clip, selection stage on seed 0 only
    uv run python scripts/run_variant.py --arch tt --ranks 2 4 8 --seeds 0 \\
        --grad-clip 1.0 --label exp_clip_selection
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from torch import nn

from mat_tt.compress import compression_report, tensorize_model
from mat_tt.data import canonicalize_and_dedupe, featurize_molecules
from mat_tt.model import ModelConfig, build_model, count_parameters
from mat_tt.run_curves import save_run_curves
from mat_tt.train import CHECKPOINT_DIR, RESULTS_DIR, TrainConfig, run_seeds, summarize

SPEC = "square"
N_MODES = 4
SECONDS_PER_EPOCH = {"dense": 1.4, "tt": 4.0}
"""Measured on this laptop, for the estimate printed before the run."""


def tt_factory(rank: int):
    """Fresh TT cores, the same construction scripts/run_task4.py uses."""

    def factory(seed: int, splits) -> nn.Module:
        model, _ = tensorize_model(
            build_model(ModelConfig()), spec=SPEC, rank=rank, mode="scratch", n_modes=N_MODES
        )
        return model

    return factory


def compression_at(rank: int) -> dict:
    dense = build_model(ModelConfig())
    tt_model, paths = tensorize_model(dense, spec=SPEC, rank=rank, n_modes=N_MODES)
    return compression_report(tt_model, dense, paths)


def dense_compression() -> dict:
    """The dense model described in the same keys, so payloads read alike."""
    total = count_parameters(build_model(ModelConfig())).total
    return {
        "n_layers": 0,
        "tt_total_params": total,
        "tt_layer_params": 0,
        "dense_total_params": total,
        "model_compression": 1.0,
        "layer_compression": 1.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["dense", "tt"], required=True)
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[2, 4, 8], help="ignored when --arch dense"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--label", required=True, help="results/<label>.json")
    parser.add_argument("--display-label", default=None, help="how tables should name this run")

    budget = TrainConfig()
    parser.add_argument("--lr", type=float, default=budget.lr)
    parser.add_argument("--grad-clip", type=float, default=budget.grad_clip)
    parser.add_argument("--patience", type=int, default=budget.patience)
    parser.add_argument("--scheduler", default=budget.scheduler)
    parser.add_argument("--max-epochs", type=int, default=budget.max_epochs)
    parser.add_argument("--batch-size", type=int, default=budget.batch_size)
    parser.add_argument(
        "--no-checkpoints", action="store_true", help="skip writing model weights to disk"
    )
    parser.add_argument(
        "--use-csv-targets",
        action="store_true",
        help="use MAT's globally standardized labels instead of fitting on the training split",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        patience=args.patience,
        scheduler=args.scheduler,
        standardize_targets=not args.use_csv_targets,
    )
    ranks = args.ranks if args.arch == "tt" else [None]
    seconds = SECONDS_PER_EPOCH[args.arch]
    print(f"{args.label}: {args.arch}, ranks {ranks}, seeds {args.seeds}")
    print(f"budget {train_config.as_dict()}")
    print(
        f"up to {len(ranks) * len(args.seeds) * args.max_epochs * seconds / 60:.0f} min, "
        f"less with early stopping\n"
    )

    dedup = canonicalize_and_dedupe()
    features = featurize_molecules(dedup.clean_df["canonical_smiles"].tolist())

    started = time.perf_counter()
    variants = []
    for rank in ranks:
        if rank is None:
            print(f"dense, {len(args.seeds)} seeds")
            factory, compression = None, dense_compression()
        else:
            print(f"TT rank {rank}, {len(args.seeds)} seeds")
            factory, compression = tt_factory(rank), compression_at(rank)

        results = run_seeds(
            dedup,
            features,
            train_config=train_config,
            seeds=tuple(args.seeds),
            progress=True,
            checkpoint_dir=None if args.no_checkpoints else CHECKPOINT_DIR,
            checkpoint_prefix=f"{args.label}_r{rank}" if rank else args.label,
            model_factory=factory,
        )
        summary = summarize(results)
        variants.append(
            {
                "rank": rank if rank is not None else 0,
                "lr": args.lr,
                "compression": compression,
                "summary": summary,
                "runs": [r.as_dict() for r in results],
            }
        )
        print(
            f"  test RMSE {summary['test_rmse_mean']:.4f} +/- {summary['test_rmse_std']:.4f}, "
            f"val {summary['val_rmse_mean']:.4f}, "
            f"per seed {[round(v, 4) for v in summary['test_rmse_per_seed']]}\n"
        )

    payload = {
        "label": args.label,
        "display_label": args.display_label or args.label,
        "arch": args.arch,
        "arm": "4b" if args.arch == "tt" else "dense",
        "spec": SPEC if args.arch == "tt" else None,
        "n_modes": N_MODES if args.arch == "tt" else None,
        "lr": args.lr,
        "seeds": list(args.seeds),
        "train_config": train_config.as_dict(),
        "variants": variants,
    }
    path = Path(RESULTS_DIR) / f"{args.label}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {path}")
    print(f"wrote {len(save_run_curves(path))} training curves")
    print(f"total wall clock: {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
