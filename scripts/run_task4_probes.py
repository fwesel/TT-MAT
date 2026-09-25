"""Which matrices should be tensorized? Two probes around the Task 4 target set.

The main run tensorizes the 12 square 256x256 matrices, which is where 98.8
percent of the parameters are. These two probes vary the target set instead of
the rank, both at rank 4 and from scratch, so they read directly against the
square-only run at rank 4 in results/task4_tt_scratch.json.

- attention, the 8 projections only, leaving the feed forward layers dense. The
  case study is titled about attention, and this is the only measurement that
  says whether the damage comes from attention or from the feed forward layers.
- square_embedding, the 12 square matrices plus src_embed.lut. The embedding is
  0.9 percent of the model, so this is not a compression story. Its input axis
  indexes kinds of chemical information and Task 3 measured it as markedly more
    structured than the square matrices, which makes it where the case study's
    hypothesis is most directly testable.

Reads the learning rate the from-scratch arm already chose, so this does not
re-sweep. Writes results/task4_probes.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time

from mat_tt.compress import compression_report, select_targets, tensorize_model
from mat_tt.data import canonicalize_and_dedupe, featurize_molecules
from mat_tt.model import ModelConfig, build_model
from mat_tt.run_curves import save_run_curves
from mat_tt.train import CHECKPOINT_DIR, RESULTS_DIR, TrainConfig, run_seeds, summarize

RANK = 4
N_MODES = 4
SEEDS = (0, 1, 2)
SPECS = ("attention", "square_embedding")


def factory_for(spec: str):
    def factory(seed, splits):
        model, _ = tensorize_model(
            build_model(ModelConfig()), spec=spec, rank=RANK, mode="scratch", n_modes=N_MODES
        )
        return model

    return factory


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    sweep_path = RESULTS_DIR / "task4_lr_sweep.json"
    if not sweep_path.is_file():
        print(f"no {sweep_path}, run scripts/run_task4.py first")
        return 1
    lr = json.loads(sweep_path.read_text())["arms"]["scratch"]["chosen_lr"]
    train_config = TrainConfig(lr=lr)
    print(f"rank {RANK}, from scratch, lr {lr:.0e} as chosen by the main from-scratch arm")

    dedup = canonicalize_and_dedupe()
    features = featurize_molecules(dedup.clean_df["canonical_smiles"].tolist())
    dense = build_model(ModelConfig())

    probes = []
    started = time.perf_counter()
    for spec in SPECS:
        targets = select_targets(dense, spec)
        tt_model, paths = tensorize_model(dense, spec=spec, rank=RANK, n_modes=N_MODES)
        compression = compression_report(tt_model, dense, paths)
        print(
            f"\n{spec}: {len(targets)} layers, {compression['tt_total_params']:,} params "
            f"({compression['model_compression']:.1f}x model)"
        )
        results = run_seeds(
            dedup,
            features,
            train_config=train_config,
            seeds=SEEDS,
            progress=True,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_prefix=f"tt_{spec}_r{RANK}",
            model_factory=factory_for(spec),
        )
        summary = summarize(results)
        probes.append(
            {
                "spec": spec,
                "rank": RANK,
                "roles": sorted({entry.role for entry in targets}),
                "compression": compression,
                "summary": summary,
                "runs": [r.as_dict() for r in results],
            }
        )
        print(
            f"  {spec}: test RMSE {summary['test_rmse_mean']:.4f} "
            f"+/- {summary['test_rmse_std']:.4f}, val {summary['val_rmse_mean']:.4f}"
        )

    path = RESULTS_DIR / "task4_probes.json"
    path.write_text(
        json.dumps(
            {
                "rank": RANK,
                "n_modes": N_MODES,
                "mode": "scratch",
                "lr": lr,
                "seeds": list(SEEDS),
                "probes": probes,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote {path}")
    print(f"wrote {len(save_run_curves(path))} training curves")
    print(f"total wall clock: {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
