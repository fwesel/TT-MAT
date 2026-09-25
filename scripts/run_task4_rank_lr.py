"""Task 4b again, with the learning rate swept separately at every TT rank.

run_task4.py gives each arm one learning rate and reuses it across ranks, on the
stated grounds that rank is then the only thing varying. That premise is wrong.
Higher rank spreads the same target variance over more cores, so the cores
themselves shrink, measured at 0.445 standard deviation at rank 2 against 0.231
at rank 8, and a fixed learning rate is a larger step relative to the parameters
at higher rank. The rank 8 runs show it: selected epoch 6 to 19 of a 150-epoch budget and
then nothing, which is the early plateau the case study names.

So under that protocol the rank trend measures rank and step size together and
cannot be read as a capacity effect, which is what Task 5 needs. This script
sweeps the learning rate per rank instead. Both protocols are kept:
results/task4_tt_scratch.json is the fixed-rate record, which is what establishes
that the compression works, and this writes
results/task4_tt_scratch_per_rank_lr.json.

Every sweep row and final run is recomputed so stored values cannot cross feature,
target-scaling, dependency, or protocol changes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from mat_tt.data import canonicalize_and_dedupe, featurize_molecules, make_splits
from mat_tt.run_curves import save_run_curves
from mat_tt.train import CHECKPOINT_DIR, RESULTS_DIR, TrainConfig, run_seeds, summarize
from mat_tt.tuning import require_interior_choice

# The experiment helpers live in the sibling script rather than being duplicated.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_task4 import N_MODES, SPEC, compression_at, scratch_factory, sweep_lr  # noqa: E402

ARM = "scratch"
RANKS = (2, 4, 8)
SEEDS = (0, 1, 2)
RANK_LR_CANDIDATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    dedup = canonicalize_and_dedupe()
    features = featurize_molecules(dedup.clean_df["canonical_smiles"].tolist())
    splits_0 = make_splits(dedup.clean_df, seed=0)

    chosen: dict[int, dict] = {}
    for rank in RANKS:
        result = sweep_lr(
            features,
            splits_0,
            RANK_LR_CANDIDATES,
            scratch_factory(rank),
            ARM,
            rank=rank,
        )
        require_interior_choice(result, f"{ARM}, rank {rank}")
        chosen[rank] = result
    sweep_path = RESULTS_DIR / "task4_rank_lr_sweep.json"
    sweep_path.write_text(
        json.dumps({"arm": ARM, "epochs": 60, "seed": 0, "by_rank": chosen}, indent=1) + "\n"
    )
    print(f"\nwrote {sweep_path}")
    print(f"chosen per rank: { ({r: c['chosen_lr'] for r, c in chosen.items()}) }")

    variants = []
    started = time.perf_counter()
    for rank in RANKS:
        lr = chosen[rank]["chosen_lr"]
        train_config = TrainConfig(lr=lr)
        print(f"\n{ARM}, TT rank {rank}, lr {lr:.0e}, {len(SEEDS)} seeds")
        results = run_seeds(
            dedup,
            features,
            train_config=train_config,
            seeds=SEEDS,
            progress=True,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_prefix=f"tt_{ARM}_perrank_r{rank}",
            model_factory=scratch_factory(rank),
        )
        summary = summarize(results)
        variants.append(
            {
                "rank": rank,
                "lr": lr,
                "compression": compression_at(rank),
                "summary": summary,
                "runs": [r.as_dict() for r in results],
            }
        )
        print(
            f"  rank {rank}: test RMSE {summary['test_rmse_mean']:.4f} "
            f"+/- {summary['test_rmse_std']:.4f}, val {summary['val_rmse_mean']:.4f}"
        )

    path = RESULTS_DIR / "task4_tt_scratch_per_rank_lr.json"
    path.write_text(
        json.dumps(
            {
                "label": "tt_scratch_per_rank_lr",
                "display_label": "4b, TT from scratch, lr swept per rank",
                "arm": "4b",
                "protocol": "learning rate swept separately at every rank",
                "spec": SPEC,
                "n_modes": N_MODES,
                "lr_per_rank": {str(r): chosen[r]["chosen_lr"] for r in RANKS},
                "seeds": list(SEEDS),
                "train_config": TrainConfig().as_dict(),
                "variants": variants,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote {path}")
    print(f"wrote {len(save_run_curves(path))} training curves")
    print(
        f"total wall clock for the retrained ranks: {(time.perf_counter() - started) / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
