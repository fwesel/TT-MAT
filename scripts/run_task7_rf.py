"""Task 7: a classical baseline to calibrate what the Transformer buys.

Fingerprints or RDKit descriptors into a random forest, on the same
deduplicated molecules and the same seeded splits as every other result here.
The featurizer is chosen on seed 0 validation, then 3 seeds are run.

Writes results/task7_random_forest.json. Costs a couple of minutes.
"""

from __future__ import annotations

import json
import logging
import sys
import time

from mat_tt.classical import ForestConfig, choose_featurizer, run_seeds, summarize
from mat_tt.data import canonicalize_and_dedupe
from mat_tt.train import RESULTS_DIR

SEEDS = (0, 1, 2)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dedup = canonicalize_and_dedupe()
    print(f"{len(dedup.clean_df)} deduplicated molecules, the same set the Transformer trains on")

    base = ForestConfig()
    started = time.perf_counter()

    print("\nfeaturizer choice, seed 0 validation only")
    selection = choose_featurizer(dedup, base)
    for row in selection["candidates"]:
        print(
            f"  {row['featurizer']:<13} {row['n_features']:>5} features  "
            f"val RMSE {row['val_rmse']:.4f}  {row['train_seconds']:.1f} s"
        )
    config = ForestConfig(**{**base.as_dict(), "featurizer": selection["chosen_featurizer"]})
    print(f"chosen: {config.featurizer}")

    print(f"\nfinal runs, {len(SEEDS)} seeds, {config.as_dict()}")
    results = run_seeds(dedup, config, SEEDS)
    summary = summarize(results)
    for result in results:
        print(
            f"  seed {result['seed']}: val {result['val_rmse']:.4f} "
            f"test {result['test_rmse']:.4f}  {result['train_seconds']:.1f} s"
        )

    payload = {
        "label": "random_forest",
        "display_label": f"random forest, {config.featurizer}",
        "config": config.as_dict(),
        "featurizer_selection": selection,
        "seeds": list(SEEDS),
        "summary": summary,
        "runs": results,
    }
    path = RESULTS_DIR / "task7_random_forest.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"\nwrote {path}")
    print(f"test RMSE {summary['test_rmse_mean']:.4f} +/- {summary['test_rmse_std']:.4f}")
    print(
        f"forest size {summary['n_nodes_mean']:,.0f} nodes against the model's 799,233 parameters"
    )
    print(f"total wall clock: {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
