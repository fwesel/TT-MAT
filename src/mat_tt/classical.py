"""A classical baseline: molecular features plus a random forest.

Task 7's first optional item, "a quick classical baseline (fingerprints or RDKit
descriptors plus a random forest) to calibrate what the Transformer buys".

It runs on the same deduplicated molecules and the same seeded splits as every
other result in this repository, through ``mat_tt.data.make_splits``, so the
comparison is on identical test molecules rather than on a similar dataset. The
featurizer is chosen on seed 0 validation only, which mirrors how the
Transformer's learning rate was chosen, and the test split is read once per seed.

Both featurizers are deliberately ordinary. Morgan fingerprints at radius 2 are
the standard substructure baseline, and the descriptor set is the usual
physicochemical panel that a chemist would reach for first. The point is to
calibrate, not to win.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor

from mat_tt.data import DedupResult, assert_no_leakage, make_splits
from mat_tt.train import TargetScaler

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

DESCRIPTORS: dict[str, Callable[[Chem.Mol], float]] = {
    "mol_weight": Descriptors.MolWt,
    "logp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "h_donors": rdMolDescriptors.CalcNumHBD,
    "h_acceptors": rdMolDescriptors.CalcNumHBA,
    "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "rings": rdMolDescriptors.CalcNumRings,
    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "heavy_atoms": lambda mol: float(mol.GetNumHeavyAtoms()),
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "heteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
    "molar_refractivity": Crippen.MolMR,
}
"""The physicochemical panel a chemist reaches for first. LogP alone predicts
solubility reasonably well, which is what makes this a fair calibration."""


def _parse(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse '{smiles}'")
    return mol


def morgan_fingerprints(smiles: Sequence[str]) -> np.ndarray:
    """Binary Morgan (ECFP-like) fingerprints, one row per molecule."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS
    )
    rows = [generator.GetFingerprintAsNumPy(_parse(s)) for s in smiles]
    return np.asarray(rows, dtype=np.uint8)


def rdkit_descriptors(smiles: Sequence[str]) -> np.ndarray:
    """The descriptor panel above, one row per molecule."""
    rows = [[float(fn(_parse(s))) for fn in DESCRIPTORS.values()] for s in smiles]
    return np.asarray(rows, dtype=np.float64)


FEATURIZERS: dict[str, Callable[[Sequence[str]], np.ndarray]] = {
    "fingerprints": morgan_fingerprints,
    "descriptors": rdkit_descriptors,
}


@dataclass(frozen=True)
class ForestConfig:
    """Random forest settings, left close to the library defaults on purpose."""

    featurizer: str = "fingerprints"
    n_estimators: int = 500
    min_samples_leaf: int = 1
    max_features: float | str = 1.0 / 3.0
    n_jobs: int = 1
    """Single-threaded on purpose. With n_jobs=-1 the per-tree averaging in
    ``predict`` reduces in a nondeterministic order, so repeated runs differ by
    around 4e-16 and are not bit-identical. That is far below anything reported
    here, but exactness costs about 5 seconds a fit, which is worth paying in a
    study about clean experimental design."""

    def as_dict(self) -> dict:
        return asdict(self)


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def forest_size(forest: RandomForestRegressor) -> dict:
    """How big the fitted forest is, for an honest size comparison.

    A forest's size is nodes, not matrix entries, so this is never plotted on the
    same axis as a parameter count. It is worth reporting anyway, because the
    answer is usually that the forest is larger than the Transformer.
    """
    nodes = sum(int(tree.tree_.node_count) for tree in forest.estimators_)
    return {"n_trees": len(forest.estimators_), "n_nodes": nodes}


def fit_one(
    dedup: DedupResult, config: ForestConfig, seed: int, *, evaluate_test: bool = True
) -> dict:
    """Fit on one training split, score validation, and optionally score test once."""
    splits = make_splits(dedup.clean_df, seed=seed)
    assert_no_leakage(splits)
    featurize = FEATURIZERS[config.featurizer]

    started = time.perf_counter()
    x_train = featurize(splits.train["canonical_smiles"].tolist())
    y_train = splits.train["y"].to_numpy(dtype=np.float64)
    scaler = TargetScaler.fit(y_train)

    forest = RandomForestRegressor(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        n_jobs=config.n_jobs,
        random_state=seed,
    )
    forest.fit(x_train, scaler.encode(y_train))
    train_seconds = time.perf_counter() - started

    scores = {}
    evaluation_splits = [("val", splits.val)]
    if evaluate_test:
        evaluation_splits.append(("test", splits.test))
    for name, frame in evaluation_splits:
        smiles = frame["canonical_smiles"].tolist()
        predictions = scaler.decode(forest.predict(featurize(smiles)))
        targets = frame["y"].to_numpy(dtype=np.float64)
        scores[name] = (rmse(predictions, targets), smiles, predictions, targets)

    val_rmse = scores["val"][0]
    result = {
        "seed": seed,
        "val_rmse": val_rmse,
        "train_seconds": train_seconds,
        "forest": forest_size(forest),
        "n_features": int(x_train.shape[1]),
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
        "target_scaler": scaler.as_dict(),
    }
    if evaluate_test:
        test_rmse, test_smiles, test_predictions, test_targets = scores["test"]
        result.update(
            {
                "test_rmse": test_rmse,
                "test_predictions": dict(zip(test_smiles, test_predictions.tolist(), strict=True)),
                "test_targets": dict(zip(test_smiles, test_targets.tolist(), strict=True)),
            }
        )
    return result


def choose_featurizer(dedup: DedupResult, config: ForestConfig, seed: int = 0) -> dict:
    """Pick fingerprints or descriptors on one seed's validation split.

    Same discipline as the Transformer's learning rate: the choice is made on
    validation, at one seed, and the test split plays no part in it.
    """
    candidates = []
    for name in FEATURIZERS:
        result = fit_one(
            dedup,
            ForestConfig(**{**config.as_dict(), "featurizer": name}),
            seed,
            evaluate_test=False,
        )
        candidates.append(
            {
                "featurizer": name,
                "val_rmse": result["val_rmse"],
                "n_features": result["n_features"],
                "train_seconds": result["train_seconds"],
            }
        )
    chosen = min(candidates, key=lambda row: row["val_rmse"])["featurizer"]
    return {"seed": seed, "candidates": candidates, "chosen_featurizer": chosen}


def run_seeds(
    dedup: DedupResult, config: ForestConfig, seeds: Sequence[int] = (0, 1, 2)
) -> list[dict]:
    return [fit_one(dedup, config, seed) for seed in seeds]


def summarize(results: list[dict]) -> dict:
    """Same keys the reporting helpers already read for the Transformer runs."""
    test = [r["test_rmse"] for r in results]
    val = [r["val_rmse"] for r in results]
    spread = statistics.stdev if len(results) > 1 else (lambda _: 0.0)
    return {
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "test_rmse_mean": statistics.fmean(test),
        "test_rmse_std": spread(test),
        "val_rmse_mean": statistics.fmean(val),
        "val_rmse_std": spread(val),
        "test_rmse_per_seed": test,
        "val_rmse_per_seed": val,
        "train_seconds_mean": statistics.fmean([r["train_seconds"] for r in results]),
        "train_seconds_total": sum(r["train_seconds"] for r in results),
        "n_nodes_mean": statistics.fmean([r["forest"]["n_nodes"] for r in results]),
    }
