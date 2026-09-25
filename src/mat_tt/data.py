"""Task 1: data preparation for the ESOL solubility dataset.

Canonicalizes SMILES, checks for duplicates (which would otherwise leak across
train/val/test splits and inflate test scores), builds a reproducible seeded
split, and featurizes molecules with MAT's featurization code (vendored in
``mat_tt.vendor.mat``), caching the result to disk.

Featurization is cached **per molecule**, keyed by canonical SMILES, so the
cache is independent of the split: changing the seed re-slices the same cached
features instead of recomputing them (or, worse, silently reusing another
split's features).

Conformer generation follows MAT's chemistry steps (add hydrogens, embed in 3D
with up to 5000 attempts, UFF-optimize, remove hydrogens), with a fixed RDKit
seed for reproducibility. It records whether embedding succeeded instead of
swallowing the failure: MAT falls back to 2D coordinates silently, which would
otherwise hide molecules whose "distance matrix" is not a real 3D interatomic
distance matrix.

Only data prep lives here -- no model code, no torch. See
``notebooks/case_study.ipynb`` for how this is used.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

from mat_tt.vendor.mat.featurization.data_utils import featurize_mol

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ESOL_CSV = REPO_ROOT / "data" / "esol" / "esol.csv"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "esol" / "cache"
CONFORMER_SEED = 0
CACHE_VERSION = 2

ConformerStatus = Literal["3d", "2d_fallback", "failed"]

MolFeatures = tuple[np.ndarray, np.ndarray, np.ndarray]
"""(node_features, adjacency_matrix, distance_matrix), as returned by MAT's featurize_mol."""


def canonicalize_smiles(smiles: str) -> str | None:
    """Return the RDKit-canonical form of a SMILES string, or None if unparsable."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


@dataclass
class DedupResult:
    """Outcome of canonicalizing + deduplicating a raw ESOL dataframe."""

    clean_df: pd.DataFrame
    """One row per unique canonical SMILES, columns: smiles, canonical_smiles, y."""
    n_raw: int
    n_whitespace_smiles: int
    """Raw rows whose SMILES had surrounding whitespace (stripped before parsing)."""
    n_unparsable: int
    unparsable_smiles: list[str]
    n_duplicate_groups: int
    n_duplicates_dropped: int
    n_duplicates_before_canonicalization: int
    """Duplicates already visible as identical raw strings, before canonicalization."""
    disagreeing_duplicate_groups: list[tuple[str, list[float]]] = field(default_factory=list)
    """Canonical SMILES whose duplicate rows carry different labels."""

    @property
    def duplicate_label_spread(self) -> float:
        """Largest within-duplicate-group label disagreement, in CSV units."""
        if not self.disagreeing_duplicate_groups:
            return 0.0
        return max(max(ys) - min(ys) for _, ys in self.disagreeing_duplicate_groups)


def canonicalize_and_dedupe(
    csv_path: Path | str = DEFAULT_ESOL_CSV,
    label_disagreement_tol: float = 1e-6,
) -> DedupResult:
    """Load the raw ESOL CSV, canonicalize SMILES, and collapse duplicates.

    Args:
        csv_path: Path to the ESOL CSV (columns: smiles, y).
        label_disagreement_tol: Duplicate groups whose y-values span more than
            this are reported as "disagreeing" (reflects measurement error in
            the underlying solubility assay).

    Returns:
        A DedupResult with one row per unique canonical SMILES.
    """
    raw_df = pd.read_csv(csv_path)
    n_raw = len(raw_df)
    stripped = raw_df["smiles"].str.strip()
    n_dupes_before_canon = int(n_raw - stripped.nunique())
    n_whitespace = int((raw_df["smiles"] != stripped).sum())

    canonical = raw_df["smiles"].map(canonicalize_smiles)
    unparsable_mask = canonical.isna()
    unparsable_smiles = raw_df.loc[unparsable_mask, "smiles"].tolist()
    if unparsable_smiles:
        logger.warning(
            "%d SMILES could not be parsed by RDKit and were dropped: %s",
            len(unparsable_smiles),
            unparsable_smiles,
        )

    df = raw_df.loc[~unparsable_mask].copy()
    df["canonical_smiles"] = canonical.loc[~unparsable_mask]

    grouped = df.groupby("canonical_smiles")["y"]
    group_sizes = grouped.size()
    n_duplicate_groups = int((group_sizes > 1).sum())
    n_duplicates_dropped = int((group_sizes - 1).clip(lower=0).sum())

    disagreeing: list[tuple[str, list[float]]] = []
    for canon_smiles, ys in grouped:
        if len(ys) > 1 and (ys.max() - ys.min()) > label_disagreement_tol:
            disagreeing.append((canon_smiles, ys.tolist()))
    if disagreeing:
        logger.warning(
            "%d duplicate SMILES groups disagree on the label by more than %g "
            "(kept the mean; source provenance is not available).",
            len(disagreeing),
            label_disagreement_tol,
        )

    clean_df = (
        df.groupby("canonical_smiles", as_index=False)
        .agg(smiles=("smiles", "first"), y=("y", "mean"))
        .loc[:, ["smiles", "canonical_smiles", "y"]]
    )

    return DedupResult(
        clean_df=clean_df,
        n_raw=n_raw,
        n_whitespace_smiles=n_whitespace,
        n_unparsable=len(unparsable_smiles),
        unparsable_smiles=unparsable_smiles,
        n_duplicate_groups=n_duplicate_groups,
        n_duplicates_dropped=n_duplicates_dropped,
        n_duplicates_before_canonicalization=n_dupes_before_canon,
        disagreeing_duplicate_groups=disagreeing,
    )


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def items(self):
        return vars(self).items()


def make_splits(
    clean_df: pd.DataFrame,
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Splits:
    """Build a reproducible random train/val/test split over deduplicated molecules.

    Splitting happens on already-deduplicated canonical SMILES (one row per
    molecule), so by construction no canonical SMILES can appear in more than
    one split. ``assert_no_leakage`` re-checks this explicitly.
    """
    train_val, test = train_test_split(clean_df, test_size=test_fraction, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_fraction / (1 - test_fraction), random_state=seed
    )
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def assert_no_leakage(splits: Splits) -> None:
    """Raise if any canonical SMILES appears in more than one split."""
    sets = {name: set(df["canonical_smiles"]) for name, df in splits.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(
                    f"{len(overlap)} molecules leak between '{a}' and '{b}' splits: {overlap}"
                )


def _embed_conformer(mol: Chem.Mol) -> tuple[Chem.Mol, ConformerStatus]:
    """Generate coordinates for a molecule, following MAT's procedure.

    Mirrors ``load_data_from_smiles`` in the vendored MAT code (add H, embed in
    3D with maxAttempts=5000, UFF-optimize, remove H; fall back to RDKit's 2D
    coordinates if that fails) but reports which path was taken rather than
    swallowing the failure.
    """
    mol_h = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol_h, maxAttempts=5000, randomSeed=CONFORMER_SEED) != 0:
            raise ValueError("EmbedMolecule found no 3D conformer")
        AllChem.UFFOptimizeMolecule(mol_h)
        return Chem.RemoveHs(mol_h), "3d"
    except (ValueError, RuntimeError, Chem.rdchem.AtomValenceException):
        AllChem.Compute2DCoords(mol)
        return mol, "2d_fallback"


@dataclass
class FeaturizedDataset:
    """Per-molecule MAT features, keyed by canonical SMILES.

    Split-independent by construction: a split is a selection over this store,
    so re-splitting with a different seed never invalidates or mismatches the
    cached features.
    """

    features: dict[str, MolFeatures]
    conformer_status: dict[str, ConformerStatus]
    failures: dict[str, str]
    """Canonical SMILES that could not be featurized at all -> reason."""

    @property
    def n_3d(self) -> int:
        return sum(s == "3d" for s in self.conformer_status.values())

    @property
    def n_2d_fallback(self) -> int:
        return sum(s == "2d_fallback" for s in self.conformer_status.values())

    @property
    def d_atom(self) -> int:
        """Length of the atom feature vector (MAT's ``d_atom``)."""
        return next(iter(self.features.values()))[0].shape[1]

    def select(self, df: pd.DataFrame) -> tuple[list[MolFeatures], list[float], list[str]]:
        """Return (X, y, canonical_smiles) for the rows of ``df``, skipping failures.

        Keeping the SMILES alongside X and y means downstream analysis (error
        breakdowns, scaffold splits) can always map a prediction back to a molecule.
        """
        x, y, smiles = [], [], []
        for row in df.itertuples(index=False):
            feats = self.features.get(row.canonical_smiles)
            if feats is None:
                continue
            x.append(feats)
            y.append(float(row.y))
            smiles.append(row.canonical_smiles)
        return x, y, smiles


def featurize_molecules(
    canonical_smiles: list[str],
    one_hot_formal_charge: bool = False,
    add_dummy_node: bool = True,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> FeaturizedDataset:
    """Featurize molecules with MAT's graph featurization, caching per molecule.

    The cache is a single pickle keyed by canonical SMILES (file name records the
    featurization options, which change the feature shape). Molecules already in
    the cache are not recomputed, and molecules absent from it are appended, so
    the cache stays valid across seeds, splits and dataset changes.

    Args:
        canonical_smiles: Canonical SMILES to featurize.
        one_hot_formal_charge: Passed through to MAT's ``featurize_mol``.
        add_dummy_node: Passed through to MAT's ``featurize_mol``.
        cache_dir: Directory holding the feature cache.
        use_cache: If False, recompute everything and do not read or write the cache.

    Returns:
        A FeaturizedDataset covering every molecule that could be featurized.
    """
    stamp = f"{'_dn' if add_dummy_node else ''}{'_ohfc' if one_hot_formal_charge else ''}"
    cache_path = Path(cache_dir) / f"features_v{CACHE_VERSION}{stamp}.p"

    features: dict[str, MolFeatures] = {}
    conformer_status: dict[str, ConformerStatus] = {}
    failures: dict[str, str] = {}

    if use_cache and cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        features, conformer_status, failures = (
            cached["features"],
            cached["status"],
            cached["failures"],
        )
        logger.info("Loaded %d cached molecules from '%s'", len(features), cache_path)

    todo = [s for s in canonical_smiles if s not in features and s not in failures]
    if todo:
        logger.info("Featurizing %d molecules (%d already cached)", len(todo), len(features))
    for smiles in todo:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("RDKit could not parse the SMILES")
            mol, status = _embed_conformer(mol)
            features[smiles] = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
            conformer_status[smiles] = status
        except Exception as exc:  # noqa: BLE001 - record why, keep going
            failures[smiles] = f"{type(exc).__name__}: {exc}"
            logger.warning("Featurization failed for '%s': %s", smiles, failures[smiles])

    if use_cache and todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(
                {"features": features, "status": conformer_status, "failures": failures}, fh
            )
        logger.info("Cached %d molecules at '%s'", len(features), cache_path)

    requested = set(canonical_smiles)
    dataset = FeaturizedDataset(
        features={s: f for s, f in features.items() if s in requested},
        conformer_status={s: st for s, st in conformer_status.items() if s in requested},
        failures={s: r for s, r in failures.items() if s in requested},
    )
    if dataset.n_2d_fallback:
        logger.warning(
            "%d molecules fell back to 2D coordinates (3D embedding failed); their "
            "distance matrix is not a true interatomic distance matrix.",
            dataset.n_2d_fallback,
        )
    if dataset.failures:
        logger.warning("%d molecules could not be featurized at all.", len(dataset.failures))
    return dataset
