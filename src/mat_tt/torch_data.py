"""Batching for MAT, turning featurized molecules into padded torch tensors.

Replaces the DataLoader machinery in MAT's ``data_utils.py``, which hardcodes
CUDA tensors and returns them in an easy-to-mis-unpack order. Padding semantics
are unchanged, this reuses MAT's ``pad_array``.

Padding is zeros for all three matrices, including the distance matrix, so padded
positions read as distance zero (meaning very close). That is only safe because
``MultiHeadedAttention.forward`` masks those columns to inf before the distance
softmax, and ``attention`` masks them to -inf before the attention softmax. The
mask is load bearing.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mat_tt.data import FeaturizedDataset, MolFeatures
from mat_tt.vendor.mat.featurization.data_utils import pad_array


class MolBatch(NamedTuple):
    """One padded batch. Shapes are (B, N, d_atom) and (B, N, N), mask is (B, N)."""

    node_features: torch.Tensor
    adjacency: torch.Tensor
    distance: torch.Tensor
    mask: torch.Tensor
    y: torch.Tensor
    smiles: list[str]

    def to(self, device: str | torch.device) -> MolBatch:
        return self._replace(
            node_features=self.node_features.to(device),
            adjacency=self.adjacency.to(device),
            distance=self.distance.to(device),
            mask=self.mask.to(device),
            y=self.y.to(device),
        )


class MolDataset(Dataset):
    """Featurized molecules as (features, label, canonical SMILES) triples."""

    def __init__(self, x: list[MolFeatures], y: list[float], smiles: list[str]):
        if not (len(x) == len(y) == len(smiles)):
            raise ValueError(f"length mismatch: x={len(x)} y={len(y)} smiles={len(smiles)}")
        self.x = x
        self.y = y
        self.smiles = smiles

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[MolFeatures, float, str]:
        return self.x[i], self.y[i], self.smiles[i]


def collate_molecules(batch: list[tuple[MolFeatures, float, str]]) -> MolBatch:
    """Pad a batch to its largest molecule and stack it into tensors."""
    max_atoms = max(features[0].shape[0] for features, _, _ in batch)
    d_atom = batch[0][0][0].shape[1]

    node_features, adjacency, distance, labels, smiles = [], [], [], [], []
    for (afm, adj, dist), y, smi in batch:
        node_features.append(pad_array(afm, (max_atoms, d_atom)))
        adjacency.append(pad_array(adj, (max_atoms, max_atoms)))
        distance.append(pad_array(dist, (max_atoms, max_atoms)))
        labels.append(y)
        smiles.append(smi)

    node_features = torch.from_numpy(np.stack(node_features))
    # An all-zero feature row is padding. The dummy node survives this because
    # featurize_mol sets its first feature to 1.
    mask = node_features.abs().sum(dim=-1) != 0

    return MolBatch(
        node_features=node_features,
        adjacency=torch.from_numpy(np.stack(adjacency)),
        distance=torch.from_numpy(np.stack(distance)),
        mask=mask,
        y=torch.tensor(labels, dtype=torch.float32).unsqueeze(-1),
        smiles=smiles,
    )


def make_loader(
    dataset: FeaturizedDataset,
    split_df: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Build a DataLoader for one split, selecting its molecules from the feature store."""
    x, y, smiles = dataset.select(split_df)
    return DataLoader(
        MolDataset(x, y, smiles),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_molecules,
        generator=generator,
    )
