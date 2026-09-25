"""Measuring how compressible a weight matrix is.

Used by ``scripts/run_task3.py`` on MAT's trained matrices, with no training
involved. The functions live here rather than in the script so positive and
negative controls use the same calculations. A tool that reports "not
compressible" is only worth trusting if it also reports "compressible" when
the structure really is there.
"""

from __future__ import annotations

import torch
from torch import nn

from mat_tt.tt import TTLinear


def relative_error(approximation: torch.Tensor, target: torch.Tensor) -> float:
    """Relative Frobenius error. 0 is exact, 1 means no better than predicting zero."""
    return float((approximation - target).norm() / target.norm())


def spectrum_summary(weight: torch.Tensor, n_values: int = 64) -> dict:
    """How concentrated a matrix's singular spectrum is.

    Stable rank is ``||W||_F^2 / ||W||_2^2``. It equals the true rank for a flat
    spectrum and falls toward 1 as a single direction dominates. Read it against a
    random matrix of the same shape: a trained matrix that looks random has little
    low rank structure of any kind to exploit, by TT or anything else.
    """
    values = torch.linalg.svdvals(weight)
    energy = float((values**2).sum())
    return {
        "stable_rank": energy / float(values[0] ** 2),
        "energy_top_1": float((values[:1] ** 2).sum()) / energy,
        "energy_top_16": float((values[:16] ** 2).sum()) / energy,
        "energy_top_64": float((values[:64] ** 2).sum()) / energy,
        "singular_values": values[:n_values].tolist(),
    }


def tt_approximation(
    weight: torch.Tensor, rank: int | list[int], n_modes: int = 4
) -> tuple[float, float, int]:
    """Report residual norm, approximation norm, and parameter count after TT-SVD."""
    out_features, in_features = weight.shape
    dense = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        dense.weight.copy_(weight)
    layer = TTLinear.from_dense(dense, rank=rank, n_modes=n_modes)
    approximation = layer.to_dense_weight()
    return (
        relative_error(approximation, weight),
        float(approximation.norm() / weight.norm()),
        layer.n_core_parameters,
    )


def plain_low_rank_approximation(weight: torch.Tensor, budget: int) -> tuple[float, int, int]:
    """Best truncated SVD that fits in ``budget`` parameters.

    The control for whether TT's structure earns its place. A rank k factorization
    of a d_out by d_in matrix costs ``k * (d_out + d_in)``, so the budget fixes k.
    Returns (relative error, rank used, parameters used), with rank 0 and NaN error
    when the budget cannot afford even rank 1.
    """
    out_features, in_features = weight.shape
    k = budget // (out_features + in_features)
    if k < 1:
        return float("nan"), 0, 0
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    approximation = (u[:, :k] * s[:k]) @ vh[:k]
    return relative_error(approximation, weight), k, k * (out_features + in_features)
