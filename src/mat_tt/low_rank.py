"""A linear layer represented by two low-rank matrix factors."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def effective_weight_rms(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """RMS entry of ``left @ right`` without constructing that matrix."""
    left_gram = left.transpose(0, 1) @ left
    right_gram = right @ right.transpose(0, 1)
    squared_norm = (left_gram * right_gram).sum()
    return torch.sqrt(squared_norm / (left.shape[0] * right.shape[1]))


class LowRankLinear(nn.Module):
    """Drop-in ``nn.Linear`` represented as ``left @ right``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if rank < 1 or rank > min(in_features, out_features):
            raise ValueError(
                f"rank must be between 1 and {min(in_features, out_features)}, got {rank}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.left = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))
        self.right = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.bias = (
            nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None
        )
        self.reset_parameters()

    @property
    def n_factor_parameters(self) -> int:
        return self.left.numel() + self.right.numel()

    @property
    def n_parameters(self) -> int:
        return self.n_factor_parameters + (self.bias.numel() if self.bias is not None else 0)

    @property
    def dense_parameters(self) -> int:
        return self.in_features * self.out_features + (
            self.out_features if self.bias is not None else 0
        )

    @property
    def compression_ratio(self) -> float:
        return self.dense_parameters / self.n_parameters

    def reset_parameters(self) -> None:
        """Initialize factors so their effective weight has Xavier RMS."""
        target_rms = math.sqrt(2.0 / (self.in_features + self.out_features))
        factor_std = math.sqrt(target_rms / math.sqrt(self.rank))
        with torch.no_grad():
            self.left.normal_(0, factor_std)
            self.right.normal_(0, factor_std)
            actual_rms = float(effective_weight_rms(self.left, self.right))
            if actual_rms > 0:
                correction = math.sqrt(target_rms / actual_rms)
                self.left.mul_(correction)
                self.right.mul_(correction)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(inputs, self.right)
        return F.linear(hidden, self.left, self.bias)

    @classmethod
    def from_dense(cls, linear: nn.Linear, rank: int) -> LowRankLinear:
        """Return the best rank-``rank`` truncated-SVD approximation."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            left, singular_values, right = torch.linalg.svd(linear.weight.data, full_matrices=False)
            scale = singular_values[:rank].sqrt()
            layer.left.copy_(left[:, :rank] * scale)
            layer.right.copy_(scale[:, None] * right[:rank])
            if linear.bias is not None:
                layer.bias.copy_(linear.bias.data)
        return layer

    @torch.no_grad()
    def to_dense_weight(self) -> torch.Tensor:
        """Reconstruct the effective weight for diagnostics only."""
        return self.left @ self.right

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, params={self.n_parameters} "
            f"({self.compression_ratio:.1f}x smaller)"
        )
