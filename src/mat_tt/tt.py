"""A reusable TT-parameterized linear layer.

The weight of an ``out_features x in_features`` linear map is stored as a chain
of small TT cores instead of a dense matrix. Both sides are factorized into
modes, so 256 becomes 4x4x4x4, and core k has shape
``(rank[k], out_modes[k], in_modes[k], rank[k + 1])``. This is the TT-matrix
format, also called MPO or BlockTT.

The forward pass contracts the input against the cores one at a time and never
builds the dense matrix, which is the property the case study asks for. Two
things make that easy to get wrong:

- ``contract_cores`` is the explicit forward implementation. TensorLy-Torch's
    factorized einsum is retained as an independent test oracle.
- initialization measures the effective weight norm by contracting core Gram
    matrices, not by reconstructing the weight.

Orientation was verified against ``nn.Linear`` rather than assumed: the library's
``tensorized_shape`` is ``(out_modes, in_modes)``, matching the ``(out_features,
in_features)`` layout of ``nn.Linear.weight``, so a dense weight is decomposed
without transposing it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from tltorch import TensorizedTensor
from tltorch.functional import factorized_linear
from torch import nn

FACTORIZATION = "blocktt"
"""TT-matrix format, where each core carries one output mode and one input mode."""

IMPLEMENTATION = "factorized"
"""Contract against the cores. Never "reconstructed", which forms the dense matrix."""


def prime_factors(n: int) -> list[int]:
    """Prime factors of n, largest first."""
    factors, remaining, divisor = [], n, 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    return sorted(factors, reverse=True)


def factorize_dim(dim: int, n_modes: int) -> tuple[int, ...]:
    """Split ``dim`` into ``n_modes`` near-equal factors whose product is ``dim``.

    Raises rather than guessing, because a silently wrong factorization would
    reshape the weight into a different matrix. MAT's embedding is 26 wide, which
    only factors as 2x13, so asking for 4 modes there is an error, not something
    to paper over.
    """
    if dim < 1 or n_modes < 1:
        raise ValueError(f"dim and n_modes must be positive, got dim={dim} n_modes={n_modes}")
    if n_modes == 1:
        return (dim,)

    root = round(dim ** (1 / n_modes))
    for candidate in (root, root + 1):
        if candidate > 1 and candidate**n_modes == dim:
            return (candidate,) * n_modes

    # Greedy balance: hand each prime factor to the smallest bucket so far.
    factors = prime_factors(dim)
    if len(factors) < n_modes:
        raise ValueError(
            f"cannot split {dim} into {n_modes} factors greater than 1, "
            f"its prime factorization is {factors}"
        )
    modes = [1] * n_modes
    for factor in factors:
        modes[modes.index(min(modes))] *= factor
    modes = tuple(sorted(modes))
    if math.prod(modes) != dim:
        raise ValueError(f"factorization {modes} does not multiply to {dim}")
    return modes


def rank_sequence(rank: int | Sequence[int], n_cores: int) -> list[int]:
    """Boundary ranks are always 1. An int means a uniform interior rank."""
    if isinstance(rank, int):
        return [1] + [rank] * (n_cores - 1) + [1]
    ranks = list(rank)
    if len(ranks) != n_cores + 1:
        raise ValueError(f"expected {n_cores + 1} ranks for {n_cores} cores, got {len(ranks)}")
    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError(f"boundary ranks must both be 1, got {ranks}")
    return ranks


def exact_rank_sequence(out_modes: Sequence[int], in_modes: Sequence[int]) -> list[int]:
    """The rank sequence at which TT-SVD is lossless.

    Each core carries ``out_modes[k] * in_modes[k]``, so the exact rank at cut k
    is the smaller of the product to its left and the product to its right. For
    256 as 4x4x4x4 this is [1, 16, 256, 16, 1], not a uniform rank, which is why
    a uniform "large" rank does not reproduce the dense matrix exactly.
    """
    sizes = [o * i for o, i in zip(out_modes, in_modes, strict=True)]
    return (
        [1] + [min(math.prod(sizes[:k]), math.prod(sizes[k:])) for k in range(1, len(sizes))] + [1]
    )


def tt_parameter_count(
    out_modes: Sequence[int], in_modes: Sequence[int], rank: int | Sequence[int]
) -> int:
    """Parameters in the cores alone, excluding bias."""
    ranks = rank_sequence(rank, len(in_modes))
    return sum(
        ranks[k] * o * i * ranks[k + 1]
        for k, (o, i) in enumerate(zip(out_modes, in_modes, strict=True))
    )


def core_std(target_std: float, ranks: Sequence[int]) -> float:
    """Per-core standard deviation that puts the reconstructed matrix at ``target_std``.

    An entry of the reconstructed matrix is a sum over interior rank indices of
    products of one entry from each core. With ``n`` cores of standard deviation
    ``s`` and ``R`` interior rank combinations, each term has standard deviation
    ``s ** n`` and the sum has roughly ``sqrt(R) * s ** n``, so invert that.

    The library's ``normal_`` uses ``(std / prod(rank)) ** (1 / order)`` with
    ``order`` fixed at 2 for a TT-matrix regardless of core count, which lands the
    reconstructed matrix around 1e-4 when 0.0625 was asked for. A layer initialized
    that small never trains, which is the early plateau the case study warns about,
    so the cores are set from this formula instead.
    """
    n_cores = len(ranks) - 1
    interior = math.prod(ranks[1:-1]) if n_cores > 1 else 1
    return (target_std / math.sqrt(interior)) ** (1.0 / n_cores)


def effective_weight_rms(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Root-mean-square entry of the represented weight, without reconstructing it."""
    gram = cores[0].new_ones((1, 1))
    n_entries = 1
    for core in cores:
        rank_in, out_mode, in_mode, rank_out = core.shape
        physical = out_mode * in_mode
        flattened = core.reshape(rank_in, physical, rank_out)
        gram = torch.einsum("ab,api,bpj->ij", gram, flattened, flattened)
        n_entries *= physical
    return torch.sqrt(gram.squeeze() / n_entries)


def contract_cores(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    in_modes: Sequence[int],
    out_modes: Sequence[int],
) -> torch.Tensor:
    """Contract the input against the TT cores one core at a time.

    This is the production forward contraction. The dense weight is never formed:
    the running state carries the output modes consumed so far, the input modes
    still to consume, and the open rank index.

    Args:
        x: Input of shape (batch, in_features).
        cores: Core k has shape (rank[k], out_modes[k], in_modes[k], rank[k + 1]).
        in_modes: Input mode sizes, multiplying to in_features.
        out_modes: Output mode sizes, multiplying to out_features.

    Returns:
        Output of shape (batch, out_features).
    """
    batch = x.shape[0]
    remaining_in = math.prod(in_modes)
    # State: (batch, output modes done, input modes left, open rank).
    state = x.reshape(batch, 1, remaining_in, 1)
    done_out = 1

    for core, in_mode, out_mode in zip(cores, in_modes, out_modes, strict=True):
        rank_in, rank_out = core.shape[0], core.shape[-1]
        remaining_in //= in_mode

        # Peel this core's input mode off the front of the modes still to consume.
        state = state.reshape(batch, done_out, in_mode, remaining_in, rank_in)
        # Put the two axes to contract last: the open rank and this input mode.
        state = state.permute(0, 1, 3, 4, 2)
        # Contract (open rank, input mode) against core axes (rank_in, in_mode).
        state = torch.tensordot(state, core, dims=([3, 4], [0, 2]))
        # Now (batch, done_out, remaining_in, out_mode, rank_out). The new output
        # mode is not adjacent to the ones already done, so move it next to them
        # before merging. Reshaping across remaining_in would reorder the data.
        state = state.permute(0, 1, 3, 2, 4)
        done_out *= out_mode
        state = state.reshape(batch, done_out, remaining_in, rank_out)

    return state.reshape(batch, done_out)


class TTLinear(nn.Module):
    """A linear layer whose weight is stored as TT cores.

    Drop-in for ``nn.Linear``: same ``in_features``, ``out_features`` and bias
    semantics, so it can be substituted at the paths ``mat_tt.model.linear_inventory``
    reports without touching the surrounding architecture.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int | Sequence[int] = 4,
        n_modes: int = 4,
        in_modes: Sequence[int] | None = None,
        out_modes: Sequence[int] | None = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_modes = tuple(in_modes) if in_modes else factorize_dim(in_features, n_modes)
        self.out_modes = tuple(out_modes) if out_modes else factorize_dim(out_features, n_modes)

        if math.prod(self.in_modes) != in_features:
            raise ValueError(f"in_modes {self.in_modes} do not multiply to {in_features}")
        if math.prod(self.out_modes) != out_features:
            raise ValueError(f"out_modes {self.out_modes} do not multiply to {out_features}")
        if len(self.in_modes) != len(self.out_modes):
            raise ValueError(
                f"each core needs one input and one output mode, got "
                f"{len(self.in_modes)} and {len(self.out_modes)}"
            )

        self.rank = rank_sequence(rank, len(self.in_modes))
        # tensorized_shape is (out_modes, in_modes), matching nn.Linear.weight.
        self.weight = TensorizedTensor.new(
            (self.out_modes, self.in_modes),
            rank=self.rank,
            factorization=FACTORIZATION,
            device=device,
            dtype=dtype,
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None
        )
        self.reset_parameters()

    @property
    def cores(self) -> list[torch.Tensor]:
        return list(self.weight.factors)

    @property
    def n_core_parameters(self) -> int:
        return sum(core.numel() for core in self.cores)

    @property
    def n_parameters(self) -> int:
        return self.n_core_parameters + (self.bias.numel() if self.bias is not None else 0)

    @property
    def dense_parameters(self) -> int:
        """What an equivalent nn.Linear would cost, for the compression ratio."""
        return self.in_features * self.out_features + (
            0 if self.bias is None else self.out_features
        )

    @property
    def compression_ratio(self) -> float:
        return self.dense_parameters / self.n_parameters

    def reset_parameters(self) -> None:
        """Initialize the cores so the reconstructed matrix has Xavier-like scale.

        The library spreads a target standard deviation across cores. Getting this
        wrong is the "early plateau when training TT cores from scratch" the case
        study warns about, so the target is set from the layer's own fan in and fan
        out rather than left at a default.
        """
        target_std = math.sqrt(2.0 / (self.in_features + self.out_features))
        std = core_std(target_std, self.rank)
        # Set the cores directly. BlockTT.normal_ rescales the value it is given
        # by (std / prod(rank)) ** (1 / order), where order is 2 for a TT-matrix
        # whatever the core count, which leaves the reconstructed matrix near zero.
        with torch.no_grad():
            for core in self.weight.factors:
                core.data.normal_(0, std)

            actual = float(effective_weight_rms(self.cores))
            if actual > 0:
                correction = (target_std / actual) ** (1.0 / len(self.cores))
                for core in self.weight.factors:
                    core.data.mul_(correction)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = contract_cores(
            x.reshape(-1, self.in_features), self.cores, self.in_modes, self.out_modes
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], self.out_features)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Same map via TensorLy-Torch's factorized einsum, used as a test oracle."""
        return factorized_linear(
            x,
            self.weight,
            bias=self.bias,
            in_features=self.in_features,
            implementation=IMPLEMENTATION,
        )

    @classmethod
    def from_dense(
        cls,
        linear: nn.Linear,
        rank: int | Sequence[int] = 4,
        n_modes: int = 4,
        in_modes: Sequence[int] | None = None,
        out_modes: Sequence[int] | None = None,
    ) -> TTLinear:
        """TT-SVD an existing dense layer, keeping its bias unchanged.

        This is the "compress the trained dense model" path. At
        ``exact_rank_sequence`` the decomposition is lossless, and below it the
        layer is an approximation that fine-tuning is expected to recover from.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            n_modes=n_modes,
            in_modes=in_modes,
            out_modes=out_modes,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        layer.weight.init_from_matrix(linear.weight.data)
        if linear.bias is not None:
            with torch.no_grad():
                layer.bias.copy_(linear.bias.data)
        return layer

    @torch.no_grad()
    def to_dense_weight(self) -> torch.Tensor:
        """Reconstruct the effective weight matrix. Diagnostics only, never in forward."""
        return (
            self.weight.to_matrix()
            if hasattr(self.weight, "to_matrix")
            else self.weight.to_tensor().reshape(self.out_features, self.in_features)
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, in_modes={self.in_modes}, out_modes={self.out_modes}, "
            f"params={self.n_parameters} ({self.compression_ratio:.0f}x smaller)"
        )
