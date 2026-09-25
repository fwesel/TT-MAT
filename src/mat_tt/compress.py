"""Replacing MAT's dense linear layers with TT-parameterized ones.

Task 4 asks for one or more of the model's weight matrices to be replaced by TT
equivalents, either trained from scratch or TT-SVD of the trained weights
followed by fine tuning. Both paths run through ``tensorize_model`` here, which
does module surgery on a model built from the vendored architecture rather than
forking that architecture.

Which matrices, and why:

- ``square``, the 12 square 256x256 matrices (the 8 attention projections plus
  the 4 feed forward layers), is the primary target. It holds 789,504 of the
  model's 799,233 parameters, so it is the only choice where the whole-model
  compression ratio is not capped by what was left dense.
- ``embedding`` is ``src_embed.lut``, 26 by 256 and 6,912 parameters. Compressing
  it can never save more than 0.9 percent of the model, so it is not a
  compression target, but its input axis indexes kinds of chemical information
  and Task 3 measured it as markedly more structured than the square matrices.
  It is run as a hypothesis probe.
- ``generator.proj`` is never a target. It maps 256 to 1, there is no output side
  to factorize, and it is 257 parameters.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

from torch import nn

from mat_tt.low_rank import LowRankLinear
from mat_tt.model import LinearEntry, count_parameters, linear_inventory
from mat_tt.tt import TTLinear, exact_rank_sequence, factorize_dim, prime_factors

TARGET_SETS: dict[str, tuple[str, ...]] = {
    "attention": ("W_Q", "W_K", "W_V", "W_O"),
    "feed_forward": ("feed_forward",),
    "square": ("W_Q", "W_K", "W_V", "W_O", "feed_forward"),
    "embedding": ("embedding",),
    "square_embedding": ("W_Q", "W_K", "W_V", "W_O", "feed_forward", "embedding"),
}
"""Named target sets, by the roles ``mat_tt.model.linear_inventory`` assigns."""

MODES = ("scratch", "svd")
"""How a replaced layer is initialized. See ``tensorize_model``."""

EXACT = "exact"
"""Rank keyword asking each layer for the rank sequence at which TT-SVD is lossless."""


def select_targets(model: nn.Module, spec: str) -> list[LinearEntry]:
    """The layers a named target set refers to, in module order.

    Roles come from ``linear_inventory`` so paths and roles stay defined in one
    place. ``generator.proj`` is excluded by construction: no target set names the
    output_head role.
    """
    if spec not in TARGET_SETS:
        raise KeyError(f"unknown target set '{spec}', expected one of {sorted(TARGET_SETS)}")
    roles = TARGET_SETS[spec]
    return [entry for entry in linear_inventory(model) if entry.role in roles]


def modes_for(
    in_features: int, out_features: int, n_modes: int = 4
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Mode sizes for both sides of a weight, as ``(out_modes, in_modes)``.

    A TT-matrix core carries one output mode and one input mode, so both sides
    need the same number of modes. ``n_modes`` is a request, capped by whichever
    side has fewer prime factors: 256 by 256 gives the 4x4x4x4 the case study
    suggests, and the 26-wide embedding gives 2 cores, (16, 16) by (2, 13),
    because 26 only factors as 2x13.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be positive, got {n_modes}")
    affordable = min(len(prime_factors(in_features)), len(prime_factors(out_features)))
    n = min(n_modes, max(affordable, 1))
    return factorize_dim(out_features, n), factorize_dim(in_features, n)


def replace_linear(model: nn.Module, path: str, layer: nn.Module) -> None:
    """Put ``layer`` where the module at dotted ``path`` was.

    MAT holds its projections in ``nn.ModuleList``, whose children are named by
    index, so that case is assigned by item rather than by attribute.
    """
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(name)] = layer
    else:
        setattr(parent, name, layer)


def tensorize_model(
    model: nn.Module,
    spec: str = "square",
    rank: int | str | Sequence[int] = 4,
    mode: str = "scratch",
    n_modes: int = 4,
) -> tuple[nn.Module, list[str]]:
    """Copy ``model`` and replace its ``spec`` layers with ``TTLinear`` equivalents.

    Args:
        model: A model built by ``mat_tt.model.build_model``. Left untouched: the
            surgery runs on a deep copy, so the dense model stays available for
            comparison and its weights are the source for ``mode="svd"``.
        spec: A key of ``TARGET_SETS``.
        rank: TT rank, uniform in the interior, or an explicit rank sequence, or
            ``"exact"`` for the lossless sequence of each layer. An explicit
            sequence only works when every target layer has the same number of
            cores, which ``"exact"`` exists to avoid: the embedding has 2 cores
            where the square matrices have 4.
        mode: "scratch" initializes fresh cores, which is Task 4b. "svd" runs
            TT-SVD on the trained weight and copies the bias, which is Task 4a.
        n_modes: Requested modes per side, capped per layer by ``modes_for``.

    Returns:
        The tensorized model and the dotted paths that were replaced.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}', expected one of {list(MODES)}")

    tt_model = copy.deepcopy(model)
    targets = select_targets(tt_model, spec)
    if not targets:
        raise ValueError(f"target set '{spec}' matched no layers")

    for entry in targets:
        dense = tt_model.get_submodule(entry.path)
        out_modes, in_modes = modes_for(entry.in_features, entry.out_features, n_modes)
        layer_rank = exact_rank_sequence(out_modes, in_modes) if rank == EXACT else rank
        if mode == "svd":
            layer = TTLinear.from_dense(
                dense, rank=layer_rank, in_modes=in_modes, out_modes=out_modes
            )
        else:
            layer = TTLinear(
                in_features=entry.in_features,
                out_features=entry.out_features,
                rank=layer_rank,
                in_modes=in_modes,
                out_modes=out_modes,
                bias=dense.bias is not None,
                device=dense.weight.device,
                dtype=dense.weight.dtype,
            )
        replace_linear(tt_model, entry.path, layer)

    return tt_model, [entry.path for entry in targets]


def low_rank_model(
    model: nn.Module,
    spec: str = "square",
    rank: int = 1,
    mode: str = "scratch",
) -> tuple[nn.Module, list[str]]:
    """Copy ``model`` and replace ``spec`` layers with ``LowRankLinear``."""
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}', expected one of {list(MODES)}")

    factorized_model = copy.deepcopy(model)
    targets = select_targets(factorized_model, spec)
    if not targets:
        raise ValueError(f"target set '{spec}' matched no layers")

    for entry in targets:
        dense = factorized_model.get_submodule(entry.path)
        if mode == "svd":
            layer = LowRankLinear.from_dense(dense, rank=rank)
        else:
            layer = LowRankLinear(
                in_features=entry.in_features,
                out_features=entry.out_features,
                rank=rank,
                bias=dense.bias is not None,
                device=dense.weight.device,
                dtype=dense.weight.dtype,
            )
        replace_linear(factorized_model, entry.path, layer)

    return factorized_model, [entry.path for entry in targets]


def compression_report(tt_model: nn.Module, dense_model: nn.Module, paths: Sequence[str]) -> dict:
    """Parameter counts and compression ratios, for the tensorized layers and the whole model.

    Both levels are reported because they say different things. The layer-level
    ratio is a property of the TT format at that rank. The model-level ratio is
    what the deployed model actually costs, and it is bounded by whatever was left
    dense.
    """
    dense_total = count_parameters(dense_model)
    tt_total = count_parameters(tt_model)

    dense_layers = sum(count_parameters(dense_model.get_submodule(p)).total for p in paths)
    tt_layers = sum(count_parameters(tt_model.get_submodule(p)).total for p in paths)

    return {
        "n_layers": len(paths),
        "paths": list(paths),
        "dense_layer_params": dense_layers,
        "tt_layer_params": tt_layers,
        "layer_compression": dense_layers / tt_layers if tt_layers else float("inf"),
        "dense_total_params": dense_total.total,
        "tt_total_params": tt_total.total,
        "tt_trainable_params": tt_total.trainable,
        "model_compression": dense_total.total / tt_total.total if tt_total.total else float("inf"),
        "untouched_params": tt_total.total - tt_layers,
    }


def low_rank_compression_report(
    factorized_model: nn.Module, dense_model: nn.Module, paths: Sequence[str]
) -> dict:
    """Parameter counts for low-rank matrix-factorized layers and the whole model."""
    dense_total = count_parameters(dense_model)
    low_rank_total = count_parameters(factorized_model)
    dense_layers = sum(count_parameters(dense_model.get_submodule(path)).total for path in paths)
    low_rank_layers = sum(
        count_parameters(factorized_model.get_submodule(path)).total for path in paths
    )
    return {
        "n_layers": len(paths),
        "paths": list(paths),
        "dense_layer_params": dense_layers,
        "low_rank_layer_params": low_rank_layers,
        "layer_compression": dense_layers / low_rank_layers,
        "dense_total_params": dense_total.total,
        "low_rank_total_params": low_rank_total.total,
        "low_rank_trainable_params": low_rank_total.trainable,
        "model_compression": dense_total.total / low_rank_total.total,
        "untouched_params": low_rank_total.total - low_rank_layers,
    }
