"""Model construction and parameter accounting for the dense MAT baseline.

The architecture itself comes from the vendored MAT code. This module only holds
the configuration, and the parameter bookkeeping that Tasks 4 to 6 need in order
to quote compression ratios.

``linear_inventory`` lists every nn.Linear with its dotted path, which are the
targets Task 4 will replace with TT-parameterized equivalents by module surgery
on a constructed model. Doing it that way means the vendored architecture never
has to be forked, and the train-from-scratch path and the TT-SVD-of-trained-weights
path run through the same code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

from torch import nn

from mat_tt.vendor.mat.transformer import make_model

DEFAULT_D_ATOM = 26
"""Atom feature vector length produced by mat_tt.data with one_hot_formal_charge=False."""


@dataclass(frozen=True)
class ModelConfig:
    """Dense MAT baseline configuration.

    Defaults follow the case study's suggested configuration: d_model 256 (the
    dimension it names for the 4x4x4x4 tensorization), 2 encoder layers, 8 heads.

    distance_matrix_kernel is "softmax" rather than "exp" because exp is
    unnormalized, which would put the distance term on a different scale from the
    attention and adjacency terms while the lambdas assume they are comparable.
    """

    d_atom: int = DEFAULT_D_ATOM
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 8
    n_dense: int = 2
    dropout: float = 0.1
    lambda_attention: float = 0.33
    lambda_distance: float = 0.33
    leaky_relu_slope: float = 0.1
    aggregation_type: str = "mean"
    distance_matrix_kernel: str = "softmax"
    dense_output_nonlinearity: str = "relu"
    init_type: str = "uniform"
    n_output: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


def build_model(config: ModelConfig | None = None) -> nn.Module:
    """Construct a dense MAT model from the vendored architecture."""
    config = config or ModelConfig()
    return make_model(
        d_atom=config.d_atom,
        N=config.n_layers,
        d_model=config.d_model,
        h=config.n_heads,
        N_dense=config.n_dense,
        dropout=config.dropout,
        lambda_attention=config.lambda_attention,
        lambda_distance=config.lambda_distance,
        leaky_relu_slope=config.leaky_relu_slope,
        aggregation_type=config.aggregation_type,
        distance_matrix_kernel=config.distance_matrix_kernel,
        dense_output_nonlinearity=config.dense_output_nonlinearity,
        init_type=config.init_type,
        n_output=config.n_output,
    )


class ParameterCounts(NamedTuple):
    total: int
    trainable: int


def count_parameters(module: nn.Module) -> ParameterCounts:
    return ParameterCounts(
        total=sum(p.numel() for p in module.parameters()),
        trainable=sum(p.numel() for p in module.parameters() if p.requires_grad),
    )


class LinearEntry(NamedTuple):
    """One nn.Linear in the model, addressable by its dotted path."""

    path: str
    in_features: int
    out_features: int
    weight_params: int
    bias_params: int
    role: str

    @property
    def params(self) -> int:
        return self.weight_params + self.bias_params

    @property
    def is_square(self) -> bool:
        return self.in_features == self.out_features


ATTENTION_ROLES = ("W_Q", "W_K", "W_V", "W_O")


def _classify(path: str) -> str:
    """Label a Linear by what it does, so reports read in MAT's own terms."""
    if path.endswith("src_embed.lut"):
        return "embedding"
    if ".self_attn.linears." in path:
        return ATTENTION_ROLES[int(path.rsplit(".", 1)[1])]
    if ".feed_forward.linears." in path:
        return "feed_forward"
    if path.startswith("generator"):
        return "output_head"
    return "other"


def linear_inventory(model: nn.Module) -> list[LinearEntry]:
    """List every nn.Linear in the model, in module order."""
    entries = []
    for path, module in model.named_modules():
        if isinstance(module, nn.Linear):
            entries.append(
                LinearEntry(
                    path=path,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    weight_params=module.weight.numel(),
                    bias_params=module.bias.numel() if module.bias is not None else 0,
                    role=_classify(path),
                )
            )
    return entries


def get_submodule_by_path(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted path from linear_inventory back to its module."""
    return model.get_submodule(path)


def parameter_report(model: nn.Module) -> dict:
    """Summarize where the parameters are, for the Task 2 table and later compression ratios."""
    counts = count_parameters(model)
    inventory = linear_inventory(model)
    square = [e for e in inventory if e.is_square]

    by_role: dict[str, dict[str, int]] = {}
    for entry in inventory:
        bucket = by_role.setdefault(entry.role, {"count": 0, "params": 0})
        bucket["count"] += 1
        bucket["params"] += entry.params

    square_weights = sum(e.weight_params for e in square)
    return {
        "total_params": counts.total,
        "trainable_params": counts.trainable,
        "n_linear_layers": len(inventory),
        "n_square_linear_layers": len(square),
        "square_weight_params": square_weights,
        "square_weight_fraction": square_weights / counts.total,
        "by_role": by_role,
    }
