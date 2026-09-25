"""Weight initialization helpers, vendored from https://github.com/ardigen/MAT.

Adapted from MAT (MIT License, Ardigen 2020). See ../NOTICE.md for what changed.

Upstream imports ``_calculate_fan_in_and_fan_out``, ``_no_grad_normal_`` and
``_no_grad_uniform_`` from ``torch.nn.init``. Those are private and have changed
signature across torch releases, and importing them at module scope would take
the whole model definition down with them. The two functions below are rewritten
on public APIs instead, keeping the same fan computation and the same resulting
distribution. Upstream's ``earily_stop`` is not vendored, it tracks an
accuracy-style higher-is-better metric which does not fit RMSE.
"""

import math

import torch


def _fan_in_and_fan_out(tensor):
    """Same computation as torch.nn.init._calculate_fan_in_and_fan_out."""
    if tensor.dim() < 2:
        raise ValueError("fan in and fan out require a tensor with 2 or more dimensions")
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    return num_input_fmaps * receptive_field_size, num_output_fmaps * receptive_field_size


def xavier_normal_small_init_(tensor, gain=1.0):
    fan_in, fan_out = _fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))

    with torch.no_grad():
        return tensor.normal_(0.0, std)


def xavier_uniform_small_init_(tensor, gain=1.0):
    fan_in, fan_out = _fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    with torch.no_grad():
        return tensor.uniform_(-a, a)
