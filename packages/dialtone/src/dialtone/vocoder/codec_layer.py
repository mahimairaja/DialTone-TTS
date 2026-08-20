"""The differentiable codec layer."""

from __future__ import annotations

import math

import torch
from handset_bench import codec

__all__ = ["MU", "codec_roundtrip", "compand", "exact_roundtrip", "expand"]

#: G.711 uses 8 bits, so mu is 255.
MU = float(codec.QUANTIZATION_CHANNELS - 1)

_LOG1P_MU = math.log1p(MU)


def compand(x: torch.Tensor) -> torch.Tensor:
    """Mu-law companding. Differentiable as written, no approximation needed."""
    return torch.sign(x) * torch.log1p(MU * x.abs()) / _LOG1P_MU


def expand(y: torch.Tensor) -> torch.Tensor:
    """Inverse companding."""
    return torch.sign(y) * torch.expm1(y.abs() * _LOG1P_MU) / MU


def codec_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Mu-law encode and decode, differentiable, forward bit-exact."""
    # expand(compand(x)) == x, so the quantiser is the only lossy step and the STE
    # reduces to an identity gradient. That lets the forward pass be the reference.
    exact = exact_roundtrip(x)
    return x + (exact - x).detach()


def exact_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """The non-differentiable reference path, used at evaluation and in tests."""
    return codec.mu_law_decode(codec.mu_law_encode(x))
