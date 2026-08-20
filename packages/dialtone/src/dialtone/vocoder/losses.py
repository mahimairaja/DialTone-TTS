"""Losses for the narrowband vocoder."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["MultiResolutionSTFTLoss", "STFT_FFT_SIZES", "STFT_HOPS", "spectral_loss"]

#: Scaled for 8kHz. The stock 24k sizes would resolve mostly empty band.
STFT_FFT_SIZES = (512, 256, 128)
STFT_HOPS = (128, 64, 32)


def spectral_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    n_fft: int,
    hop: int,
) -> torch.Tensor:
    """Log-magnitude plus spectral-convergence loss at one resolution."""
    window = torch.hann_window(n_fft, device=predicted.device)
    kwargs = {
        "n_fft": n_fft,
        "hop_length": hop,
        "win_length": n_fft,
        "window": window,
        "return_complex": True,
        "center": True,
    }
    pred_spec = torch.stft(predicted, **kwargs).abs()
    target_spec = torch.stft(target, **kwargs).abs()

    convergence = torch.norm(target_spec - pred_spec, p="fro") / (
        torch.norm(target_spec, p="fro") + 1e-8
    )
    log_magnitude = nn.functional.l1_loss(
        torch.log(pred_spec + 1e-7), torch.log(target_spec + 1e-7)
    )
    return convergence + log_magnitude


class MultiResolutionSTFTLoss(nn.Module):
    """Sum of spectral losses across several resolutions."""

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = STFT_FFT_SIZES,
        hops: tuple[int, ...] = STFT_HOPS,
    ):
        super().__init__()
        if len(fft_sizes) != len(hops):
            raise ValueError("fft_sizes and hops must be the same length")
        self.fft_sizes = fft_sizes
        self.hops = hops

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted = predicted.squeeze(1) if predicted.dim() == 3 else predicted
        target = target.squeeze(1) if target.dim() == 3 else target

        # An ISTFT head can return a frame more or fewer than the reference; the
        length = min(predicted.shape[-1], target.shape[-1])
        predicted, target = predicted[..., :length], target[..., :length]

        total = predicted.new_zeros(())
        for n_fft, hop in zip(self.fft_sizes, self.hops, strict=True):
            if length < n_fft:
                continue
            total = total + spectral_loss(predicted, target, n_fft, hop)
        return total / max(len(self.fft_sizes), 1)
