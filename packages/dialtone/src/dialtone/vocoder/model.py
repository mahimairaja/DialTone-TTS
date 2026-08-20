"""The narrowband vocoder generator."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["MEL_FPS", "NarrowbandVocos", "hop_for", "output_fps"]

#: ZipVoice's VocosFbank: 24000 Hz, hop 256. Do not change; the model is frozen.
MEL_SAMPLE_RATE = 24000
MEL_HOP = 256
MEL_BINS = 100
MEL_FPS = MEL_SAMPLE_RATE / MEL_HOP  # 93.75

TELEPHONY_RATE = 8000


def hop_for(n: int) -> int:
    """ISTFT hop for a temporal resample factor of `n / 3`."""
    if n < 1 or 256 % n != 0:
        raise ValueError(
            f"n must be a positive divisor of 256, got {n}. The rate relation is "
            "h = 256 / n, and a non-integer hop drifts against the mel: n = 3 "
            "would need hop 85.33, which is about 32 ms of desync over ten "
            "seconds."
        )
    return 256 // n


def output_fps(n: int) -> float:
    return TELEPHONY_RATE / hop_for(n)


class _ConvNeXtBlock(nn.Module):
    """Depthwise convolution, layer norm, pointwise expansion. Vocos backbone."""

    def __init__(self, dim: int, expansion: int = 3):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pointwise2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x).transpose(1, 2)
        x = self.pointwise2(self.act(self.pointwise1(self.norm(x))))
        return residual + x.transpose(1, 2)


class NarrowbandVocos(nn.Module):
    """Mel at 93.75 fps to an 8 kHz waveform."""

    def __init__(
        self,
        n_mels: int = MEL_BINS,
        dim: int = 384,
        n_blocks: int = 8,
        n: int = 4,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n = n
        self.hop = hop_for(n)
        self.n_fft = self.hop * 4  # the stock Vocos ratio, 1024 / 256
        self.sample_rate = TELEPHONY_RATE

        self.input_proj = nn.Conv1d(n_mels, dim, kernel_size=7, padding=3)
        self.backbone = nn.Sequential(*[_ConvNeXtBlock(dim) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(dim)

        # Temporal resample by n/3, done as two integer stages so no interpolation
        self.upsample = nn.ConvTranspose1d(
            dim, dim, kernel_size=n * 2, stride=n, padding=n // 2
        )
        self.downsample = nn.Conv1d(dim, dim, kernel_size=6, stride=3, padding=2)

        # ISTFT head predicts magnitude and phase for n_fft // 2 + 1 bins.
        self.n_bins = self.n_fft // 2 + 1
        self.head = nn.Conv1d(dim, self.n_bins * 2, kernel_size=1)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def expected_samples(self, mel_frames: int) -> int:
        """Output length for a given mel length: `mel_frames * 256 / n * n / 3`."""
        return int(round(mel_frames * self.hop * self.n / 3))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() != 3 or mel.shape[1] != self.n_mels:
            raise ValueError(
                f"expected mel of shape (batch, {self.n_mels}, frames), "
                f"got {tuple(mel.shape)}"
            )
        target_frames = int(round(mel.shape[-1] * self.n / 3))

        x = self.input_proj(mel)
        x = self.backbone(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        x = self.upsample(x)
        x = self.downsample(x)

        # Integer stride arithmetic can leave a frame either side; trim or pad so
        if x.shape[-1] > target_frames:
            x = x[..., :target_frames]
        elif x.shape[-1] < target_frames:
            x = nn.functional.pad(x, (0, target_frames - x.shape[-1]))

        magnitude_phase = self.head(x)
        magnitude, phase = magnitude_phase.chunk(2, dim=1)
        magnitude = torch.exp(magnitude.clamp(max=12.0))
        spec = torch.polar(magnitude, phase)

        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=False,
        )
        return wav.unsqueeze(1)
