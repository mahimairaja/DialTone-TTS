"""The phone-line codec chain."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import torch
import torchaudio.functional as AF
import torchaudio.transforms as AT

__all__ = [
    "ASR_RATE",
    "BAND_HIGH_HZ",
    "BAND_LOW_HZ",
    "BAND_SECTIONS",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "QUANTIZATION_CHANNELS",
    "TELEPHONY_RATE",
    "ZERO_CODE",
    "band_limit",
    "condition_seed",
    "drop_frames",
    "mu_law_decode",
    "mu_law_encode",
    "phone_line",
    "resample",
    "to_asr_rate",
]

#: The PSTN passband. Everything outside this is discarded by the carrier.
BAND_LOW_HZ = 300.0
BAND_HIGH_HZ = 3400.0

#: G.711 operates at 8kHz.
TELEPHONY_RATE = 8000

#: Both Parakeet and faster-whisper expect 16kHz input.
ASR_RATE = 16000

#: G.711 is 8-bit.
QUANTIZATION_CHANNELS = 256

#: A 20ms frame at 8kHz. The unit a real network loses.
FRAME_MS = 20
FRAME_SAMPLES = TELEPHONY_RATE * FRAME_MS // 1000  # 160

#: Cascaded biquad sections in the band-limit. One biquad is 12 dB/octave, which
BAND_SECTIONS = 4

#: What silence encodes to. Mu-law has no exact zero: decoding 128 yields 8.62e-5,
ZERO_CODE = 128


@lru_cache(maxsize=32)
def _resampler(src_sr: int, dst_sr: int) -> AT.Resample:
    """Cached resampler. Construction builds a kernel, so reuse it across calls."""
    return AT.Resample(orig_freq=src_sr, new_freq=dst_sr, dtype=torch.float32)


@lru_cache(maxsize=8)
def _butterworth_qs(sections: int) -> tuple[float, ...]:
    """Q values for a Butterworth cascade of order 2*sections."""
    order = 2 * sections
    return tuple(
        1.0 / (2.0 * math.cos((2 * k + 1) * math.pi / (2 * order)))
        for k in range(sections)
    )


def band_limit(
    x: torch.Tensor,
    sr: int,
    low_hz: float = BAND_LOW_HZ,
    high_hz: float = BAND_HIGH_HZ,
    sections: int = BAND_SECTIONS,
) -> torch.Tensor:
    """Restrict `x` to the PSTN passband, at the native rate."""
    # Applied at the native rate: the 3400 Hz lowpass is below the 8kHz Nyquist,
    # so it doubles as the anti-alias filter.
    y = x
    for q in _butterworth_qs(sections):
        y = AF.highpass_biquad(y, sr, cutoff_freq=low_hz, Q=q)
    for q in _butterworth_qs(sections):
        y = AF.lowpass_biquad(y, sr, cutoff_freq=high_hz, Q=q)
    return y


def resample(x: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return x
    return _resampler(src_sr, dst_sr)(x)


def mu_law_encode(x: torch.Tensor) -> torch.Tensor:
    """Compand and quantise to 8-bit codes in [0, 255]."""
    return AF.mu_law_encoding(x, QUANTIZATION_CHANNELS)


def mu_law_decode(codes: torch.Tensor) -> torch.Tensor:
    """Expand 8-bit codes back to float32 in [-1, 1]."""
    return AF.mu_law_decoding(codes, QUANTIZATION_CHANNELS).float()


def condition_seed(utterance_id: str, condition_name: str, run_seed: int) -> int:
    """Derive a stable per-utterance seed."""
    payload = f"{utterance_id}|{condition_name}|{run_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def drop_frames(
    codes: torch.Tensor,
    p: float,
    seed: int,
    frame_samples: int = FRAME_SAMPLES,
) -> torch.Tensor:
    """Drop whole 20ms frames independently at rate `p`, filling with `ZERO_CODE`."""
    if p <= 0.0:
        return codes

    n_full = codes.numel() // frame_samples
    if n_full == 0:
        return codes

    generator = torch.Generator().manual_seed(seed)
    drop_mask = torch.rand(n_full, generator=generator) < p

    out = codes.clone()
    body = out[: n_full * frame_samples].view(n_full, frame_samples)
    body[drop_mask] = ZERO_CODE
    return out


def phone_line(
    x: torch.Tensor,
    src_sr: int,
    *,
    loss_p: float = 0.0,
    seed: int = 0,
) -> torch.Tensor:
    """Push `x` through a telephone line. Returns float32 at 8000 Hz."""
    if x.numel() == 0:
        raise ValueError("phone_line received empty audio")

    x = x.float()
    if x.ndim > 1:
        x = x.reshape(-1)

    limited = band_limit(x, src_sr)
    narrow = resample(limited, src_sr, TELEPHONY_RATE)
    # Loss goes after encoding: a network drops packets, and a packet carries codes.
    codes = mu_law_encode(narrow)
    lossy = drop_frames(codes, p=loss_p, seed=seed)
    return mu_law_decode(lossy)


def to_asr_rate(x: torch.Tensor, src_sr: int) -> torch.Tensor:
    """Normalise to the 16kHz both ASR backends expect."""
    return resample(x.float(), src_sr, ASR_RATE)
