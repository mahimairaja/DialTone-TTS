"""Tests for the phone-line codec chain."""

import math

import pytest
import torch
import torchaudio.functional as AF
from handset_bench import codec


def tone(sr: int, freq: float, secs: float = 1.0, amp: float = 0.5) -> torch.Tensor:
    t = torch.arange(int(sr * secs), dtype=torch.float32) / sr
    return amp * torch.sin(2 * math.pi * freq * t)


def steady(x: torch.Tensor) -> torch.Tensor:
    """Drop the first half of a signal."""
    return x[x.numel() // 2 :]


def rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt())


def band_energy_ratio(x: torch.Tensor, sr: int, above_hz: float) -> float:
    x = steady(x)
    spec = torch.fft.rfft(x).abs()
    freqs = torch.fft.rfftfreq(x.numel(), 1 / sr)
    return float(spec[freqs > above_hz].pow(2).sum() / spec.pow(2).sum())


# ---------------------------------------------------------------- mu-law


def test_zero_encodes_to_code_128():
    """mu-law has no exact zero code. 128 is what silence encodes to."""
    assert int(codec.mu_law_encode(torch.zeros(1))[0]) == 128
    assert codec.ZERO_CODE == 128


def test_decoding_the_zero_code_is_inaudible_but_not_exactly_zero():
    """Documents a real property of mu-law rather than asserting a convenient lie."""
    value = float(codec.mu_law_decode(torch.tensor([codec.ZERO_CODE]))[0])
    assert value != 0.0
    assert 20 * math.log10(abs(value)) < -80.0


def test_mu_law_encode_matches_torchaudio_exactly():
    x = (torch.rand(4096) * 2 - 1).float()
    assert torch.equal(codec.mu_law_encode(x), AF.mu_law_encoding(x, 256))


def test_mu_law_decode_matches_torchaudio():
    codes = torch.randint(0, 256, (4096,))
    assert torch.allclose(
        codec.mu_law_decode(codes), AF.mu_law_decoding(codes, 256), atol=1e-6
    )


def test_mu_law_round_trip_is_close_to_the_original():
    x = (torch.rand(4096) * 2 - 1).float()
    y = codec.mu_law_decode(codec.mu_law_encode(x))
    assert (y - x).abs().max() < 0.05


# ---------------------------------------------------------------- band-limit


def test_band_limit_removes_energy_above_3400():
    """Measured on a mixture, because a ratio needs in-band content to be a ratio."""
    sr = 24000
    mixed = tone(sr, 1000, amp=0.4) + tone(sr, 6000, amp=0.4)
    assert band_energy_ratio(codec.band_limit(mixed, sr), sr, 3400) < 0.01


def test_band_limit_removes_energy_below_300():
    sr = 24000
    x = tone(sr, 80)
    assert rms(steady(codec.band_limit(x, sr))) < 0.1 * rms(steady(x))


def test_band_limit_preserves_in_band_energy():
    sr = 24000
    x = tone(sr, 1000)
    assert rms(steady(codec.band_limit(x, sr))) > 0.9 * rms(steady(x))


# ---------------------------------------------------------------- frame drop


def test_zero_loss_drops_nothing():
    codes = torch.full((1600,), 200, dtype=torch.int64)
    assert torch.equal(codec.drop_frames(codes, p=0.0, seed=7), codes)


def test_drop_rate_is_approximately_p():
    n_frames = 1500
    codes = torch.full((n_frames * codec.FRAME_SAMPLES,), 200, dtype=torch.int64)
    out = codec.drop_frames(codes, p=0.03, seed=7)
    dropped_frames = int((out == codec.ZERO_CODE).sum().item()) // codec.FRAME_SAMPLES
    assert 0.015 < dropped_frames / n_frames < 0.05


def test_drop_frames_is_deterministic_for_the_same_seed():
    codes = torch.full((16000,), 200, dtype=torch.int64)
    a = codec.drop_frames(codes, p=0.03, seed=99)
    b = codec.drop_frames(codes, p=0.03, seed=99)
    assert torch.equal(a, b)


def test_drop_frames_differs_across_seeds():
    codes = torch.full((16000,), 200, dtype=torch.int64)
    a = codec.drop_frames(codes, p=0.05, seed=1)
    b = codec.drop_frames(codes, p=0.05, seed=2)
    assert not torch.equal(a, b)


def test_dropped_frames_are_whole_frames():
    """Packet loss removes packets, not individual samples."""
    codes = torch.full((16000,), 200, dtype=torch.int64)
    out = codec.drop_frames(codes, p=0.2, seed=3)
    n_full = out.numel() // codec.FRAME_SAMPLES
    frames = out[: n_full * codec.FRAME_SAMPLES].view(n_full, codec.FRAME_SAMPLES)
    for frame in frames:
        uniq = set(frame.tolist())
        assert uniq == {200} or uniq == {codec.ZERO_CODE}


# ---------------------------------------------------------------- full chain


def test_phone_line_output_is_8k_float32():
    out = codec.phone_line(tone(24000, 440), 24000)
    assert out.dtype == torch.float32
    assert out.numel() == 8000


def test_phone_line_from_22050_also_gives_8k():
    """Piper's medium voices are 22050 Hz. Native rate must not matter."""
    out = codec.phone_line(tone(22050, 440), 22050)
    assert abs(out.numel() - 8000) <= 2


def test_phone_line_is_deterministic():
    x = tone(24000, 440)
    a = codec.phone_line(x, 24000, loss_p=0.03, seed=1234)
    b = codec.phone_line(x, 24000, loss_p=0.03, seed=1234)
    assert torch.equal(a, b)


def test_phone_line_differs_across_seeds_under_loss():
    x = tone(24000, 440)
    a = codec.phone_line(x, 24000, loss_p=0.03, seed=1)
    b = codec.phone_line(x, 24000, loss_p=0.03, seed=2)
    assert not torch.equal(a, b)


def test_phone_line_strips_out_of_band_content_and_keeps_in_band():
    """Feed both an in-band and an out-of-band tone through the whole chain."""
    sr = 24000
    mixed = tone(sr, 1000, amp=0.4) + tone(sr, 6000, amp=0.4)
    out = codec.phone_line(mixed, sr)
    assert band_energy_ratio(out, 8000, 3400) < 0.05
    assert out.abs().max() > 0.1  # the 1000 Hz component survived


def test_band_limit_is_steep_enough_to_be_realistic():
    """A single biquad at 12 dB/octave is not a telephone line."""
    sr = 24000
    x = tone(sr, 6000)
    db_down = 20 * math.log10(rms(steady(codec.band_limit(x, sr))) / rms(steady(x)))
    assert db_down < -30.0


def test_band_limit_is_minus_3db_at_the_nominal_cutoff():
    """The corner must land where it is documented."""
    sr = 24000
    x = tone(sr, int(codec.BAND_HIGH_HZ))
    db_down = 20 * math.log10(rms(steady(codec.band_limit(x, sr))) / rms(steady(x)))
    assert -3.5 < db_down < -2.5


def test_phone_line_rejects_empty_input():
    with pytest.raises(ValueError):
        codec.phone_line(torch.zeros(0), 24000)


# ---------------------------------------------------------------- asr rate


def test_to_asr_rate_upsamples_8k_to_16k():
    assert codec.to_asr_rate(tone(8000, 440), 8000).numel() == 16000


def test_to_asr_rate_downsamples_24k_to_16k():
    assert codec.to_asr_rate(tone(24000, 440), 24000).numel() == 16000


def test_to_asr_rate_is_identity_at_16k():
    x = tone(16000, 440)
    assert torch.equal(codec.to_asr_rate(x, 16000), x)
