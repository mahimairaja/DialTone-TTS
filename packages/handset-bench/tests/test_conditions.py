"""Tests for the named condition presets."""

import math

import torch
from handset_bench import codec
from handset_bench.conditions import CONDITIONS, apply, resolve


def tone(sr: int, freq: float, secs: float = 1.0) -> torch.Tensor:
    t = torch.arange(int(sr * secs), dtype=torch.float32) / sr
    return 0.5 * torch.sin(2 * math.pi * freq * t)


def test_exactly_four_conditions():
    assert set(CONDITIONS) == {"wideband", "clean", "loss_1pct", "loss_3pct"}


def test_wideband_is_the_pre_codec_control():
    """The control condition applies no phone line at all."""
    c = CONDITIONS["wideband"]
    assert c.apply_codec is False
    assert c.band_limit is False
    assert c.loss_p == 0.0


def test_loss_rates_match_their_names():
    assert CONDITIONS["clean"].loss_p == 0.0
    assert CONDITIONS["loss_1pct"].loss_p == 0.01
    assert CONDITIONS["loss_3pct"].loss_p == 0.03


def test_every_condition_states_its_degradation_in_plain_terms():
    """The exact degradation is stated on the scorecard."""
    for c in CONDITIONS.values():
        assert len(c.description) > 20
        assert "  " not in c.description


def test_loss_descriptions_name_the_frame_size_and_the_fill():
    for name in ("loss_1pct", "loss_3pct"):
        d = CONDITIONS[name].description
        assert "20ms" in d
        assert "silence" in d.lower()


def test_codec_conditions_return_8k():
    for name in ("clean", "loss_1pct", "loss_3pct"):
        out = apply(CONDITIONS[name], tone(24000, 440), 24000, seed=1)
        assert out.sample_rate == codec.TELEPHONY_RATE
        assert out.pcm.numel() == 8000


def test_wideband_returns_the_native_rate_untouched():
    x = tone(24000, 440)
    out = apply(CONDITIONS["wideband"], x, 24000, seed=1)
    assert out.sample_rate == 24000
    assert torch.equal(out.pcm, x)


def test_apply_is_deterministic():
    x = tone(24000, 440)
    a = apply(CONDITIONS["loss_3pct"], x, 24000, seed=42)
    b = apply(CONDITIONS["loss_3pct"], x, 24000, seed=42)
    assert torch.equal(a.pcm, b.pcm)


def test_loss_conditions_actually_alter_the_audio():
    """Guards against a degraded-line condition silently becoming a no-op."""
    x = tone(24000, 440, secs=30.0)
    clean = apply(CONDITIONS["clean"], x, 24000, seed=7).pcm
    for name in ("loss_1pct", "loss_3pct"):
        lossy = apply(CONDITIONS[name], x, 24000, seed=7).pcm
        differing = int((lossy - clean).abs().gt(1e-6).sum())
        assert differing > 0, f"{name} left the audio untouched"


def test_heavier_loss_degrades_more_than_lighter_loss():
    x = tone(24000, 440, secs=60.0)
    clean = apply(CONDITIONS["clean"], x, 24000, seed=11).pcm

    def changed(name: str) -> int:
        lossy = apply(CONDITIONS[name], x, 24000, seed=11).pcm
        return int((lossy - clean).abs().gt(1e-6).sum())

    assert changed("loss_3pct") > changed("loss_1pct")


def test_clean_applies_no_loss_at_all():
    x = tone(24000, 440, secs=30.0)
    a = apply(CONDITIONS["clean"], x, 24000, seed=1).pcm
    b = apply(CONDITIONS["clean"], x, 24000, seed=999).pcm
    assert torch.equal(a, b), "clean must not depend on the seed"


def test_resolve_rejects_an_unknown_condition():
    import pytest

    with pytest.raises(KeyError):
        resolve("loss_50pct")


def test_asr_input_is_16k_for_both_arms():
    """The control and the codec arm normalise identically."""
    x = tone(24000, 440)
    for name in ("wideband", "clean"):
        out = apply(CONDITIONS[name], x, 24000, seed=1)
        assert codec.to_asr_rate(out.pcm, out.sample_rate).numel() == 16000
