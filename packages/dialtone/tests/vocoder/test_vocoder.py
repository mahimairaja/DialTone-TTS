"""Tests for the codec layer and the narrowband generator."""

import math

import pytest
import torch
from dialtone.vocoder.codec_layer import (
    MU,
    codec_roundtrip,
    compand,
    exact_roundtrip,
    expand,
)
from dialtone.vocoder.model import MEL_FPS, NarrowbandVocos, hop_for, output_fps
from handset_bench import codec

# ------------------------------------------------------------- codec layer


def test_forward_is_bit_identical_to_handset_bench():
    """THE test for this feature."""
    x = (torch.rand(8192) * 2 - 1).float()
    assert torch.equal(codec_roundtrip(x), exact_roundtrip(x))


def test_forward_identity_holds_at_the_extremes():
    x = torch.tensor([-1.0, -0.999, 0.0, 0.999, 1.0])
    assert torch.equal(codec_roundtrip(x), exact_roundtrip(x))


def test_gradient_flows_through_the_quantiser():
    x = (torch.rand(1024) * 2 - 1).float().requires_grad_(True)
    codec_roundtrip(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_gradient_is_not_dead():
    """A zero gradient everywhere would mean the straight-through is miswired."""
    x = (torch.rand(1024) * 2 - 1).float().requires_grad_(True)
    codec_roundtrip(x).sum().backward()
    assert x.grad.abs().sum() > 0


def test_companding_alone_needs_no_approximation():
    """Documents why the blueprint's stated main risk mostly dissolves."""
    x = torch.tensor([0.5], requires_grad=True)
    compand(x).backward()
    assert torch.isfinite(x.grad).all()


def test_compand_and_expand_are_inverses():
    """The algebra the whole layer rests on."""
    x = (torch.rand(4096) * 2 - 1).float()
    assert torch.allclose(expand(compand(x)), x, atol=1e-6)


def test_quantisation_is_the_only_thing_the_roundtrip_loses():
    """The forward carries real quantisation error, so the loss can see it."""
    x = (torch.rand(4096) * 2 - 1).float()
    error = (codec_roundtrip(x) - x).abs()
    assert error.max() > 0.0
    assert error.max() < 0.05


def test_mu_matches_the_shared_quantisation_channels():
    assert MU == codec.QUANTIZATION_CHANNELS - 1 == 255


# ------------------------------------------------------------ rate relation


@pytest.mark.parametrize(
    ("n", "hop", "fps"),
    [(1, 256, 31.25), (2, 128, 62.5), (4, 64, 125.0), (8, 32, 250.0)],
)
def test_rate_relation_holds(n, hop, fps):
    assert hop_for(n) == hop
    assert output_fps(n) == fps
    assert abs(fps - MEL_FPS * n / 3) < 1e-9


def test_mel_frame_rate_is_93_75():
    """24000 / 256. The number the whole architecture is built around."""
    assert MEL_FPS == 93.75


def test_n_equals_3_is_rejected():
    """hop would be 85.33, which drifts about 32 ms over ten seconds."""
    with pytest.raises(ValueError, match="divisor of 256"):
        hop_for(3)


@pytest.mark.parametrize("bad", [0, 3, 5, 6, 7, 9, 100])
def test_non_divisors_are_rejected(bad):
    with pytest.raises(ValueError):
        hop_for(bad)


# ---------------------------------------------------------------- generator


def test_output_sample_rate_is_8000():
    assert NarrowbandVocos(n=4).sample_rate == 8000


def test_mel_input_dims_match_zipvoice():
    """The acoustic model is frozen. 100 bins, no exceptions."""
    assert NarrowbandVocos().n_mels == 100


def test_output_length_follows_the_rate_relation():
    model = NarrowbandVocos(n=4).eval()
    mel_frames = 120
    with torch.no_grad():
        wav = model(torch.randn(1, 100, mel_frames))
    expected = mel_frames * 256 / 3  # 93.75 fps in, 8000 Hz out
    assert abs(wav.shape[-1] - expected) <= model.hop


@pytest.mark.parametrize("n", [1, 2, 4])
def test_every_variant_produces_the_right_length(n):
    model = NarrowbandVocos(n=n).eval()
    mel_frames = 90
    with torch.no_grad():
        wav = model(torch.randn(1, 100, mel_frames))
    assert abs(wav.shape[-1] - mel_frames * 256 / 3) <= model.hop * 2


def test_eval_mode_is_deterministic():
    """Same text and voice twice gives identical audio."""
    model = NarrowbandVocos(n=4).eval()
    mel = torch.randn(1, 100, 50)
    with torch.no_grad():
        assert torch.equal(model(mel), model(mel))


def test_rejects_wrong_mel_bin_count():
    with pytest.raises(ValueError, match="100"):
        NarrowbandVocos().eval()(torch.randn(1, 80, 50))


def test_output_cannot_contain_energy_above_nyquist():
    """At 8 kHz out, Nyquist is 4 kHz, so the bound holds by construction."""
    model = NarrowbandVocos(n=4).eval()
    with torch.no_grad():
        wav = model(torch.randn(1, 100, 60)).squeeze()
    freqs = torch.fft.rfftfreq(wav.numel(), 1 / model.sample_rate)
    assert float(freqs.max()) <= 4000.0


def test_gradients_reach_the_input_projection():
    """The smoke check that the graph is connected end to end."""
    model = NarrowbandVocos(n=4)
    loss = model(torch.randn(1, 100, 40)).abs().mean()
    loss.backward()
    grad = model.input_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_codec_layer_composes_with_the_generator():
    """The training objective path: mel -> wav -> codec -> loss, differentiable."""
    model = NarrowbandVocos(n=4)
    wav = model(torch.randn(1, 100, 40))
    passed = codec_roundtrip(wav.clamp(-1, 1))
    loss = passed.abs().mean()
    loss.backward()
    assert torch.isfinite(model.input_proj.weight.grad).all()


def test_parameter_count_is_sane():
    """A smoke bound. Wildly wrong sizing usually means a config mistake."""
    n_params = sum(p.numel() for p in NarrowbandVocos().parameters())
    assert 1e6 < n_params < 1e8
    assert math.isfinite(n_params)


# ------------------------------------------------------------------- losses


def test_stft_sizes_are_the_8k_set_not_the_24k_set():
    """Stock 24k sizes at 8kHz would resolve mostly band that does not exist."""
    from dialtone.vocoder.losses import STFT_FFT_SIZES, MultiResolutionSTFTLoss

    assert MultiResolutionSTFTLoss().fft_sizes == STFT_FFT_SIZES == (512, 256, 128)
    assert 2048 not in STFT_FFT_SIZES
    assert max(STFT_FFT_SIZES) <= 512


def test_identical_signals_give_near_zero_loss():
    from dialtone.vocoder.losses import MultiResolutionSTFTLoss

    x = torch.randn(1, 8000)
    assert float(MultiResolutionSTFTLoss()(x, x)) < 1e-4


def test_different_signals_give_positive_loss():
    from dialtone.vocoder.losses import MultiResolutionSTFTLoss

    loss = MultiResolutionSTFTLoss()(torch.randn(1, 8000), torch.randn(1, 8000))
    assert float(loss) > 0.0


def test_loss_tolerates_a_one_frame_length_mismatch():
    from dialtone.vocoder.losses import MultiResolutionSTFTLoss

    a, b = torch.randn(1, 8000), torch.randn(1, 8064)
    assert torch.isfinite(MultiResolutionSTFTLoss()(a, b))


@pytest.mark.slow
def test_training_step_reduces_loss_on_a_single_batch():
    """The overfit gate, in miniature and on CPU."""
    from dialtone.vocoder.codec_layer import codec_roundtrip
    from dialtone.vocoder.losses import MultiResolutionSTFTLoss

    torch.manual_seed(0)
    model = NarrowbandVocos(dim=64, n_blocks=2, n=4)
    criterion = MultiResolutionSTFTLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)

    mel = torch.randn(1, 100, 24)
    target = torch.tanh(torch.randn(1, model.expected_samples(24)) * 0.3)

    losses = []
    for _ in range(60):
        optimiser.zero_grad()
        wav = model(mel).squeeze(1).clamp(-1, 1)
        loss = criterion(codec_roundtrip(wav), target)
        loss.backward()
        optimiser.step()
        losses.append(loss.detach().item())

    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]} -> {losses[-1]}"
