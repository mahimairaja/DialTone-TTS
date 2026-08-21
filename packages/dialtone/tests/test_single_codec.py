"""There must be exactly one implementation of the telephony chain."""

import pathlib

import dialtone
import handset_bench
import torch
from dialtone.vocoder.codec_layer import exact_roundtrip
from handset_bench import codec


def test_codec_comes_from_the_installed_package():
    """Not from a vendored copy inside this repo.

    Two copies can drift, and if training and evaluation conditions drift apart
    every quality claim built on them becomes unfalsifiable.
    """
    repo = pathlib.Path(dialtone.__file__).resolve().parents[4]
    package = pathlib.Path(handset_bench.__file__).resolve()
    assert "site-packages" in package.parts, f"handset_bench resolved to {package}"
    assert not (repo / "packages" / "handset-bench").exists()


def test_the_two_agree_bit_for_bit():
    signal = torch.randn(2048) * 0.3
    assert torch.equal(
        exact_roundtrip(signal),
        codec.mu_law_decode(codec.mu_law_encode(signal)),
    )
