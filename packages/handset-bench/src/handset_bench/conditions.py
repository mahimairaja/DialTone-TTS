"""Named condition presets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from handset_bench import codec

__all__ = ["CONDITIONS", "Condition", "ConditionedAudio", "apply", "resolve"]


@dataclass(frozen=True)
class Condition:
    """One row of the scorecard's condition axis."""

    name: str
    band_limit: bool
    apply_codec: bool
    loss_p: float
    description: str


@dataclass(frozen=True)
class ConditionedAudio:
    """Audio after a condition has been applied, with the rate it now carries."""

    pcm: torch.Tensor
    sample_rate: int
    condition: Condition


def _loss_description(percent: float) -> str:
    return (
        f"20ms frames dropped independently at {percent:g} percent, "
        "each replaced with the mu-law code for silence (about -81 dBFS)."
    )


CONDITIONS: dict[str, Condition] = {
    "wideband": Condition(
        name="wideband",
        band_limit=False,
        apply_codec=False,
        loss_p=0.0,
        description=(
            "No phone line applied. The system's own output, used as the pre-codec "
            "control so the size of the drop is visible."
        ),
    ),
    "clean": Condition(
        name="clean",
        band_limit=True,
        apply_codec=True,
        loss_p=0.0,
        description=(
            "Band-limited to 300-3400 Hz, resampled to 8kHz, G.711 mu-law encoded "
            "and decoded. No packet loss."
        ),
    ),
    "loss_1pct": Condition(
        name="loss_1pct",
        band_limit=True,
        apply_codec=True,
        loss_p=0.01,
        description=(
            "Band-limited to 300-3400 Hz, resampled to 8kHz, G.711 mu-law encoded "
            "and decoded. " + _loss_description(1)
        ),
    ),
    "loss_3pct": Condition(
        name="loss_3pct",
        band_limit=True,
        apply_codec=True,
        loss_p=0.03,
        description=(
            "Band-limited to 300-3400 Hz, resampled to 8kHz, G.711 mu-law encoded "
            "and decoded. " + _loss_description(3)
        ),
    ),
}


def resolve(name: str) -> Condition:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}. Known: {sorted(CONDITIONS)}")
    return CONDITIONS[name]


def apply(
    condition: Condition,
    pcm: torch.Tensor,
    sample_rate: int,
    *,
    seed: int,
) -> ConditionedAudio:
    """Apply a condition to native-rate audio."""
    if not condition.apply_codec:
        return ConditionedAudio(pcm=pcm, sample_rate=sample_rate, condition=condition)

    out = codec.phone_line(pcm, sample_rate, loss_p=condition.loss_p, seed=seed)
    return ConditionedAudio(
        pcm=out, sample_rate=codec.TELEPHONY_RATE, condition=condition
    )
