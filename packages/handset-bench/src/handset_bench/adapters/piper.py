"""Piper adapter."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from handset_bench.adapters.base import AudioChunk, SynthResult, Timings

__all__ = ["DEFAULT_VOICE", "PiperAdapter"]

DEFAULT_VOICE = "en_US-lessac-medium"


class PiperAdapter:
    """Wraps `piper-tts`. One voice per instance."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        *,
        download_dir: str | Path | None = None,
        use_cuda: bool = False,
        deterministic: bool = True,
    ):
        self.voice = voice
        self.use_cuda = use_cuda
        self.deterministic = deterministic
        self._download_dir = Path(download_dir) if download_dir else None
        self._voice = None
        self._piper_version: str | None = None
        self._syn_config = None

    # ------------------------------------------------------------- lifecycle

    def _ensure_loaded(self):
        if self._voice is not None:
            return self._voice

        from importlib.metadata import version as pkg_version

        from piper import PiperVoice
        from piper.download_voices import download_voice

        self._piper_version = pkg_version("piper-tts")

        target = (
            self._download_dir or Path.home() / ".cache" / "handset-bench" / "piper"
        )
        target.mkdir(parents=True, exist_ok=True)
        model_path = target / f"{self.voice}.onnx"
        if not model_path.exists():
            download_voice(self.voice, target)

        self._voice = PiperVoice.load(model_path, use_cuda=self.use_cuda)

        if self.deterministic:
            # Piper is VITS-based and samples noise for its stochastic duration
            # predictor, so the same text gives a different waveform every call.
            # Zeroing both noise terms makes synthesis reproducible, which a
            # benchmark needs more than it needs prosodic variation.
            try:
                from piper import SynthesisConfig

                self._syn_config = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)
            except Exception:  # noqa: BLE001 - older piper has no SynthesisConfig
                self._syn_config = None
        return self._voice

    def _synthesize_chunks(self, engine, text: str):
        """Yield audio chunks, passing the synthesis config when piper supports one."""
        if self._syn_config is not None:
            return engine.synthesize(text, syn_config=self._syn_config)
        return engine.synthesize(text)

    # -------------------------------------------------------------- protocol

    def version_string(self) -> str:
        """Pins both the engine and the voice."""
        if self._piper_version is None:
            from importlib.metadata import version as pkg_version

            self._piper_version = pkg_version("piper-tts")
        # The noise mode is part of the identity: it changes the waveform, so a
        # record made with sampled noise is not comparable to a deterministic one.
        mode = "det" if self.deterministic else "sampled"
        return f"piper-{self._piper_version}+{self.voice}+{mode}"

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthResult:
        engine = self._ensure_loaded()
        submitted_ns = time.perf_counter_ns()
        first_audio_ns: int | None = None
        parts: list[np.ndarray] = []
        sample_rate: int | None = None

        try:
            for chunk in self._synthesize_chunks(engine, text):
                if first_audio_ns is None:
                    first_audio_ns = time.perf_counter_ns()
                if sample_rate is None:
                    sample_rate = chunk.sample_rate
                parts.append(np.asarray(chunk.audio_float_array, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 - a failed system is a result, not a crash
            now = time.perf_counter_ns()
            return SynthResult(
                pcm=np.zeros(0, dtype=np.float32),
                sample_rate=sample_rate or 22050,
                timings=Timings(submitted_ns, now, now),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        completed_ns = time.perf_counter_ns()
        if not parts or first_audio_ns is None:
            return SynthResult(
                pcm=np.zeros(0, dtype=np.float32),
                sample_rate=sample_rate or 22050,
                timings=Timings(submitted_ns, completed_ns, completed_ns),
                status="empty",
            )

        return SynthResult(
            pcm=np.concatenate(parts),
            sample_rate=sample_rate or 22050,
            timings=Timings(submitted_ns, first_audio_ns, completed_ns),
            status="ok",
        )

    def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> Iterator[AudioChunk]:
        engine = self._ensure_loaded()
        for chunk in self._synthesize_chunks(engine, text):
            yield AudioChunk(
                pcm=np.asarray(chunk.audio_float_array, dtype=np.float32),
                sample_rate=chunk.sample_rate,
                received_ns=time.perf_counter_ns(),
            )
