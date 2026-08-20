"""ZipVoice adapter, exposing `zipvoice` and `zipvoice_distill` as separate entries."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from handset_bench.adapters.base import AudioChunk, SynthResult, Timings

__all__ = ["MODEL_DEFAULTS", "STOCK_VOCODER", "ZIPVOICE_REPO", "ZipVoiceAdapter"]

ZIPVOICE_REPO = "k2-fsa/ZipVoice"

#: The stock vocoder ZipVoice ships with, and the thing this project replaces.
STOCK_VOCODER = "charactr/vocos-mel-24khz"

#: Upstream's own defaults, taken from `infer_zipvoice.py`.
MODEL_DEFAULTS = {
    "zipvoice": {"num_step": 16, "guidance_scale": 1.0},
    "zipvoice_distill": {"num_step": 8, "guidance_scale": 3.0},
}

SAMPLING_RATE = 24000


class ZipVoiceAdapter:
    """Wraps k2-fsa ZipVoice. One model variant per instance."""

    def __init__(
        self,
        model_name: str = "zipvoice_distill",
        *,
        prompt_wav: str | Path | None = None,
        prompt_text: str | None = None,
        num_step: int | None = None,
        guidance_scale: float | None = None,
        device: str | None = None,
    ):
        if model_name not in MODEL_DEFAULTS:
            raise ValueError(
                f"unknown model {model_name!r}. Known: {sorted(MODEL_DEFAULTS)}"
            )
        self.model_name = model_name
        self.prompt_wav = str(prompt_wav) if prompt_wav else None
        self.prompt_text = prompt_text
        defaults = MODEL_DEFAULTS[model_name]
        self.num_step = num_step if num_step is not None else defaults["num_step"]
        self.guidance_scale = (
            guidance_scale if guidance_scale is not None else defaults["guidance_scale"]
        )
        self._device_hint = device
        self._loaded = None
        self._prompt_cache = None

    # ------------------------------------------------------------- lifecycle

    def _ensure_loaded(self):
        if self._loaded is not None:
            return self._loaded

        import json

        import safetensors.torch
        import torch
        from huggingface_hub import hf_hub_download
        from zipvoice.models.zipvoice import ZipVoice
        from zipvoice.models.zipvoice_distill import ZipVoiceDistill
        from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
        from zipvoice.utils.checkpoint import load_checkpoint
        from zipvoice.utils.feature import VocosFbank

        checkpoint = hf_hub_download(
            ZIPVOICE_REPO, filename=f"{self.model_name}/model.pt"
        )
        config_path = hf_hub_download(
            ZIPVOICE_REPO, filename=f"{self.model_name}/model.json"
        )
        token_file = hf_hub_download(
            ZIPVOICE_REPO, filename=f"{self.model_name}/tokens.txt"
        )

        tokenizer = EmiliaTokenizer(token_file=token_file)
        tokenizer_config = {
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
        }
        with open(config_path) as handle:
            model_config = json.load(handle)

        cls = ZipVoice if self.model_name == "zipvoice" else ZipVoiceDistill
        model = cls(**model_config["model"], **tokenizer_config)

        if str(checkpoint).endswith(".safetensors"):
            safetensors.torch.load_model(model, checkpoint)
        else:
            load_checkpoint(filename=checkpoint, model=model, strict=True)

        if self._device_hint:
            device = torch.device(self._device_hint)
        elif torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")

        model = model.to(device).eval()

        from vocos import Vocos

        vocoder = Vocos.from_pretrained(STOCK_VOCODER).to(device).eval()

        self._loaded = {
            "model": model,
            "vocoder": vocoder,
            "tokenizer": tokenizer,
            "features": VocosFbank(),
            "device": device,
            "torch": torch,
        }
        return self._loaded

    # -------------------------------------------------------------- protocol

    def version_string(self) -> str:
        """Includes the sampling configuration, because it changes the output."""
        return f"{self.model_name}+k2fsa+step{self.num_step}+cfg{self.guidance_scale:g}"

    def _prompt(self) -> tuple[str, str]:
        if not self.prompt_wav or not self.prompt_text:
            raise ValueError(
                "ZipVoice is prompt-conditioned and needs prompt_wav plus "
                "prompt_text. The benchmark supplies three fixed heldout "
                "LibriTTS-R speakers, identical across every cloning-capable "
                "system, so the comparison is like for like."
            )
        return self.prompt_wav, self.prompt_text

    def _prompt_state(self):
        """Prompt features and tokens, computed once and reused."""
        if self._prompt_cache is not None:
            return self._prompt_cache

        state = self._ensure_loaded()
        torch = state["torch"]
        device = state["device"]

        from zipvoice.utils.infer import load_prompt_wav

        prompt_wav_path, prompt_text = self._prompt()
        feat_scale, target_rms = 0.1, 0.1

        prompt_wav = load_prompt_wav(prompt_wav_path, sampling_rate=SAMPLING_RATE)
        rms = prompt_wav.pow(2).mean().sqrt()
        if rms < target_rms:
            prompt_wav = prompt_wav * target_rms / rms

        prompt_features = (
            state["features"]
            .extract(prompt_wav, sampling_rate=SAMPLING_RATE)
            .to(device)
        )
        prompt_features = prompt_features.unsqueeze(0) * feat_scale
        prompt_lens = torch.tensor([prompt_features.size(1)], device=device)
        prompt_tokens = state["tokenizer"].texts_to_token_ids([prompt_text])

        self._prompt_cache = (prompt_features, prompt_lens, prompt_tokens)
        return self._prompt_cache

    def _synthesize_mel(self, text: str):
        """Text to mel. The acoustic model only; no vocoder runs here."""
        state = self._ensure_loaded()
        torch = state["torch"]
        model, tokenizer, features = (
            state["model"],
            state["tokenizer"],
            state["features"],
        )
        device = state["device"]

        prompt_features, prompt_lens, prompt_tokens = self._prompt_state()
        feat_scale = 0.1

        tokens = tokenizer.texts_to_token_ids([text])
        _ = features, device

        with torch.inference_mode():
            pred, _, _, _ = model.sample(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_lens,
                speed=1.0,
                t_shift=0.5,
                duration="predict",
                num_step=self.num_step,
                guidance_scale=self.guidance_scale,
            )
        return pred.permute(0, 2, 1) / feat_scale

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthResult:
        state = self._ensure_loaded()
        torch = state["torch"]
        submitted_ns = time.perf_counter_ns()

        try:
            mel = self._synthesize_mel(text)
            with torch.inference_mode():
                wav = state["vocoder"].decode(mel).squeeze(1).clamp(-1, 1)
            pcm = wav.squeeze(0).float().cpu().numpy().astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - a failed system is a result
            now = time.perf_counter_ns()
            return SynthResult(
                pcm=np.zeros(0, dtype=np.float32),
                sample_rate=SAMPLING_RATE,
                timings=Timings(submitted_ns, now, now),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        completed_ns = time.perf_counter_ns()
        if pcm.size == 0:
            return SynthResult(
                pcm=pcm,
                sample_rate=SAMPLING_RATE,
                timings=Timings(submitted_ns, completed_ns, completed_ns),
                status="empty",
            )

        # Non-autoregressive: no audio exists until the whole chunk is generated,
        return SynthResult(
            pcm=pcm,
            sample_rate=SAMPLING_RATE,
            timings=Timings(submitted_ns, completed_ns, completed_ns),
            status="ok",
        )

    def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> Iterator[AudioChunk]:
        """One chunk. ZipVoice cannot emit audio before the chunk is complete."""
        result = self.synthesize(text, voice=voice)
        if result.status == "ok":
            yield AudioChunk(
                pcm=result.pcm,
                sample_rate=result.sample_rate,
                received_ns=result.timings.first_audio_ns,
            )
