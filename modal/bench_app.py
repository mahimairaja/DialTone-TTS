"""Modal app for the handset-bench matrix."""

from __future__ import annotations

import json
import os
import pathlib

import modal

MINUTES = 60

#: Raises the Hub rate limit and speeds up weight downloads. Optional: a machine
try:
    HF_SECRET = [modal.Secret.from_name("huggingface")]
except Exception:  # noqa: BLE001 - absence is a valid state, not an error
    HF_SECRET = []

REPO_ROOT = pathlib.Path(__file__).parent.parent
BENCH_SRC = REPO_ROOT / "packages" / "handset-bench" / "src" / "handset_bench"

app = modal.App("dialtone-bench")

audio_vol = modal.Volume.from_name("handset-bench-audio", create_if_missing=True)
cache_vol = modal.Volume.from_name("handset-bench-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("dialtone-data", create_if_missing=True)

CONDITION_NAMES = ("wideband", "clean", "loss_1pct", "loss_3pct")

_base = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch==2.13.0",
    "torchaudio==2.11.0",
    "numpy==2.5.2",
    "soundfile==0.14.0",
    "jiwer==4.0.0",
    "whisper-normalizer==0.1.15",
)

# add_local_dir must be the LAST step in each chain. Modal mounts these files at
_MOUNT = ("/root/handset_bench",)

piper_image = _base.uv_pip_install("piper-tts==1.6.1").add_local_dir(
    BENCH_SRC, "/root/handset_bench"
)

# ZipVoice is installed from a pinned git checkout, not from PyPI.
ZIPVOICE_SHA = "2f7326fbfe999a3ad179e3f1af82a424d4a62819"

zipvoice_image = (
    _base.apt_install("espeak-ng", "git", "ffmpeg")
    .run_commands(
        "git clone https://github.com/k2-fsa/ZipVoice.git /opt/ZipVoice",
        f"cd /opt/ZipVoice && git checkout {ZIPVOICE_SHA}",
    )
    .uv_pip_install(
        # From the upstream requirements.txt, minus `k2`, which is required for
        "piper_phonemize==1.2.0",
        # torchaudio routes audio loading through TorchCodec, which upstream
        "torchcodec",
        "lhotse",
        "tensorboard",
        "vocos",
        "pydub",
        "cn2an",
        "inflect",
        "jieba",
        "pypinyin",
        "huggingface_hub>=0.34",
        "safetensors>=0.4",
        "setuptools<81",
        # Named explicitly because the packages that import them do not declare
        "urllib3",
        "requests",
        "tqdm",
        "packaging",
        "intervaltree",
        "cytoolz",
        "click",
        find_links="https://k2-fsa.github.io/icefall/piper_phonemize.html",
    )
    .env({"PYTHONPATH": "/opt/ZipVoice:/root"})
    .add_local_dir(BENCH_SRC, "/root/handset_bench")
)

_SITE = "/usr/local/lib/python3.12/site-packages"

# faster-whisper 1.2.0 imports `requests` and pulls weights through
whisper_image = (
    _base.apt_install("ffmpeg")
    .uv_pip_install(
        "faster-whisper==1.2.0",
        "requests>=2.32",
        "huggingface_hub>=0.34",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    )
    .run_commands(
        f"echo {_SITE}/nvidia/cublas/lib > /etc/ld.so.conf.d/nvidia-wheels.conf",
        f"echo {_SITE}/nvidia/cudnn/lib >> /etc/ld.so.conf.d/nvidia-wheels.conf",
        "ldconfig",
    )
    .add_local_dir(BENCH_SRC, "/root/handset_bench")
)

# NeMo is a large image and pins its own torch and numpy. It is built from a bare
parakeet_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "nemo_toolkit[asr]==2.6.0",
        "soundfile",
        "jiwer",
        "whisper-normalizer",
        "huggingface_hub>=0.34",
    )
    .add_local_dir(BENCH_SRC, "/root/handset_bench")
)

# Parakeet is opt-in, and the reason is structural rather than stylistic.
PARAKEET_ENABLED = os.environ.get("DIALTONE_PARAKEET", "0") == "1"


def _k2_available() -> bool:
    """Whether k2 is importable in this container."""
    try:
        import k2  # noqa: F401
    except Exception:
        return False
    return True


def _load_prompt() -> tuple[str, str] | tuple[None, None]:
    """The fixed prompt reference for cloning-capable systems."""
    prompts_path = pathlib.Path("/data/prompts/prompts.json")
    if not prompts_path.exists():
        return None, None
    prompts = json.loads(prompts_path.read_text())
    if not prompts:
        return None, None
    speaker = sorted(prompts)[0]
    return prompts[speaker]["wav"], prompts[speaker]["text"]


def _run_generation(adapter, system: str, utterances, out_root: pathlib.Path) -> dict:
    """Synthesise once, then derive every condition from that single waveform."""
    import numpy as np
    import torch
    from handset_bench import codec
    from handset_bench.conditions import CONDITIONS, apply

    manifest: list[dict] = []
    for utterance in utterances:
        result = adapter.synthesize(utterance.text)
        entry = {
            "utterance_id": utterance.utterance_id,
            "text": utterance.text,
            "status": result.status,
            "error": result.error,
            "ttfb_generation_ms": result.timings.ttfb_generation_ms,
            "total_ms": result.timings.total_ms,
            "audio_seconds": (
                result.pcm.size / result.sample_rate if result.pcm.size else 0.0
            ),
            "paths": {},
        }
        if result.status == "ok":
            pcm = torch.from_numpy(result.pcm.copy()).float()
            for name in CONDITION_NAMES:
                seed = codec.condition_seed(utterance.utterance_id, name, 0)
                conditioned = apply(
                    CONDITIONS[name], pcm, result.sample_rate, seed=seed
                )
                for_asr = codec.to_asr_rate(conditioned.pcm, conditioned.sample_rate)
                directory = out_root / name
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{utterance.utterance_id}.npy"
                np.save(path, for_asr.numpy().astype(np.float32))
                entry["paths"][name] = str(path)
        manifest.append(entry)

    manifest_path = out_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    audio_vol.commit()

    return {
        "system": system,
        "manifest_path": str(manifest_path),
        "n_utterances": len(manifest),
        "n_ok": sum(1 for e in manifest if e["status"] == "ok"),
        "errors": [e["error"] for e in manifest if e["error"]][:3],
        # Per-utterance generation status and timing, returned so the local side
        "index": {
            e["utterance_id"]: {
                "status": e["status"],
                "error": e["error"],
                "ttfb_generation_ms": e["ttfb_generation_ms"],
                "total_ms": e["total_ms"],
                "audio_seconds": e["audio_seconds"],
            }
            for e in manifest
        },
    }


@app.function(
    image=piper_image,
    cpu=4.0,
    timeout=30 * MINUTES,
    max_containers=2,
    volumes={"/audio": audio_vol, "/cache": cache_vol},
    secrets=HF_SECRET,
)
def generate_piper(voice: str = "en_US-lessac-medium", limit: int = 0) -> dict:
    import sys

    sys.path.insert(0, "/root")
    from handset_bench.adapters.piper import PiperAdapter
    from handset_bench.textset import loader

    utterances = loader.sample(loader.load(), limit)
    adapter = PiperAdapter(voice=voice, download_dir="/cache/piper")
    adapter.synthesize("warm up the engine please")

    version = adapter.version_string()
    out = pathlib.Path("/audio") / "piper" / version.replace("/", "_")
    result = _run_generation(adapter, "piper", utterances, out)
    result["version"] = version
    return result


@app.function(
    image=zipvoice_image,
    gpu="A10G",
    timeout=60 * MINUTES,
    max_containers=2,
    volumes={"/audio": audio_vol, "/cache": cache_vol, "/data": data_vol},
    secrets=HF_SECRET,
)
def generate_zipvoice(model_name: str = "zipvoice_distill", limit: int = 0) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import os

    os.environ.setdefault("HF_HOME", "/cache/hf")

    from handset_bench.adapters.zipvoice import ZipVoiceAdapter
    from handset_bench.textset import loader

    data_vol.reload()
    prompt_wav, prompt_text = _load_prompt()
    if not prompt_wav:
        return {
            "system": model_name,
            "status": "unavailable",
            "reason": (
                "no prompt reference found on the data volume. Run "
                "modal/data_app.py first: ZipVoice is prompt-conditioned and the "
                "benchmark uses three fixed heldout LibriTTS-R speakers."
            ),
        }

    utterances = loader.sample(loader.load(), limit)
    adapter = ZipVoiceAdapter(
        model_name, prompt_wav=prompt_wav, prompt_text=prompt_text
    )
    adapter.synthesize("warm up the model please")

    version = adapter.version_string()
    out = pathlib.Path("/audio") / model_name / version.replace("/", "_")
    result = _run_generation(adapter, model_name, utterances, out)
    result["version"] = version
    cache_vol.commit()
    return result


@app.function(
    image=zipvoice_image,
    gpu="A10G",
    timeout=30 * MINUTES,
    volumes={"/cache": cache_vol, "/data": data_vol},
    secrets=HF_SECRET,
)
def measure_chunking(model_name: str = "zipvoice_distill") -> dict:
    """The load-bearing measurement for the whole base-model choice."""
    import sys

    sys.path.insert(0, "/root")
    import os

    os.environ.setdefault("HF_HOME", "/cache/hf")

    from handset_bench.adapters.zipvoice import ZipVoiceAdapter
    from handset_bench.chunking import ChunkedDriver, split_sentences

    data_vol.reload()
    prompt_wav, prompt_text = _load_prompt()
    if not prompt_wav:
        return {"status": "unavailable", "reason": "no prompt reference available"}

    adapter = ZipVoiceAdapter(
        model_name, prompt_wav=prompt_wav, prompt_text=prompt_text
    )
    adapter.synthesize("warm up the model please")

    # Multi-sentence agent turns, the shape a phone assistant actually emits.
    messages = [
        "Thanks for calling. Let me pull up your account. One moment please.",
        "Your appointment is confirmed. You will get a text shortly. "
        "Is there anything else I can help with?",
        "I can see the payment went through this morning. "
        "The balance is now zero. Have a good day.",
    ]

    driver = ChunkedDriver(adapter)
    rows = []
    for message in messages:
        chunked = driver.synthesize_chunked(message)
        whole = adapter.synthesize(message)
        audio_s = chunked.pcm.size / chunked.sample_rate if chunked.pcm.size else 0.0
        rows.append(
            {
                "message": message,
                "n_chunks": chunked.n_chunks,
                "chunks": split_sentences(message),
                "ttfb_first_chunk_ms": round(chunked.ttfb_first_chunk_ms, 1),
                "ttft_full_message_ms": round(chunked.ttft_full_message_ms, 1),
                "unchunked_total_ms": round(whole.timings.total_ms, 1),
                "audio_seconds": round(audio_s, 2),
                "realtime_factor": (
                    round(audio_s / (chunked.ttft_full_message_ms / 1000), 1)
                    if chunked.ttft_full_message_ms
                    else None
                ),
            }
        )

    first_chunk = [r["ttfb_first_chunk_ms"] for r in rows]
    return {
        "status": "ok",
        "model": model_name,
        "version": adapter.version_string(),
        "rows": rows,
        "median_ttfb_first_chunk_ms": sorted(first_chunk)[len(first_chunk) // 2],
        "worst_ttfb_first_chunk_ms": max(first_chunk),
    }


@app.function(
    image=zipvoice_image,
    gpu="A10G",
    timeout=30 * MINUTES,
    volumes={"/cache": cache_vol},
    secrets=HF_SECRET,
)
def smoke_zipvoice(model_name: str = "zipvoice_distill") -> dict:
    """Prove the ZipVoice image and adapter work, with no corpus dependency."""
    import sys
    import wave

    sys.path.insert(0, "/root")
    import os

    os.environ.setdefault("HF_HOME", "/cache/hf")

    import numpy as np
    from handset_bench.adapters.base import validate_adapter
    from handset_bench.adapters.zipvoice import ZipVoiceAdapter

    prompt_wav = "/tmp/smoke_prompt.wav"
    rate = 24000
    t = np.arange(int(rate * 4.0)) / rate
    tone = 0.2 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    with wave.open(prompt_wav, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((tone * 32767).astype("<i2").tobytes())

    adapter = ZipVoiceAdapter(
        model_name, prompt_wav=prompt_wav, prompt_text="this is a test prompt"
    )
    validate_adapter(adapter)

    result = adapter.synthesize("Your confirmation code is 4 7 B 2 9 K.")
    summary = {
        "version": adapter.version_string(),
        "status": result.status,
        "error": result.error,
        "sample_rate": result.sample_rate,
        "n_samples": int(result.pcm.size),
        "audio_seconds": round(result.pcm.size / result.sample_rate, 2)
        if result.pcm.size
        else 0.0,
        "total_ms": round(result.timings.total_ms, 1),
        "k2_available": _k2_available(),
    }
    print("SMOKE RESULT:", json.dumps(summary))
    return summary


def _transcribe_with(model_fn, manifest_path: str, conditions: list[str]) -> dict:
    import numpy as np

    audio_vol.reload()
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    out: dict[str, dict[str, str]] = {}
    for condition in conditions:
        hypotheses: dict[str, str] = {}
        for entry in manifest:
            if entry["status"] != "ok":
                hypotheses[entry["utterance_id"]] = ""
                continue
            audio = np.load(entry["paths"][condition]).astype(np.float32)
            hypotheses[entry["utterance_id"]] = model_fn(audio)
        out[condition] = hypotheses
    return out


@app.function(
    image=whisper_image,
    gpu="A10G",
    timeout=60 * MINUTES,
    max_containers=2,
    volumes={"/audio": audio_vol, "/cache": cache_vol},
    secrets=HF_SECRET,
)
def transcribe_whisper(manifest_path: str, conditions: list[str]) -> dict:
    """Second-column listener. Greedy, no temperature fallback, no VAD filter."""
    import sys

    sys.path.insert(0, "/root")
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "large-v3", device="cuda", compute_type="float16", download_root="/cache/fw"
    )

    def run(audio):
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(s.text for s in segments).strip()

    by_condition = _transcribe_with(run, manifest_path, conditions)
    cache_vol.commit()
    return {
        "asr_backend": "faster-whisper-large-v3",
        "asr_revision": "1.2.0",
        "by_condition": by_condition,
    }


def _transcribe_parakeet(manifest_path: str, conditions: list[str]) -> dict:
    """Headline listener."""
    import sys

    sys.path.insert(0, "/root")
    import os
    import tempfile

    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("NEMO_CACHE_DIR", "/cache/nemo")

    import nemo.collections.asr as nemo_asr
    import soundfile as sf

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    model.eval()

    def run(audio):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
            sf.write(handle.name, audio, 16000)
            output = model.transcribe([handle.name], batch_size=1, verbose=False)
        first = output[0]
        return (getattr(first, "text", None) or str(first)).strip()

    by_condition = _transcribe_with(run, manifest_path, conditions)
    cache_vol.commit()
    return {
        "asr_backend": "nvidia/parakeet-tdt-0.6b-v2",
        "asr_revision": "nemo-2.6.0",
        "by_condition": by_condition,
    }


# Registered only when enabled, so the NeMo image is not built otherwise. Modal
transcribe_parakeet = (
    app.function(
        image=parakeet_image,
        gpu="A10G",
        timeout=60 * MINUTES,
        max_containers=2,
        volumes={"/audio": audio_vol, "/cache": cache_vol},
    )(_transcribe_parakeet)
    if PARAKEET_ENABLED
    else None
)

# --------------------------------------------------------------------- local

#: Conservative per-system cost estimates in US dollars, for the pre-flight guard.
_COST_ESTIMATE_USD = {
    "piper": 0.15,
    "zipvoice_distill": 2.50,
    "zipvoice": 4.00,
    "chunking": 0.60,
    "whisper": 1.20,
    "parakeet": 2.00,
}

GENERATORS = {
    "piper": ("generate_piper", {}),
    "zipvoice_distill": ("generate_zipvoice", {"model_name": "zipvoice_distill"}),
    "zipvoice": ("generate_zipvoice", {"model_name": "zipvoice"}),
}


@app.local_entrypoint()
def run_matrix(
    systems: str = "piper,zipvoice_distill",
    listeners: str = "whisper",
    limit: int = 0,
    max_cost_usd: float = 30.0,
    chunking: bool = True,
):
    """Run the matrix, cheap-first, refusing to launch beyond the cost ceiling."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "packages" / "handset-bench" / "src"))

    from handset_bench import report, scoring
    from handset_bench.textset import loader

    wanted = [s.strip() for s in systems.split(",") if s.strip()]
    listener_names = [s.strip() for s in listeners.split(",") if s.strip()]

    unknown = [s for s in wanted if s not in GENERATORS]
    if unknown:
        raise SystemExit(f"unknown systems {unknown}. Known: {sorted(GENERATORS)}")

    estimate = sum(_COST_ESTIMATE_USD.get(s, 1.0) for s in wanted)
    estimate += sum(_COST_ESTIMATE_USD.get(name, 1.0) for name in listener_names)
    if chunking:
        estimate += _COST_ESTIMATE_USD["chunking"]
    scale = (limit / 300) if limit else 1.0
    estimate *= max(scale, 0.1)

    print(f"estimated cost ${estimate:.2f}, ceiling ${max_cost_usd:.2f}")
    if estimate > max_cost_usd:
        raise SystemExit(
            f"refusing to launch: estimate ${estimate:.2f} exceeds ceiling "
            f"${max_cost_usd:.2f}"
        )

    # Cheap-first is a hard sequence, not a preference. Piper gates the rest.
    ordered = [s for s in ("piper", "zipvoice_distill", "zipvoice") if s in wanted]
    utterances = loader.sample(loader.load(), limit)
    results_root = REPO_ROOT / "results"
    records: list[dict] = []

    for system in ordered:
        fn_name, kwargs = GENERATORS[system]
        print(f"\n=== generating: {system} ===")
        generated = globals()[fn_name].remote(limit=limit, **kwargs)

        if generated.get("status") == "unavailable":
            print(f"  UNAVAILABLE: {generated['reason']}")
            records.append(
                {
                    "system": system,
                    "version": "unavailable",
                    "condition": "clean",
                    "mode": "quality",
                    "status": "unavailable",
                    "unavailable_reason": generated["reason"],
                    "asr_backend": "none",
                    "aggregate": {},
                    "per_utterance": [],
                }
            )
            continue

        print(f"  version={generated['version']} ok={generated['n_ok']}")
        if generated.get("errors"):
            print(f"  sample errors: {generated['errors']}")

        # The audio manifest stays on the Volume; the returned index carries the
        gen_index = generated.get("index") or {}

        for listener in listener_names:
            fn = transcribe_parakeet if listener == "parakeet" else transcribe_whisper
            if fn is None:
                print(
                    f"  {listener} is not enabled, skipping. Set "
                    "DIALTONE_PARAKEET=1 to build the NeMo image "
                    "(run_baseline.sh --parakeet does it)."
                )
                continue
            print(f"  transcribing with {listener}")
            scored = fn.remote(generated["manifest_path"], list(CONDITION_NAMES))

            for condition in CONDITION_NAMES:
                record = scoring.build_quality_record(
                    system=system,
                    version=generated["version"],
                    condition=condition,
                    utterances=utterances,
                    hypotheses=scored["by_condition"][condition],
                    generation=gen_index,
                    asr_backend=scored["asr_backend"],
                    asr_revision=scored["asr_revision"],
                )
                records.append(record)
                directory = (
                    results_root / system / generated["version"].replace("/", "_")
                )
                directory.mkdir(parents=True, exist_ok=True)
                suffix = "" if listener == "parakeet" else f".{listener}"
                (directory / f"{condition}{suffix}.json").write_text(
                    json.dumps(record, indent=2, sort_keys=True)
                )
                print(f"    {condition}: WER {record['aggregate']['wer'] * 100:.2f}%")

        latency = scoring.build_latency_record(
            system=system, version=generated["version"], generation=gen_index
        )
        records.append(latency)
        directory = results_root / system / generated["version"].replace("/", "_")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "clean.latency.json").write_text(
            json.dumps(latency, indent=2, sort_keys=True)
        )

    if chunking and any(s.startswith("zipvoice") for s in ordered):
        print("\n=== chunked-driver measurement ===")
        measured = measure_chunking.remote("zipvoice_distill")
        (results_root / "CHUNKING.json").write_text(json.dumps(measured, indent=2))
        if measured.get("status") == "ok":
            print(
                f"  median time to first chunk: "
                f"{measured['median_ttfb_first_chunk_ms']} ms"
            )
            print(f"  worst: {measured['worst_ttfb_first_chunk_ms']} ms")
            for row in measured["rows"]:
                print(
                    f"    {row['n_chunks']} chunks | first "
                    f"{row['ttfb_first_chunk_ms']} ms | whole "
                    f"{row['ttft_full_message_ms']} ms | rtf {row['realtime_factor']}"
                )
        else:
            print(f"  UNAVAILABLE: {measured.get('reason')}")

    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "SCORECARD.md").write_text(report.render_scorecard(records))
    (results_root / "scorecard.csv").write_text(report.render_csv(records))
    (results_root / "NUMBERS_TO_BEAT.md").write_text(
        report.render_numbers_to_beat(records)
    )
    print(f"\nwrote {results_root / 'SCORECARD.md'}")
