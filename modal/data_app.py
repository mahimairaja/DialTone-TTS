"""Modal app for corpus ingest: LibriTTS-R, licence gate, frozen split, provenance."""

from __future__ import annotations

import json
import pathlib

import modal

MINUTES = 60

REPO_ROOT = pathlib.Path(__file__).parent.parent
DIALTONE_SRC = REPO_ROOT / "packages" / "dialtone" / "src" / "dialtone"

# Pinned exactly. dialtone imports the codec chain from here and never
# reimplements it, so a drift would separate training from evaluation silently.
BENCH_PIN = "handset-bench==0.1.0"

app = modal.App("dialtone-data")

data_vol = modal.Volume.from_name("dialtone-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "tar")
    .uv_pip_install("numpy==2.5.2", "soundfile==0.14.0", BENCH_PIN)
    .add_local_dir(DIALTONE_SRC, "/root/dialtone")
)

#: Subsets ingested by default.
DEFAULT_SUBSETS = ("dev-clean", "test-clean", "train-clean-100")


@app.function(
    image=image,
    cpu=8.0,
    timeout=60 * MINUTES,
    max_containers=3,
    volumes={"/data": data_vol},
)
def download_subset(subset: str) -> dict:
    """Fetch and extract one subset onto the Volume. Idempotent."""
    import subprocess
    import sys

    sys.path.insert(0, "/root")
    from dialtone.data.sources.libritts_r import tarball_url

    root = pathlib.Path("/data/libritts_r")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / f".{subset}.complete"

    if marker.exists():
        return {"subset": subset, "cached": True}

    url = tarball_url(subset)
    archive = pathlib.Path(f"/tmp/{subset}.tar.gz")
    print(f"downloading {url}")
    subprocess.run(["curl", "-sSL", "-o", str(archive), url], check=True)

    print(f"extracting {archive}")
    # The tarball contains a LibriTTS_R/ top level directory; strip it so the
    subprocess.run(
        ["tar", "-xzf", str(archive), "-C", str(root), "--strip-components=1"],
        check=True,
    )
    archive.unlink(missing_ok=True)
    marker.write_text("ok")
    data_vol.commit()

    n_wav = sum(1 for _ in (root / subset).rglob("*.wav"))
    return {"subset": subset, "cached": False, "n_wav": n_wav}


@app.function(
    image=image,
    cpu=4.0,
    timeout=45 * MINUTES,
    max_containers=12,
    volumes={"/data": data_vol},
)
def scan_shard(spec: tuple[str, list[str]]) -> list[dict]:
    """Walk one shard of one subset and return its records."""
    import sys
    from dataclasses import asdict

    sys.path.insert(0, "/root")
    from dialtone.data.sources.libritts_r import ingest_subset

    subset, speakers = spec
    data_vol.reload()
    root = pathlib.Path("/data/libritts_r")
    records = [asdict(r) for r in ingest_subset(root, subset, speakers)]
    print(f"{subset} [{len(speakers)} speakers]: {len(records)} utterances")
    return records


@app.function(
    image=image,
    cpu=2.0,
    timeout=30 * MINUTES,
    volumes={"/data": data_vol},
)
def plan_shards(subsets: list[str], per_shard: int = 40) -> list[tuple]:
    """List speakers per subset and chunk them into shards."""
    import sys

    sys.path.insert(0, "/root")
    from dialtone.data.sources.libritts_r import list_speakers

    data_vol.reload()
    root = pathlib.Path("/data/libritts_r")
    shards: list[tuple] = []
    for subset in subsets:
        speakers = list_speakers(root, subset)
        for start in range(0, len(speakers), per_shard):
            shards.append((subset, speakers[start : start + per_shard]))
        print(f"{subset}: {len(speakers)} speakers")
    print(f"planned {len(shards)} shards")
    return shards


@app.function(
    image=image,
    cpu=4.0,
    timeout=30 * MINUTES,
    volumes={"/data": data_vol},
)
def build_manifest(scanned: list[list[dict]]) -> dict:
    """Gate on licence, freeze the split, select voices, render provenance."""
    import sys

    sys.path.insert(0, "/root")
    from dialtone.data.licence import gate
    from dialtone.data.manifest import ManifestRecord, write_manifest
    from dialtone.data.provenance import render_data_card
    from dialtone.data.split import make_split, write_split
    from dialtone.data.voices import select_prompt_speakers, select_voices

    data_vol.reload()

    records = [ManifestRecord(**row) for batch in scanned for row in batch]
    print(f"scanned {len(records)} utterances across {len(scanned)} subsets")

    accepted, rejected = gate(records)
    print(f"licence gate: {len(accepted)} accepted, {len(rejected)} rejected")

    out = pathlib.Path("/data/manifests")
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(accepted, out / "libritts_r.jsonl")

    split = make_split(accepted)
    write_split(split, out / "split_v1.json")

    voices = select_voices(accepted)
    prompts = select_prompt_speakers(accepted, split)
    data_card = render_data_card(accepted, removals=[])
    data_vol.commit()

    return {
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "total_hours": round(sum(r.duration_s for r in accepted) / 3600, 2),
        "n_speakers": len({r.speaker_id for r in accepted}),
        "split": {
            "heldout_basis": split.heldout_basis,
            "counts": split.counts,
            "manifest_hash": split.manifest_hash,
            "heldout_speaker_ids": list(split.heldout_speaker_ids),
            "train_speaker_ids": list(split.train_speaker_ids),
            "heldout_utterance_ids": list(split.heldout_utterance_ids),
        },
        "voices": [
            {
                "id": v.id,
                "speaker_id": v.speaker_id,
                "subset": v.subset,
                "total_duration_s": v.total_duration_s,
                "selection_rule": v.selection_rule,
            }
            for v in voices
        ],
        "prompt_speakers": prompts,
        "data_card": data_card,
    }


@app.function(
    image=image,
    cpu=4.0,
    timeout=30 * MINUTES,
    volumes={"/data": data_vol},
)
def export_prompts(prompt_speakers: list[str]) -> dict:
    """Copy one reference utterance per prompt speaker to a stable location."""
    import shutil
    import sys

    sys.path.insert(0, "/root")
    from dialtone.data.manifest import read_manifest

    data_vol.reload()
    records = read_manifest(pathlib.Path("/data/manifests/libritts_r.jsonl"))
    root = pathlib.Path("/data/libritts_r")
    out = pathlib.Path("/data/prompts")
    out.mkdir(parents=True, exist_ok=True)

    exported = {}
    for speaker in prompt_speakers:
        candidates = [r for r in records if r.speaker_id == speaker]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda r: abs(r.duration_s - 5.0))
        target_wav = out / f"{speaker}.wav"
        shutil.copyfile(root / chosen.audio_path, target_wav)
        (out / f"{speaker}.txt").write_text(chosen.text, encoding="utf-8")
        exported[speaker] = {
            "wav": str(target_wav),
            "text": chosen.text,
            "duration_s": chosen.duration_s,
            "utterance_id": chosen.utterance_id,
        }

    (out / "prompts.json").write_text(json.dumps(exported, indent=2))
    data_vol.commit()
    return exported


@app.local_entrypoint()
def ingest(subsets: str = ",".join(DEFAULT_SUBSETS)):
    """Run corpus ingest end to end and write the frozen artefacts into the repo."""
    from dialtone.data.split import dumps_split

    names = [s.strip() for s in subsets.split(",") if s.strip()]

    print(f"downloading subsets: {names}")
    for result in download_subset.map(names):
        print(f"  {result}")

    print("planning scan shards")
    shards = plan_shards.remote(names)

    print(f"scanning {len(shards)} shards in parallel")
    scanned = list(scan_shard.map(shards))

    print("licence gate, frozen split, voice selection")
    summary = build_manifest.remote(scanned)
    print(
        f"  {summary['n_accepted']} utterances, {summary['total_hours']}h, "
        f"{summary['n_speakers']} speakers, {summary['n_rejected']} rejected"
    )
    print(f"  split: {summary['split']['counts']}")
    print(f"  prompt speakers: {summary['prompt_speakers']}")

    exported = export_prompts.remote(summary["prompt_speakers"])
    print(f"  exported {len(exported)} prompt references")

    # Write the frozen artefacts into the repository, where they are committed.
    (REPO_ROOT / "manifests").mkdir(exist_ok=True)
    split_payload = dict(summary["split"])
    split_payload["seed"] = 20260814
    (REPO_ROOT / "manifests" / "split_v1.json").write_text(dumps_split(split_payload))

    (REPO_ROOT / "configs").mkdir(exist_ok=True)
    voices_yaml = ["# Generated by modal/data_app.py. Do not edit by hand.", "voices:"]
    for voice in summary["voices"]:
        voices_yaml += [
            f"  - id: {voice['id']}",
            f'    speaker_id: "{voice["speaker_id"]}"',
            f"    subset: {voice['subset']}",
            f"    total_duration_s: {voice['total_duration_s']}",
        ]
    voices_yaml += [
        f'selection_rule: "{summary["voices"][0]["selection_rule"]}"',
        "prompt_speakers:",
    ]
    voices_yaml += [f'  - "{s}"' for s in summary["prompt_speakers"]]
    (REPO_ROOT / "configs" / "voices.yaml").write_text("\n".join(voices_yaml) + "\n")

    (REPO_ROOT / "DATA_CARD.md").write_text(summary["data_card"])

    print("\nwrote manifests/split_v1.json, configs/voices.yaml, DATA_CARD.md")
