#!/usr/bin/env python3
"""Prepare private Qwen3-ASR fine-tuning JSONLs from existing PD splits.

Adapted from the team's ``prepare_qwen_finetuning.py`` (not a replacement for
speaker splitting or transcript cleaning). Source data and audio are read-only.

The added ``category`` field is REQUIRED by the shared category-aware sampler.
Train categories can be explicitly filtered with --train-categories so the
team can choose a common category policy. Validation categories are not filtered
by that option. This script does not silently discard unconfigured categories.

Output: train.jsonl, val.jsonl, train_pilot.jsonl, preparation_report.json.
Output JSONL rows: {"audio", "text", "speaker_id", "category"}.
All outputs are RESTRICTED: do not commit or upload them.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path, PurePosixPath

QWEN_PREFIX = "language English<asr_text>"


def resolve_audio_path(src: dict, *, audio_root: Path | None) -> str:
    """Resolve an audio path without changing the original PD split manifest.

    If the dataset was moved, use audio_relpath (recorded during PD preparation)
    relative to the NEW SAP dataset root. This never copies or edits audio.
    """
    if audio_root is None:
        return src["audio_path"].strip()

    rel = src.get("audio_relpath")
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("--audio-root requires audio_relpath in each source row")
    relative = PurePosixPath(rel.replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or any(
        part in ("", ".", "..") or ":" in part for part in relative.parts
    ):
        raise ValueError("Invalid audio_relpath; refusing to read outside the audio root")
    resolved = audio_root.joinpath(*relative.parts).resolve()
    if audio_root != resolved and audio_root not in resolved.parents:
        raise ValueError("Invalid audio_relpath; refusing to read outside the audio root")
    return str(resolved)


def load_prepared_split(
    path: Path,
    split: str,
    *,
    verify_audio: bool = False,
    max_duration: float = 0.0,
    allowed_categories: set[str] | None = None,
    audio_root: Path | None = None,
) -> tuple[list[dict], set[str], set[str], dict]:
    """Read the prepared PD split, preserving the original audio and transcripts.

    The returned speaker/path sets describe the FULL source split, before filters,
    so overlap detection cannot be bypassed by filtering recordings.
    """
    rows: list[dict] = []
    source_speakers: set[str] = set()
    source_audio_keys: set[str] = set()
    observed_categories: Counter[str] = Counter()
    excluded_categories: Counter[str] = Counter()
    excluded_long: Counter[str] = Counter()
    missing_audio: list[str] = []

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            src = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
        if not isinstance(src, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        required = ("audio_path", "speaker_id", "text", "category")
        if not all(isinstance(src.get(key), str) and src[key].strip() for key in required):
            raise ValueError(f"{path}:{line_no}: audio_path, speaker_id, text, category are required")
        if str(src.get("split", split)).casefold() != split.casefold():
            raise ValueError(f"{path}:{line_no}: expected split={split!r}")

        audio = resolve_audio_path(src, audio_root=audio_root)
        speaker = src["speaker_id"].strip()
        category = src["category"].strip()
        text = src["text"].strip()
        audio_key = audio.replace("\\", "/").casefold()
        if audio_key in source_audio_keys:
            raise ValueError(f"{path}:{line_no}: duplicate audio_path")
        if text.startswith(QWEN_PREFIX):
            raise ValueError(f"{path}:{line_no}: source text already has Qwen prefix")

        source_audio_keys.add(audio_key)
        source_speakers.add(speaker)
        observed_categories[category] += 1

        # Intentionally filter only the TRAIN split when requested. The caller
        # separately passes allowed_categories=None for validation.
        if allowed_categories is not None and category not in allowed_categories:
            excluded_categories[category] += 1
            continue

        if verify_audio or max_duration > 0:
            audio_file = Path(audio)
            if not audio_file.is_file():
                missing_audio.append(audio)
                continue
            if max_duration > 0:
                import soundfile as sf  # lazily required only when measuring duration
                info = sf.info(audio_file)
                if info.samplerate <= 0:
                    raise ValueError(f"{path}:{line_no}: invalid audio sample rate")
                if info.frames / info.samplerate > max_duration:
                    excluded_long[category] += 1
                    continue

        rows.append({
            "audio": audio,
            "text": QWEN_PREFIX + text,
            "speaker_id": speaker,
            "category": category,
        })

    if missing_audio:
        raise ValueError(f"{path}: {len(missing_audio)} missing audio files; first: {missing_audio[0]}")
    if not rows:
        raise ValueError(f"{path}: no eligible recordings after filtering")

    report = {
        "source_recordings": sum(observed_categories.values()),
        "source_speakers": len(source_speakers),
        "retained_recordings": len(rows),
        "retained_speakers": len({r["speaker_id"] for r in rows}),
        "source_categories": dict(sorted(observed_categories.items())),
        "retained_categories": dict(sorted(Counter(r["category"] for r in rows).items())),
        "excluded_by_category": dict(sorted(excluded_categories.items())),
        "excluded_by_duration": dict(sorted(excluded_long.items())),
    }
    return rows, source_speakers, source_audio_keys, report


def make_pilot(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Select at least one recording per retained speaker; deterministic by seed."""
    if size <= 0 or size >= len(rows):
        return list(rows)
    by_speaker: dict[str, list[dict]] = {}
    for row in rows:
        by_speaker.setdefault(row["speaker_id"], []).append(row)
    if size < len(by_speaker):
        raise ValueError(f"pilot-size {size} < retained speaker count {len(by_speaker)}")
    rng = random.Random(seed)
    selected = [rng.choice(by_speaker[s]) for s in sorted(by_speaker)]
    selected_paths = {r["audio"].replace("\\", "/").casefold() for r in selected}
    remaining = [r for r in rows if r["audio"].replace("\\", "/").casefold() not in selected_paths]
    rng.shuffle(remaining)
    selected.extend(remaining[:size - len(selected)])
    return sorted(selected, key=lambda r: r["audio"].replace("\\", "/").casefold())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="existing PD train.jsonl")
    parser.add_argument("--val", type=Path, required=True, help="existing PD val.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="NEW restricted output directory")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="drop recordings longer than this in both splits; 0 disables")
    parser.add_argument("--verify-audio", action="store_true")
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="actual SAP dataset root if the original absolute WAV paths have moved; uses audio_relpath")
    parser.add_argument("--pilot-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-categories", nargs="+", default=None,
                        help="EXPLICITLY restrict training categories; validation retains all categories")
    parser.add_argument("--allow-speaker-drop", action="store_true",
                        help="permit filters to remove all recordings of a TRAIN speaker (not recommended)")
    args = parser.parse_args(argv)

    if args.max_duration < 0 or args.pilot_size < 0:
        parser.error("--max-duration and --pilot-size must be >= 0")
    if any(not p.is_file() for p in (args.train, args.val)):
        parser.error("Train and validation input manifests must both exist")
    output = args.out.resolve()
    if output in (args.train.resolve().parent, args.val.resolve().parent):
        parser.error("Output cannot be a source manifest directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory is not empty; use a fresh directory to prevent stale artifacts")

    audio_root = args.audio_root.resolve() if args.audio_root is not None else None
    if audio_root is not None and not audio_root.is_dir():
        parser.error("--audio-root does not exist or is not a directory")

    categories = None
    if args.train_categories is not None:
        categories = set(args.train_categories)
        if len(categories) != len(args.train_categories):
            parser.error("Duplicate values in --train-categories")

    try:
        train, train_speakers, train_audio, train_report = load_prepared_split(
            args.train, "train", verify_audio=args.verify_audio,
            max_duration=args.max_duration, allowed_categories=categories,
            audio_root=audio_root,
        )
        val, val_speakers, val_audio, val_report = load_prepared_split(
            args.val, "val", verify_audio=args.verify_audio, max_duration=args.max_duration,
            audio_root=audio_root,
        )
        if train_speakers & val_speakers:
            raise ValueError("Train/Validation source speaker overlap")
        if train_audio & val_audio:
            raise ValueError("Train/Validation source audio overlap")
        if categories is not None:
            retained = {row["category"] for row in train}
            if retained != categories:
                raise ValueError(f"Requested categories absent after filtering: {sorted(categories - retained)}")
        if train_report["source_speakers"] != train_report["retained_speakers"] and not args.allow_speaker_drop:
            raise ValueError(
                "Some TRAIN speakers have zero retained recordings; adjust filters or explicitly "
                "pass --allow-speaker-drop after reviewing the effect on the study"
            )
        pilot = make_pilot(train, args.pilot_size, args.seed)
    except ValueError as exc:
        parser.error(str(exc))

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "val.jsonl", val)
    write_jsonl(output / "train_pilot.jsonl", pilot)
    report = {
        "format": "Qwen3-ASR audio/text plus speaker_id/category for shared sampling",
        "qwen_prefix": QWEN_PREFIX,
        "source": {"train": str(args.train.resolve()), "val": str(args.val.resolve())},
        "max_duration_seconds": args.max_duration,
        "audio_root_override": str(audio_root) if audio_root is not None else None,
        "audio_exists_checked": bool(args.verify_audio or args.max_duration > 0),
        "train_categories_allowed": sorted(categories) if categories else "ALL",
        "train": train_report,
        "validation": val_report,
        "pilot": {"recordings": len(pilot), "speakers": len({r["speaker_id"] for r in pilot})},
        "speaker_overlap": 0,
        "audio_overlap": 0,
        "seed": args.seed,
        "privacy": "PRIVATE: manifest JSONLs include restricted source paths, transcripts, and speaker IDs",
    }
    (output / "preparation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Train: {len(train):,} recordings / {train_report['retained_speakers']} speakers")
    print(f"Validation: {len(val):,} recordings / {val_report['retained_speakers']} speakers")
    print(f"Pilot: {len(pilot):,} recordings")
    print(f"Private Qwen manifests written to: {output}")


if __name__ == "__main__":
    main()
