#!/usr/bin/env python3
"""Prepare private Qwen3-ASR fine-tuning manifests from PD split manifests.

The source manifests and audio are read-only. Output rows contain only the two
fields expected by Qwen's official fine-tuning script: ``audio`` and ``text``.
Speaker IDs remain in memory only for validation and summary counts.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


QWEN_PREFIX = "language English<asr_text>"


def read_manifest(path, expected_split, verify_audio, max_duration):
    rows = []
    speakers = set()
    audio_paths = set()
    categories = Counter()
    missing_audio = []
    dropped_too_long = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            required = {"audio_path", "speaker_id", "text"}
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields: {', '.join(missing)}")
            audio = str(row["audio_path"]).strip()
            speaker = str(row["speaker_id"]).strip()
            text = str(row["text"]).strip()
            split = str(row.get("split", expected_split)).strip().casefold()
            if not audio or not speaker or not text:
                raise ValueError(f"{path}:{line_number}: audio_path, speaker_id and text must be nonempty")
            if split != expected_split.casefold():
                raise ValueError(
                    f"{path}:{line_number}: expected split '{expected_split}', found '{row.get('split')}'"
                )
            key = audio.replace("\\", "/").casefold()
            if key in audio_paths:
                raise ValueError(f"{path}:{line_number}: duplicate audio_path: {audio}")
            audio_paths.add(key)
            speakers.add(speaker)
            if verify_audio or max_duration > 0:
                audio_file = Path(audio)
                if not audio_file.is_file():
                    missing_audio.append(audio)
                    continue
                if max_duration > 0:
                    import soundfile as sf
                    info = sf.info(audio_file)
                    duration = info.frames / info.samplerate
                    if duration > max_duration:
                        dropped_too_long += 1
                        continue
            categories[str(row.get("category") or "MISSING")] += 1
            rows.append({
                "audio": audio,
                "text": QWEN_PREFIX + text,
                "_speaker_id": speaker,
                "_category": str(row.get("category") or "MISSING"),
                "_audio_key": key,
            })
    if not rows:
        raise ValueError(f"{path}: no data rows")
    if missing_audio:
        sample = "\n  ".join(missing_audio[:5])
        raise ValueError(f"{path}: {len(missing_audio)} audio files are missing; first paths:\n  {sample}")
    return rows, speakers, audio_paths, categories, dropped_too_long


def make_pilot(rows, size, seed):
    """Deterministic natural-distribution sample with every speaker represented."""
    if size <= 0 or size >= len(rows):
        return list(rows)
    by_speaker = {}
    for row in rows:
        by_speaker.setdefault(row["_speaker_id"], []).append(row)
    if size < len(by_speaker):
        raise ValueError(
            f"--pilot-size {size} is smaller than the {len(by_speaker)} Train speakers; "
            "increase it so every speaker can be represented"
        )
    rng = random.Random(seed)
    selected_keys = set()
    pilot = []
    for speaker in sorted(by_speaker):
        choice = rng.choice(by_speaker[speaker])
        selected_keys.add(choice["_audio_key"])
        pilot.append(choice)
    remaining = [row for row in rows if row["_audio_key"] not in selected_keys]
    rng.shuffle(remaining)
    pilot.extend(remaining[: size - len(pilot)])
    return sorted(pilot, key=lambda row: row["_audio_key"])


def public_row(row):
    return {"audio": row["audio"], "text": row["text"]}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(public_row(row), ensure_ascii=False) + "\n")


def describe(rows):
    return {
        "recordings": len(rows),
        "speakers": len({row["_speaker_id"] for row in rows}),
        "categories": dict(sorted(Counter(row["_category"] for row in rows).items())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="prepared train.jsonl")
    parser.add_argument("--val", type=Path, required=True, help="prepared val.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="new private output directory")
    parser.add_argument("--pilot-size", type=int, default=8000,
                        help="Train pilot rows; 0 means the full Train manifest (default: 8000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-audio", action="store_true",
                        help="require every source audio_path to exist on this computer")
    parser.add_argument("--max-duration", type=float, default=0,
                        help="drop clips longer than this many seconds; 0 keeps all clips")
    args = parser.parse_args(argv)
    if args.pilot_size < 0 or args.max_duration < 0:
        parser.error("--pilot-size and --max-duration must be nonnegative")
    for path in (args.train, args.val):
        if not path.is_file():
            parser.error(f"Manifest does not exist: {path}")
    out = args.out.resolve()
    if out in (args.train.resolve().parent, args.val.resolve().parent):
        parser.error("--out must be a separate directory, not a source-manifest directory")

    try:
        train, train_speakers, train_audio, _, train_too_long = read_manifest(
            args.train, "train", args.verify_audio, args.max_duration
        )
        val, val_speakers, val_audio, _, val_too_long = read_manifest(
            args.val, "val", args.verify_audio, args.max_duration
        )
        speaker_overlap = train_speakers & val_speakers
        audio_overlap = train_audio & val_audio
        if speaker_overlap:
            raise ValueError(f"Train/Validation speaker overlap: {len(speaker_overlap)}")
        if audio_overlap:
            raise ValueError(f"Train/Validation audio overlap: {len(audio_overlap)}")
        pilot = make_pilot(train, args.pilot_size, args.seed)
    except ValueError as exc:
        parser.error(str(exc))

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "train_pilot.jsonl", pilot)
    report = {
        "format": {
            "fields": ["audio", "text"],
            "text_prefix": QWEN_PREFIX,
            "target_trainer": "QwenLM/Qwen3-ASR finetuning/qwen3_asr_sft.py",
        },
        "source": {"train": str(args.train.resolve()), "val": str(args.val.resolve())},
        "seed": args.seed,
        "pilot_size_requested": args.pilot_size,
        "audio_existence_verified": args.verify_audio,
        "max_duration_seconds": args.max_duration,
        "dropped_over_max_duration": {"train": train_too_long, "validation": val_too_long},
        "integrity": {"speaker_overlap": 0, "audio_overlap": 0},
        "train": describe(train),
        "validation": describe(val),
        "pilot": describe(pilot),
        "privacy": "Outputs contain restricted audio paths and transcripts; do not commit them.",
    }
    (out / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Train:      {len(train):,} recordings, {len(train_speakers)} speakers")
    print(f"Validation: {len(val):,} recordings, {len(val_speakers)} speakers")
    print(f"Pilot:      {len(pilot):,} recordings, {describe(pilot)['speakers']} speakers")
    if args.max_duration:
        print(f"Dropped over {args.max_duration:g}s: Train={train_too_long:,}, Validation={val_too_long:,}")
    print(f"Wrote private Qwen manifests: {out}")


if __name__ == "__main__":
    main()
