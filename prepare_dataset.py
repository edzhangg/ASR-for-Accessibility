#!/usr/bin/env python3
r"""Prepare speaker-disjoint disease-specific ASR JSONL manifests from original SAP data.

Read-only with respect to the SAP dataset. No audio is copied or changed.

Usage (PowerShell):
  py -u prepare_disease_dataset.py `
    --root "C:\SAP downsampled database" `
    --out ".\data\disease_prepared" `
    --seed 42

Outputs: train.jsonl, val.jsonl, test.jsonl, speaker_split.json,
         cleaning_report.json, split_summary.csv.

IMPORTANT: manifests contain transcripts, speaker identifiers and file paths.
Keep them in approved restricted storage, not a public repository.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Change this constant to another SAP Etiology label, e.g. "ALS", "Stroke",
# "Cerebral Palsy", or "Down Syndrome".
TARGET_ETIOLOGY = "Parkinson's Disease"

SHARD_PATTERN = re.compile(r"^SpeechAccessibility_\d{4}-\d{2}-\d{2}_\d{3}$")
SQUARE_PATTERN = re.compile(r"\[[^\[\]]*\]")
PAREN_PATTERN = re.compile(r"\(([^()]*)\)")
SPLITS = ("train", "val", "test")


def normalize_etiology(value):
    """Compare disease labels case-insensitively, ignoring common formatting differences."""
    text = str(value or "").strip().replace("’", "'").casefold()
    text = re.sub(r"[\s_-]+", " ", text)
    return text.replace("'", "")


def matches_target_etiology(value):
    return normalize_etiology(value) == normalize_etiology(TARGET_ETIOLOGY)


def clean_transcript(original):
    """Return (cleaned, exclude_reason) using the agreed project policy.

    [...] -> remove text and brackets
    (...) -> retain text, remove brackets
    any { or } -> exclude entire utterance (includes malformed annotations)
    """
    if not isinstance(original, str) or not original.strip():
        return None, "empty_raw_transcript"
    if "{" in original or "}" in original:
        return None, "curly_braces"
    text = SQUARE_PATTERN.sub(" ", original)
    # Repeat for safely nested simple parentheses, though such input is unusual.
    while True:
        new = PAREN_PATTERN.sub(lambda match: match.group(1), text)
        if new == text:
            break
        text = new
    # Unbalanced markup should not silently become evaluation reference text.
    if any(symbol in text for symbol in "[]()"):
        return None, "unbalanced_square_or_parentheses"
    text = " ".join(text.split())
    if not text:
        return None, "empty_after_cleaning"
    return text, None


def list_original_shards(root, split):
    base = root / split
    if not base.is_dir():
        raise ValueError(f"Missing original split directory: {base}")
    shards = sorted(p for p in base.iterdir() if p.is_dir() and SHARD_PATTERN.fullmatch(p.name))
    if not shards:
        raise ValueError(f"No original SAP shards found under {base}")
    return shards


def stable_audio_key(path, root):
    return path.relative_to(root).as_posix().casefold()


def scan_original_data(root):
    """Scan only official Train and Dev shards; never walk derived subset folders."""
    records = []
    speaker_sources = defaultdict(set)
    seen_audio = set()
    issues = Counter()
    sources = {}
    for original_split in ("Train", "Dev"):
        shards = list_original_shards(root, original_split)
        sources[original_split] = [p.name for p in shards]
        print(f"[{original_split}] scanning {len(shards)} original shards...", flush=True)
        for shard in shards:
            for metadata_path in sorted(shard.rglob("*.json")):
                try:
                    with metadata_path.open("r", encoding="utf-8") as handle:
                        metadata = json.load(handle)
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    issues["json_read_or_parse_error"] += 1
                    print(f"WARNING: cannot parse {metadata_path}: {exc}", file=sys.stderr)
                    continue
                if not isinstance(metadata, dict) or not matches_target_etiology(metadata.get("Etiology")):
                    continue
                speaker = str(metadata.get("Contributor ID") or "").strip()
                items = metadata.get("Files")
                if not speaker or not isinstance(items, list):
                    issues["invalid_target_metadata"] += 1
                    continue
                speaker_sources[speaker].add(original_split)
                for entry in items:
                    if not isinstance(entry, dict):
                        issues["invalid_record"] += 1
                        continue
                    filename = entry.get("Filename")
                    if (not isinstance(filename, str) or Path(filename).name != filename
                            or not filename.lower().endswith(".wav")):
                        issues["invalid_audio_filename"] += 1
                        continue
                    audio_path = metadata_path.parent / filename
                    key = stable_audio_key(audio_path, root)
                    if key in seen_audio:
                        issues["duplicate_audio_reference"] += 1
                        continue
                    seen_audio.add(key)
                    if not audio_path.is_file():
                        issues["missing_audio_file"] += 1
                        continue
                    prompt = entry.get("Prompt")
                    if not isinstance(prompt, dict):
                        issues["invalid_prompt"] += 1
                        continue
                    transcript = prompt.get("Transcript")
                    cleaned, exclusion = clean_transcript(transcript)
                    records.append({
                        "audio_path": str(audio_path.resolve()),
                        "audio_relpath": audio_path.relative_to(root).as_posix(),
                        "speaker_id": speaker,
                        "source_split": original_split,
                        "category": str(prompt.get("Category Description") or "MISSING"),
                        "subcategory": str(prompt.get("Sub Category Description") or "MISSING"),
                        "raw_transcript": transcript if isinstance(transcript, str) else "",
                        "text": cleaned,
                        "exclude_reason": exclusion,
                    })
        print(f"[{original_split}] complete", flush=True)
    return records, speaker_sources, issues, sources


def determine_assignments(speaker_sources, seed, validation_speakers, existing_file):
    overlaps = {k: sorted(v) for k, v in speaker_sources.items() if len(v) != 1}
    if overlaps:
        raise ValueError(f"Speaker IDs appear in multiple source splits: {len(overlaps)}")
    original_train = sorted(k for k, v in speaker_sources.items() if "Train" in v)
    original_dev = sorted(k for k, v in speaker_sources.items() if "Dev" in v)
    if not original_train or not original_dev:
        raise ValueError("No target-disease speakers found in Train or Dev")
    if not 1 <= validation_speakers < len(original_train):
        raise ValueError(f"--val-speakers must be between 1 and {len(original_train)-1}")
    if existing_file.is_file():
        with existing_file.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        saved_target = saved.get("target_etiology")
        if saved_target is not None and normalize_etiology(saved_target) != normalize_etiology(TARGET_ETIOLOGY):
            raise ValueError(
                f"Existing speaker_split.json is for {saved_target!r}, not {TARGET_ETIOLOGY!r}; "
                "use a separate --out directory for each disease"
            )
        # Legacy split files may lack the target label; the speaker-population
        # and seed checks below must still pass before reusing them.
        assignments = saved.get("speaker_assignments")
        if not isinstance(assignments, dict):
            raise ValueError(f"Malformed existing speaker_split.json: {existing_file}")
        if saved.get("seed") != seed or saved.get("val_speakers_requested") != validation_speakers:
            raise ValueError("Existing speaker_split.json uses different seed/val-speakers; use another output folder")
        if set(assignments) != set(speaker_sources):
            raise ValueError("Speaker population changed since existing speaker_split.json; refusing to overwrite")
        if any(assignments[k] not in SPLITS for k in assignments):
            raise ValueError("Invalid split value in existing speaker assignments")
        if any(assignments[k] != "test" for k in original_dev):
            raise ValueError("Existing assignments place an official Dev speaker outside test")
        if sum(assignments[k] == "val" for k in original_train) != validation_speakers:
            raise ValueError("Existing assignments have incorrect validation speaker count")
        if any(assignments[k] == "test" for k in original_train):
            raise ValueError("Existing assignments put an original Train speaker in test")
        print("Reusing saved speaker_split.json", flush=True)
        return assignments
    # Random sample on a canonical sorted speaker list for repeatability.
    rng = random.Random(seed)
    val_set = set(rng.sample(original_train, validation_speakers))
    return {speaker: ("test" if speaker in original_dev else
                      "val" if speaker in val_set else "train")
            for speaker in sorted(speaker_sources)}


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="C:\\SAP downsampled database")
    parser.add_argument("--out", required=True, type=Path, help="Restricted output directory, outside the SAP root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-speakers", type=int, default=25)
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    if not root.is_dir():
        parser.error(f"Dataset root does not exist: {root}")
    if out == root or root in out.parents:
        parser.error("Output directory must be outside original SAP root")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Target disease: {TARGET_ETIOLOGY}", flush=True)
    records, speaker_sources, issues, shards = scan_original_data(root)
    if not records:
        raise RuntimeError(f"Found no recordings for {TARGET_ETIOLOGY!r}; check dataset path and Etiology label")
    # Fail rather than silently create manifests from incomplete/broken source data.
    if issues:
        raise RuntimeError(f"Source integrity problems encountered; manifests NOT written: {dict(issues)}")

    assignment_file = out / "speaker_split.json"
    assignments = determine_assignments(speaker_sources, args.seed, args.val_speakers, assignment_file)
    split_records = {s: [] for s in SPLITS}
    clean_drops = Counter()
    clean_drops_by_split = Counter()
    before_by_source = Counter()
    for record in records:
        before_by_source[record["source_split"]] += 1
        split = assignments[record["speaker_id"]]
        if record["exclude_reason"]:
            clean_drops[record["exclude_reason"]] += 1
            clean_drops_by_split[(split, record["exclude_reason"])] += 1
            continue
        item = {k: v for k, v in record.items() if k != "exclude_reason"}
        item["split"] = split
        split_records[split].append(item)

    # Integrity checks; allow a few speakers to have no usable utterances but report them.
    speaker_sets = {s: {r["speaker_id"] for r in split_records[s]} for s in SPLITS}
    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i+1:]:
            if speaker_sets[split_a] & speaker_sets[split_b]:
                raise RuntimeError(f"Speaker overlap between {split_a} and {split_b}")
    all_clean_audio = [r["audio_relpath"].casefold() for s in SPLITS for r in split_records[s]]
    if len(all_clean_audio) != len(set(all_clean_audio)):
        raise RuntimeError("Duplicate audio path found across output splits")
    if not split_records["train"] or not split_records["val"] or not split_records["test"]:
        raise RuntimeError("At least one output split has no usable recordings")

    speaker_assignments = {k: assignments[k] for k in sorted(assignments)}
    assignment_contents = {
        "description": "Speaker-disjoint disease-specific split; original SAP Dev reserved as test",
        "target_etiology": TARGET_ETIOLOGY,
        "seed": args.seed,
        "val_speakers_requested": args.val_speakers,
        "speaker_assignments": speaker_assignments,
    }
    # Persist source assignment before manifests; on subsequent runs it is validated and reused.
    if not assignment_file.is_file():
        write_json(assignment_file, assignment_contents)

    summary = {s: {
        "assigned_speakers": sum(v == s for v in assignments.values()),
        "speakers_with_usable_recordings": len(speaker_sets[s]),
        "usable_recordings": len(split_records[s]),
        "category_counts": dict(sorted(Counter(r["category"] for r in split_records[s]).items())),
        "dropped_recordings_by_reason": dict(sorted(
            (reason, n) for (split, reason), n in clean_drops_by_split.items() if split == s)),
    } for s in SPLITS}
    report = {
        "target_etiology": TARGET_ETIOLOGY,
        "source_root": str(root),
        "original_shards": shards,
        "rules": {
            "square_brackets": "remove bracketed text entirely",
            "parentheses": "remove brackets only; preserve spoken contents",
            "curly_braces": "exclude any recording with { or }",
            "empty_cleaned_transcript": "exclude recording",
            "unbalanced_square_or_parentheses": "exclude for manual inspection",
        },
        "seed": args.seed,
        "val_speakers_requested": args.val_speakers,
        "raw_target_recordings": len(records),
        "raw_target_recordings_by_source": dict(sorted(before_by_source.items())),
        "total_dropped": sum(clean_drops.values()),
        "dropped_by_reason": dict(sorted(clean_drops.items())),
        "total_usable_recordings": sum(len(x) for x in split_records.values()),
        "splits": summary,
        "speaker_overlap": 0,
        "source_integrity_issues": dict(issues),
        "notes": [
            "No WAV files or original metadata were modified or copied.",
            "Audio paths are absolute Windows paths; audio_relpath enables relocation.",
            "This is transcript markup cleaning, not WER/CER linguistic normalization.",
            "Test is original Dev; ensure it was not used to select models/tune hyperparameters.",
        ],
    }

    for split in SPLITS:
        target = out / f"{split}.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as f:
            for r in sorted(split_records[split], key=lambda x: (x["speaker_id"], x["audio_relpath"])):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split:5s}: {len(split_records[split]):,} recordings; "
              f"{len(speaker_sets[split])} active speakers", flush=True)
    write_json(out / "cleaning_report.json", report)
    with (out / "split_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "assigned_speakers", "speakers_with_usable_recordings", "usable_recordings", "dropped_recordings"])
        for s in SPLITS:
            writer.writerow([s, summary[s]["assigned_speakers"],
                             summary[s]["speakers_with_usable_recordings"],
                             summary[s]["usable_recordings"],
                             sum(n for (x, _), n in clean_drops_by_split.items() if x == s)])
    print(f"Dropped {sum(clean_drops.values()):,} recordings: {dict(clean_drops)}", flush=True)
    print(f"Wrote manifests and reports to: {out}", flush=True)


if __name__ == "__main__":
    main()
