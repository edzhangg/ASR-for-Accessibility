#!/usr/bin/env python3
"""Read-only Parkinson's subset inventory for SAP downsampled database.

Scans only original Train/Dev SpeechAccessibility_YYYY-MM-DD_NNN shards.
Does not copy, rename, or modify original data. No third-party dependencies.
Default outputs are aggregate statistics only; use --write-manifest only inside
approved restricted storage (manifest contains speaker IDs, paths, transcripts).
"""
import argparse
import csv
import json
import re
import statistics
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

SHARD_RE = re.compile(r"^SpeechAccessibility_\d{4}-\d{2}-\d{2}_\d{3}$")
PATTERNS = {
    "square_brackets": re.compile(r"\[[^\]]*\]"),
    "parentheses": re.compile(r"\([^)]*\)"),
    "curly_braces": re.compile(r"\{[^}]*\}"),
}


def is_pd(value):
    # Conservative aliases; print raw etiologies to flag unexpected variants.
    key = re.sub(r"[\s_-]+", " ", str(value or "").replace("’", "'").strip().casefold())
    return key in {"parkinson's disease", "parkinsons disease", "parkinson disease"}


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * p / 100
    lo, hi = int(x), min(int(x) + 1, len(values) - 1)
    return round(values[lo] * (hi - x) + values[hi] * (x - lo), 3)


def distribution(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values), "min": round(min(values), 3),
        "p25": percentile(values, 25), "median": percentile(values, 50),
        "p75": percentile(values, 75), "p95": percentile(values, 95),
        "max": round(max(values), 3), "mean": round(statistics.mean(values), 3),
    }


def write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="SAP downsampled database root")
    ap.add_argument("--out", type=Path, required=True, help="Output directory OUTSIDE dataset root")
    ap.add_argument("--audio-headers", action="store_true", help="Check every PD WAV header (slower)")
    ap.add_argument("--write-manifest", action="store_true", help="Write restricted PD utterance manifest with paths/IDs/text")
    args = ap.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    if not root.is_dir():
        ap.error(f"Dataset root does not exist: {root}")
    if out == root or root in out.parents:
        ap.error("--out must be outside original SAP root")
    for split in ("Train", "Dev"):
        if not (root / split).is_dir():
            ap.error(f"Missing {split} directory under {root}")
    out.mkdir(parents=True, exist_ok=True)

    etiologies = Counter()
    pd_speakers = defaultdict(set)
    speakers_by_split = defaultdict(set)
    pd_utterances = Counter()
    categories = Counter()
    subcategories = Counter()
    category_split = Counter()
    rating_dims = Counter()
    rating_levels = Counter()
    rating_category_dims = Counter()
    rating_presence = Counter()
    markup_counts = Counter()
    markup_value_counts = {k: Counter() for k in PATTERNS}
    transcript_stats = []
    durations = []
    sample_rates = Counter()
    channels = Counter()
    per_speaker_n = Counter()
    per_speaker_secs = Counter()
    file_dups = set()
    seen_keys = set()
    issues = Counter()
    examples = defaultdict(list)
    written_manifest = 0
    manifest = None
    if args.write_manifest:
        manifest = (out / "manifest_pd.jsonl").open("w", encoding="utf-8")
    shard_count = Counter()
    json_count = Counter()

    try:
        for split in ("Train", "Dev"):
            shards = sorted(p for p in (root / split).iterdir() if p.is_dir() and SHARD_RE.fullmatch(p.name))
            shard_count[split] = len(shards)
            print(f"[{split}] original shards={len(shards)}", flush=True)
            for shard in shards:
                for jp in sorted(shard.rglob("*.json")):
                    json_count[split] += 1
                    if json_count[split] % 100 == 0:
                        print(f"[{split}] JSON scanned={json_count[split]}, PD recordings={pd_utterances[split]:,}", flush=True)
                    try:
                        with jp.open(encoding="utf-8") as handle:
                            meta = json.load(handle)
                        if not isinstance(meta, dict) or not isinstance(meta.get("Files"), list):
                            issues["invalid_json_schema"] += 1
                            continue
                    except (OSError, ValueError, UnicodeError):
                        issues["json_read_or_parse_error"] += 1
                        continue
                    etiology = str(meta.get("Etiology") or "MISSING")
                    etiologies[etiology] += 1
                    if not is_pd(etiology):
                        continue
                    sid = str(meta.get("Contributor ID") or "").strip()
                    if not sid:
                        issues["missing_speaker_id"] += 1
                        continue
                    pd_speakers[split].add(sid)
                    speakers_by_split[sid].add(split)
                    folder = jp.parent
                    for item in meta["Files"]:
                        if not isinstance(item, dict):
                            issues["invalid_file_record"] += 1
                            continue
                        name = item.get("Filename")
                        if not isinstance(name, str) or not name.lower().endswith(".wav") or Path(name).name != name:
                            issues["bad_filename"] += 1
                            continue
                        key = (split, sid, name)
                        if key in seen_keys:
                            issues["duplicate_record_key"] += 1
                            continue
                        seen_keys.add(key)
                        audio = folder / name
                        if not audio.is_file():
                            issues["missing_wav"] += 1
                            continue
                        prompt = item.get("Prompt") or {}
                        if not isinstance(prompt, dict):
                            issues["invalid_prompt_schema"] += 1
                            continue
                        transcript = prompt.get("Transcript")
                        transcript = transcript if isinstance(transcript, str) else ""
                        category = prompt.get("Category Description") or "MISSING"
                        subcategory = prompt.get("Sub Category Description") or "MISSING"
                        cat = str(category)
                        sub = str(subcategory)
                        pd_utterances[split] += 1
                        per_speaker_n[(split, sid)] += 1
                        categories[cat] += 1
                        category_split[(split, cat)] += 1
                        subcategories[sub] += 1
                        transcript_stats.append(len(transcript.split()))
                        if not transcript.strip():
                            issues["empty_transcript"] += 1
                        if prompt.get("Prompt Text") == transcript:
                            issues["prompt_matches_transcript"] += 1
                        else:
                            issues["prompt_differs_transcript"] += 1
                        for kind, pattern in PATTERNS.items():
                            vals = pattern.findall(transcript)
                            if vals:
                                markup_counts[kind] += 1
                                markup_value_counts[kind].update(vals)
                        ratings = item.get("Ratings")
                        if isinstance(ratings, list) and ratings:
                            rating_presence["recordings_with_ratings"] += 1
                            for rating in ratings:
                                if not isinstance(rating, dict):
                                    issues["non_dict_rating"] += 1
                                    continue
                                dim = str(rating.get("Dimension Description") or "MISSING")
                                level = str(rating.get("Level") if rating.get("Level") is not None else "MISSING")
                                dimcat = str(rating.get("Dimension Category Description") or "MISSING")
                                rating_dims[dim] += 1
                                rating_levels[(dim, level)] += 1
                                rating_category_dims[(dimcat, dim)] += 1
                        else:
                            rating_presence["recordings_without_ratings"] += 1
                        duration = None
                        if args.audio_headers:
                            try:
                                with wave.open(str(audio), "rb") as wav:
                                    sr = wav.getframerate()
                                    ch = wav.getnchannels()
                                    duration = wav.getnframes() / sr
                                sample_rates[sr] += 1
                                channels[ch] += 1
                                durations.append(duration)
                                per_speaker_secs[(split, sid)] += duration
                            except (OSError, EOFError, wave.Error, ZeroDivisionError):
                                issues["unreadable_wav_header"] += 1
                        if manifest is not None:
                            row = {"audio_path": str(audio), "speaker_id": sid,
                                   "source_split": split, "etiology": etiology,
                                   "category": cat, "subcategory": sub,
                                   "transcript": transcript, "prompt_text": prompt.get("Prompt Text") or "",
                                   "duration_s": round(duration, 3) if duration is not None else None}
                            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                            written_manifest += 1
            print(f"[{split}] JSON={json_count[split]}, PD speakers={len(pd_speakers[split])}, PD recordings={pd_utterances[split]:,}", flush=True)
    finally:
        if manifest is not None:
            manifest.close()

    overlap = pd_speakers["Train"] & pd_speakers["Dev"]
    summary = {
        "scope": "Parkinson's Disease only; original Train/Dev shards only",
        "original_shards_by_split": dict(shard_count),
        "json_files_scanned_by_split": dict(json_count),
        "all_etiology_json_counts": dict(etiologies),
        "pd_speakers_by_split": {k: len(v) for k, v in pd_speakers.items()},
        "pd_unique_speakers_total": len(pd_speakers["Train"] | pd_speakers["Dev"]),
        "pd_recordings_by_split": dict(pd_utterances),
        "pd_recordings_total": sum(pd_utterances.values()),
        "pd_train_dev_speaker_overlap_count": len(overlap),
        "categories": dict(categories),
        "transcript_word_count": distribution(transcript_stats),
        "markup_recordings": dict(markup_counts),
        "ratings_presence": dict(rating_presence),
        "unique_rating_dimensions": len(rating_dims),
        "audio_headers_enabled": args.audio_headers,
        "pd_audio_duration_seconds": distribution(durations) if args.audio_headers else None,
        "pd_total_audio_hours": round(sum(durations) / 3600, 3) if args.audio_headers else None,
        "sample_rates": dict(sample_rates),
        "channels": dict(channels),
        "issues": dict(issues),
        "manifest_written": args.write_manifest,
        "notes": [
            "Original audio is read-only; derivatives under Dev/Train are ignored.",
            "Aggregated reports intentionally omit speaker IDs, full transcript, and audio paths.",
            "Ratings counts refer to rating objects; one recording may have multiple ratings.",
            "WAV header validation does not decode audio waveforms.",
            "Source splits are preserved and no Train/Validation/Test reassignment is performed.",
        ],
    }
    (out / "pd_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "pd_category_distribution.csv", ["split", "category", "recordings"],
              ((s, cat, n) for (s, cat), n in sorted(category_split.items())))
    write_csv(out / "pd_subcategory_distribution.csv", ["subcategory", "recordings"],
              sorted(subcategories.items(), key=lambda x: (-x[1], x[0])))
    write_csv(out / "pd_rating_dimensions.csv", ["dimension_category", "dimension", "rating_objects"],
              ((dc, dim, n) for (dc, dim), n in sorted(rating_category_dims.items(), key=lambda x: (-x[1], x[0]))))
    write_csv(out / "pd_rating_levels.csv", ["dimension", "level", "rating_objects"],
              ((dim, level, n) for (dim, level), n in sorted(rating_levels.items(), key=lambda x: (-x[1], x[0]))))
    write_csv(out / "pd_markup_examples_aggregate.csv", ["markup_type", "markup_value", "occurrences"],
              ((kind, value, count) for kind, counter in markup_value_counts.items()
               for value, count in counter.most_common(30)))
    # Per-speaker output is intentionally omitted unless user explicitly requests restricted manifest.
    print("\n=== PD INVENTORY ===")
    print(f"PD speakers: Train={len(pd_speakers['Train'])}, Dev={len(pd_speakers['Dev'])}, overlap={len(overlap)}")
    print(f"PD recordings: Train={pd_utterances['Train']:,}, Dev={pd_utterances['Dev']:,}, total={sum(pd_utterances.values()):,}")
    if args.audio_headers:
        print(f"PD audio hours: {summary['pd_total_audio_hours']}")
    print("Speech categories:", dict(categories))
    print("Recordings with ratings:", rating_presence["recordings_with_ratings"])
    print("Distinct rating dimensions:", len(rating_dims))
    print("Issues:", dict(issues))
    print("Saved reports to:", out)
    if args.write_manifest:
        print(f"Restricted manifest rows: {written_manifest}. Do NOT commit manifest to public git.")


if __name__ == "__main__":
    main()
