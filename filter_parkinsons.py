#!/usr/bin/env python3
"""Copy the Parkinson's-disease speakers out of the Speech Accessibility Project (SAP)
dataset and build a manifest.

The SAP root is only ever READ. Files are copied (or hard-linked) into --out.

Outputs (written into --out):
  <relative speaker folders>/...   copied audio + json, same structure as the source
  manifest_pd.jsonl                one line per utterance
  speakers_pd.csv                  speaker_id, split, n_utts, total_minutes
  split_pd.json                    speaker -> split (only if the source has no Train/Dev folders)

Example:
  python filter_parkinsons.py --src /data/SAP --out /data/SAP_PD --dry-run
  python filter_parkinsons.py --src /data/SAP --out /data/SAP_PD --hardlink
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import wave
from collections import defaultdict
from pathlib import Path

OFFICIAL_SPLITS = {"train": "Train", "dev": "Dev", "test": "Test"}


def is_parkinsons(etiology):
    e = (etiology or "").replace("’", "'").strip().lower()
    return e == "parkinson's disease"


def find_speaker_jsons(src):
    """Yield every speaker JSON under src (one per speaker folder)."""
    for dirpath, _, files in os.walk(src):
        for f in files:
            if f.lower().endswith(".json"):
                yield Path(dirpath) / f


def official_split(rel_folder):
    """Return 'Train'/'Dev'/'Test' if the speaker sits under such a folder, else None."""
    top = rel_folder.parts[0].lower() if rel_folder.parts else ""
    return OFFICIAL_SPLITS.get(top)


def wav_duration(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def make_speaker_split(speaker_ids, seed, split_file):
    """Speaker-level 70/15/15 split. Reuses split_file if it already exists."""
    if split_file.exists():
        saved = json.loads(split_file.read_text())
        if set(saved) >= set(speaker_ids):
            print(f"Reusing existing speaker split: {split_file}")
            return saved
    ids = sorted(speaker_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    # "dev" folder on the SAP downsampled dataset should be our "test" dataset. Run baselines and tuned models on this dataset
    # n_dev = max(1, round(0.15 * n)) if n >= 3 else 0
    n_dev = 0
    n_test = max(1, round(1 * n)) if n >= 3 else 0
    split = {}
    for i, spk in enumerate(ids):
        split[spk] = "Dev" if i < n_dev else "Test" if i < n_dev + n_test else "Train"
    return split


def copy_folder(src_dir, dst_dir, hardlink, stats):
    """Copy (or hard-link) all files of one speaker folder. Skips same-size files."""
    for dirpath, _, files in os.walk(src_dir):
        rel = Path(dirpath).relative_to(src_dir)
        (dst_dir / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            s, d = Path(dirpath) / f, dst_dir / rel / f
            size = s.stat().st_size
            if d.exists() and d.stat().st_size == size:
                stats["skipped"] += 1
                continue
            if d.exists():
                d.unlink()  # partial file from an interrupted run (this is in OUT, not SRC)
            if hardlink:
                try:
                    os.link(s, d)
                except OSError:
                    shutil.copy2(s, d)  # e.g. different filesystem
                    stats["link_fallback"] += 1
            else:
                shutil.copy2(s, d)
            stats["copied"] += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="SAP dataset root (read-only)")
    ap.add_argument("--out", required=True, help="output folder; must be OUTSIDE --src")
    ap.add_argument("--dry-run", action="store_true", help="report counts/GB, copy nothing")
    ap.add_argument("--hardlink", action="store_true",
                    help="hard-link instead of copy (saves disk; never edit those files in place)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the speaker split")
    args = ap.parse_args()

    src, out = Path(args.src).resolve(), Path(args.out).resolve()
    if not src.is_dir():
        sys.exit(f"--src is not a directory: {src}")
    if out == src or src in out.parents:
        sys.exit(f"Refusing to run: output {out} is inside the source {src}")

    # 1. find Parkinson's speakers
    speakers = []  # (speaker_id, source folder, json dict)
    etiologies = defaultdict(int)
    for jp in find_speaker_jsons(src):
        try:
            meta = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"WARNING: unreadable json {jp}: {e}")
            continue
        etiologies[meta.get("Etiology")] += 1
        if is_parkinsons(meta.get("Etiology")):
            speakers.append((meta["Contributor ID"], jp.parent, meta))
    print("Etiologies found (speakers):", dict(etiologies))
    if not speakers:
        sys.exit("No Parkinson's speakers found.")

    # 2. decide the split of each speaker
    official = {sid: official_split(folder.relative_to(src)) for sid, folder, _ in speakers}
    if all(official.values()):
        split_of = official
        print("Using the dataset's official Train/Dev/Test folders.")
    else:
        split_file = out / "split_pd.json"
        split_of = make_speaker_split([s for s, _, _ in speakers], args.seed, split_file)
        print(f"NOTE: no official split found, created a speaker-level split (seed={args.seed}); "
              f"saved to {split_file}" + (" (dry run: not written)" if args.dry_run else ""))
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            split_file.write_text(json.dumps(split_of, indent=2, sort_keys=True))

    # 3. build manifest rows (audio durations are read from the wav headers)
    rows, warnings, total_bytes = [], [], 0
    for sid, folder, meta in speakers:
        dst_folder = out / folder.relative_to(src)
        total_bytes += sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        for item in meta.get("Files", []):
            wav = folder / item["Filename"]
            prompt = item.get("Prompt", {})
            transcript = prompt.get("Transcript", "")
            if not wav.exists():
                warnings.append(f"missing audio: {wav.name}")
                continue
            if not transcript.strip():
                warnings.append(f"missing transcript: {wav.name}")
                continue
            try:
                dur = wav_duration(wav)
            except (wave.Error, EOFError) as e:
                warnings.append(f"unreadable audio {wav.name}: {e}")
                continue
            rows.append({
                "audio_path": str(dst_folder / item["Filename"]),
                "speaker_id": sid,
                "split": split_of[sid],
                "etiology": meta["Etiology"],
                "category": prompt.get("Category Description", ""),
                "transcript": transcript,
                "duration_s": round(dur, 3),
            })

    # 4. summary
    per_split = defaultdict(lambda: {"spk": set(), "utts": 0, "sec": 0.0})
    per_spk = defaultdict(lambda: {"n": 0, "sec": 0.0})
    for r in rows:
        s = per_split[r["split"]]
        s["spk"].add(r["speaker_id"]); s["utts"] += 1; s["sec"] += r["duration_s"]
        per_spk[r["speaker_id"]]["n"] += 1; per_spk[r["speaker_id"]]["sec"] += r["duration_s"]
    print(f"\n{'split':<8}{'speakers':>10}{'utterances':>12}{'hours':>8}")
    for name in sorted(per_split):
        s = per_split[name]
        print(f"{name:<8}{len(s['spk']):>10}{s['utts']:>12}{s['sec'] / 3600:>8.2f}")
    print(f"{'TOTAL':<8}{len(speakers):>10}{len(rows):>12}{sum(r['duration_s'] for r in rows) / 3600:>8.2f}")
    print(f"\nParkinson's data size on disk: {total_bytes / 1e9:.2f} GB")
    free = shutil.disk_usage(out if out.exists() else out.parent if out.parent.exists() else Path("/")).free
    print(f"Free space at output location: {free / 1e9:.1f} GB"
          + ("  (hard-linking: needs ~0)" if args.hardlink else ""))
    if warnings:
        print(f"\n{len(warnings)} warnings (first 10):")
        for w in warnings[:10]:
            print("  -", w)

    if args.dry_run:
        print("\nDry run: nothing copied or written.")
        return
    if not args.hardlink and total_bytes > free:
        sys.exit(f"Not enough free space: need {total_bytes / 1e9:.2f} GB, have {free / 1e9:.2f} GB. "
                 "Try --hardlink or a different --out.")

    # 5. copy + write manifest
    out.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(int)
    for sid, folder, _ in speakers:
        copy_folder(folder, out / folder.relative_to(src), args.hardlink, stats)
    print(f"\nFiles copied: {stats['copied']}, skipped (already there): {stats['skipped']}"
          + (f", hardlink fell back to copy: {stats['link_fallback']}" if stats["link_fallback"] else ""))

    with open(out / "manifest_pd.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "speakers_pd.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "split", "n_utts", "total_minutes"])
        for sid in sorted(per_spk):
            w.writerow([sid, split_of[sid], per_spk[sid]["n"], round(per_spk[sid]["sec"] / 60, 2)])
    print(f"Wrote {out / 'manifest_pd.jsonl'} and {out / 'speakers_pd.csv'}")


if __name__ == "__main__":
    main()
