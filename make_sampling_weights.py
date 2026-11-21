#!/usr/bin/env python3
"""Build per-speaker sampling probabilities for the fine-tuning experiments.

Input : a CSV with at least speaker_id, difficulty_wer_percent, n_manifest
        (per-speaker zero-shot WER on the TRAIN speakers, and recordings per speaker).
Output: the same CSV with difficulty_rank, hard_top25 and probability columns for

  e1  uniform over recordings        (natural distribution)
  e2  uniform over speakers
  e3  hard-speaker 50/50             (top 25% hardest speakers get 50% of the mass)
  e4  difficulty-weighted            p_speaker ∝ clip(WER, floor, cap)

Per-recording probability = speaker probability / n_manifest for every strategy.
Speaker probabilities are what the trainer relies on; it re-splits them over the
recordings that actually exist in the (possibly duration-filtered) manifest.

  python make_sampling_weights.py --in weights.csv --out weights_v2.csv [--verify]

--verify additionally checks that e1-e3 recomputed here match the columns already
present in the input file. CSVs contain speaker IDs: keep them out of git.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

HARD_FRACTION = 0.25
E3_HARD_MASS = 0.5
E4_CAP_DEFAULT = 30.0
BASE_COLUMNS = ["speaker_id", "difficulty_rank", "difficulty_wer_percent", "hard_top25", "n_manifest"]
OUT_COLUMNS = BASE_COLUMNS + [
    "e1_uniform_recording_p_speaker", "e2_speaker_balanced_p_speaker",
    "e3_hard50_p_speaker", "e4_difficulty_p_speaker",
    "e1_per_recording_probability", "e2_per_recording_probability",
    "e3_per_recording_probability", "e4_per_recording_probability",
]


def compute_columns(speakers, hard_fraction=HARD_FRACTION, e4_floor=None, e4_cap=E4_CAP_DEFAULT):
    """speakers: list of dicts with speaker_id, wer (percent), n. Returns list of row dicts.

    e4_floor=None -> the median speaker WER (data-derived).
    """
    if not speakers:
        raise ValueError("no speakers")
    for s in speakers:
        if s["n"] <= 0 or not math.isfinite(s["wer"]) or s["wer"] < 0:
            raise ValueError(f"invalid row for {s['speaker_id']}")
    ids = [s["speaker_id"] for s in speakers]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate speaker_id")

    S = len(speakers)
    N = sum(s["n"] for s in speakers)
    n_hard = math.ceil(hard_fraction * S)           # 25% of 221 -> 56
    if n_hard >= S:
        raise ValueError("hard group must be smaller than the full speaker set")
    # Hardest first; ties broken by speaker_id so the result is deterministic.
    order = sorted(speakers, key=lambda s: (-s["wer"], s["speaker_id"]))
    rank = {s["speaker_id"]: i + 1 for i, s in enumerate(order)}

    floor = statistics.median(s["wer"] for s in speakers) if e4_floor is None else e4_floor
    if floor > e4_cap:
        raise ValueError("e4 floor exceeds cap")
    raw = {s["speaker_id"]: min(max(s["wer"], floor), e4_cap) for s in speakers}
    raw_total = sum(raw.values())

    out = []
    for s in speakers:
        sid, n = s["speaker_id"], s["n"]
        hard = rank[sid] <= n_hard
        p = {
            "e1": n / N,
            "e2": 1 / S,
            "e3": E3_HARD_MASS / n_hard if hard else (1 - E3_HARD_MASS) / (S - n_hard),
            "e4": raw[sid] / raw_total,
        }
        row = {"speaker_id": sid, "difficulty_rank": rank[sid],
               "difficulty_wer_percent": s["wer"], "hard_top25": int(hard), "n_manifest": n}
        row["e1_uniform_recording_p_speaker"] = p["e1"]
        row["e2_speaker_balanced_p_speaker"] = p["e2"]
        row["e3_hard50_p_speaker"] = p["e3"]
        row["e4_difficulty_p_speaker"] = p["e4"]
        for k in ("e1", "e2", "e3", "e4"):
            row[f"{k}_per_recording_probability"] = p[k] / n
        out.append(row)
    out.sort(key=lambda r: r["difficulty_rank"])
    return out


def read_input(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        need = {"speaker_id", "difficulty_wer_percent", "n_manifest"}
        if not reader.fieldnames or not need.issubset(reader.fieldnames):
            raise ValueError(f"input CSV needs columns: {sorted(need)}")
        rows = list(reader)
    speakers = [{"speaker_id": r["speaker_id"].strip(), "wer": float(r["difficulty_wer_percent"]),
                 "n": int(r["n_manifest"])} for r in rows]
    return speakers, rows


def verify_against(rows, computed, tol=1e-6):
    """Compare recomputed e1-e3 and hard flags with columns already in the input CSV."""
    by_id = {r["speaker_id"]: r for r in computed}
    checked, problems = 0, []
    for old in rows:
        new = by_id[old["speaker_id"].strip()]
        for col in OUT_COLUMNS[3:]:
            if col.startswith("e4") or col not in old:
                continue
            checked += 1
            if col == "hard_top25":
                same = int(old[col]) == new[col]
            else:
                same = math.isclose(float(old[col]), new[col], rel_tol=tol, abs_tol=1e-12)
            if not same:
                problems.append((old["speaker_id"], col, old[col], new[col]))
    return checked, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--e4-floor", type=float, default=None, help="default: median speaker WER")
    ap.add_argument("--e4-cap", type=float, default=E4_CAP_DEFAULT, help="WER (percent) cap for e4")
    ap.add_argument("--verify", action="store_true",
                    help="check recomputed e1-e3 columns against those already in --in")
    args = ap.parse_args(argv)
    if args.out.resolve() == args.inp.resolve():
        ap.error("--out must differ from --in")

    speakers, raw_rows = read_input(args.inp)
    computed = compute_columns(speakers, e4_floor=args.e4_floor, e4_cap=args.e4_cap)

    if args.verify:
        checked, problems = verify_against(raw_rows, computed)
        print(f"verify: {checked} values compared, {len(problems)} mismatches")
        for p in problems[:10]:
            print("  MISMATCH", p)
        if problems:
            raise SystemExit(1)

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for r in computed:
            w.writerow({k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in r.items()})

    hard_n = sum(r["hard_top25"] for r in computed)
    e4_hard = sum(r["e4_difficulty_p_speaker"] for r in computed if r["hard_top25"])
    print(f"wrote {args.out}: {len(computed)} speakers, {hard_n} hard; "
          f"e4 puts {e4_hard:.1%} of draws on the hard group")


if __name__ == "__main__":
    main()
