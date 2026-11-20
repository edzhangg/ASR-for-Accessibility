#!/usr/bin/env python3
"""Rescore saved ASR predictions under an explicitly documented text policy.

This tool never runs a model and never changes the source run. It reads the
``scored_predictions.jsonl`` written by ``run_asr_baselines.py`` and writes a
separate comparison between:

1. the normalized reference/hypothesis saved by the original run; and
2. those saved normalized strings after narrowly canonicalizing English time
   abbreviations that the original normalizer split into two letter tokens.

The extra policy is intentionally conservative:

* normalized standalone ``a m``/``p m`` -> ``am``/``pm``

It does NOT equate different words, rewrite arbitrary initialisms, remove
disfluencies, repair partial words, or change number words. Additions to this
policy should be justified from validation data and applied symmetrically to
references and hypotheses.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


POLICY_NAME = "whisper_english_plus_time_abbreviations_v1"
POLICY_RULES = [{
    "id": "spaced_am_pm_tokens",
    "description": (
        "After the saved Whisper normalization, canonicalize standalone two-token "
        "a m/p m forms to the same am/pm tokens used for uppercase AM/PM."
    ),
    "examples": ["a m -> am", "p m -> pm"],
    "caution": (
        "This assumes standalone a m/p m in this ASR task denotes a time abbreviation. "
        "Review normalization_changes.csv before adopting the policy."
    ),
}]

SPACED_AM_PM = re.compile(r"(?<!\w)([ap])\s+m(?!\w)")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def apply_policy_to_saved_normalization(text):
    """Return an adjusted saved-normalized string and replacement count."""
    text = str(text or "")
    return SPACED_AM_PM.subn(lambda match: match.group(1) + "m", text)


def aggregate(rows, ref_key, hyp_key):
    import jiwer

    pairs = [(row[ref_key], row[hyp_key]) for row in rows if row[ref_key].strip()]
    if not pairs:
        return {
            "n": 0, "wer": None, "cer": None, "substitutions": 0,
            "deletions": 0, "insertions": 0, "ref_words": 0,
        }
    references, hypotheses = map(list, zip(*pairs))
    words = jiwer.process_words(references, hypotheses)
    return {
        "n": len(pairs),
        "wer": round(100 * words.wer, 4),
        "cer": round(100 * jiwer.cer(references, hypotheses), 4),
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "ref_words": words.hits + words.substitutions + words.deletions,
    }


def summarize(rows, ref_key, hyp_key):
    by_category = defaultdict(list)
    by_speaker = defaultdict(list)
    for row in rows:
        by_category[row.get("category") or "MISSING"].append(row)
        by_speaker[row.get("speaker_id") or "MISSING"].append(row)
    return {
        "overall": aggregate(rows, ref_key, hyp_key),
        "by_category": {
            name: aggregate(group, ref_key, hyp_key)
            for name, group in sorted(by_category.items())
        },
        "per_speaker": {
            name: aggregate(group, ref_key, hyp_key)
            for name, group in sorted(by_speaker.items())
        },
    }


def write_category_csv(path, summaries):
    fields = [
        "policy", "category", "n", "wer_percent", "cer_percent",
        "substitutions", "deletions", "insertions", "ref_words",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for policy_name, summary in summaries.items():
            for category, metrics in summary["by_category"].items():
                writer.writerow({
                    "policy": policy_name,
                    "category": category,
                    "n": metrics["n"],
                    "wer_percent": metrics["wer"],
                    "cer_percent": metrics["cer"],
                    "substitutions": metrics["substitutions"],
                    "deletions": metrics["deletions"],
                    "insertions": metrics["insertions"],
                    "ref_words": metrics["ref_words"],
                })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-predictions", type=Path, required=True,
        help="scored_predictions.jsonl from a completed baseline run",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="new output directory; must not be the input file's parent",
    )
    args = parser.parse_args(argv)

    source = args.scored_predictions.resolve()
    out = args.out.resolve()
    if not source.is_file():
        parser.error(f"Input does not exist: {source}")
    if out == source.parent:
        parser.error("--out must be a separate subdirectory, not the source run directory")

    rows = []
    changed_rows = []
    rule_counts = Counter()
    required = {"text", "hyp", "ref_norm", "hyp_norm"}
    for line_number, row in read_jsonl(source):
        missing = sorted(required - row.keys())
        if missing:
            parser.error(f"{source}:{line_number}: missing fields: {', '.join(missing)}")

        adjusted_ref_norm, reference_count = apply_policy_to_saved_normalization(row["ref_norm"])
        adjusted_hyp_norm, hypothesis_count = apply_policy_to_saved_normalization(row["hyp_norm"])
        rule_counts["reference_spaced_am_pm_tokens"] += reference_count
        rule_counts["hypothesis_spaced_am_pm_tokens"] += hypothesis_count

        scored = {
            **row,
            "adjusted_ref_norm": adjusted_ref_norm,
            "adjusted_hyp_norm": adjusted_hyp_norm,
        }
        rows.append(scored)
        if adjusted_ref_norm != row["ref_norm"] or adjusted_hyp_norm != row["hyp_norm"]:
            changed_rows.append({
                "line_number": line_number,
                "audio_path": row.get("audio_path", ""),
                "speaker_id": row.get("speaker_id", ""),
                "category": row.get("category", ""),
                "original_ref_norm": row["ref_norm"],
                "adjusted_ref_norm": adjusted_ref_norm,
                "original_hyp_norm": row["hyp_norm"],
                "adjusted_hyp_norm": adjusted_hyp_norm,
            })

    if not rows:
        parser.error("Input contains no scored prediction rows")

    summaries = {
        "original_saved_normalization": summarize(rows, "ref_norm", "hyp_norm"),
        POLICY_NAME: summarize(rows, "adjusted_ref_norm", "adjusted_hyp_norm"),
    }
    original = summaries["original_saved_normalization"]["overall"]
    adjusted = summaries[POLICY_NAME]["overall"]
    comparison = {
        "source": str(source),
        "n_input_rows": len(rows),
        "n_rows_with_normalized_text_changes": len(changed_rows),
        "policy_name": POLICY_NAME,
        "policy_rules": POLICY_RULES,
        "rule_application_counts": dict(sorted(rule_counts.items())),
        "explicit_non_rules": [
            "Different lexical words remain errors.",
            "Arbitrary initialisms and spaced spoken letters are not collapsed.",
            "Disfluencies and partial words are not removed or repaired.",
            "Raw text is not renormalized, preventing package-version drift.",
            "Number words and digits remain exactly as saved by the original normalization.",
        ],
        "original": summaries["original_saved_normalization"],
        "adjusted": summaries[POLICY_NAME],
        "delta_adjusted_minus_original": {
            "wer_percentage_points": round(adjusted["wer"] - original["wer"], 4),
            "cer_percentage_points": round(adjusted["cer"] - original["cer"], 4),
            "substitutions": adjusted["substitutions"] - original["substitutions"],
            "deletions": adjusted["deletions"] - original["deletions"],
            "insertions": adjusted["insertions"] - original["insertions"],
            "ref_words": adjusted["ref_words"] - original["ref_words"],
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_category_csv(out / "category_scores.csv", summaries)
    with (out / "rescored_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "normalization_changes.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "line_number", "audio_path", "speaker_id", "category",
            "original_ref_norm", "adjusted_ref_norm",
            "original_hyp_norm", "adjusted_hyp_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changed_rows)

    print(f"Rows rescored: {len(rows):,}")
    print(f"Rows changed by policy: {len(changed_rows):,}")
    print(f"Original WER/CER: {original['wer']:.4f}% / {original['cer']:.4f}%")
    print(f"Adjusted WER/CER: {adjusted['wer']:.4f}% / {adjusted['cer']:.4f}%")
    print(f"WER delta: {comparison['delta_adjusted_minus_original']['wer_percentage_points']:+.4f} points")
    print(f"Policy: {POLICY_NAME}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
