#!/usr/bin/env python3
r"""Full SAP Parkinson's Qwen3-ASR baseline + speaker difficulty reports.

Uses the SAME Qwen inference, Whisper English normalization and WER/CER
conventions as run_asr_baselines_v2.py, which was already validated on 10,733
SAP PD validation utterances. This script does not train or alter dataset splits.

Windows PowerShell, from project root, with .venv activated:

  python -u .\run_qwen_full_baseline.py --splits train --out .\runs\qwen_full_baseline

Optional: import already-finished validation predictions rather than rerun:

  python -u .\run_qwen_full_baseline.py --splits val --out .\runs\qwen_full_baseline `
    --reuse-val-from .\runs\baseline_v2_full\qwen\predictions.jsonl

Test is SEALED by default; explicit --include-test required to run it.
All output contains protected speech transcripts, speaker IDs, file locations.
Do not upload outputs to public GitHub or share raw transcripts publicly.

Dependencies: torch, transformers, accelerate, numpy, soundfile, scipy,
              jiwer, whisper-normalizer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_VERSION = "1.0.0"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
NORMALIZER = "whisper_normalizer.english.EnglishTextNormalizer"
SPLITS = ("train", "val", "test")
REPORT_COLS = [
    "split", "speaker_id", "category", "subcategory", "audio_path", "duration_s",
    "rms", "peak", "inference_method", "num_chunks", "native_long_error",
    "text", "hyp", "ref_norm", "hyp_norm", "ref_words", "hyp_words",
    "hits", "s", "d", "i", "word_errors", "ref_chars", "char_errors",
    "wer_percent", "cer_percent", "inference_seconds", "review_flags",
]


def echo(msg: str) -> None:
    print(msg, flush=True)


def json_dump(path: Path, obj: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def jsonl_append(path: Path, obj: dict) -> None:
    # Each prediction is committed immediately to permit restart after interruption.
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def csv_dump(path: Path, columns: list[str], records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Malformed JSONL in {path} line {lineno}: {exc}") from exc
                if not isinstance(record, dict):
                    raise RuntimeError(f"Non-object JSONL in {path} line {lineno}")
                yield record


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path, split: str):
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    for i, r in enumerate(rows, 1):
        for key in ("audio_path", "speaker_id", "category", "text"):
            if key not in r:
                raise ValueError(f"Missing field {key!r} in {path}, row {i}")
        if not str(r["text"]).strip():
            raise ValueError(f"Empty reference in {path}, row {i}")
        if r.get("split", split) != split:
            raise ValueError(f"Manifest row {i} has split={r.get('split')}, expected {split}")
        if not str(r["speaker_id"]).strip() or not str(r["audio_path"]).strip():
            raise ValueError(f"Empty speaker ID or path in {path}, row {i}")
        key = str(r["audio_path"])
        if key in seen:
            raise ValueError(f"Duplicate audio_path in {path}: {key}")
        seen.add(key)
    return sorted(rows, key=lambda r: str(r["audio_path"]).casefold())


def load_audio(path: str):
    """Same mono 16k preprocessing as v2. Also compute *signal*, not VAD, stats."""
    from math import gcd
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio):
        raise ValueError("Empty WAV")
    mono = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = gcd(sr, 16000)
        mono = resample_poly(mono, 16000 // g, sr // g)
    mono = np.asarray(mono, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError("WAV has NaN/Inf values")
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(mono)))
    return mono, float(len(mono) / 16000), rms, peak


class QwenRunner:
    """Identical Qwen model, prompt and decoding to successful v2 Qwen baseline."""
    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected. Refusing expensive CPU full-set run.")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID, device_map="auto", dtype=dtype
        ).eval()

    def __call__(self, audio, max_new_tokens, is_long=False):
        p, m, torch = self.processor, self.model, self.torch
        inputs = p.apply_transcription_request([audio], language=["English"]).to(m.device, m.dtype)
        token_limit = (
            max(max_new_tokens, min(2048, int(len(audio) / 16000 * 15)))
            if is_long else max_new_tokens
        )
        with torch.inference_mode():
            out = m.generate(**inputs, max_new_tokens=token_limit, do_sample=False)
        generated = out[:, inputs["input_ids"].shape[1]:]
        decoded = p.decode(generated, return_format="transcription_only")
        return str(decoded[0] if isinstance(decoded, (list, tuple)) else decoded)


def chunk_audio(audio, seconds):
    window = int(seconds * 16000)
    return [audio[i:i + window] for i in range(0, len(audio), window)]


def transcribe(runner, audio, duration, max_new_tokens, chunk_seconds):
    if duration <= chunk_seconds:
        return runner(audio, max_new_tokens, is_long=False), "native_short", 1, ""
    try:
        return runner(audio, max_new_tokens, is_long=True), "native_long", 1, ""
    except Exception as exc:
        cause = f"{type(exc).__name__}: {exc}"
        echo(f"  [qwen] native long-form failed; using non-overlapping chunks: {cause}")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        pieces = chunk_audio(audio, chunk_seconds)
        transcribed = [runner(chunk, max_new_tokens, is_long=False).strip() for chunk in pieces]
        return " ".join(x for x in transcribed if x), "fixed_chunks_no_overlap", len(pieces), cause


@dataclass
class Stats:
    n: int = 0
    ref_words: int = 0
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_chars: int = 0
    char_errors: int = 0
    audio_seconds: float = 0.0
    inference_seconds: float = 0.0
    empty_hyp_count: int = 0
    high_wer_count: int = 0

    def add(self, r):
        self.n += 1
        self.ref_words += r["ref_words"]
        self.hits += r["hits"]
        self.substitutions += r["s"]
        self.deletions += r["d"]
        self.insertions += r["i"]
        self.ref_chars += r["ref_chars"]
        self.char_errors += r["char_errors"]
        self.audio_seconds += float(r.get("duration_s") or 0)
        self.inference_seconds += float(r.get("inference_seconds") or 0)
        self.empty_hyp_count += (not r["hyp_norm"].strip())
        self.high_wer_count += (r["wer_percent"] >= 100)

    def record(self):
        err = self.substitutions + self.deletions + self.insertions
        return {
            "n": self.n,
            "ref_words": self.ref_words,
            "word_errors": err,
            "wer": round(100 * err / self.ref_words, 4) if self.ref_words else None,
            "cer": round(100 * self.char_errors / self.ref_chars, 4) if self.ref_chars else None,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "ref_chars": self.ref_chars,
            "char_errors": self.char_errors,
            "substitution_rate_percent": round(100 * self.substitutions / self.ref_words, 4) if self.ref_words else None,
            "deletion_rate_percent": round(100 * self.deletions / self.ref_words, 4) if self.ref_words else None,
            "insertion_rate_percent": round(100 * self.insertions / self.ref_words, 4) if self.ref_words else None,
            "audio_hours": round(self.audio_seconds / 3600, 4),
            "inference_hours": round(self.inference_seconds / 3600, 4),
            "empty_hyp_count": self.empty_hyp_count,
            "wer_ge_100_count": self.high_wer_count,
        }


def get_stats(scored):
    result = Stats()
    for r in scored:
        result.add(r)
    return result.record()


def per_group(scored, *keys):
    groups = defaultdict(Stats)
    for r in scored:
        key = tuple(r.get(k, "MISSING") for k in keys)
        groups[key].add(r)
    return [dict(zip(keys, key), **agg.record()) for key, agg in sorted(groups.items())]


def duration_bin(seconds):
    if seconds < 5: return "00-05s"
    if seconds < 10: return "05-10s"
    if seconds < 25: return "10-25s"
    if seconds < 45: return "25-45s"
    if seconds < 90: return "45-90s"
    return "90s+"


def review_flags(r):
    flags = []
    if r["wer_percent"] >= 100:
        flags.append("WER_GE_100")
    if r["ref_words"] >= 8 and r["d"] / r["ref_words"] >= .5:
        flags.append("HIGH_DELETION_RATE")
    if r["i"] >= 20:
        flags.append("MANY_INSERTIONS")
    if r["ref_words"] >= 3 and not r["hyp_norm"].strip():
        flags.append("EMPTY_PREDICTION")
    if r["ref_words"] <= 2 and r["wer_percent"] >= 100:
        flags.append("SHORT_REFERENCE_HIGH_WER")
    if r["rms"] is not None and r["rms"] < .002:
        flags.append("LOW_RMS_CHECK_AUDIO")
    if r["duration_s"] is not None and r["duration_s"] > 60:
        flags.append("LONG_RECORDING")
    return ";".join(flags)


def score_predictions(rows, preds, normalizer):
    import jiwer
    scored = []
    n_empty_refs = 0
    for r in rows:
        p = preds.get(r["audio_path"])
        if p is None:
            continue
        ref_norm = normalizer(str(r["text"]))
        hyp_norm = normalizer(str(p["hyp"]))
        if not ref_norm.strip():
            n_empty_refs += 1
            continue
        words = jiwer.process_words(ref_norm, hyp_norm)
        chars = jiwer.process_characters(ref_norm, hyp_norm)
        ref_words = words.hits + words.substitutions + words.deletions
        ref_chars = chars.hits + chars.substitutions + chars.deletions
        word_errors = words.substitutions + words.deletions + words.insertions
        char_errors = chars.substitutions + chars.deletions + chars.insertions
        out = {
            "split": r.get("split", ""),
            "speaker_id": r["speaker_id"],
            "category": r.get("category", "MISSING"),
            "subcategory": r.get("subcategory", "MISSING"),
            "audio_path": r["audio_path"],
            "duration_s": p.get("duration_s"),
            "rms": p.get("rms"),
            "peak": p.get("peak"),
            "inference_method": p.get("inference_method", "unknown"),
            "num_chunks": p.get("num_chunks", 1),
            "native_long_error": p.get("native_long_error", ""),
            "text": r["text"], "hyp": p["hyp"],
            "ref_norm": ref_norm, "hyp_norm": hyp_norm,
            "ref_words": ref_words, "hyp_words": len(hyp_norm.split()),
            "hits": words.hits, "s": words.substitutions, "d": words.deletions,
            "i": words.insertions, "word_errors": word_errors,
            "ref_chars": ref_chars, "char_errors": char_errors,
            "wer_percent": round(100 * word_errors / ref_words, 4),
            "cer_percent": round(100 * char_errors / ref_chars, 4) if ref_chars else None,
            "inference_seconds": p.get("inference_seconds", 0),
        }
        out["review_flags"] = review_flags(out)
        scored.append(out)
    return scored, n_empty_refs


def make_report(split, rows, preds, folder, normalizer, *, partial):
    echo(f"[{split}] calculating WER/CER and reports from {len(preds):,} cached predictions...")
    scored, empty_refs = score_predictions(rows, preds, normalizer)
    if not scored:
        json_dump(folder / "summary.json", {"split": split, "n_scored": 0, "n_selected": len(rows)})
        return {"split": split, "n_scored": 0, "n_selected": len(rows)}

    for r in scored:
        r["split"] = split
    overall = get_stats(scored)
    by_speaker = per_group(scored, "speaker_id")
    by_category = per_group(scored, "category")
    by_speaker_category = per_group(scored, "speaker_id", "category")
    by_subcategory = per_group(scored, "category", "subcategory")
    by_method = per_group(scored, "inference_method")
    for r in scored:
        r["duration_bin"] = duration_bin(float(r.get("duration_s") or 0))
    by_duration = per_group(scored, "duration_bin")

    by_speaker.sort(key=lambda r: (-(r["wer"] or 0), r["speaker_id"]))
    # Category-balanced speaker WER is supplementary: averaging category-level
    # *corpus* WER gives equal weight to observed categories, rather than to
    # utterances/words. It may be unstable when a speaker has few samples in a
    # category, so the primary difficulty rank remains overall speaker WER.
    category_wer_by_speaker = defaultdict(list)
    for g in by_speaker_category:
        if g["wer"] is not None:
            category_wer_by_speaker[g["speaker_id"]].append(g["wer"])
    manifest_counts = Counter(r["speaker_id"] for r in rows)
    for r in by_speaker:
        cwers = category_wer_by_speaker[r["speaker_id"]]
        r["category_equal_wer_percent"] = round(statistics.mean(cwers), 4) if cwers else None
        r["n_categories_scored"] = len(cwers)
        r["error_share_percent"] = round(100 * r["word_errors"] / overall["word_errors"], 4) if overall["word_errors"] else 0.0
    speaker_wers = [r["wer"] for r in by_speaker if r["wer"] is not None]
    k = max(1, math.ceil(.25 * len(speaker_wers))) if speaker_wers else 0
    worst_group = by_speaker[:k]
    worst_ids = {r["speaker_id"] for r in worst_group}
    overall_speakers = len(by_speaker)
    selected_paths = {r["audio_path"] for r in rows}
    n_predicted = sum(path in preds for path in selected_paths)
    # Some valid references may normalize to empty (e.g. filler-only text).
    # These are explicitly counted/excluded from WER, not inference failures.
    complete = n_predicted == len(rows) and (len(scored) + empty_refs) == len(rows)

    for rank, record in enumerate(by_speaker, 1):
        record["difficulty_rank"] = rank
        record["difficulty_percentile"] = round(100 * (overall_speakers - rank + 1) / overall_speakers, 2)
        record["top25pct_hard_candidate"] = int(record["speaker_id"] in worst_ids)
        record["difficulty_group"] = "hard_top25" if record["speaker_id"] in worst_ids else "other"
        record["ranking_status"] = "FINAL" if complete else "PROVISIONAL_INCOMPLETE"
        record["n_manifest"] = manifest_counts[record["speaker_id"]]
        record["coverage_percent"] = round(100 * record["n"] / record["n_manifest"], 2)

    summary = {
        "script_version": SCRIPT_VERSION, "split": split, "model_id": MODEL_ID,
        "n_manifest": len(rows), "n_selected": len(rows), "n_predicted": n_predicted, "n_scored": len(scored),
        "n_missing_predictions": len(rows) - n_predicted,
        "n_empty_normalized_references": empty_refs,
        "n_speakers": overall_speakers, "complete": complete, "partial_or_interrupted": partial,
        "overall": overall,
        "macro_speaker_wer": round(statistics.mean(speaker_wers), 4) if speaker_wers else None,
        "median_speaker_wer": round(statistics.median(speaker_wers), 4) if speaker_wers else None,
        "p90_speaker_wer": round(sorted(speaker_wers)[min(len(speaker_wers)-1, math.ceil(.9*len(speaker_wers))-1)], 4) if speaker_wers else None,
        "worst_quartile_macro_speaker_wer": round(statistics.mean(x["wer"] for x in worst_group), 4) if worst_group else None,
        "worst_quartile_n_speakers": k,
        "worst_quartile_speaker_ids": sorted(worst_ids),
        "by_category": {r["category"]: {k: v for k, v in r.items() if k != "category"} for r in by_category},
        "inference_methods": dict(Counter(p.get("inference_method", "unknown") for k, p in preds.items() if k in selected_paths)),
        "review_flag_counts": dict(Counter(flag for r in scored for flag in r["review_flags"].split(";") if flag)),
        "metric_definition": "WER/CER are corpus micro rates; macro speaker WER is mean of per-speaker corpus WER. Normalize with Whisper EnglishTextNormalizer, as in v2.",
        "caution": "High baseline WER denotes ASR difficulty, NOT clinical PD severity. Do not tune on test. If incomplete, difficulty ranking is provisional.",
    }
    json_dump(folder / "summary.json", summary)

    with (folder / "scored_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps({k: r.get(k) for k in REPORT_COLS}, ensure_ascii=False) + "\n")
    csv_dump(folder / "recording_scores.csv", REPORT_COLS, scored)
    csv_dump(folder / "speaker_scores.csv", list(by_speaker[0]), by_speaker)
    csv_dump(folder / "speaker_difficulty.csv", list(by_speaker[0]), by_speaker)
    csv_dump(folder / "category_scores.csv", list(by_category[0]), by_category)
    csv_dump(folder / "speaker_category_scores.csv", list(by_speaker_category[0]), by_speaker_category)
    csv_dump(folder / "subcategory_scores.csv", list(by_subcategory[0]), by_subcategory)
    csv_dump(folder / "duration_scores.csv", list(by_duration[0]), by_duration)
    csv_dump(folder / "inference_method_scores.csv", list(by_method[0]), by_method)

    review = [r for r in scored if r["review_flags"]]
    # Absolute error counts prioritize clips with substantive transcription errors.
    review.sort(key=lambda r: (-r["word_errors"], -r["ref_words"], -r["wer_percent"]))
    csv_dump(folder / "review_candidates.csv", REPORT_COLS, review[:1000])

    # E1/E2/E3 sampling probabilities; derived from completed Train baseline only.
    if split == "train" and not complete:
        (folder / "training_sampling_weights.csv").unlink(missing_ok=True)
    if split == "train" and complete:
        n_speakers = len(by_speaker)
        n_hard = len(worst_ids)
        n_other = n_speakers - n_hard
        n_train_utts = len(rows)
        weights = []
        for s in by_speaker:
            n_record = s["n_manifest"]
            hard = bool(s["top25pct_hard_candidate"])
            p_e1_speaker = n_record / n_train_utts
            p_e2_speaker = 1 / n_speakers
            p_e3_speaker = (0.5 / n_hard if hard else 0.5 / n_other) if n_other else 1/n_speakers
            weights.append({
                "speaker_id": s["speaker_id"], "difficulty_rank": s["difficulty_rank"],
                "difficulty_wer_percent": s["wer"], "hard_top25": int(hard),
                "n_manifest": n_record,
                "e1_uniform_recording_p_speaker": round(p_e1_speaker, 12),
                "e2_speaker_balanced_p_speaker": round(p_e2_speaker, 12),
                "e3_hard50_p_speaker": round(p_e3_speaker, 12),
                "e1_per_recording_probability": round(1 / n_train_utts, 14),
                "e2_per_recording_probability": round(p_e2_speaker / n_record, 14),
                "e3_per_recording_probability": round(p_e3_speaker / n_record, 14),
            })
        csv_dump(folder / "training_sampling_weights.csv", list(weights[0]), weights)

    (folder / "HOW_TO_READ_RESULTS.txt").write_text(
        "Qwen3-ASR Full SAP PD Baseline (Whisper-normalized)\n"
        "- summary.json: corpus micro WER/CER, macro speaker WER, worst-quartile summary.\n"
        "- predictions.jsonl: checkpoint; raw hypotheses; resume from here.\n"
        "- recording_scores.csv: per-utterance S/D/I, CER, duration, signal RMS, method.\n"
        "- speaker_difficulty.csv: full speaker ranking; FINAL only when all rows scored.\n"
        "- speaker_category_scores.csv: errors per speaker and category.\n"
        "- review_candidates.csv: top 1,000 rule-flagged clips; inspect audio manually.\n"
        "- training_sampling_weights.csv: E1 uniform-recording, E2 speaker-balanced,\n"
        "  E3 hard-speaker-50% proposal; produced only for COMPLETE train run.\n"
        "Notes: WER may exceed 100%. Low waveform RMS is not a silence diagnosis.\n"
        "Model-specific WER is not clinical disease-severity measurement.\n"
        "Do not commit results containing transcripts, speaker IDs, or private paths.\n",
        encoding="utf-8",
    )
    echo(f"[{split}] {len(scored):,}/{len(rows):,} scored; WER={overall['wer']}% CER={overall['cer']}% ")
    echo(f"[{split}] speaker macro WER={summary['macro_speaker_wer']}%; worst-quartile macro={summary['worst_quartile_macro_speaker_wer']}%")
    return summary


def existing_predictions(path: Path, expected: set[str]):
    cached = {}
    if path.exists():
        for record in read_jsonl(path):
            p = record.get("audio_path")
            if record.get("model_id") != MODEL_ID:
                raise RuntimeError(f"Wrong model_id in prediction cache {path}: {record.get('model_id')}")
            if p not in expected:
                raise RuntimeError(f"Unexpected audio_path in cache {path}: {p}")
            if p in cached:
                raise RuntimeError(f"Duplicate cached prediction for {p} in {path}")
            if not isinstance(record.get("hyp"), str):
                raise RuntimeError(f"Invalid cached hypothesis: {p}")
            cached[p] = record
    return cached


def check_prior_val_config(source: Path, val_manifest: Path, args):
    config_file = source.parent / "run_config.json"
    if not config_file.is_file():
        raise RuntimeError(f"Old val run_config.json not found beside {source}; refusing unverified import")
    info = json.loads(config_file.read_text(encoding="utf-8"))
    checks = {
        "model_id": MODEL_ID,
        "manifest_sha256": sha256(val_manifest),
        "chunk_seconds": args.chunk_seconds,
        "max_new_tokens": args.max_new_tokens,
        "normalizer": NORMALIZER,
    }
    for key, value in checks.items():
        if info.get(key) != value:
            raise RuntimeError(f"Existing val predictions mismatch on {key}: {info.get(key)!r} != {value!r}")
    if info.get("long_audio") != "native_first_then_nonoverlap_chunks":
        raise RuntimeError("Old val decoding mode does not match this script")


def run_split(split, rows, manifest, out_root, args, normalizer):
    folder = out_root / split
    folder.mkdir(parents=True, exist_ok=True)
    path_preds = folder / "predictions.jsonl"
    path_errors = folder / "errors.jsonl"
    config_path = folder / "run_config.json"
    config = {
        "script_version": SCRIPT_VERSION, "split": split, "model_id": MODEL_ID,
        "manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
        "normalizer": NORMALIZER, "max_new_tokens": args.max_new_tokens,
        "chunk_seconds": args.chunk_seconds,
        "long_audio": "native_first_then_nonoverlap_chunks",
        "generation": "greedy, Qwen apply_transcription_request(language=English)",
    }
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if config != old:
            raise RuntimeError(f"Existing {config_path} has different settings! Use a fresh --out")
    else:
        json_dump(config_path, config)

    expected = {r["audio_path"] for r in rows}
    cached = existing_predictions(path_preds, expected)
    if split == "val" and args.reuse_val_from:
        source = args.reuse_val_from.resolve()
        if source == path_preds.resolve():
            raise RuntimeError("--reuse-val-from must be different from output predictions file")
        check_prior_val_config(source, manifest, args)
        importable = existing_predictions(source, expected)
        for r in rows:
            p = importable.get(r["audio_path"])
            if p and r["audio_path"] not in cached:
                jsonl_append(path_preds, p)
                cached[r["audio_path"]] = p
        echo(f"[val] imported compatible existing predictions: {len(importable):,}")

    selected = rows[:args.max_utts] if args.max_utts else rows
    pending = [r for r in selected if r["audio_path"] not in cached]
    echo(f"[{split}] manifest={len(rows):,}, selected={len(selected):,}, cached={len(cached):,}, pending={len(pending):,}")
    runner = None
    consecutive_errors = 0
    interrupted = False
    aborted_for_errors = False
    error_types = Counter()
    started = time.perf_counter()

    try:
        if not args.score_only:
            for idx, r in enumerate(pending, 1):
                try:
                    wav, duration, rms, peak = load_audio(r["audio_path"])
                    if runner is None:
                        echo(f"[{split}] loading {MODEL_ID} (will use CUDA)...")
                        try:
                            runner = QwenRunner()
                        except Exception as exc:
                            raise RuntimeError(f"MODEL_LOAD_FAILED: {exc}") from exc
                    t0 = time.perf_counter()
                    hyp, method, nchunks, long_error = transcribe(
                        runner, wav, duration, args.max_new_tokens, args.chunk_seconds
                    )
                    elapsed = time.perf_counter() - t0
                    if not isinstance(hyp, str):
                        raise TypeError(f"Qwen returned {type(hyp)}, expected str")
                    p = {
                        "model_id": MODEL_ID, "split": split,
                        "audio_path": r["audio_path"], "speaker_id": r["speaker_id"],
                        "category": r.get("category", "MISSING"), "hyp": hyp,
                        "duration_s": round(duration, 3), "rms": round(rms, 8),
                        "peak": round(peak, 8), "inference_seconds": round(elapsed, 4),
                        "inference_method": method, "num_chunks": nchunks,
                        "native_long_error": long_error,
                    }
                    jsonl_append(path_preds, p)
                    cached[p["audio_path"]] = p
                    consecutive_errors = 0
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    consecutive_errors += 1
                    error_types[type(exc).__name__] += 1
                    jsonl_append(path_errors, {
                        "split": split, "audio_path": r["audio_path"],
                        "error_type": type(exc).__name__, "error": str(exc),
                        "epoch_time": time.time(),
                    })
                    print(f"[{split}] ERROR {idx}/{len(pending)}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    if "MODEL_LOAD_FAILED:" in str(exc) or consecutive_errors >= args.max_consecutive_errors:
                        aborted_for_errors = True
                        echo(f"[{split}] aborting after model-load failure or {consecutive_errors} consecutive errors")
                        break
                    try:
                        import torch
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    except Exception:
                        pass
                if idx == 1 or idx % args.print_every == 0 or idx == len(pending):
                    elapsed_wall = max(time.perf_counter() - started, 1e-9)
                    echo(f"[{split}] {idx:,}/{len(pending):,} pending attempted; "
                         f"{len(cached):,}/{len(rows):,} total cached; {elapsed_wall/60:.1f} min elapsed")
                    json_dump(folder / "progress.json", {
                        "n_manifest": len(rows), "n_selected": len(selected),
                        "n_cached": len(cached), "n_remaining_selected": len(selected) - sum(r["audio_path"] in cached for r in selected),
                        "errors_this_run": dict(error_types), "elapsed_minutes": round(elapsed_wall/60, 2),
                    })
    except KeyboardInterrupt:
        interrupted = True
        echo(f"[{split}] Ctrl+C detected; saved predictions retained; generating partial reports...")
    finally:
        summary = make_report(split, rows, cached, folder, normalizer,
                              partial=interrupted or len(cached) < len(rows))
        summary["errors_this_run"] = dict(error_types)
        json_dump(folder / "summary.json", summary)
    return summary, interrupted or aborted_for_errors


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/pd_prepared"),
                   help="Existing PD split manifests; no source audio files are modified")
    p.add_argument("--splits", choices=SPLITS, nargs="+", default=["train"],
                   help="default=train; can request train val test, but test requires --include-test")
    p.add_argument("--include-test", action="store_true", help="Explicitly unseal TEST; do NOT use for method selection")
    p.add_argument("--out", type=Path, default=Path("runs/qwen_full_baseline"))
    p.add_argument("--max-utts", type=int, default=0,
                   help="0=FULL split, otherwise first N sorted paths for a quick smoke. Never a balanced sample.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--chunk-seconds", type=float, default=25.0)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--max-consecutive-errors", type=int, default=12)
    p.add_argument("--score-only", action="store_true", help="Recalculate reports from saved predictions, no GPU inference")
    p.add_argument("--reuse-val-from", type=Path, default=None,
                   help="Import prior v2 Qwen VAL predictions after verifying manifest hash and generation settings")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.splits = list(dict.fromkeys(args.splits))
    if "test" in args.splits and not args.include_test:
        raise SystemExit("TEST is held out. To run it explicitly use --include-test; do not select/tune models on TEST.")
    if args.max_utts < 0 or args.max_new_tokens <= 0 or args.chunk_seconds <= 0:
        raise SystemExit("Invalid --max-utts, --max-new-tokens or --chunk-seconds")
    if args.print_every <= 0 or args.max_consecutive_errors <= 0:
        raise SystemExit("--print-every and --max-consecutive-errors must be positive")
    if args.reuse_val_from and "val" not in args.splits:
        raise SystemExit("--reuse-val-from is valid only if --splits includes val")

    manifests, records = {}, {}
    for split in args.splits:
        manifest = (args.data_dir / f"{split}.jsonl").resolve()
        manifests[split] = manifest
        records[split] = load_manifest(manifest, split)
    seen_speakers = {}
    for split, rows in records.items():
        ids = {r["speaker_id"] for r in rows}
        for old_split, old_ids in seen_speakers.items():
            overlap = old_ids & ids
            if overlap:
                raise SystemExit(f"Speaker leakage between {old_split}/{split}: {len(overlap)} IDs")
        seen_speakers[split] = ids
        echo(f"[{split}] {len(rows):,} recordings, {len(ids)} speakers; manifest SHA256={sha256(manifests[split])[:12]}...")
    args.out.mkdir(parents=True, exist_ok=True)
    from whisper_normalizer.english import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer()
    summaries = []
    for split in args.splits:
        summary, interrupted = run_split(split, records[split], manifests[split], args.out, args, normalizer)
        summaries.append({"split": split, "n_scored": summary["n_scored"],
                          "n_selected": summary["n_selected"], "n_speakers": summary.get("n_speakers"),
                          "wer": summary.get("overall", {}).get("wer"),
                          "cer": summary.get("overall", {}).get("cer"),
                          "macro_speaker_wer": summary.get("macro_speaker_wer"),
                          "complete": summary.get("complete")})
        if interrupted:
            break
    csv_dump(args.out / "split_comparison.csv", list(summaries[0]), summaries)
    echo(f"Done. Results: {args.out.resolve()}")
    echo("Privacy: reports contain private transcripts and speaker IDs; do not commit to a public repo.")


if __name__ == "__main__":
    main()
