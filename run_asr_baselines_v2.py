#!/usr/bin/env python3
r"""Zero-shot ASR benchmarks on a prepared SAP train/val/test JSONL manifest.

This script does NOT train models, re-split speakers, or edit the SAP source data.
It supports Whisper-small, Qwen3-ASR, Cohere Transcribe, and Higgs Audio STT.

Typical PowerShell commands (after activating .venv):
  python .\run_asr_baselines_v2.py --manifest .\data\pd_prepared\val.jsonl --models whisper qwen --max-utts 20 --out .\runs\baseline_v2_smoke
  python .\run_asr_baselines_v2.py --manifest .\data\pd_prepared\val.jsonl --models whisper qwen --max-utts 0 --out .\runs\baseline_v2_full
  python .\run_asr_baselines_v2.py --manifest .\data\pd_prepared\val.jsonl --models whisper qwen --max-utts 0 --out .\runs\baseline_v2_full --score-only

Required: torch, transformers, accelerate, numpy, soundfile, scipy, jiwer,
          whisper-normalizer. Higgs also requires boson_multimodal, and Cohere
          requires accepting its Hugging Face gated model license.

Privacy: reports and manifests contain speaker IDs, transcripts and file paths.
Keep the output in your approved restricted environment; do not commit them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

MODEL_IDS = {
    "whisper": "openai/whisper-small",
    "qwen": "Qwen/Qwen3-ASR-1.7B-hf",
    "cohere": "CohereLabs/cohere-transcribe-03-2026",
    "higgs": "bosonai/higgs-audio-v3-stt",
}
MODEL_NAMES = tuple(MODEL_IDS)
NORMALIZER_NAME = "whisper_normalizer.english.EnglishTextNormalizer"


# ------------------------ I/O and reproducible selection ---------------------
def read_jsonl(path: Path):
    """Yield JSON objects; include useful line numbers if an input is malformed."""
    with path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}, line {n}: {exc}") from exc


def append_jsonl(path: Path, record: dict):
    """Persist each successful prediction immediately so interrupted runs resume."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, columns, records):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(records)


def select_rows(rows, max_utts, seed):
    """Small runs sample categories evenly; --max-utts 0 uses ALL manifest rows.

    Sampling is deterministic and shared across models. It is not speaker-balanced.
    Do not treat a 20-recording smoke test as a representative WER estimate.
    """
    rows = sorted(rows, key=lambda r: str(r["audio_path"]).casefold())
    if max_utts <= 0 or max_utts >= len(rows):
        return rows
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("category", "MISSING")].append(row)
    for g in groups.values():
        rng.shuffle(g)
    chosen = []
    while len(chosen) < max_utts and any(groups.values()):
        for category in sorted(groups):
            if groups[category] and len(chosen) < max_utts:
                chosen.append(groups[category].pop())
    return sorted(chosen, key=lambda r: str(r["audio_path"]).casefold())


def load_audio(path: str):
    """Read full WAV as float32 mono 16 kHz, resampling only if necessary."""
    import numpy as np
    import soundfile as sf
    from math import gcd

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio):
        raise ValueError("empty WAV")
    mono = audio.mean(axis=1)
    if rate != 16000:
        from scipy.signal import resample_poly
        g = gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // g, rate // g)
    mono = np.asarray(mono, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError("WAV contains NaN/Inf")
    return mono, len(mono) / 16000


# ------------------------ Model-specific inference --------------------------
class WhisperRunner:
    """HF Whisper pipeline. For long recordings use Whisper timestamp-based generation.

    The pipeline can feed full long audio to Whisper's native sequential
    long-form decoding when return_timestamps=True (no 30-second skip).
    """
    def __init__(self):
        import torch
        from transformers import pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition", model=MODEL_IDS["whisper"],
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    def __call__(self, audio, max_new_tokens, is_long=False):
        opts = {"language": "english", "task": "transcribe"}
        # Long-form generates multiple segments, so per-call token caps can be
        # problematic. Keep generation settings native for long recordings.
        if not is_long:
            opts["max_new_tokens"] = max_new_tokens
        out = self.pipe(
            {"raw": audio, "sampling_rate": 16000},
            generate_kwargs=opts,
            return_timestamps=True if is_long else False,
        )
        return str(out["text"])


class QwenRunner:
    """Qwen3-ASR Transformers model, using its official transcription request."""
    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_IDS["qwen"])
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_IDS["qwen"], device_map="auto", dtype=dtype
        ).eval()

    def __call__(self, audio, max_new_tokens, is_long=False):
        p, m, torch = self.processor, self.model, self.torch
        inputs = p.apply_transcription_request([audio], language=["English"]).to(m.device, m.dtype)
        # Long inputs need more output tokens; this is a ceiling, not a target.
        token_limit = max(max_new_tokens, min(2048, int(len(audio) / 16000 * 15))) if is_long else max_new_tokens
        with torch.inference_mode():
            out = m.generate(**inputs, max_new_tokens=token_limit, do_sample=False)
        generated = out[:, inputs["input_ids"].shape[1]:]
        decoded = p.decode(generated, return_format="transcription_only")
        return str(decoded[0] if isinstance(decoded, (list, tuple)) else decoded)


class CohereRunner:
    """Cohere Transcribe via the *native* Transformers >=5.4 model classes.

    Why not a generic ASR pipeline with trust_remote_code=True?
    Older custom Cohere code can conflict with Transformers' native loader
    (e.g., list|set TypeError or unexpected code_revision argument).

    Cohere's processor natively chunks sufficiently long audio and its
    decode(audio_chunk_index=...) reassembles the transcript. This is preferable
    to doing an extra, arbitrary fixed-length split in our outer pipeline.
    """

    def __init__(self):
        import torch
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("5.4.0"):
            raise RuntimeError(
                "Cohere Transcribe requires transformers>=5.4.0 for its "
                "native ASR class; installed version is " + transformers.__version__
                + ". Use a separate .venv-cohere if upgrading would affect other models."
            )

        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        model_id = MODEL_IDS["cohere"]
        self.torch = torch
        # No trust_remote_code: use the model class included in Transformers.
        self.processor = AutoProcessor.from_pretrained(model_id)

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model = CohereAsrForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype, device_map="auto"
            ).eval()
        else:
            self.model = CohereAsrForConditionalGeneration.from_pretrained(
                model_id, device_map="cpu"
            ).eval()

    def __call__(self, audio, max_new_tokens, is_long=False):
        # Input audio from load_audio() is float32 mono waveform at 16 kHz.
        # This processor may internally create multiple chunks for long audio.
        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt", language="en"
        )
        chunk_index = inputs.get("audio_chunk_index")
        inputs.to(self.model.device, dtype=self.model.dtype)

        with self.torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # The chunk index is necessary for combining native long-form output.
        decode_kwargs = {"skip_special_tokens": True}
        if chunk_index is not None:
            decode_kwargs.update(audio_chunk_index=chunk_index, language="en")
        transcription = self.processor.decode(tokens, **decode_kwargs)
        if isinstance(transcription, (tuple, list)):
            if len(transcription) != 1:
                raise RuntimeError(
                    f"Cohere returned {len(transcription)} transcripts for one audio input"
                )
            transcription = transcription[0]
        return str(transcription).strip()


class HiggsRunner:
    """Higgs' custom audio path. Requires boson_multimodal and adequate VRAM.

    The custom API is version-sensitive; validate with a short smoke test.
    Chunking only helps input length, not a model that cannot load into VRAM.
    """
    def __init__(self):
        import torch
        from dataclasses import asdict
        from functools import partial
        from transformers import AutoConfig, AutoModel, AutoTokenizer, WhisperProcessor
        from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
        from boson_multimodal.data_types import ChatMLSample, AudioContent, Message
        from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample_qwen
        self.torch, self.asdict = torch, asdict
        self.partial, self.ChatMLSample = partial, ChatMLSample
        self.AudioContent, self.Message = AudioContent, Message
        self.ChatMLDatasetSample = ChatMLDatasetSample
        self.prepare_chatml_sample_qwen = prepare_chatml_sample_qwen
        mid = MODEL_IDS["higgs"]
        config = AutoConfig.from_pretrained(mid, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            mid, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="eager", device_map="cuda:0",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(mid)
        self.model.audio_out_bos_token_id = self.tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
        self.model.audio_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|audio_eos|>")
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=WhisperProcessor.from_pretrained("openai/whisper-large-v3"),
            audio_in_token_id=config.audio_in_token_idx,
            audio_out_token_id=config.audio_out_token_idx,
            audio_stream_bos_id=config.audio_stream_bos_id,
            audio_stream_eos_id=config.audio_stream_eos_id,
            encode_whisper_embed=config.encode_whisper_embed,
            pad_token_id=config.pad_token_id,
            return_audio_in_tokens=config.encode_audio_in_tokens,
            use_delay_pattern=config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=config.audio_num_codebooks,
            chunk_size_seconds=getattr(config, "chunk_size_seconds", 30),
            encoder_padding_method=getattr(config, "encoder_padding_method", "max_length"),
        )

    def __call__(self, audio, max_new_tokens, is_long=False):
        torch = self.torch
        prompt = "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."
        messages = [self.Message(role="user", content=[prompt, self.AudioContent(audio_url="placeholder")])]
        chatml = self.ChatMLSample(messages=messages)
        prep = self.partial(self.prepare_chatml_sample_qwen, enable_thinking=True)
        input_tokens, _, _, _ = prep(chatml, self.tokenizer, add_generation_prompt=True)
        sample = self.ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens), label_ids=None,
            audio_ids_concat=None, audio_ids_start=None,
            audio_waveforms_concat=torch.tensor(audio, dtype=torch.float32),
            audio_waveforms_start=torch.tensor([0]),
            audio_sample_rate=torch.tensor([16000]),
            audio_speaker_indices=torch.tensor([0]),
        )
        batch = self.asdict(self.collator([sample]))
        device = next(self.model.parameters()).device
        batch = {k: v.to(device).contiguous() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **batch, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False,
                stop_strings=["<|im_end|>", "<|endoftext|>"], tokenizer=self.tokenizer,
            )
        ids = output[0] if isinstance(output, tuple) else output
        full = self.tokenizer.decode(ids[0], skip_special_tokens=False)
        parts = full.split("assistant\n")
        hyp = parts[-1] if len(parts) > 1 else full
        hyp = re.sub(r"<think>.*?</think>", "", hyp, flags=re.DOTALL)
        return re.sub(r"<\|.*?\|>", "", hyp).strip()


def make_runner(name):
    return {"whisper": WhisperRunner, "qwen": QwenRunner,
            "cohere": CohereRunner, "higgs": HiggsRunner}[name]()


def chunk_audio(audio, seconds):
    """Partition into contiguous non-overlapping pieces, preserving all samples.

    No overlap is used intentionally: naive overlapping chunk transcription
    often duplicates words and changes WER. Boundary words may still suffer.
    """
    n = int(seconds * 16000)
    if n <= 0:
        raise ValueError("chunk size must be positive")
    return [audio[i:i+n] for i in range(0, len(audio), n)]


def transcribe_full(runner, name, audio, duration, max_new_tokens, chunk_seconds):
    """Native full-audio first; for long-audio errors, retry contiguous chunks.

    Returns (transcript, method, num_chunks, native_error).
    The last field is useful for auditing why fallback was invoked.
    """
    if duration <= chunk_seconds:
        return runner(audio, max_new_tokens, is_long=False), "native_short", 1, ""
    try:
        return runner(audio, max_new_tokens, is_long=True), "native_long", 1, ""
    except Exception as exc:
        original_error = f"{type(exc).__name__}: {exc}"
        print(f"  [{name}] native long-form failed; trying chunks: {original_error}", flush=True)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        pieces = chunk_audio(audio, chunk_seconds)
        parts = []
        for idx, piece in enumerate(pieces, 1):
            # Short, independent windows do not share linguistic context.
            part = runner(piece, max_new_tokens, is_long=False)
            parts.append(part.strip())
        return " ".join(x for x in parts if x), "fixed_chunks_no_overlap", len(pieces), original_error


# ------------------------ Metric calculation --------------------------------
def word_alignment(ref, hyp):
    """Per-recording S/D/I counts for inspection, not per-recording mean WER."""
    import jiwer
    return jiwer.process_words(ref, hyp)


def aggregate(rows):
    """Corpus WER=(sum S+D+I)/(sum reference words), not mean clip WER."""
    import jiwer
    if not rows:
        return {"n": 0, "wer": None, "cer": None,
                "substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0}
    ref = [r["ref_norm"] for r in rows]
    hyp = [r["hyp_norm"] for r in rows]
    w = jiwer.process_words(ref, hyp)
    return {
        "n": len(rows), "wer": round(w.wer * 100, 4),
        "cer": round(jiwer.cer(ref, hyp) * 100, 4),
        "substitutions": w.substitutions, "deletions": w.deletions,
        "insertions": w.insertions,
        "ref_words": w.hits + w.substitutions + w.deletions,
    }


def score_predictions(selected, predictions, normalizer):
    """Only successful predictions are scored; missing and empty refs are counted.

    Predictions are stored raw. Recompute normalized metrics later with
    --score-only, without re-running any GPU inference.
    """
    valid, empty_refs = [], 0
    for row in selected:
        pred = predictions.get(row["audio_path"])
        if pred is None:
            continue
        ref_norm = normalizer(row["text"])
        hyp_norm = normalizer(pred["hyp"])
        if not ref_norm.strip():
            empty_refs += 1
            continue
        aligned = word_alignment(ref_norm, hyp_norm)
        valid.append({
            **row, "hyp": pred["hyp"], "ref_norm": ref_norm, "hyp_norm": hyp_norm,
            "duration_s": pred.get("duration_s"),
            "inference_method": pred.get("inference_method", "unknown"),
            "num_chunks": pred.get("num_chunks", 1),
            "s": aligned.substitutions, "d": aligned.deletions, "i": aligned.insertions,
            "ref_words": aligned.hits + aligned.substitutions + aligned.deletions,
            "wer_percent": round(aligned.wer * 100, 4),
            "cer_percent": round(__import__("jiwer").cer(ref_norm, hyp_norm) * 100, 4),
        })
    return valid, empty_refs


def group_scores(scored, field):
    groups = defaultdict(list)
    for row in scored:
        groups[row[field]].append(row)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def emit_reports(folder, model, selected, predictions, normalizer, args, failures):
    """Write readable CSVs and per-speaker summaries for manual review."""
    scored, empty_refs = score_predictions(selected, predictions, normalizer)
    overall = aggregate(scored)
    by_category = group_scores(scored, "category")
    by_speaker = group_scores(scored, "speaker_id")
    by_speaker_category = defaultdict(list)
    for r in scored:
        by_speaker_category[(r["speaker_id"], r.get("category", "MISSING"))].append(r)
    sp_wers = [v["wer"] for v in by_speaker.values() if v["wer"] is not None]
    expected = {r["audio_path"] for r in selected}
    predicted_paths = expected & set(predictions)
    scored_paths = {r["audio_path"] for r in scored}
    summary = {
        "model_name": model, "model_id": MODEL_IDS[model],
        "manifest": str(args.manifest.resolve()), "n_selected": len(selected),
        "n_predicted": len(predicted_paths), "n_scored": len(scored),
        "n_unpredicted": len(selected) - len(predicted_paths),
        "n_empty_normalized_references": empty_refs,
        "overall": overall, "by_category": by_category, "per_speaker": by_speaker,
        "mean_speaker_wer": round(statistics.mean(sp_wers), 4) if sp_wers else None,
        "median_speaker_wer": round(statistics.median(sp_wers), 4) if sp_wers else None,
        "inference_methods": dict(Counter(predictions[k].get("inference_method", "unknown") for k in predicted_paths)),
        "failures_this_invocation": dict(failures),
        "warning": "Incomplete model outputs must not be compared as if full and identical.",
    }
    write_json(folder / "summary.json", summary)

    # The JSONL files are detailed and lossless; CSVs are more convenient in Excel.
    with (folder / "scored_predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in scored:
            cols = ("audio_path", "speaker_id", "category", "text", "hyp",
                    "ref_norm", "hyp_norm", "duration_s", "inference_method",
                    "num_chunks", "s", "d", "i", "ref_words", "wer_percent", "cer_percent")
            fh.write(json.dumps({c: r.get(c) for c in cols}, ensure_ascii=False) + "\n")

    stat_cols = ["n", "wer", "cer", "substitutions", "deletions", "insertions", "ref_words"]
    def make_stats(label, mapping):
        return [{label: key, **{col: v[col] for col in stat_cols}} for key, v in mapping.items()]

    write_csv(folder / "category_scores.csv", ["category"] + stat_cols, make_stats("category", by_category))
    write_csv(folder / "speaker_scores.csv", ["speaker_id"] + stat_cols,
              sorted(make_stats("speaker_id", by_speaker), key=lambda v: -(v["wer"] or 0)))
    write_csv(folder / "speaker_category_scores.csv", ["speaker_id", "category"] + stat_cols,
              [{"speaker_id": s, "category": c, **aggregate(rows)}
               for (s, c), rows in sorted(by_speaker_category.items())])

    inspect_cols = ["speaker_id", "category", "audio_path", "duration_s", "inference_method",
                    "num_chunks", "text", "hyp", "ref_norm", "hyp_norm", "wer_percent",
                    "cer_percent", "s", "d", "i", "ref_words"]
    recording_rows = [{k: r.get(k) for k in inspect_cols} for r in scored]
    write_csv(folder / "recording_scores.csv", inspect_cols, recording_rows)
    write_csv(folder / "error_examples.csv", inspect_cols,
              sorted((r for r in recording_rows if (r["s"]+r["d"]+r["i"])>0),
                     key=lambda r: (-r["wer_percent"], -r["ref_words"])))
    # CSV sorted by per-recording WER can prioritize short 1-word utterances.
    # Use speaker_scores.csv and ref_words to assess practical importance.

    return summary, scored


# ------------------------ Run/resume and model comparison -------------------
def manifest_fingerprint(path: Path):
    """Hash manifest bytes to reject resuming against silently modified splits."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def run_model(model, selected, args, normalizer, fingerprint):
    out = args.out / model
    out.mkdir(parents=True, exist_ok=True)
    pred_path, err_path = out / "predictions.jsonl", out / "errors.jsonl"
    config_path = out / "run_config.json"
    config = {
        "model": model, "model_id": MODEL_IDS[model],
        "manifest": str(args.manifest.resolve()), "manifest_sha256": fingerprint,
        "max_new_tokens": args.max_new_tokens,
        "long_audio": "native_first_then_nonoverlap_chunks",
        "chunk_seconds": args.chunk_seconds,
        "normalizer": NORMALIZER_NAME,
    }
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise RuntimeError(f"Existing {config_path} uses different settings; choose another --out")
    else:
        write_json(config_path, config)

    cached = {}
    if pred_path.exists():
        for p in read_jsonl(pred_path):
            if p.get("model_id") != MODEL_IDS[model]:
                raise RuntimeError(f"Prediction checkpoint mismatch in {pred_path}")
            cached[p["audio_path"]] = p

    pending = [r for r in selected if r["audio_path"] not in cached]
    print(f"[{model}] {len(selected):,} selected; {len(pending):,} pending; {len(cached):,} cached", flush=True)
    failures = Counter()
    runner = None
    if args.score_only:
        pending = []

    for idx, row in enumerate(pending, 1):
        try:
            audio, duration = load_audio(row["audio_path"])
            if runner is None:
                print(f"[{model}] loading {MODEL_IDS[model]}", flush=True)
                try:
                    runner = make_runner(model)
                except Exception as exc:
                    raise RuntimeError(f"MODEL_LOAD_FAILED: {exc}") from exc
            hyp, method, chunks, native_error = transcribe_full(
                runner, model, audio, duration, args.max_new_tokens, args.chunk_seconds
            )
            if not isinstance(hyp, str):
                raise TypeError(f"Non-string hypothesis: {type(hyp)}")
            record = {
                "model_id": MODEL_IDS[model], "audio_path": row["audio_path"],
                "speaker_id": row["speaker_id"], "category": row.get("category", "MISSING"),
                "hyp": hyp, "duration_s": round(duration, 3),
                "inference_method": method, "num_chunks": chunks,
                "native_long_error": native_error,
            }
            append_jsonl(pred_path, record)
            cached[row["audio_path"]] = record
        except Exception as exc:
            failures[type(exc).__name__] += 1
            append_jsonl(err_path, {
                "audio_path": row["audio_path"], "model_id": MODEL_IDS[model],
                "error_type": type(exc).__name__, "error": str(exc), "time": time.time()
            })
            print(f"[{model}] ERROR {idx}/{len(pending)}: {exc}", file=sys.stderr, flush=True)
            if "MODEL_LOAD_FAILED:" in str(exc):
                print(f"[{model}] aborting: could not load model", file=sys.stderr)
                break
        if idx == 1 or idx % 25 == 0 or idx == len(pending):
            print(f"[{model}] {idx}/{len(pending)} attempted", flush=True)

    summary, scored = emit_reports(out, model, selected, cached, normalizer, args, failures)
    print(f"[{model}] {summary['n_scored']}/{summary['n_selected']} scored; "
          f"WER={summary['overall']['wer']}% CER={summary['overall']['cer']}%", flush=True)
    return summary, scored


def compare_models(outputs, out: Path):
    """Compare only intersection of successfully scored audio paths (fair set)."""
    rows = []
    good = [(model, summary, scored) for model, (summary, scored) in outputs.items() if scored]
    all_sets = [{r["audio_path"] for r in scored} for _, _, scored in good]
    common = set.intersection(*all_sets) if all_sets else set()
    for model, summary, scored in good:
        shared = [r for r in scored if r["audio_path"] in common]
        raw = summary["overall"]
        fair = aggregate(shared)
        rows.append({"model": model, "n_selected": summary["n_selected"],
                     "n_scored": summary["n_scored"], "n_common": len(common),
                     "wer_all_success": raw["wer"], "cer_all_success": raw["cer"],
                     "wer_common": fair["wer"], "cer_common": fair["cer"]})
    write_csv(out / "comparison.csv", ["model", "n_selected", "n_scored", "n_common",
                                       "wer_all_success", "cer_all_success", "wer_common", "cer_common"], rows)

    # Compare speaker WER across only common recordings, not unmatched sets.
    speakers = sorted({r["speaker_id"] for _, _, scored in good for r in scored if r["audio_path"] in common})
    matrix = []
    for speaker in speakers:
        item = {"speaker_id": speaker}
        for model, _, scored in good:
            relevant = [r for r in scored if r["audio_path"] in common and r["speaker_id"] == speaker]
            result = aggregate(relevant)
            item[f"{model}_n"] = result["n"]
            item[f"{model}_wer"] = result["wer"]
            item[f"{model}_cer"] = result["cer"]
        matrix.append(item)
    columns = ["speaker_id"] + [f"{m}_{suffix}" for m, _, _ in good for suffix in ("n", "wer", "cer")]
    write_csv(out / "speaker_comparison.csv", columns, matrix)
    return common


def save_distribution_plot(outputs, out: Path):
    """Optional histogram of all individual-speaker WERs; needs matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping speaker_wer_distribution.png")
        return
    plt.figure(figsize=(9, 5))
    has_data = False
    for model, (summary, _scored) in outputs.items():
        vals = [v["wer"] for v in summary["per_speaker"].values() if v["wer"] is not None]
        if vals:
            plt.hist(vals, bins=12, alpha=0.45, label=model)
            has_data = True
    if has_data:
        plt.xlabel("Per-speaker corpus WER (%)")
        plt.ylabel("Number of speakers")
        plt.title("Validation baseline: speaker-level WER distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "speaker_wer_distribution.png", dpi=160)
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/pd_prepared/val.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/baseline_v2_full"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["whisper", "qwen"])
    parser.add_argument("--max-utts", type=int, default=20,
                        help="20 for balanced smoke; 0 for FULL manifest (recommended for validation)")
    parser.add_argument("--chunk-seconds", type=float, default=25.0,
                        help="Fallback chunk length when native long audio fails (no overlap)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-only", action="store_true",
                        help="Re-score stored predictions without loading ASR models")
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error(f"Missing manifest: {args.manifest}")
    if args.max_utts < 0 or args.chunk_seconds <= 0 or args.max_new_tokens <= 0:
        parser.error("Require max-utts>=0, chunk-seconds>0, max-new-tokens>0")

    from whisper_normalizer.english import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer()
    rows = list(read_jsonl(args.manifest))
    if not rows:
        parser.error("Empty manifest")
    paths = [r.get("audio_path") for r in rows]
    if len(set(paths)) != len(paths):
        parser.error("Manifest has duplicate audio_path values")
    for row in rows:
        if not all(k in row for k in ("audio_path", "text", "speaker_id", "category")):
            parser.error("Missing manifest fields audio_path/text/speaker_id/category")
        if not str(row["text"]).strip():
            parser.error("Manifest contains empty reference text")
    args.out.mkdir(parents=True, exist_ok=True)
    fingerprint = manifest_fingerprint(args.manifest)
    print(f"Manifest: {args.manifest} ({len(rows):,} rows); selected models: {args.models}")
    selected = select_rows(rows, args.max_utts, args.seed)
    print(f"Selected {len(selected):,} audio recordings (max-utts={args.max_utts})")

    outputs = {}
    for model in dict.fromkeys(args.models):
        try:
            outputs[model] = run_model(model, selected, args, normalizer, fingerprint)
        except Exception as exc:
            print(f"[{model}] FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
            # Fatal errors are preserved in a separate file for debugging.
            write_json(args.out / f"{model}_fatal.json", {"error": str(exc)})
        finally:
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    if outputs:
        common = compare_models(outputs, args.out)
        save_distribution_plot(outputs, args.out)
        print(f"Common scored recordings across completed models: {len(common):,}")
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
