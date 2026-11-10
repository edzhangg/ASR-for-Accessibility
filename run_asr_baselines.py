#!/usr/bin/env python3
"""Run independent zero-shot ASR baselines over the manifest made by filter_parkinsons.py.

Models: Qwen3-ASR, Cohere Transcribe (gated repo: accept its terms on Hugging Face first),
Higgs Audio STT, Whisper-small. Each model gets its own run folder under --out.
Original audio/metadata are never modified. This script does not perform training.

  python run_asr_baselines.py --manifest SAP_PD_Filtered/manifest_pd.jsonl --split Test --max-utts 20 --models qwen

Report headline numbers on --split Test only, so every model (zero-shot or fine-tuned) is scored on the
same held-out speakers. Every model uses the same --max-duration, so they are scored on the same clips.

Requires: torch (CUDA build), transformers>=5.13 (Qwen) / >=5.4 (Cohere), accelerate, soundfile, scipy,
          numpy, jiwer, whisper-normalizer, and for Higgs the boson_multimodal package.
"""
import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

MODEL_IDS = {
    "qwen": "Qwen/Qwen3-ASR-1.7B-hf",
    "cohere": "CohereLabs/cohere-transcribe-03-2026",
    "higgs": "bosonai/higgs-audio-v3-stt",
    "whisper": "openai/whisper-small",
}
MODEL_NAMES = tuple(MODEL_IDS)


# ----------------------------------------------------------------------------- reference cleaning
def _strip_markup(t):
    """Removal shared by both reference versions."""
    t = t.replace(" ", " ").replace("’", "'")
    t = re.sub(r"\[[^\]]*\]", " ", t)                 # [bracketed notes: prompt text, PII]
    t = re.sub(r"\{u:?\s*\}", " ", t)                 # {u:} unintelligible
    t = re.sub(r"\{\w+:\s*([^}]*)\}", r" \1 ", t)     # {g: guess} -> keep the guessed words
    t = re.sub(r"~[A-Za-z]\b", " ", t)                # ~A style codes
    t = re.sub(r"#dis\b", " ", t)
    return t


def reference_clean(t):
    """Bracketed notes AND parenthesized disfluencies removed."""
    return re.sub(r"\([^)]*\)", " ", _strip_markup(t))


def reference_verbatim(t):
    """Bracketed notes removed; disfluency words kept (parentheses dropped)."""
    t = re.sub(r"\(([^)]*)\)", r" \1 ", _strip_markup(t))
    return t.replace("*", " ")


VERSIONS = {"clean": reference_clean, "verbatim": reference_verbatim}


# ----------------------------------------------------------------------------- io helpers
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_number}: {e}") from e
                yield row


def append_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def select_rows(rows, max_utts, seed):
    """Balanced *sampling* across categories for smoke tests only.

    Full evaluation (--max-utts 0, the default) preserves the actual dataset distribution.
    """
    rows = sorted(rows, key=lambda r: str(r["audio_path"]).casefold())
    if max_utts <= 0 or max_utts >= len(rows):
        return rows
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("category", "MISSING")].append(r)
    for g in groups.values():
        rng.shuffle(g)
    selected = []
    names = sorted(groups)
    while len(selected) < max_utts and any(groups.values()):
        for name in names:
            if groups[name] and len(selected) < max_utts:
                selected.append(groups[name].pop())
    return sorted(selected, key=lambda r: str(r["audio_path"]).casefold())


def load_audio(path):
    import numpy as np
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if a.shape[0] == 0:
        raise ValueError("empty audio")
    a = a.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, 16000)
        a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
    return np.asarray(a, dtype=np.float32), len(a) / 16000


# ----------------------------------------------------------------------------- model runners
class ModelLoadError(RuntimeError):
    pass


def _first(decoded):
    return decoded[0] if isinstance(decoded, (list, tuple)) else str(decoded)


class QwenRunner:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        self.torch = torch
        mid = MODEL_IDS["qwen"]
        self.processor = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            mid, device_map="auto", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ).eval()

    def __call__(self, audio, max_new_tokens):
        p, m, torch = self.processor, self.model, self.torch
        inputs = p.apply_transcription_request([audio], language=["English"]).to(m.device, m.dtype)
        with torch.inference_mode():
            output = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        token_ids = output[:, inputs["input_ids"].shape[1]:]
        return _first(p.decode(token_ids, return_format="transcription_only"))


class CohereRunner:
    """Native Transformers classes, as in the model card's Quick Start (needs transformers>=5.4)."""
    def __init__(self):
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        self.torch = torch
        mid = MODEL_IDS["cohere"]
        self.processor = AutoProcessor.from_pretrained(mid)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(mid, device_map="auto").eval()

    def __call__(self, audio, max_new_tokens):
        p, m, torch = self.processor, self.model, self.torch
        inputs = p(audio, sampling_rate=16000, return_tensors="pt", language="en")
        chunk_index = inputs.get("audio_chunk_index")  # set when long audio was split into chunks
        inputs.to(m.device, dtype=m.dtype)
        with torch.inference_mode():
            outputs = m.generate(**inputs, max_new_tokens=max_new_tokens)
        kwargs = {"audio_chunk_index": chunk_index, "language": "en"} if chunk_index is not None else {}
        return _first(p.decode(outputs, skip_special_tokens=True, **kwargs))


class WhisperRunner:
    def __init__(self):
        import torch
        from transformers import pipeline
        self.pipe = pipeline(
            task="automatic-speech-recognition", model=MODEL_IDS["whisper"], device=0,
            dtype=torch.float16,
        )

    def __call__(self, audio, max_new_tokens):
        # Explicit English transcription, not English translation.
        out = self.pipe(
            {"raw": audio, "sampling_rate": 16000},
            generate_kwargs={"language": "english", "task": "transcribe", "max_new_tokens": max_new_tokens},
        )
        return str(out["text"])


class HiggsRunner:
    """Based on bosonai/higgs-audio-v3-stt model card's custom inference API.

    This path needs boson_multimodal and trust_remote_code=True. It is intentionally
    independent of Qwen/Cohere ASR wrappers.
    """
    def __init__(self):
        import torch
        from dataclasses import asdict
        from functools import partial
        from transformers import AutoConfig, AutoModel, AutoTokenizer, WhisperProcessor
        from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
        from boson_multimodal.data_types import ChatMLSample, AudioContent, Message
        from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample_qwen
        self.torch = torch
        self.asdict = asdict
        self.partial = partial
        self.ChatMLSample = ChatMLSample
        self.AudioContent = AudioContent
        self.Message = Message
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
        whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_proc,
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

    def __call__(self, audio, max_new_tokens):
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


RUNNERS = {"qwen": QwenRunner, "cohere": CohereRunner, "higgs": HiggsRunner, "whisper": WhisperRunner}


# ----------------------------------------------------------------------------- scoring
def aggregate(rows, ref_key):
    import jiwer
    if not rows:
        return {"n": 0, "wer": None, "cer": None, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0}
    refs = [r[ref_key] for r in rows]
    hyps = [r["hyp_norm"] for r in rows]
    w = jiwer.process_words(refs, hyps)
    return {
        "n": len(rows), "wer": round(100 * w.wer, 4),
        "cer": round(100 * jiwer.cer(refs, hyps), 4),
        "substitutions": w.substitutions, "deletions": w.deletions,
        "insertions": w.insertions, "ref_words": w.hits + w.substitutions + w.deletions,
    }


def summarize(selected, pred_by_path, normalizer):
    """Score every predicted row against both the clean and the verbatim reference."""
    valid = []
    for row in selected:
        p = pred_by_path.get(row["audio_path"])
        if p is None:
            continue
        scored = {**row, "hyp": p["hyp"], "hyp_norm": normalizer(p["hyp"])}
        for version, clean_fn in VERSIONS.items():
            scored[f"ref_{version}"] = normalizer(clean_fn(row["transcript"]))
        valid.append(scored)

    results = {}
    for version in VERSIONS:
        key = f"ref_{version}"
        rows = [r for r in valid if r[key].strip()]  # a reference can become empty after cleaning
        by_category, by_speaker = defaultdict(list), defaultdict(list)
        for r in rows:
            by_category[r.get("category", "MISSING")].append(r)
            by_speaker[r["speaker_id"]].append(r)
        speaker_scores = {s: aggregate(v, key) for s, v in sorted(by_speaker.items())}
        wers = [x["wer"] for x in speaker_scores.values() if x["wer"] is not None]
        results[version] = {
            "overall": aggregate(rows, key),
            "by_category": {c: aggregate(v, key) for c, v in sorted(by_category.items())},
            "per_speaker": speaker_scores,
            "speaker_wer": {"mean": round(statistics.mean(wers), 4), "median": round(statistics.median(wers), 4),
                            "best": min(wers), "worst": max(wers), "n_speakers": len(wers)} if wers else None,
            "n_empty_references": len(valid) - len(rows),
        }
    results["n_selected"] = len(selected)
    results["n_predicted"] = len(valid)
    results["n_unpredicted"] = len(selected) - len(valid)
    return results, valid


def print_table(name, results):
    f = lambda x: "    -" if x is None else f"{x:6.1f}"
    for version in VERSIONS:
        res = results[version]
        print(f"\n[{name}] reference: {version}")
        print(f"  {'':<32}{'n':>6}{'WER%':>8}{'CER%':>8}")
        o = res["overall"]
        print(f"  {'OVERALL':<32}{o['n']:>6}{f(o['wer']):>8}{f(o['cer']):>8}")
        for cat, v in res["by_category"].items():
            print(f"    {cat:<30}{v['n']:>6}{f(v['wer']):>8}{f(v['cer']):>8}")
        s = res["speaker_wer"]
        if s:
            print(f"  per-speaker WER ({s['n_speakers']} speakers): median {s['median']:.1f}  "
                  f"best {s['best']:.1f}  worst {s['worst']:.1f}  mean {s['mean']:.1f}")


def run_metadata():
    try:
        import torch
        import transformers
        return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__, "transformers": transformers.__version__}
    except ImportError:
        return {"gpu": None}


# ----------------------------------------------------------------------------- one model
def run_one(name, rows, args, normalizer):
    t0 = time.time()
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    pred_file = out / "predictions.jsonl"
    err_file = out / "errors.jsonl"
    config_file = out / "run_config.json"
    selected = select_rows(rows, args.max_utts, args.seed)
    signature = {
        "model_name": name, "model_id": MODEL_IDS[name],
        "manifest": str(args.manifest.resolve()), "split": args.split,
        "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "max_duration": args.max_duration,
        "normalizer": "whisper_normalizer.english.EnglishTextNormalizer",
    }
    if config_file.exists():
        previous = json.loads(config_file.read_text(encoding="utf-8"))
        if previous != signature:
            raise RuntimeError(f"Existing run config mismatch in {out}. Choose a new --out (or restore original settings).")
    else:
        config_file.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    predictions = {}
    if pred_file.exists():
        for p in read_jsonl(pred_file):
            if p.get("model_id") != MODEL_IDS[name]:
                raise RuntimeError(f"Model mismatch in {pred_file}")
            predictions[p["audio_path"]] = p
    pending = [r for r in selected if r["audio_path"] not in predictions]
    print(f"\n[{name}] {len(selected)} selected; {len(pending)} pending; {len(predictions)} cached", flush=True)
    runner = None
    failures = Counter()
    for idx, row in enumerate(pending, 1):
        try:
            import soundfile as sf
            header = sf.info(row["audio_path"])
            duration = header.frames / header.samplerate
            if duration > args.max_duration:
                raise ValueError(f"duration {duration:.2f}s > max_duration {args.max_duration}s")
            audio, _ = load_audio(row["audio_path"])
            if runner is None:
                print(f"[{name}] loading model {MODEL_IDS[name]}...", flush=True)
                try:
                    runner = RUNNERS[name]()
                except Exception as load_exc:
                    append_jsonl(err_file, {"model_id": MODEL_IDS[name], "audio_path": row["audio_path"],
                                            "error_type": "model_load_failed", "error": str(load_exc), "time": time.time()})
                    raise ModelLoadError(f"Failed to initialize {name}: {load_exc}") from load_exc
            hypothesis = runner(audio, args.max_new_tokens)
            if not isinstance(hypothesis, str):
                raise TypeError(f"model returned {type(hypothesis)}")
            prediction = {"model_id": MODEL_IDS[name], "audio_path": row["audio_path"], "hyp": hypothesis,
                          "speaker_id": row["speaker_id"], "category": row.get("category", "MISSING")}
            append_jsonl(pred_file, prediction)
            predictions[row["audio_path"]] = prediction
        except Exception as exc:
            typ = type(exc).__name__
            failures[typ] += 1
            append_jsonl(err_file, {"model_id": MODEL_IDS[name], "audio_path": row["audio_path"],
                                    "error_type": typ, "error": str(exc), "time": time.time()})
            print(f"[{name}] failed {idx}/{len(pending)}: {typ}: {exc}", file=sys.stderr, flush=True)
            # Model loading failures should not be repeated for every sample.
            if isinstance(exc, ModelLoadError):
                print(f"[{name}] model failed to load; aborting this model run", file=sys.stderr)
                break
        if idx == 1 or idx % 10 == 0 or idx == len(pending):
            print(f"[{name}] processed {idx}/{len(pending)}", flush=True)

    results, scored = summarize(selected, predictions, normalizer)
    results.update({
        "model_name": name, "model_id": MODEL_IDS[name],
        "manifest": str(args.manifest.resolve()), "split": args.split, "max_utts": args.max_utts,
        "seed": args.seed, "max_duration": args.max_duration,
        "failures_this_run": dict(failures), "runtime_s_this_invocation": round(time.time() - t0, 1),
        **run_metadata(),
        "warning": "Corpus metrics use only successful predictions; inspect n_unpredicted and errors before comparing models.",
    })
    (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "scored_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps({k: r[k] for k in ("audio_path", "speaker_id", "category", "transcript", "hyp",
                                                  "ref_clean", "ref_verbatim", "hyp_norm")}, ensure_ascii=False) + "\n")
    with (out / "category_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reference", "category", "n", "wer_percent", "cer_percent", "substitutions", "deletions",
                         "insertions", "ref_words"])
        for version in VERSIONS:
            for k, r in results[version]["by_category"].items():
                writer.writerow([version, k, r["n"], r["wer"], r["cer"], r["substitutions"], r["deletions"],
                                 r["insertions"], r["ref_words"]])
    print_table(name, results)
    print(f"[{name}] scored={results['n_predicted']}/{results['n_selected']} | {out}", flush=True)
    for r in scored[:5]:
        print(f"  REF: {r['ref_clean']}\n  HYP: {r['hyp_norm']}\n")
    return results


# ----------------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="manifest_pd.jsonl from filter_parkinsons.py")
    parser.add_argument("--split", default="Test",
                        help="Train/Dev/Test or 'all'. Report headline numbers on Test only.")
    parser.add_argument("--out", type=Path, default=Path("runs/baselines_test"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["qwen", "whisper"])
    parser.add_argument("--max-utts", type=int, default=0,
                        help="Category-balanced smoke-test sample size; 0 (default) means the whole split")
    parser.add_argument("--max-duration", type=float, default=120.0,
                        help="Longer clips are logged as skipped, not truncated (same cap for every model)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true", help="run even without a CUDA GPU (very slow)")
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error(f"Manifest missing: {args.manifest}")
    if args.max_utts < 0 or args.max_duration <= 0 or args.max_new_tokens <= 0:
        parser.error("Invalid max-utts, max-duration, or max-new-tokens")

    all_rows = list(read_jsonl(args.manifest))
    if not all_rows:
        parser.error("Manifest is empty")
    for row in all_rows:
        if not all(k in row for k in ("audio_path", "speaker_id", "transcript", "split")):
            parser.error("Manifest missing required fields: audio_path, speaker_id, transcript, split")
    paths = [r["audio_path"] for r in all_rows]
    if len(paths) != len(set(paths)):
        parser.error("Duplicate audio_path in manifest")
    rows = [r for r in all_rows if args.split.lower() == "all" or r["split"].lower() == args.split.lower()]
    if not rows:
        parser.error(f"No utterances in split '{args.split}' (splits present: {sorted({r['split'] for r in all_rows})})")

    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if not has_gpu and not args.allow_cpu:
        sys.exit("No CUDA GPU found (or torch is not installed). Run on the NVIDIA machine, or pass --allow-cpu.")

    from whisper_normalizer.english import EnglishTextNormalizer
    norm = EnglishTextNormalizer()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Manifest: {args.manifest} | split '{args.split}': {len(rows)} utterances | "
          f"models: {', '.join(args.models)}", flush=True)
    outputs = {}
    for name in dict.fromkeys(args.models):
        try:
            outputs[name] = run_one(name, rows, args, norm)
        except Exception as e:
            print(f"[{name}] FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            outputs[name] = {"fatal_error": str(e)}
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    with (args.out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scored", "selected", "clean_wer", "clean_cer", "verbatim_wer", "verbatim_cer",
                         "unpredicted", "error"])
        for name, r in outputs.items():
            c, v = r.get("clean", {}).get("overall", {}), r.get("verbatim", {}).get("overall", {})
            writer.writerow([name, r.get("n_predicted", 0), r.get("n_selected", ""), c.get("wer"), c.get("cer"),
                             v.get("wer"), v.get("cer"), r.get("n_unpredicted", ""), r.get("fatal_error", "")])
    print(f"\nWrote comparison: {args.out / 'comparison.csv'}", flush=True)
    if any("fatal_error" in r or r.get("n_unpredicted", 0) for r in outputs.values()):
        print("WARNING: Some models did not finish. Do not compare partial-set WER as equivalent.", flush=True)


if __name__ == "__main__":
    main()
