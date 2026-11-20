#!/usr/bin/env python3
"""Run independent zero-shot ASR baselines over prepared SAP JSONL manifests.

Models: Qwen3-ASR, Cohere Transcribe, Higgs Audio STT, Whisper-small.
Original audio/metadata are never modified. Each checkpoint has a separate run folder.
Start with --max-utts 20. This script does not perform training.

Requires: torch, transformers, accelerate, soundfile, numpy, jiwer,
          whisper-normalizer; Higgs additionally requires boson_multimodal.
Some checkpoints have version-specific requirements and may require GPU memory.
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

    Full evaluation (--max-utts 0) preserves the actual dataset distribution.
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


class ModelLoadError(RuntimeError):
    pass


class QwenTransformersRunner:
    """Transformers-native runner used by the published ``-hf`` checkpoint."""

    def __init__(self, model_id):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        self.torch = torch
        mid = model_id
        self.processor = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            mid, device_map="auto", torch_dtype=torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16,
        ).eval()

    def __call__(self, audio, max_new_tokens):
        p, m, torch = self.processor, self.model, self.torch
        inputs = p.apply_transcription_request([audio], language=["English"]).to(m.device, m.dtype)
        with torch.inference_mode():
            output = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        token_ids = output[:, inputs["input_ids"].shape[1]:]
        decoded = p.decode(token_ids, return_format="transcription_only")
        return decoded[0] if isinstance(decoded, (list, tuple)) else str(decoded)


class QwenFineTunedRunner:
    """Official qwen-asr runner used by fine-tuned Qwen3-ASR checkpoints."""

    def __init__(self, model_id, max_new_tokens, adapter_path=None):
        import torch
        from qwen_asr import Qwen3ASRModel
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="cuda:0",
            max_inference_batch_size=1,
            max_new_tokens=max_new_tokens,
        )
        if adapter_path:
            from peft import PeftModel
            adapted = PeftModel.from_pretrained(self.model.model, adapter_path)
            self.model.model = adapted.merge_and_unload().eval()

    def __call__(self, audio, max_new_tokens):
        # qwen-asr accepts a (waveform, sample_rate) pair and handles long inputs.
        results = self.model.transcribe(audio=(audio, 16000), language="English")
        if not results:
            raise RuntimeError("qwen-asr returned no transcription")
        return str(results[0].text)


class CohereRunner:
    def __init__(self):
        import torch
        from transformers import pipeline
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=MODEL_IDS["cohere"],
            trust_remote_code=True,
            device=0 if torch.cuda.is_available() else -1,
        )

    def __call__(self, audio, max_new_tokens):
        # Cohere uses its own processor/generation settings. Do not force Qwen's decoding API.
        result = self.pipe({"raw": audio, "sampling_rate": 16000})
        if isinstance(result, dict):
            return str(result["text"])
        raise RuntimeError(f"Unexpected Cohere pipeline output type: {type(result)}")


class WhisperRunner:
    def __init__(self):
        import torch
        from transformers import pipeline
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=MODEL_IDS["whisper"],
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=dtype,
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


def model_id_for(name, args):
    if name != "qwen":
        return MODEL_IDS[name]
    local_path = Path(args.qwen_model)
    return str(local_path.resolve()) if local_path.exists() else args.qwen_model


def model_identity_for(name, args):
    model_id = model_id_for(name, args)
    if name == "qwen" and args.qwen_adapter:
        adapter = Path(args.qwen_adapter)
        adapter_id = str(adapter.resolve()) if adapter.exists() else args.qwen_adapter
        return f"{model_id} + LoRA:{adapter_id}"
    return model_id


def make_runner(name, args):
    if name == "qwen":
        model_id = model_id_for(name, args)
        if args.qwen_backend == "qwen-asr":
            return QwenFineTunedRunner(model_id, args.max_new_tokens, args.qwen_adapter or None)
        return QwenTransformersRunner(model_id)
    if name == "cohere":
        return CohereRunner()
    if name == "higgs":
        return HiggsRunner()
    if name == "whisper":
        return WhisperRunner()
    raise ValueError(name)


def count_errors(ref, hyp):
    """Word-level error decomposition matching standard corpus WER."""
    import jiwer
    p = jiwer.process_words(ref, hyp)
    return p.substitutions, p.deletions, p.insertions, p.hits


def aggregate(rows):
    import jiwer
    if not rows:
        return {"n": 0, "wer": None, "cer": None, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0}
    refs = [r["ref_norm"] for r in rows]
    hyps = [r["hyp_norm"] for r in rows]
    w = jiwer.process_words(refs, hyps)
    return {
        "n": len(rows), "wer": round(100 * w.wer, 4),
        "cer": round(100 * jiwer.cer(refs, hyps), 4),
        "substitutions": w.substitutions, "deletions": w.deletions,
        "insertions": w.insertions, "ref_words": w.hits + w.substitutions + w.deletions,
    }


def summarize(selected, pred_by_path, normalizer):
    valid, empty_normalized = [], 0
    for row in selected:
        p = pred_by_path.get(row["audio_path"])
        if p is None:
            continue
        ref_norm = normalizer(row["text"])
        hyp_norm = normalizer(p["hyp"])
        if not ref_norm.strip():
            empty_normalized += 1
            continue
        valid.append({**row, "hyp": p["hyp"], "ref_norm": ref_norm, "hyp_norm": hyp_norm})
    by_category = defaultdict(list)
    by_speaker = defaultdict(list)
    for row in valid:
        by_category[row.get("category", "MISSING")].append(row)
        by_speaker[row["speaker_id"]].append(row)
    category_scores = {c: aggregate(v) for c, v in sorted(by_category.items())}
    speaker_scores = {s: aggregate(v) for s, v in sorted(by_speaker.items())}
    speaker_wers = [x["wer"] for x in speaker_scores.values() if x["wer"] is not None]
    return {
        "overall": aggregate(valid),
        "by_category": category_scores, "per_speaker": speaker_scores,
        "mean_speaker_wer": round(statistics.mean(speaker_wers), 4) if speaker_wers else None,
        "median_speaker_wer": round(statistics.median(speaker_wers), 4) if speaker_wers else None,
        "n_empty_normalized_references": empty_normalized,
        "n_selected": len(selected), "n_predicted": sum(r["audio_path"] in pred_by_path for r in selected),
        "n_unpredicted": sum(r["audio_path"] not in pred_by_path for r in selected),
    }, valid


def run_one(name, rows, args, normalizer):
    import torch
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    pred_file = out / "predictions.jsonl"
    err_file = out / "errors.jsonl"
    config_file = out / "run_config.json"
    selected = select_rows(rows, args.max_utts, args.seed)
    model_id = model_identity_for(name, args)
    signature = {
        "model_name": name, "model_id": model_id,
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "max_duration": args.max_duration,
        "normalizer": "whisper_normalizer.english.EnglishTextNormalizer",
    }
    if name == "qwen":
        signature["qwen_backend"] = args.qwen_backend
    if config_file.exists():
        previous = json.loads(config_file.read_text(encoding="utf-8"))
        if previous != signature:
            raise RuntimeError(f"Existing run config mismatch in {out}. Choose a new --out (or restore original settings).")
    else:
        config_file.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    predictions = {}
    if pred_file.exists():
        for p in read_jsonl(pred_file):
            if p.get("model_id") != model_id:
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
                print(f"[{name}] loading model {model_id}...", flush=True)
                try:
                    runner = make_runner(name, args)
                except Exception as load_exc:
                    append_jsonl(err_file, {"model_id": model_id, "audio_path": row["audio_path"],
                                            "error_type": "model_load_failed", "error": str(load_exc), "time": time.time()})
                    raise ModelLoadError(f"Failed to initialize {name}: {load_exc}") from load_exc
            hypothesis = runner(audio, args.max_new_tokens)
            if not isinstance(hypothesis, str):
                raise TypeError(f"model returned {type(hypothesis)}")
            prediction = {"model_id": model_id, "audio_path": row["audio_path"], "hyp": hypothesis,
                          "speaker_id": row["speaker_id"], "category": row.get("category", "MISSING")}
            append_jsonl(pred_file, prediction)
            predictions[row["audio_path"]] = prediction
        except Exception as exc:
            typ = type(exc).__name__
            failures[typ] += 1
            append_jsonl(err_file, {"model_id": model_id, "audio_path": row["audio_path"],
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
        "model_name": name, "model_id": model_id,
        "selected_manifest": str(args.manifest.resolve()), "max_utts": args.max_utts,
        "seed": args.seed, "max_duration": args.max_duration,
        "failures_this_run": dict(failures),
        "warning": "Corpus metrics use only successful predictions; inspect n_unpredicted and errors before comparing models.",
    })
    (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "scored_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps({k: r[k] for k in ("audio_path", "speaker_id", "category", "text", "hyp", "ref_norm", "hyp_norm")}, ensure_ascii=False) + "\n")
    with (out / "category_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "n", "wer_percent", "cer_percent", "substitutions", "deletions", "insertions", "ref_words"])
        for k, r in results["by_category"].items():
            writer.writerow([k, r["n"], r["wer"], r["cer"], r["substitutions"], r["deletions"], r["insertions"], r["ref_words"]])
    print(f"[{name}] scored={results['overall']['n']}/{results['n_selected']} "
          f"WER={results['overall']['wer']}% CER={results['overall']['cer']}% | {out}", flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/pd_prepared/val.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/baselines_val"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["qwen", "cohere", "higgs", "whisper"])
    parser.add_argument(
        "--qwen-model", default=MODEL_IDS["qwen"],
        help="Qwen Hugging Face model ID or local fine-tuned checkpoint directory",
    )
    parser.add_argument(
        "--qwen-backend", choices=("transformers", "qwen-asr"), default="transformers",
        help="Use 'qwen-asr' for checkpoints produced by Qwen's official fine-tuning script",
    )
    parser.add_argument(
        "--qwen-adapter", default="",
        help="optional PEFT/LoRA adapter checkpoint; requires --qwen-backend qwen-asr",
    )
    parser.add_argument("--max-utts", type=int, default=20, help="Smoke test count; 0 means entire manifest")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Longer clips are logged as skipped, not truncated")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error(f"Manifest missing: {args.manifest}")
    if args.max_utts < 0 or args.max_duration <= 0 or args.max_new_tokens <= 0:
        parser.error("Invalid max-utts, max-duration, or max-new-tokens")
    if args.qwen_adapter and args.qwen_backend != "qwen-asr":
        parser.error("--qwen-adapter requires --qwen-backend qwen-asr")
    from whisper_normalizer.english import EnglishTextNormalizer
    norm = EnglishTextNormalizer()
    rows = list(read_jsonl(args.manifest))
    if not rows:
        parser.error("Manifest is empty")
    paths = [r["audio_path"] for r in rows]
    if len(paths) != len(set(paths)):
        parser.error("Duplicate audio_path in manifest")
    for row in rows:
        if not all(k in row for k in ("audio_path", "speaker_id", "text")):
            parser.error("Manifest missing required fields: audio_path, speaker_id, text")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Manifest: {args.manifest} ({len(rows)} rows) | models: {', '.join(args.models)}", flush=True)
    outputs = {}
    for name in dict.fromkeys(args.models):
        try:
            outputs[name] = run_one(name, rows, args, norm)
        except Exception as e:
            print(f"[{name}] FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            outputs[name] = {"fatal_error": str(e)}
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    with (args.out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scored", "selected", "wer_percent", "cer_percent", "unpredicted", "error"])
        for name, r in outputs.items():
            overall = r.get("overall", {})
            writer.writerow([name, overall.get("n", 0), r.get("n_selected", ""), overall.get("wer"), overall.get("cer"), r.get("n_unpredicted", ""), r.get("fatal_error", "")])
    print(f"\nWrote comparison: {args.out / 'comparison.csv'}", flush=True)
    if any("fatal_error" in r or r.get("n_unpredicted", 0) for r in outputs.values()):
        print("WARNING: Some models did not finish. Do not compare partial-set WER as equivalent.", flush=True)


if __name__ == "__main__":
    main()
