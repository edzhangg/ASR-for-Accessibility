#!/usr/bin/env python3
"""Zero-shot ASR baseline on the Parkinson's manifest (made by filter_parkinsons.py).

  python baseline_eval.py --manifest /data/SAP_PD/manifest_pd.jsonl --split Test --max-utts 20 --out runs/smoke

Writes into --out: predictions.jsonl (incremental, rerun resumes), errors.jsonl, summary.json.
Scoring is done from predictions.jsonl, so changing the text cleaning never needs a re-run of the model.
"""
import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# model id -> how to call it (taken from each Hugging Face model card)
MODELS = {
    "Qwen/Qwen3-ASR-1.7B-hf": {"cast_to_model_dtype": True, "min_vram_gb": 0},
    "CohereLabs/cohere-transcribe-03-2026f": {"cast_to_model_dtype": True, "min_vram_gb": 0},
    "bosonai/higgs-audio-v3-stt": {"cast_to_model_dtype": True, "min_vram_gb": 0},
    # 34B params in BF16 -> ~68 GB of weights; the card's deployment baseline is one H100 80 GB
    # "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B": {"cast_to_model_dtype": False, "min_vram_gb": 75},
}


# ----------------------------------------------------------------------------- text cleaning
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


# ----------------------------------------------------------------------------- audio
def load_audio(path, target_sr=16000):
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # mono
    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    return wav


# ----------------------------------------------------------------------------- models
def load_model(model_id, force=False):
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. This script is meant to run on an NVIDIA GPU.")
    vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9
    need = MODELS[model_id]["min_vram_gb"]
    if vram < need and not force:
        sys.exit(f"{model_id} needs ~{need} GB of GPU memory; found {vram:.0f} GB "
                 f"({torch.cuda.get_device_name(0)}). Use a bigger GPU (or --force to try anyway).")
    print(f"transformers {transformers.__version__}, GPU: {torch.cuda.get_device_name(0)} ({vram:.0f} GB)")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return processor, model


def transcribe(model_id, processor, model, audios, max_new_tokens):
    """audios: list of 16 kHz mono float32 arrays -> list of strings."""
    import torch
    if model_id.startswith("Qwen/"):
        inputs = processor.apply_transcription_request(audios, language=["English"] * len(audios))
        inputs = inputs.to(model.device, model.dtype)
    else:  # Nemotron
        inputs = processor.apply_transcription_request(audios).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    if model_id.startswith("Qwen/"):
        return processor.decode(gen, return_format="transcription_only")
    return processor.decode(gen, skip_special_tokens=True)


# ----------------------------------------------------------------------------- scoring
def score(refs, hyps):
    """Corpus-level WER and CER (in %). Utterances whose reference is empty are skipped."""
    import jiwer
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return {"wer": None, "cer": None, "n": 0}
    r, h = zip(*pairs)
    return {"wer": 100 * jiwer.wer(list(r), list(h)), "cer": 100 * jiwer.cer(list(r), list(h)), "n": len(pairs)}


def summarize(preds):
    from whisper_normalizer.english import EnglishTextNormalizer
    norm = EnglishTextNormalizer()
    versions = {"clean": reference_clean, "verbatim": reference_verbatim}
    for p in preds:
        p["hyp_norm"] = norm(p["hyp"])
        for name, fn in versions.items():
            p[f"ref_{name}"] = norm(fn(p["ref_raw"]))

    result = {}
    for name in versions:
        refs = lambda rows: [r[f"ref_{name}"] for r in rows]
        hyps = lambda rows: [r["hyp_norm"] for r in rows]
        res = {"overall": score(refs(preds), hyps(preds)), "by_category": {}, "per_speaker": {}}
        for key, field in (("by_category", "category"), ("per_speaker", "speaker_id")):
            groups = defaultdict(list)
            for p in preds:
                groups[p[field]].append(p)
            for g, rows in sorted(groups.items()):
                res[key][g] = score(refs(rows), hyps(rows))
        wers = [v["wer"] for v in res["per_speaker"].values() if v["wer"] is not None]
        if wers:
            res["speaker_wer_stats"] = {"median": statistics.median(wers), "best": min(wers), "worst": max(wers),
                                        "n_speakers": len(wers)}
        result[name] = res
    return result


def print_table(result):
    f = lambda x: "   -  " if x is None else f"{x:6.1f}"
    for name, res in result.items():
        print(f"\n=== reference: {name} ===")
        print(f"{'':<34}{'n':>6}{'WER%':>8}{'CER%':>8}")
        o = res["overall"]
        print(f"{'OVERALL':<34}{o['n']:>6}{f(o['wer']):>8}{f(o['cer']):>8}")
        for cat, v in res["by_category"].items():
            print(f"  {cat:<32}{v['n']:>6}{f(v['wer']):>8}{f(v['cer']):>8}")
        s = res.get("speaker_wer_stats")
        if s:
            print(f"  per-speaker WER ({s['n_speakers']} speakers): median {s['median']:.1f}  "
                  f"best {s['best']:.1f}  worst {s['worst']:.1f}")


# ----------------------------------------------------------------------------- main
def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="manifest_pd.jsonl from filter_parkinsons.py")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B-hf", choices=list(MODELS))
    ap.add_argument("--split", default="Test",
                    help="Train/Dev/Test, or 'all'. Report headline numbers on Test only so every model "
                         "(zero-shot or fine-tuned) is scored on the same held-out speakers.")
    ap.add_argument("--max-utts", type=int, default=None, help="random subsample (fixed --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-duration", type=float, default=120.0, help="skip clips longer than this (s)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--force", action="store_true", help="skip the GPU-memory check")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # pick utterances (sorted first so the sample only depends on the seed)
    rows = [r for r in read_jsonl(Path(args.manifest))
            if args.split.lower() == "all" or r["split"].lower() == args.split.lower()]
    rows.sort(key=lambda r: r["audio_path"])
    if args.max_utts and args.max_utts < len(rows):
        rows = random.Random(args.seed).sample(rows, args.max_utts)
    print(f"{len(rows)} utterances from split '{args.split}'")

    pred_path, err_path = out / "predictions.jsonl", out / "errors.jsonl"
    done = {p["audio_path"] for p in read_jsonl(pred_path)}
    skipped = {e["audio_path"] for e in read_jsonl(err_path)}
    todo = [r for r in rows if r["audio_path"] not in done and r["audio_path"] not in skipped]
    print(f"{len(done)} already done, {len(todo)} to run")

    def log_error(row, why):
        print(f"  SKIP {Path(row['audio_path']).name}: {why}")
        with open(err_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"audio_path": row["audio_path"], "error": why}) + "\n")

    if todo:
        processor, model = load_model(args.model, args.force)
        todo.sort(key=lambda r: r["duration_s"])  # similar lengths per batch
        with open(pred_path, "a", encoding="utf-8") as pf:
            for i in range(0, len(todo), args.batch_size):
                batch, audios = [], []
                for r in todo[i:i + args.batch_size]:
                    if r["duration_s"] > args.max_duration:
                        log_error(r, f"too long ({r['duration_s']:.0f}s)")
                        continue
                    try:
                        audios.append(load_audio(r["audio_path"]))
                        batch.append(r)
                    except Exception as e:  # noqa: BLE001 - log and continue
                        log_error(r, f"audio load failed: {e}")
                if not batch:
                    continue
                try:
                    hyps = transcribe(args.model, processor, model, audios, args.max_new_tokens)
                except Exception as e:  # noqa: BLE001 - retry one by one (e.g. OOM on a big batch)
                    import torch
                    torch.cuda.empty_cache()
                    hyps = []
                    for r, a in zip(batch, audios):
                        try:
                            hyps.append(transcribe(args.model, processor, model, [a], args.max_new_tokens)[0])
                        except Exception as e2:  # noqa: BLE001
                            log_error(r, f"inference failed: {e2}")
                            hyps.append(None)
                for r, h in zip(batch, hyps):
                    if h is None:
                        continue
                    pf.write(json.dumps({"audio_path": r["audio_path"], "speaker_id": r["speaker_id"],
                                         "category": r["category"], "duration_s": r["duration_s"],
                                         "ref_raw": r["transcript"], "hyp": h}, ensure_ascii=False) + "\n")
                pf.flush()
                print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}")

    # score every selected utterance that has a prediction
    wanted = {r["audio_path"] for r in rows}
    preds = [p for p in read_jsonl(pred_path) if p["audio_path"] in wanted]
    if not preds:
        sys.exit("No predictions to score.")
    result = summarize(preds)
    print_table(result)

    try:
        import torch
        import transformers
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        versions = {"transformers": transformers.__version__, "torch": torch.__version__}
    except ImportError:
        gpu, versions = None, {}
    summary = {"model": args.model, "split": args.split, "seed": args.seed, "max_utts": args.max_utts,
               "batch_size": args.batch_size, "n_scored": len(preds),
               "n_skipped": len(read_jsonl(err_path)), "gpu": gpu, **versions,
               "runtime_s_this_invocation": round(time.time() - t0, 1), "results": result}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out / 'summary.json'}  (skipped/failed: {summary['n_skipped']}, see errors.jsonl)")

    print("\nExample reference/prediction pairs (clean ref):")
    for p in preds[:5]:
        print(f"  REF: {p['ref_clean']}\n  HYP: {p['hyp_norm']}\n")


if __name__ == "__main__":
    main()
