# ASR for Parkinson's speech (SAP dataset)

Two scripts: extract the Parkinson's speakers from the Speech Accessibility Project (SAP) data, then run a
zero-shot ASR baseline on them. The SAP folder is only ever read.

## Setup (NVIDIA GPU machine)
```bash
py -m venv .venv; .\.venv\Scripts\Activate.ps1      # Python 3.10+
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or the CUDA build for your driver
pip install -r requirements.txt
```

## 1. Extract the Parkinson's subset
`--out` must be outside `--src`. Use the same filesystem for `--hardlink`.
```bash
# see counts and GB first, nothing is copied
python filter_parkinsons.py --src /path/to/SAP --out /path/to/SAP_PD --dry-run

# real run (add --hardlink to save disk; safe to re-run, same-size files are skipped)
python filter_parkinsons.py --src /path/to/SAP --out /path/to/SAP_PD
```
Writes to `--out`: the copied speaker folders, `manifest_pd.jsonl`, `speakers_pd.csv`, and, if the dataset has no
Train/Dev/Test folders, `split_pd.json` (speaker-level ~70/15/15 split, `--seed 0`, no speaker in two splits).
Hard-linked files share data with the originals: never edit them in place.

## 2. Baseline evaluation
```bash
# smoke test
python baseline_eval.py --manifest /path/to/SAP_PD/manifest_pd.jsonl --split Test --max-utts 20 --out runs/qwen_smoke

# headline result: full Test split (default)
python baseline_eval.py --manifest /path/to/SAP_PD/manifest_pd.jsonl --batch-size 8 --out runs/qwen_test

# optional, descriptive only (not comparable to fine-tuned models): all speakers
python baseline_eval.py --manifest /path/to/SAP_PD/manifest_pd.jsonl --split all --out runs/qwen_all
```
**Report headline numbers on `--split Test` only.** Zero-shot and fine-tuned models must be scored on the same
held-out speakers; `--split all` includes the speakers a fine-tuned model trains on.
Re-running the same command resumes from `predictions.jsonl`. Outputs: `predictions.jsonl`, `errors.jsonl`
(skipped clips), `summary.json` (WER/CER overall, per category, per speaker, plus GPU / model / transformers version / runtime).

Models: `Qwen/Qwen3-ASR-1.7B-hf` (default, a few GB of VRAM) and
`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` (34B params, needs an ~80 GB GPU; the script refuses on smaller
GPUs unless `--force`). Select with `--model`.

## Scoring conventions
Both references are normalized with Whisper's `EnglishTextNormalizer`, as are the hypotheses.
- Removed from both: `[bracketed notes]` (prompt text, `[PII]`), `{u:}`, `~X` codes, `#dis`. `{g: words}` keeps the words.
- **clean**: `(parenthesized disfluencies)` removed.
- **verbatim**: the words inside `(...)` are kept, e.g. `(s-) show` -> `s- show`.
WER/CER are corpus-level (total errors / total reference words). Utterances whose reference becomes empty are skipped.

## Prepare Qwen fine-tuning manifests

Convert the existing speaker-disjoint Train and Validation manifests to the
official Qwen3-ASR `audio`/`text` format without copying or modifying audio:

```powershell
py .\prepare_qwen_finetuning.py `
  --train .\data\pd_prepared\train.jsonl `
  --val .\data\pd_prepared\val.jsonl `
  --out .\data\qwen_finetuning `
  --pilot-size 8000 `
  --max-duration 30 `
  --verify-audio
```

The command validates speaker and audio separation, writes full
Train/Validation manifests and a deterministic speaker-complete pilot, and
creates `preparation_report.json`. These outputs contain restricted paths and
transcripts and are ignored by Git.

The official Qwen trainer uses `Qwen/Qwen3-ASR-1.7B` and requires `qwen-asr`
and `datasets`. Full-parameter fine-tuning requires substantially more GPU
memory than inference; verify the training method and memory budget before
launching it on a 12 GB card.

### Experimental LoRA/QLoRA pilot for a 12 GB GPU

`train_qwen_lora.py` is a separate, project-owned parameter-efficient trainer.
It follows Qwen's official preprocessing but freezes the base model and trains
small PEFT LoRA adapters. Its defaults use 4-bit base weights, batch size 1,
gradient checkpointing, and a memory-efficient optimizer. This path is
experimental because Qwen does not currently document LoRA in its official ASR
trainer.

Use a dedicated environment: `qwen-asr` currently requires a different
Transformers version from the native `Qwen3-ASR-1.7B-hf` baseline environment.
Python 3.10 or 3.11 is preferable when available.

```powershell
py -m venv .venv-qwen-lora
.\.venv-qwen-lora\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install qwen-asr datasets peft bitsandbytes librosa accelerate
```

First run a two-step smoke test. It verifies that the model, quantization,
collator, backward pass, and adapter checkpoint fit before committing to the
pilot:

```powershell
python .\train_qwen_lora.py `
  --train-file .\data\qwen_finetuning\train_pilot.jsonl `
  --output-dir .\checkpoints\qwen_pd_lora_smoke `
  --max-steps 2 `
  --save-steps 1 `
  --save-total-limit 2
```

If peak VRAM remains safe, run the one-epoch pilot:

```powershell
python .\train_qwen_lora.py `
  --train-file .\data\qwen_finetuning\train_pilot.jsonl `
  --eval-file .\data\qwen_finetuning\val.jsonl `
  --output-dir .\checkpoints\qwen_pd_lora_pilot `
  --epochs 1 `
  --batch-size 1 `
  --grad-acc 8 `
  --save-steps 200
```

In-training evaluation is disabled by default to save time and memory. Measure
WER for selected adapter checkpoints with `run_asr_baselines.py` instead.

Evaluate a checkpoint made by the official trainer with its `qwen-asr` loader:

```powershell
py .\run_asr_baselines.py `
  --manifest .\data\pd_prepared\val.jsonl `
  --models qwen `
  --qwen-model .\checkpoints\qwen_pd_pilot\checkpoint-200 `
  --qwen-backend qwen-asr `
  --max-utts 0 `
  --max-duration 120 `
  --out .\runs\qwen_pd_checkpoint_200_val
```

Evaluate a LoRA adapter checkpoint against the same Validation manifest:

```powershell
py .\run_asr_baselines.py `
  --manifest .\data\pd_prepared\val.jsonl `
  --models qwen `
  --qwen-model Qwen/Qwen3-ASR-1.7B `
  --qwen-adapter .\checkpoints\qwen_pd_lora_pilot\checkpoint-200 `
  --qwen-backend qwen-asr `
  --max-utts 0 `
  --max-duration 120 `
  --out .\runs\qwen_pd_lora_200_val
```
