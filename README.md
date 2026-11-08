# ASR for Parkinson's speech (SAP dataset)

Two scripts: extract the Parkinson's speakers from the Speech Accessibility Project (SAP) data, then run a
zero-shot ASR baseline on them. The SAP folder is only ever read.

## Setup (NVIDIA GPU machine)
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
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
