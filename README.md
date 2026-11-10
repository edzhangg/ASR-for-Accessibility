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

## 2. Baseline evaluation (`run_asr_baselines.py`)
```bash
# smoke test: 20 category-balanced Test utterances, one model
python run_asr_baselines.py --manifest /path/to/SAP_PD/manifest_pd.jsonl --split Test --max-utts 20 --models qwen --out runs/smoke

# headline result: the full Test split (the default) for several models
python run_asr_baselines.py --manifest /path/to/SAP_PD/manifest_pd.jsonl --models qwen whisper cohere higgs --out runs/baselines_test

# optional, descriptive only (not comparable to fine-tuned models): all speakers
python run_asr_baselines.py --manifest /path/to/SAP_PD/manifest_pd.jsonl --split all --models qwen --out runs/qwen_all
```
**Report headline numbers on `--split Test` only.** Zero-shot and fine-tuned models must be scored on the same
held-out speakers; `--split all` includes the speakers a fine-tuned model trains on. Every model uses the same
`--max-duration` (120 s), so models are scored on the same clips; check `n_unpredicted` before comparing.

Each model gets `--out/<model>/` with `predictions.jsonl` (rerun resumes), `errors.jsonl`, `summary.json`
(clean + verbatim WER/CER overall, per category, per speaker with median/best/worst, plus GPU, torch/transformers
versions and runtime), `scored_predictions.jsonl` and `category_scores.csv`. `--out/comparison.csv` lines the models up.
It exits if there is no CUDA GPU (override with `--allow-cpu`).

Models (`--models`): `qwen` = `Qwen/Qwen3-ASR-1.7B-hf` (transformers >= 5.13), `whisper` = `openai/whisper-small`,
`cohere` = `CohereLabs/cohere-transcribe-03-2026` (gated: accept its terms on Hugging Face and `huggingface-cli login`;
transformers >= 5.4), `higgs` = `bosonai/higgs-audio-v3-stt` (needs the `boson_multimodal` package and
`trust_remote_code`). A model that fails to load is logged and skipped; the others still run.

`baseline_eval.py` is the earlier Qwen/Nemotron-only script, kept for reference. Nemotron-3.5-Transcribe
(34B parameters, ~80 GB GPU) does not fit on a 12 GB card.

## Scoring conventions
Both references are normalized with Whisper's `EnglishTextNormalizer`, as are the hypotheses.
- Removed from both: `[bracketed notes]` (prompt text, `[PII]`), `{u:}`, `~X` codes, `#dis`. `{g: words}` keeps the words.
- **clean**: `(parenthesized disfluencies)` removed.
- **verbatim**: the words inside `(...)` are kept, e.g. `(s-) show` -> `s- show`.
WER/CER are corpus-level (total errors / total reference words). Utterances whose reference becomes empty are skipped.
