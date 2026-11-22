# Fine-tuning guide: Qwen3-ASR on Parkinson's speech (shared pipeline)

This document is the single reference for the fine-tuning experiments. It explains the
design, what each script does, how to run one experiment or the whole grid, where the
results go, and the known caveats.

> **Privacy.** Manifests, weight CSVs, predictions and adapters contain speaker IDs, audio
> paths and transcripts of restricted data. `data/`, `runs/`, `checkpoints/` and the weight
> CSVs are in `.gitignore`. Never commit them or upload them to external services.

---

## 1. The experiment

Goal: improve ASR for **unseen** Parkinson's speakers, especially those with high baseline WER.
Two factors are varied; everything else is identical across runs.

### Factor 1: how training recordings are sampled (`--sampling`)

All strategies use the same 221 training speakers (~91k recordings). The probability that a
recording is drawn differs:

| Flag | Name | Per-speaker probability | In the 2×2? |
|---|---|---|---|
| `none` | Normal training | Trainer's usual shuffled epochs (no weighting) | control / smoke tests |
| `e1` | Uniform over recordings | proportional to the speaker's recording count (natural distribution) | no |
| `e2` | Uniform over speakers | 1/221 for every speaker | no |
| **`e3`** | **Hard-speaker oversampling** | hardest 25% of speakers (56) share **50%** of all draws; the other 165 share the other 50%. Speakers are equal *within* each group | **yes** |
| **`e4`** | **Difficulty-weighted** | proportional to `clip(speaker zero-shot WER, median WER, 30%)` | **yes** |

"Hard" = highest zero-shot WER of Qwen3-ASR-1.7B on the training speakers. Every speaker stays
in the pool in every strategy. A speaker's probability is split evenly across that speaker's
recordings.

E4 puts ~48.5% of draws on the same 56 hard speakers E3 targets, so E3 vs E4 compares
*a hard cut-off* with *a continuous gradient* at roughly the same amount of oversampling. The floor (median WER)
keeps easy speakers in the mix; the 30% cap stops the 2 worst speakers (~84% and ~82% WER) from dominating.
Change them with `make_sampling_weights.py --e4-floor/--e4-cap`.

### Factor 2: which weights get LoRA adapters (`--lora`)

| Flag | Name | Adapted modules |
|---|---|---|
| `L1` | Decoder-only | attention projections (`q/k/v/o_proj`) of the text decoder; audio encoder frozen |
| `L2` | Audio + decoder | **everything in L1** plus attention projections (`q/k/v/out_proj`) of the audio encoder |

L2 is a strict superset of L1 so the *only* difference is the audio-side adapters.
Rank 16, alpha 32, dropout 0.05 for both (constants in `lora_configs.py`).

### The grid

| Run folder | Sampling | LoRA |
|---|---|---|
| `E3_L1` | hard-speaker 50/50 | decoder-only |
| `E3_L2` | hard-speaker 50/50 | audio + decoder |
| `E4_L1` | difficulty-weighted | decoder-only |
| `E4_L2` | difficulty-weighted | audio + decoder |

Research questions: does hard-speaker oversampling help difficult unseen speakers; does
audio-side adaptation beat decoder-only; does combining both add benefit.

---

## 2. Repository map

| File | Purpose |
|---|---|
| `prepare_pd_dataset.py` | (data) cleans SAP transcripts and writes speaker-disjoint `train/val/test.jsonl` |
| `prepare_qwen_finetuning.py` | converts those manifests to the trainer format (`audio`, `text`, **`speaker_id`**, `category`); drops clips over `--max-duration` |
| `make_sampling_weights.py` | per-speaker WER → probability columns for E1–E4 (`weights CSV`) |
| `hard_speaker_sampler.py` | reads the weights CSV and builds the PyTorch `WeightedRandomSampler` |
| `lora_configs.py` | L1/L2 definitions: picks target modules by name, applies LoRA, checks what is trainable |
| `train_qwen_lora.py` | the one shared trainer (4-bit base + LoRA, `--sampling`, `--lora-config`) |
| `run_asr_baselines.py` | the evaluator used for **both** zero-shot and fine-tuned models (unchanged scoring) |
| `compare_runs.py` | final tables: overall / hard / other WER, per-speaker improvement, GPU peak |
| `run_experiment.py` | **master runner**: launches any stage of any experiment from one command |
| `configs/shared.json` | the single place for every setting that must be equal across runs |
| `rescore_asr_predictions.py` | optional re-scoring of saved predictions under a documented text policy |
| `tests/` | unit tests (no GPU / data needed, except one that needs `torch`) |

Older/other scripts: `baseline_eval.py` (original zero-shot script, **not** used for the
experiments; it uses a different cleaning and batch settings), `filter_parkinsons.py`,
`prepare_dataset.py`, `data-inspection/`.

---

## 3. Running the pipeline

All commands are run from the repo root. Examples are single-line so they work in both
PowerShell and bash.

### 3.0 Environment (training machine)

```
python -m venv .venv-qwen-lora
python -m pip install -r requirements-finetune.txt
```
Install a **CUDA** build of PyTorch first (see the README: a CPU-only torch makes the trainer stop with
"CUDA GPU is required"). Use a dedicated environment: `qwen-asr` pins a different `transformers`
version than the native `-hf` baseline.

### 3.1 Prepare data (once)

```
python prepare_qwen_finetuning.py --train data/pd_prepared/train.jsonl --val data/pd_prepared/val.jsonl --out data/qwen_finetuning --pilot-size 8000 --max-duration 30 --verify-audio
```
Re-run this if your manifests were made before `speaker_id` was added to the output
(weighted samplers need it; category-first and legacy category-stratified sampling
also need `category`). Keep `--max-duration 30`: it must match the evaluator's
`eval.max_duration` so training and evaluation drop the same long clips.

### 3.2 Build the sampling weights (once, seconds, CPU only)

```
python make_sampling_weights.py --in training_sampling_weights.csv --out data/qwen_finetuning/training_sampling_weights_v2.csv --verify
```
* **Input** needs `speaker_id`, `difficulty_wer_percent`, `n_manifest` (per-speaker zero-shot WER on the
  *training* speakers and recordings per speaker).
* **`--verify`** recomputes E1–E3 and the hard flags and checks they equal the columns already in the input file.
* The output holds all four strategies; the trainer picks one column set via `--sampling`.
* The trainer recomputes per-recording probabilities from the *actual* (duration-filtered) training manifest, so the
  recording counts in the CSV may be larger than in the manifest (never smaller).

### 3.3 Smoke test (do this first on the GPU machine)

```
python run_experiment.py --sampling e3 --lora L1 --set training.max_steps=2 --set training.save_steps=1 --runs-dir runs_smoke
python run_experiment.py --sampling e3 --lora L2 --set training.max_steps=2 --set training.save_steps=1 --runs-dir runs_smoke
```
The trainer also runs a **gradient-flow check** on the first backward pass (`gradient_check.json`): it stops the run if
the adapters that should train (decoder; plus audio for L2) receive no gradient. This protects L2 from silently
training only the decoder because of gradient checkpointing. (`--skip-grad-check` disables it for debugging only.)

Check in the output: the trainable-parameter line (`[L1] … decoder + 0 audio`, `[L2] … decoder + N audio`),
`runs_smoke/*/gpu_usage.json` (peak memory; confirms L2 fits), and that evaluation produces a `scored_predictions.jsonl`.
The first real run is also where the LoRA module names are matched against the actual model: if the pattern finds no
modules the trainer stops with a clear error.

### 3.4 Zero-shot baseline (once per split)

```
python run_experiment.py --stages zero-shot
```
Uses the same evaluator and settings as the fine-tuned runs, through the `qwen-asr` backend with no adapter.
This is the baseline that `compare_runs.py` uses, and it defines which validation speakers count as "hard".

### 3.5 Run experiments

One experiment (each team member takes one):
```
python run_experiment.py --sampling e4 --lora L1
```
The whole 2×2, then the final table:
```
python run_experiment.py --grid --stages zero-shot train eval compare
```
See what would run without running it: add `--dry-run`.

For category-first E3, opt in on a single E3 run by adding
`--category-config configs/category_sampling.json`. This fixes the category mix
at 50/30/20 and splits each category's mass equally between hard and non-hard
speakers. See [the category-first E3 guide](docs/category-first-e3.md) for the
probability definition, validation commands, and limitations. The default grid
continues to use original E3 and E4 unless this option is passed.

Useful flags:

| Flag | Meaning |
|---|---|
| `--stages` | any of `zero-shot train eval compare` (default `train eval`) |
| `--sampling` / `--lora` | one or more values; every combination runs (`--sampling e3 e4 --lora L1 L2`) |
| `--grid` | shorthand for the project's 2×2 (`e3 e4` × `L1 L2`) |
| `--set section.key=value` | override a shared setting for this launch (e.g. `training.learning_rate=5e-5`) |
| `--checkpoint checkpoint-600` | evaluate a specific checkpoint instead of `adapter-final` |
| `--resume-latest` | resume an interrupted training run |
| `--force` | retrain even if `adapter-final` exists (otherwise it is skipped) |
| `--keep-going` | in a grid, continue after a failed experiment |
| `--split val\|test` | evaluation split; `test` needs `--final` and cannot be combined with training |

> If you change a setting with `--set`, **every team member must use the same value** or the
> runs are not comparable. Each run folder saves the exact settings as `config_used.json`; diff them to check.
> Better: edit `configs/shared.json` once, commit it, and don't use `--set` for real runs.

### 3.6 Compare

```
python compare_runs.py --runs-dir runs --split val
```
(also run as the `compare` stage). Final test-set numbers, once, after the models are chosen on validation:
```
python run_experiment.py --stages zero-shot --split test --final
python run_experiment.py --sampling e3 e4 --lora L1 L2 --stages eval --split test --final
python run_experiment.py --stages compare --split test --final
```

---

## 4. Output layout (identical for every run)

```
runs/
  zero_shot/eval_val/qwen/          predictions, scored_predictions.jsonl, summary.json ...
  E3_L1/
    adapter-final/  checkpoint-*/   LoRA weights only (the base model is never saved)
    training_config.json            arguments, GPU, package versions, LoRA/sampling info
    gpu_usage.json                  peak GPU memory (allocated / reserved)
    gradient_check.json             proof that the adapters received gradient on step 1
    log_history.json                loss / learning-rate curve
    config_used.json                the shared config this run used
    run_meta.json                   git commit, every stage's command / exit code / duration
    logs/train.log  logs/eval_val.log
    eval_val/qwen/                  predictions + scores for this model
  comparison_val/
    comparison_val.md               the headline table
    comparison_summary_val.csv      one row per model
    comparison_per_speaker_val.csv  one row per speaker, with change vs zero-shot
    comparison_val.json             everything, machine-readable
```

---

## 5. How evaluation and comparison work

**Validation is scored exactly as the baseline was**: the same `run_asr_baselines.py`, the same manifest
(`val.jsonl`), the same Whisper `EnglishTextNormalizer` on reference and hypothesis, corpus-level WER/CER
(total errors ÷ total reference words), greedy decoding, same `max_new_tokens` and `max_duration`.
`compare_runs.py` only *re-reads* the saved `scored_predictions.jsonl` files; it never re-runs a model.

What it adds:

* **Common utterances.** Runs are compared on the utterances every run scored, and the number dropped is printed.
* **Hard vs other speakers.** The hardest 25% of evaluation speakers by **zero-shot** WER (`ceil(0.25 × speakers)`;
  7 of the 25 validation speakers; ties broken by speaker ID) are "hard". Because the split comes from the zero-shot run, the same
  speakers are hard for all models. With so few speakers, the hard-speaker number is noisy: read it together with the per-speaker table.
* **Per-speaker improvement** = fine-tuned WER − zero-shot WER in percentage points (negative is better), with
  the count of speakers that improved and the median change.
* **Peak GPU memory** of each training run.

---

## 6. Design notes and caveats

* **One backend.** Training and fine-tuned evaluation use the `qwen-asr` package
  (`Qwen/Qwen3-ASR-1.7B`). The zero-shot baseline must also use `--qwen-backend qwen-asr` (the default in
  `configs/shared.json`) so that zero-shot and fine-tuned numbers come from the same inference code. Older
  zero-shot numbers made with the `transformers` `-hf` backend may differ slightly.
* **4-bit training, bf16 evaluation.** Training uses a 4-bit (NF4) base for memory; the evaluator merges the
  adapter into a full-precision base. This is standard QLoRA practice, but the adapter is merged into slightly different
  weights than it was trained on. To remove the effect set `training.load_in_4bit=0` if the GPU has the memory.
* **Sampling semantics.** Weighted strategies draw with replacement; one epoch = as many draws as there are
  training recordings, so every strategy sees the same number of samples per step budget. The random generator is
  seeded once and continues across epochs (different but reproducible draws). Speakers with few recordings are repeated
  many times (a 44-recording hard speaker is drawn ~18× more often per recording than under uniform sampling), which can
  encourage memorization.
* **Count mismatch is handled.** Dropping long clips makes the manifest smaller than the one the CSV was made from. The
  trainer keeps each speaker's probability and re-splits it over the recordings that exist, so E3 still gives hard
  speakers exactly 50%. A manifest with *more* recordings than the CSV is rejected.
* **LoRA module names are discovered, not hard-coded.** `lora_configs.select_targets` matches names like
  `…self_attn.q_proj` (decoder) and `…audio_tower…self_attn.q_proj` (audio). If the real Qwen3-ASR module names differ,
  the trainer fails immediately with "No decoder attention projections found". This has not yet been run on the real model.
* **No in-training validation** (saves memory). Pick checkpoints by evaluating a few on validation
  (`--checkpoint checkpoint-N`), never on test.
* **`--max-duration`**: Monica's earlier README examples use 120 s; the shared config uses 30 s to match the training
  manifest preparation. Use one value for all runs.

---

## 7. Tests

```
PYTHONPATH=. python -m unittest discover -s tests          # bash
$env:PYTHONPATH="."; python -m unittest discover -s tests  # PowerShell
```
Covers LoRA target selection, sampler and weight-CSV validation, weight computation (including a check that E1–E3 reproduce the original
CSV logic), the comparison metrics (hard/other split, per-speaker deltas, intersection), and the runner's commands
(including that the runner only changes `--sampling`, `--lora-config` and the output folder between experiments, and that every
flag it passes is accepted by the real scripts). `test_sampled_share` needs `torch`; it is skipped-by-error on machines without it.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CUDA GPU is required` | CPU-only torch; install the CUDA wheel (README) |
| `No decoder attention projections found` | module names differ from the expected pattern; print `model.named_modules()` and adjust `lora_configs.py` |
| `Recording count exceeds CSV` | the training manifest has more rows than the weights CSV; rebuild the CSV from the current manifest |
| `Gradient-flow check FAILED` | audio (L2) or decoder adapters get no gradient; try `--set training.gradient_checkpointing=0` (more memory) |
| `Weights CSV is missing required columns` | run `make_sampling_weights.py` to add the E1–E4 columns |
| `weighted sampling needs a speaker_id` | re-run `prepare_qwen_finetuning.py` (older manifests have no `speaker_id`) |
| `--split test requires --final` | the test set is only for the one final evaluation |
| `Existing run config mismatch` (evaluator) | a run folder was reused with different settings; choose a new `--out` / runs dir |
| compare says references differ | runs used different manifests or normalizers; re-evaluate with the same command |
