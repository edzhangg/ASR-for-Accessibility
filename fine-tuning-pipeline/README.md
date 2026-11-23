# PD ASR Fine-Tuning Pipeline — Team Handoff

**Status:** end-to-end **smoke-tested** on November 23, 2026; full-scale training and controlled comparisons are **not yet complete**.

This folder is our shared interface for Parkinson's disease (PD) ASR fine-tuning. It combines **difficulty-aware, category-controlled sampling**, a **shared Hugging Face training loop**, and a **model-specific ASR backend**. The first backend is **Qwen3-ASR-1.7B**; the structure leaves room for another ASR model later.

> **Start here:** If you just pulled this branch, read **[Files and ownership](#2-files-and-ownership)**, **[Quick start](#5-quick-start-windows-powershell)**, and **[Team TODOs](#8-team-todos-and-integration-contract)**. Do not run a full training job until the experiment settings have been agreed upon.

## 1. Big picture

```text
Existing speaker-disjoint PD splits (produced outside this folder)
  data/pd_prepared/train.jsonl, val.jsonl
          │
          ▼
  fine_tune_data_prepare.py
    - resolves relocated audio paths
    - filters training categories / optional duration cutoff
    - creates Qwen-format train, val, pilot JSONLs
          │
          ├───────────────────────────────┐
          ▼                               │
  sampling.py                             │
    + existing TRAIN zero-shot WER CSV    │
    + shared 50/30/20 category policy     │
    → weighted indices (with replacement)│
          │                               │
          └──────────────┬────────────────┘
                         ▼
                      train.py
                shared Hugging Face Trainer
                         │
                         ▼
                    qwen_backend.py
              Qwen load / collator / LoRA
                         │
                         ▼
                private LoRA adapter
                         │
                         ▼
           existing run_asr_baselines.py
          reload adapter; evaluate on PD validation
```

**Key distinction:** `sampling.py` does **not** generate another manifest. It returns a PyTorch `WeightedRandomSampler`, with indices matching the **original order** of the selected training JSONL. Sampling happens dynamically during training. `train.py` owns the DataLoader/Trainer integration and saves the adapter. WER evaluation remains in the team's existing evaluator; it is **not** computed inside `train.py`.

## 2. Files and ownership

| File | Responsibility | Current status / owner |
| --- | --- | --- |
| `fine_tune_data_prepare.py` | Convert existing PD train/val manifests into Qwen-format JSONLs; resolve `audio_relpath` against a relocated SAP root; explicitly filter training categories; validate splits, audio, duration; make a pilot subset. | Implemented; adapted from the team's original Qwen preparation. |
| `sampling.py` | Shared `build_sampler()` interface. Read **TRAIN-speaker** zero-shot WER CSV, calculate per-recording sampling weights, enforce category proportions, return `WeightedRandomSampler`. | **`continuous` implemented**; **`hard_threshold` TODO** for the teammate. |
| `train.py` | Parse experiment settings; read manifest; check train/val speaker separation; construct sampler; load backend; run HF Trainer; check gradients; save adapter and run metadata. | Implemented; both LoRA modes smoke-tested with Qwen. |
| `qwen_backend.py` | Qwen3-ASR model/processor, 4-bit NF4 loading, ASR collator/label masking, decoder/audio-encoder LoRA targets, first-backward gradient audit. | Implemented; adapted from team `train_qwen_lora.py` / `lora_configs.py`, **not an import of those scripts**. |
| `category_probs.json` | Shared target category distribution for **both** sampling strategies. | Current proposed policy: 50% commands, 30% novel, 20% spontaneous; freeze before formal experiments. |
| `README.md` | This handoff, reproducible commands, limitations and integration checklist. | Keep updated alongside code changes. |

**Related code outside this folder (do not delete or silently replace):**

- `prepare_pd_dataset.py`: upstream PD extraction, transcript cleaning, and speaker-disjoint split. **Not performed again** by this folder.
- `run_asr_baselines.py`: team's shared zero-shot/fine-tuned inference and WER/CER evaluator. For LoRA adapters, use `--qwen-backend qwen-asr`.
- Existing baseline artifacts: one **TRAIN-only** Qwen zero-shot speaker WER CSV, containing at least `speaker_id,difficulty_wer_percent`. **Do not rerun expensive baseline inference if the existing result is complete and valid.**
- Original `prepare_qwen_finetuning.py`, `make_sampling_weights.py`, `hard_speaker_sampler.py`, `train_qwen_lora.py`, `lora_configs.py`, `run_experiment.py`, and comparison scripts: useful prior implementations for teammate integration and comparison. This folder is the new interface; it is **not yet wired into the old experiment runner**.

### Backend abstraction

`train.py` has a backend registry (`BACKEND_MODULES`) and calls `create_backend(args)` from the chosen backend module. The backend provides a model, processor, `collate(rows)`, `check_first_gradients(...)`, and `save_processor(...)`. To add a new model, create `<model>_backend.py` and register it; model-specific preprocessing may also need its own preparation adapter. **Only Qwen is currently registered and tested**, so “model-agnostic” describes the design boundary, not verified multi-model compatibility.

## 3. Input/output contracts

### Upstream PD manifest (private)

Expected fields include `audio_path`, `audio_relpath`, `text`, `speaker_id`, and `category`; optional split/duration metadata may also be present. Its paths may reflect the **old** SAP root. The new preparation step can resolve them without modifying upstream manifests or audio.

### Qwen training manifest (private)

Output of `fine_tune_data_prepare.py`:

```text
data/<PRIVATE_PREPARED_DIRECTORY>/
├── train.jsonl              # full retained training subset
├── train_pilot.jsonl        # deterministic smoke-test subset
├── val.jsonl                # Qwen-format validation; overlap check
└── preparation_report.json # counts, filters, split checks
```

Each Qwen row has:

```json
{
  "audio": "<LOCAL_PRIVATE_WAV_PATH>",
  "text": "language English<asr_text><TRANSCRIPT>",
  "speaker_id": "<PRIVATE_ID>",
  "category": "Digital Assistant Commands"
}
```

**Do not commit or share real rows.** The above is only a schema example. Pilot generation preserves at least one recording per retained training speaker when `pilot-size` permits it, but it is **not** a category-balanced sample by itself; the sampler controls category exposure during training.

### Speaker WER CSV (private)

`sampling.py` requires columns `speaker_id` and `difficulty_wer_percent` and ignores unrelated old E1/E2/E3 columns. **WER is stored as a percentage**: `15.0` means 15%, which the loader converts to `0.15`. It must come from **training speakers only**, using the team’s documented zero-shot Qwen baseline, and must cover every speaker in the training manifest. No validation/test WER may be used to assign training sampling weights.

### Model artifacts (private)

A successful run writes, among other HF checkpoint artifacts:

```text
runs/<NEW_RUN_NAME>/
├── training_config.json          # settings + input/code fingerprints + local paths
├── gradient_check.json           # nonzero LoRA B gradients by group
├── gpu_usage.json                # PyTorch allocated/reserved peak memory
├── log_history.json
├── checkpoint-*/                 # depends on save interval
└── adapter-final/               # PEFT adapter and processor
```

**Never commit a `runs/` folder.** Even the JSON metadata may include private paths and references.

## 4. Sampling and LoRA design decisions

### Category balancing is a shared control

Current `category_probs.json`:

| Category (exact string match) | Probability mass |
| --- | ---: |
| Digital Assistant Commands | 50% |
| Novel Sentences | 30% |
| Spontaneous Speech Prompts | 20% |

These percentages are **target proportions of all training draws**, **not** multipliers applied directly to each recording. Other categories must be **explicitly handled at preparation time** (currently excluded from training via `--train-categories`); validation is not category-balanced, and `--train-categories` does **not** filter validation categories. The agreed policy must be identical for Continuous and Hard Threshold comparisons.

### Continuous difficulty sampling — implemented

For recording `i`, with speaker `s(i)` and category `c(i)`:

```text
speaker difficulty:  d_s = TRAIN zero-shot speaker WER, as a fraction
recording weight:   w_i = (epsilon + min(d_s, max_wer)) ** alpha
final probability:  P(i) = q[c(i)] * w_i / sum(w_j for all j in category c(i))
```

Current defaults: `alpha=1.0`, `epsilon=0.02`, `max_wer=0.30`. This is **recording-level weighting**: a speaker with more retained recordings has more aggregate probability mass, all else equal. At `alpha=0`, sampling is uniform **within each category**, while the 50/30/20 category quotas still apply.

The sampler uses `WeightedRandomSampler(replacement=True)` with a seeded generator. An epoch is `len(train_rows)` **draws**, not necessarily `len(train_rows)` **unique** recordings. Exact sampler state restoration after resuming a checkpoint is **not implemented**.

### Hard-threshold sampling — teammate TODO

`sampling.py::_hard_threshold_weights(...)` deliberately raises `NotImplementedError`. The teammate should return **one positive, finite per-recording weight in the original manifest order**, using the same external `build_sampler(...)` API. The shared category-normalization step then enforces 50/30/20.

**Important mathematical pitfall:** Simply assigning hard speakers 50% mass **before** category normalization can change the **final** hard-speaker mass. If the hard-threshold study requires **both** (a) overall 50/30/20 category mass and (b) 50% hard-speaker mass, implement/test the **joint** constraints (e.g., hard/easy × category stratification where every cell is populated, or a documented constrained balancing scheme). Do **not** claim both guarantees solely from the initial speaker weights. Freeze the hard-speaker definition, threshold, and retained speaker set before comparing arms.

### Two LoRA modes — implemented

| CLI flag | Adapted attention modules | Frozen parameters |
| --- | --- | --- |
| `--lora decoder_only` | Text decoder attention only | Base weights and audio encoder |
| `--lora encoder_decoder` | Text decoder **and** audio encoder attention | Base weights |

Both use PEFT LoRA and the same configurable `r`, `lora-alpha`, `lora-dropout`. Qwen backend loads in 4-bit NF4 by default. The module matcher is intentionally restricted to `thinker.model.layers.*.self_attn.*` and `thinker.audio_tower.layers.*.self_attn.*`, rather than matching unrelated attention layers. The existing team's architecture/PEFT ideas were retained and reorganized; this is **not a direct import** of their `lora_configs.py`.

## 5. Quick start (Windows PowerShell)

Run from the **repository root**. The paths below are **examples/placeholders**, not portable paths. Do not check in private path settings.

### A. Preflight

```powershell
# Update for your local Python environment.
$py = ".\.venv-qwen-lora\Scripts\python.exe"
$pipeline = ".\fine-tuning-pipeline"
$trainSource = ".\data\pd_prepared\train.jsonl"
$valSource = ".\data\pd_prepared\val.jsonl"
$audioRoot = "<ACTUAL_LOCAL_SAP_DATASET_ROOT>"  # parent containing Train/ and Dev/
$werCsv = "<EXISTING_PRIVATE_TRAIN_ZERO_SHOT_WER_CSV>"
$categoryPolicy = "$pipeline\category_probs.json"
$prepared = ".\data\qwen_finetuning_category_smoke"

# Confirm files/locations BEFORE running.
@($py, "$pipeline\train.py", "$pipeline\sampling.py",
  "$pipeline\qwen_backend.py", "$pipeline\fine_tune_data_prepare.py",
  $trainSource, $valSource, $audioRoot, $werCsv, $categoryPolicy) |
  ForEach-Object { if (!(Test-Path $_)) { Write-Host "MISSING prerequisite (check locally)" } }

& $py -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Required environment: installed Qwen ASR and HF training dependencies (including compatible `torch`, `transformers`, `qwen-asr`, `peft`, `bitsandbytes`, `accelerate`, `librosa`, `soundfile`), plus CUDA for QLoRA. **Dependency versions are not yet frozen in a single team-owned lockfile**; verify against the successfully tested machine before reinstalling. Don't infer that every `.venv` has the correct packages. For CLI options use `& $py "$pipeline\train.py" --help`.

### B. Prepare the Qwen manifest (one-time, CPU)

```powershell
& $py "$pipeline\fine_tune_data_prepare.py" `
  --train $trainSource `
  --val $valSource `
  --out $prepared `
  --audio-root $audioRoot `
  --max-duration 30 `
  --pilot-size 8000 `
  --verify-audio `
  --train-categories `
    "Digital Assistant Commands" `
    "Novel Sentences" `
    "Spontaneous Speech Prompts"

if ($LASTEXITCODE -ne 0) { throw "Preparation failed" }
Test-Path "$prepared\train_pilot.jsonl"
```

**Duration policy is a decision, not an inherent 30-second model limit.** The **verified short-audio smoke configuration** used `--max-duration 30`. To include longer audio in **training**, create a **new** prepared directory with `--max-duration 120` (or `0` to disable filtering); do not modify/reuse an existing prepared directory. This may increase GPU memory needs. Evaluation has its **own independent** `--max-duration` flag; increasing it does not restore long recordings already filtered from the training manifest. Train and validation cutoff choices should be documented and consistent across experimental arms.

`--audio-root` resolves `audio_relpath` if the SAP folder moved. Do **not** remove `--verify-audio` to hide broken paths. Verify your chosen root contains `Train/` and `Dev/`. Keep the original PD manifests unchanged.

### C. Two-step Continuous + Encoder/Decoder LoRA smoke test (GPU)

```powershell
& $py "$pipeline\train.py" `
  --backend qwen `
  --train-manifest "$prepared\train_pilot.jsonl" `
  --val-manifest "$prepared\val.jsonl" `
  --speaker-wer-csv $werCsv `
  --sampling continuous `
  --category-probs $categoryPolicy `
  --alpha 1.0 --epsilon 0.02 --max-wer 0.30 `
  --lora encoder_decoder `
  --load-in-4bit 1 --gradient-checkpointing 1 `
  --batch-size 1 --grad-acc 1 `
  --max-steps 2 --save-steps 1 --log-steps 1 `
  --output-dir ".\runs\YOUR_NEW_L2_SMOKE_RUN"

if ($LASTEXITCODE -ne 0) { throw "Training smoke test failed" }
```

For a **Decoder-only** smoke test, change just `--lora decoder_only` and use a **different new** `--output-dir`. The Trainer intentionally rejects nonempty output directories. The smoke test trains only **2 optimizer steps**, not an entire pilot epoch, and does not measure validation WER.

### D. Check artifacts without exposing restricted content

```powershell
$run = ".\runs\YOUR_NEW_L2_SMOKE_RUN"
Test-Path "$run\adapter-final\adapter_config.json"
Get-Content "$run\gradient_check.json"
Get-Content "$run\gpu_usage.json"
```

For `encoder_decoder`, require nonzero gradients in **both** decoder and audio groups. A completed step/gradient audit is **not** proof that every adapter tensor changed or that the model improved on validation. Peak allocated/reserved GPU memory does **not** equal total device VRAM use and may be much higher for longer recordings.

### E. Reload adapter and evaluate (separate existing evaluator)

The team's `run_asr_baselines.py` expects an **original PD-format** evaluation JSONL (`audio_path`, **unprefixed** `text`, `speaker_id`, `category`). **Do not pass Qwen-format `val.jsonl` from this folder**: its `text` has the `language English<asr_text>` prefix, which is a training instruction, not a reference transcription.

If source `audio_path` values point to an old SAP location, make a **local private copy** of the original validation JSONL with `audio_path = current_audio_root / audio_relpath`, preserving all other fields. Use that fixed original validation manifest for evaluation; do not change the held-out split.

```powershell
$evalManifest = "<PRIVATE_VALIDATION_MANIFEST_WITH_VALID_AUDIO_PATHS>"
$adapter = ".\runs\YOUR_NEW_L2_SMOKE_RUN\adapter-final"

& $py .\run_asr_baselines.py `
  --manifest $evalManifest `
  --models qwen `
  --qwen-backend qwen-asr `
  --qwen-model Qwen/Qwen3-ASR-1.7B `
  --qwen-adapter $adapter `
  --max-utts 20 `
  --max-duration 120 `
  --max-new-tokens 512 `
  --seed 42 `
  --out ".\runs\YOUR_NEW_L2_EVAL_RUN"
```

`--max-duration 120` permits the 54–60 s recordings that were skipped in our original `--max-duration 30` evaluation. It does not guarantee every longer recording succeeds. Check `n_selected`, `n_predicted`, `n_unpredicted`, and error counts in `qwen/summary.json`. **An evaluation run with missing predictions must not be directly compared with a run on a different subset.**

For a proper **zero-shot vs LoRA** comparison, run the **same `qwen-asr` backend, model family, evaluation manifest, seed, decoding settings and maximum duration** once **without** `--qwen-adapter`, and once **with** it, into separate output directories. Compare exactly matched successful recordings, ideally with full eligible coverage. Do not treat the older Transformers `-hf` baseline as an automatically matched comparator to `qwen-asr`.

## 6. What has actually been verified?

These are **local smoke/integration observations**, not final scientific results.

| Check | Observed result | Interpretation |
| --- | --- | --- |
| Pilot manifest | 8,000 recordings, 221 speakers, 0 missing WAVs | Local data/score integration passed. |
| Continuous α sensitivity | α=0: 9.49%; α=0.5: 12.17%; α=1: 15.73%; α=2: 24.70% | Expected **sampled speaker zero-shot WER** rises with α; **not fine-tuned WER**. |
| 8,000 weighted draws | 49.2% commands, 30.2% novel, 20.6% spontaneous; 4,344 unique recordings | Real seeded sampling approximates 50/30/20; replacement causes repeats. |
| Static target-selection test | Decoder set included in Encoder+Decoder; unrelated modules excluded | Target-name selection logic behaves as intended on synthetic names. |
| GPU: Encoder+Decoder | 112 decoder + 96 audio target modules; 2 optimizer steps completed | Qwen 4-bit training path works for the tested short batch. |
| Gradient audit | 112/112 decoder and 96/96 audio `lora_B` groups with nonzero first-backward gradients | Both adapter groups receive gradient; does not alone prove full training quality. |
| GPU memory in short L2 smoke | 3.48 GB peak PyTorch allocated; 3.733 GB reserved | Limited to measured short samples; **not** guaranteed capacity for 120 s audio. |
| GPU: Decoder-only | 112 decoder + 0 audio targets; 2 steps completed | Second LoRA mode also follows the training path. |
| Adapter reload / ASR | Private L2 adapter reloaded and transcribed; 18/20 scored at 30 s, WER 5.1181%, CER 1.941% | Two selected clips (54.4/60.4 s) exceeded the eval cutoff; **not a performance claim**. |

**Not verified yet:** 120-second **training** stability, strict adapter **parameter delta** audit, broad validation performance, reproducible resume, category-hard-mass joint constraints, multi-seed results, full-model/model-size adaptations, and formal statistical comparison. The in-file docstrings mentioning “not GPU verified” are now outdated for the short smoke path; they should be revised at the next code cleanup.

## 7. Important experiment safeguards

1. **Privacy:** Never push SAP WAVs, private manifests, transcripts, speaker IDs, zero-shot predictions/weight CSVs, adapters, checkpoints, `runs/`, or private run metadata. Git-stage **source and non-sensitive policy only**; verify `.gitignore` and inspect `git diff --cached --name-only` before pushing.
2. **Speaker isolation:** Original PD split creation is upstream. Sampling weights must come **only from TRAIN speaker** zero-shot WER. Validation is for model selection and should be evaluated without rebalancing; Test is held out until the agreed final evaluation.
3. **Data order:** `train_rows[i]` and `train_dataset[i]` must identify the **same recording**. The current `ManifestDataset` preserves JSONL order; do not shuffle rows before using the WeightedRandomSampler without realigning weights.
4. **Equal experimental settings:** Freeze source data, duration filtering, category quotas, Qwen base model/backend, LoRA rank/alpha/dropout, optimizer, learning rate, total optimization budget, seed policy and evaluation subsets when comparing sampling strategies or LoRA modes. LoRA mode is the intentional difference in that comparison.
5. **Hard speaker vs category quotas:** A final E3-style 50% hard mass is a **separate constraint** from category 50/30/20. Validate both on the **final probability vector** after normalization, not on intermediate weights.
6. **No implicit resume or overwrite:** `train.py` refuses a nonempty run directory. Sampler RNG-state restoration is not implemented. A new run name avoids stale metadata/cache confusion.
7. **Generation warnings:** `pad_token_id`/`use_cache`/Triton FLOP warnings occurred during a successful smoke and were not fatal. Actual errors, adapter-load failures and missing predictions still require investigation.
8. **30 s vs longer audio:** The 30-second limit was chosen for initial smoke safety, **not** as a dataset requirement. For 60–120 s training use a new prepared manifest and a long-audio forward/backward+memory check before launching full training. Evaluate every comparison arm with the same duration policy.
9. **Reproducibility:** Training logs fingerprint inputs and key code files, but do not prove a reproducible full research result. Preserve the Git commit, package versions, run config, shared weight-source identity, exact train/val set and sampling distribution report for every final run.


### Stable integration API (please do not fork it)

```python
from sampling import load_speaker_wer_csv, build_sampler

speaker_wer = load_speaker_wer_csv(TRAIN_ZERO_SHOT_CSV)  # fractions, e.g. 0.15
sampler = build_sampler(
    train_rows=rows,                  # same order as train_dataset
    speaker_wer=speaker_wer,
    strategy="continuous",            # or "hard_threshold" once implemented
    category_probs=shared_policy,    # must sum to 1 and cover retained categories
    alpha=1.0, epsilon=0.02, max_wer=0.30,
    seed=42,
    num_samples=len(rows),
)
# Shared HF Trainer owns the sampler/data loader and backend.
```


## 9. Team catch-up / Git hygiene

After pulling this branch:

```powershell
git branch --show-current
git status --short
& $py "$pipeline\train.py" --help
& $py "$pipeline\fine_tune_data_prepare.py" --help
& $py -m py_compile "$pipeline\sampling.py" "$pipeline\train.py" "$pipeline\qwen_backend.py" "$pipeline\fine_tune_data_prepare.py"
```

Make sure private `data/`, `runs/`, pretrained weights and adapters exist only on approved local storage and are Git-ignored. A successful `git pull` downloads **code**, not SAP audio or private experiment artifacts.

To commit source updates deliberately:

```powershell
git add -- fine-tuning-pipeline/README.md fine-tuning-pipeline/sampling.py fine-tuning-pipeline/train.py fine-tuning-pipeline/qwen_backend.py fine-tuning-pipeline/fine_tune_data_prepare.py fine-tuning-pipeline/category_probs.json
git diff --cached --name-only   # Inspect before commit
git commit -m "Document shared ASR fine-tuning pipeline and handoff"
git push
```

*Last documented smoke-test status: November 23, 2026. Update the verified/untested sections when a full experiment or new backend is actually run.*
