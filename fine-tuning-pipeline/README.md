# PD ASR Fine-Tuning Pipeline — Team Handoff

**Last updated: November 24, 2026**

**Current status:** Both category-stratified sampling strategies are implemented and have passed all 17 CPU-based tests. The original shared Qwen training pipeline was GPU smoke-tested on November 23. GPU validation of the updated sampling implementation and full-scale controlled experiments are still pending.

This folder provides our shared interface for Parkinson's disease (PD) ASR fine-tuning, combining:

- Difficulty-aware, category-stratified speaker-level sampling.
- A shared Hugging Face training loop.
- A model-specific ASR backend.
- Configurable decoder-only and encoder-decoder LoRA.
- Reproducible sampling and training diagnostics.

The current backend is Qwen3-ASR-1.7B.

---

## 1. What's New — November 24 Update

This update builds directly on the shared fine-tuning pipeline implemented and smoke-tested on November 23. The existing training architecture, Qwen backend, data preparation workflow, category policy, and evaluation interface have been retained.

The main changes are:

| Component | Original Implementation | Updated Implementation |
|---|---|---|
| Continuous Sampling | Category-controlled recording-level difficulty weighting | Category-stratified speaker-level difficulty weighting |
| Hard Threshold Sampling | Not implemented | Fully implemented using precomputed `hard_top25` labels |
| Speaker exposure | Influenced by the number of recordings per speaker | Controlled explicitly at the speaker level within each category |
| Alpha = 0 | Uniform recording selection within category | Uniform speaker selection within category |
| Hard/other balance | Not implemented | Exactly 50/50 expected probability within each category |
| Metadata validation | Speaker WER loading | Shared WER validation and hard-label loading |
| Sampling diagnostics | No dedicated summary function | Expected category, group, and speaker-level probabilities |
| Testing | Original local smoke tests | 17 passing CPU-based tests covering both strategies |

### Files Changed

| File | Changes |
|---|---|
| `sampling.py` | Updated Continuous Sampling, implemented Hard Threshold Sampling, added hard-label loading, shared metadata validation, and sampling diagnostics. |
| `train.py` | Integrated hard-label loading and sampling diagnostics into the existing training workflow. |
| `tests/test_sampling.py` | Added synthetic unit and integration tests for both sampling strategies. |
| `README.md` | Updated documentation, experiment status, and integration instructions. |

### Files Retained Without Modification

- `fine_tune_data_prepare.py`
- `qwen_backend.py`
- `category_probs.json`
- Existing baseline and evaluation scripts.

No changes were made to the model architecture, LoRA target modules, optimizer, ASR collator, or upstream dataset preparation.

The updated implementation preserves the existing `build_sampler()` interface, with an additional optional `hard_labels` argument.

---

## 2. Pipeline Overview

```text
Existing speaker-disjoint PD splits
    data/pd_prepared/train.jsonl
    data/pd_prepared/val.jsonl
                |
                v
    fine_tune_data_prepare.py
    - Resolve relocated audio paths
    - Filter training categories
    - Apply optional duration cutoff
    - Create Qwen-format manifests
                |
                v
           sampling.py
    + TRAIN zero-shot speaker WER
    + Precomputed hard-speaker labels
    + Shared category probabilities
                |
       +--------+--------+
       |                 |
       v                 v
   Continuous       Hard Threshold
   Sampling         Sampling
       |                 |
       +--------+--------+
                |
                v
       WeightedRandomSampler
                |
                v
             train.py
       Hugging Face Trainer
                |
                v
         qwen_backend.py
       Qwen / QLoRA / Collator
                |
                v
        Private LoRA adapter
                |
                v
       run_asr_baselines.py
          WER/CER evaluation
```

**Important:** `sampling.py` does not generate a new manifest. It returns a PyTorch `WeightedRandomSampler` whose indices correspond to the original training manifest order.

Sampling happens dynamically during training. The shared Trainer owns the training loop, and WER/CER evaluation remains in the existing evaluation script.

---

## 3. Repository Structure

```text
fine-tuning-pipeline/
├── fine_tune_data_prepare.py
├── sampling.py
├── train.py
├── qwen_backend.py
├── category_probs.json
├── tests/
│   └── test_sampling.py
└── README.md
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `fine_tune_data_prepare.py` | Convert existing PD manifests into Qwen-format manifests, resolve audio paths, validate inputs, and create pilot subsets. |
| `sampling.py` | Load speaker difficulty metadata, calculate sampling probabilities, enforce category constraints, construct weighted samplers, and summarize expected exposure. |
| `train.py` | Parse experiment settings, validate speaker separation, construct the sampler, initialize the backend, run training, and save artifacts. |
| `qwen_backend.py` | Qwen model loading, 4-bit NF4 quantization, ASR collation, label masking, LoRA configuration, and gradient checks. |
| `category_probs.json` | Shared category probability policy for both sampling strategies. |
| `tests/test_sampling.py` | CPU-based synthetic tests for sampling correctness, reproducibility, metadata loading, and integration. |

### Related Existing Code

The following scripts remain useful and should not be deleted or silently replaced:

- `prepare_pd_dataset.py`
- `run_asr_baselines.py`
- `prepare_qwen_finetuning.py`
- `make_sampling_weights.py`
- `hard_speaker_sampler.py`
- `train_qwen_lora.py`
- `lora_configs.py`
- `run_experiment.py`

The shared pipeline is not yet wired into the original experiment runner.

### Backend Abstraction

The backend registry in `train.py` delegates model-specific operations to `create_backend(args)`.

Only Qwen is currently registered and tested. The architecture supports future backend extensions, but compatibility with additional ASR models has not been verified.

---

## 4. Sampling Design

### 4.1 Shared Category Distribution

Both sampling strategies use the same category probabilities defined in `category_probs.json`.

| Category | Target Probability |
|---|---:|
| Digital Assistant Commands | 50% |
| Novel Sentences | 30% |
| Spontaneous Speech Prompts | 20% |

These values represent the expected proportions of all training draws, not per-recording weight multipliers.

Training categories must be explicitly handled during data preparation. Validation data is not rebalanced.

Both strategies use:

- The same category probability policy.
- Sampling with replacement.
- A configurable random seed.
- The same number of draws per epoch.
- TRAIN-only speaker difficulty metadata.
- The original training manifest order.

By default, an epoch consists of `len(train_rows)` draws, not necessarily that many unique recordings.

### 4.2 Continuous Difficulty-Weighted Sampling

**Sampling hierarchy:**

```text
Category
    |
    v
Difficulty-weighted speaker
    |
    v
Uniform recording within speaker-category pair
```

The original implementation assigned difficulty weights directly to recordings. Consequently, speakers with more recordings received more aggregate probability mass, even when their WER was identical.

The updated implementation first defines a speaker-level difficulty weight:

```text
w_s = (epsilon + min(d_s, max_wer)) ** alpha
```

where:

- `d_s` is the speaker's TRAIN zero-shot WER as a fraction.
- `epsilon` prevents zero sampling weights.
- `max_wer` caps the contribution of extreme WER values.
- `alpha` controls the strength of difficulty weighting.

Default parameters:

| Parameter | Default |
|---|---:|
| `alpha` | 1.0 |
| `epsilon` | 0.02 |
| `max_wer` | 0.30 |

For a recording belonging to speaker `s` and category `c`:

```text
P(recording) =
    q_c
    * w_s / sum(w_j for eligible speakers j in category c)
    * 1 / n_sc
```

Here, `q_c` is the target category probability and `n_sc` is the number of recordings belonging to speaker `s` in category `c`.

This guarantees that:

- Speakers with identical difficulty weights receive equal probability within the same category, regardless of recording count.
- Recordings are sampled uniformly within each speaker-category pair.
- Increasing `alpha` increases the relative preference for higher difficulty weights.
- `alpha = 0` gives uniform speaker selection within each category.
- Category probabilities remain unchanged.

Global speaker probabilities may differ because speakers can participate in different categories.

### 4.3 Hard Threshold Sampling

**Sampling hierarchy:**

```text
Category
    |
    +------------------+
    |                  |
    v                  v
Hard group         Other group
   50%                 50%
    |                  |
    v                  v
Uniform speaker    Uniform speaker
    |                  |
    v                  v
Uniform recording  Uniform recording
```

Hard speakers are identified using the precomputed `hard_top25` column in the existing TRAIN difficulty CSV.

The sampler does not recompute the difficulty ranking after category filtering.

For a recording belonging to speaker `s`, category `c`, and difficulty group `g`:

```text
P(recording) =
    q_c
    * 0.5
    * 1 / N_cg
    * 1 / n_sc
```

where:

- `q_c` is the target category probability.
- `N_cg` is the number of eligible speakers in category `c` and group `g`.
- `n_sc` is the number of recordings belonging to speaker `s` in category `c`.

The implementation guarantees the following expected joint probabilities:

| Category | Hard Group | Other Group | Total |
|---|---:|---:|---:|
| Digital Assistant Commands | 25% | 25% | 50% |
| Novel Sentences | 15% | 15% | 30% |
| Spontaneous Speech Prompts | 10% | 10% | 20% |
| **Total** | **50%** | **50%** | **100%** |

Each category must contain at least one hard speaker and one other speaker. If either group is empty, the sampler raises a `ValueError` rather than silently changing the experiment policy.

The 50/50 guarantee applies to probability mass, not exact realized recording counts in a finite epoch.

### 4.4 Comparison Between Strategies

| Property | Continuous | Hard Threshold |
|---|---|---|
| Category selection | Fixed 50/30/20 | Fixed 50/30/20 |
| Difficulty representation | Continuous WER | Binary hard label |
| Speaker selection | Difficulty-weighted | Uniform within difficulty group |
| Recording selection | Uniform within speaker-category pair | Uniform within speaker-category pair |
| Hard-group probability | Determined by difficulty weights | Fixed at 50% |
| Replacement | Yes | Yes |
| Draws per epoch | Same | Same |

The two strategies share the same experimental controls so that their difficulty-selection mechanisms can be compared without introducing recording-count bias.

---

## 5. Input and Output Contracts

### 5.1 Training Manifests

The preparation script produces:

```text
data/<PRIVATE_PREPARED_DIRECTORY>/
├── train.jsonl
├── train_pilot.jsonl
├── val.jsonl
└── preparation_report.json
```

Each Qwen-format row contains:

```json
{
    "audio": "<LOCAL_PRIVATE_WAV_PATH>",
    "text": "language English<asr_text><TRANSCRIPT>",
    "speaker_id": "<PRIVATE_ID>",
    "category": "Digital Assistant Commands"
}
```

The preparation script retains the existing upstream speaker-disjoint splits.

Pilot generation preserves speaker coverage when possible, but does not itself enforce the target category distribution. Category balancing is handled by the sampler.

### 5.2 Speaker Difficulty CSV

The existing TRAIN zero-shot speaker difficulty CSV is reused.

Required columns:

| Column | Continuous | Hard Threshold |
|---|---|---|
| `speaker_id` | Required | Required |
| `difficulty_wer_percent` | Required | Required |
| `hard_top25` | Optional | Required |

Example using synthetic identifiers:

```csv
speaker_id,difficulty_wer_percent,hard_top25
A,60.0,1
B,25.0,0
C,10.0,0
D,5.0,0
```

WER is stored as a percentage in the CSV and converted to a fraction during loading. For example, `15.0` becomes `0.15`.

Hard labels are converted from `0`/`1` into Python booleans.

Every eligible training speaker must have a valid WER. Hard Threshold additionally requires a hard label for every eligible speaker.

Do not use validation or test WER to assign training difficulty.

Do not rerun expensive baseline inference if the existing TRAIN difficulty artifact is complete and valid.

### 5.3 Training Artifacts

The existing training output structure is unchanged:

```text
runs/<NEW_RUN_NAME>/
├── training_config.json
├── gradient_check.json
├── gpu_usage.json
├── log_history.json
├── checkpoint-*/
└── adapter-final/
```

All manifests, difficulty CSVs, model artifacts, and run metadata must remain on approved private storage.

---

## 6. LoRA Configurations

The original LoRA implementation is unchanged.

| CLI Flag | Adapted Modules | Frozen Parameters |
|---|---|---|
| `--lora decoder_only` | Text decoder attention | Base weights and audio encoder |
| `--lora encoder_decoder` | Text decoder and audio encoder attention | Base weights |

Both modes use PEFT LoRA with configurable rank, alpha, and dropout.

The Qwen backend uses 4-bit NF4 loading by default. LoRA target matching remains restricted to the intended decoder and audio encoder attention modules.

Both LoRA configurations completed two-step GPU smoke tests in the original November 23 pipeline.

---

## 7. Sampling Diagnostics

The updated pipeline prints sampling diagnostics before model training.

Reported information includes:

- Sampling strategy.
- Number of training recordings.
- Number of training speakers.
- Expected category probability distribution.
- Minimum and maximum speaker-level sampling probabilities.
- Overall hard-speaker probability in Hard Threshold mode.
- Conditional hard-speaker probability within each category in Hard Threshold mode.

Expected example for Hard Threshold:

```text
--- Sampling Diagnostics ---
Strategy: hard_threshold
Recordings: ...
Speakers: ...

Expected category distribution:
  Digital Assistant Commands: 50.00%
  Novel Sentences: 30.00%
  Spontaneous Speech Prompts: 20.00%

Overall hard probability: 50.00%

Hard probability within each category:
  Digital Assistant Commands: 50.00%
  Novel Sentences: 50.00%
  Spontaneous Speech Prompts: 50.00%

Minimum speaker probability: ...
Maximum speaker probability: ...
----------------------------
```

The diagnostics summarize the actual probability vector stored in the sampler.

They do not change sampling behavior or report realized sampling frequencies.

For a speaker with probability `p_s`, the expected number of draws per epoch is:

```text
expected_draws = num_samples * p_s
```

Individual speaker probabilities are available internally through `summarize_sampling()`, but real speaker identifiers are not printed in the training diagnostics.

Continuous mode currently does not load hard labels, so its training diagnostics do not report hard-group probability.

---

## 8. Testing and Verification

### 8.1 Updated Sampling Tests — November 24

All 17 tests passed locally on macOS using Python 3.13.1 and pytest 8.3.4.

The tests cover:

| Test Area | Status |
|---|---|
| Equal-WER speaker exposure | Passed |
| Uniform speaker selection at `alpha=0` | Passed |
| Alpha sensitivity | Passed |
| Category probability preservation | Passed |
| Hard/other group balancing | Passed |
| Hard Threshold speaker uniformity | Passed |
| Recording uniformity | Passed |
| Multi-category joint probabilities | Passed |
| WER clipping | Passed |
| Empty-group error handling | Passed |
| Missing speaker WER handling | Passed |
| Random seed reproducibility | Passed |
| Sampling with replacement | Passed |
| CSV metadata loading | Passed |
| CSV-to-sampler integration | Passed |
| Sampling diagnostics | Passed |

Some test functions cover multiple properties, and parameterized tests run separately for both strategies.

The tests use synthetic data and require no private audio, manifests, or GPU.

### Run Tests on macOS

From the repository root:

```bash
python3 -m pytest fine-tuning-pipeline/tests/test_sampling.py -v
```

Expected result:

```text
17 passed
```

Check training script syntax:

```bash
python3 -m py_compile fine-tuning-pipeline/train.py
```

### Run Tests on Windows PowerShell

Use the appropriate local Python environment:

```powershell
$py = ".\.venv-qwen-lora\Scripts\python.exe"

& $py -m pytest .\fine-tuning-pipeline\tests\test_sampling.py -v
```

PyTorch and pytest are required for the complete test suite.

### 8.2 Original Pipeline Verification — November 23

The following observations were documented before the sampling update:

| Check | Original Observed Result |
|---|---|
| Pilot manifest | 8,000 recordings, 221 speakers, 0 missing WAVs |
| Weighted sampling | 49.2% commands, 30.2% novel, 20.6% spontaneous |
| Encoder-decoder LoRA | 112 decoder and 96 audio target modules; 2 optimizer steps completed |
| Decoder-only LoRA | 112 decoder and 0 audio target modules; 2 optimizer steps completed |
| Gradient audit | Nonzero first-backward LoRA B gradients in all targeted groups |
| Short-audio GPU memory | 3.48 GB peak allocated; 3.733 GB reserved |
| Adapter reload | Successfully reloaded and transcribed audio |
| Small evaluation smoke | 18/20 recordings scored; WER 5.1181%, CER 1.941% |

These are historical smoke-test observations, not final scientific results.

The original Continuous Sampling exposure statistics were generated using recording-level weighting and must not be treated as results for the updated speaker-level implementation.

### 8.3 Remaining Verification

The updated sampling implementation has passed CPU-based tests, but the complete updated training pipeline has not yet been GPU smoke-tested.

Full-scale training, long-audio training stability, broad validation performance, multi-seed comparisons, and reproducible checkpoint resume remain unverified.

---

## 9. Quick Start — Windows PowerShell

Run all commands from the repository root.

The paths below are placeholders or examples. Update them locally without committing private paths.

### A. Configure Environment

```powershell
$py = ".\.venv-qwen-lora\Scripts\python.exe"

$pipeline = ".\fine-tuning-pipeline"

$trainSource = ".\data\pd_prepared\train.jsonl"

$valSource = ".\data\pd_prepared\val.jsonl"

$audioRoot = "<ACTUAL_LOCAL_SAP_DATASET_ROOT>"

$werCsv = "<EXISTING_PRIVATE_TRAIN_ZERO_SHOT_WER_CSV>"

$categoryPolicy = "$pipeline\category_probs.json"

$prepared = ".\data\qwen_finetuning_category_smoke"
```

Check CUDA:

```powershell
& $py -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Required training dependencies include compatible versions of:

- `torch`
- `transformers`
- `qwen-asr`
- `peft`
- `bitsandbytes`
- `accelerate`
- `librosa`
- `soundfile`

Dependency versions are not yet frozen in a shared lockfile. Reuse the successfully tested GPU environment rather than reinstalling packages unnecessarily.

### B. Prepare Qwen Manifests

Skip this step if the existing prepared manifests are valid and match the agreed experimental settings.

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

if ($LASTEXITCODE -ne 0) {
    throw "Preparation failed"
}
```

The 30-second cutoff is an initial smoke-test configuration, not an inherent model limitation.

For longer training recordings, create a new prepared directory with an appropriate duration policy. Do not overwrite an existing prepared dataset.

### C. Continuous Sampling GPU Smoke Test

```powershell
& $py "$pipeline\train.py" `
  --backend qwen `
  --train-manifest "$prepared\train_pilot.jsonl" `
  --val-manifest "$prepared\val.jsonl" `
  --speaker-wer-csv $werCsv `
  --sampling continuous `
  --category-probs $categoryPolicy `
  --alpha 1.0 --epsilon 0.02 --max-wer 0.30 `
  --lora decoder_only `
  --load-in-4bit 1 --gradient-checkpointing 1 `
  --batch-size 1 --grad-acc 1 `
  --max-steps 2 --save-steps 1 --log-steps 1 `
  --seed 42 `
  --output-dir ".\runs\continuous_decoder_smoke"

if ($LASTEXITCODE -ne 0) {
    throw "Continuous smoke test failed"
}
```

### D. Hard Threshold GPU Smoke Test

Use the same prepared manifests, category policy, LoRA configuration, seed, and optimization budget.

The difficulty CSV must contain `hard_top25`.

```powershell
& $py "$pipeline\train.py" `
  --backend qwen `
  --train-manifest "$prepared\train_pilot.jsonl" `
  --val-manifest "$prepared\val.jsonl" `
  --speaker-wer-csv $werCsv `
  --sampling hard_threshold `
  --category-probs $categoryPolicy `
  --lora decoder_only `
  --load-in-4bit 1 --gradient-checkpointing 1 `
  --batch-size 1 --grad-acc 1 `
  --max-steps 2 --save-steps 1 --log-steps 1 `
  --seed 42 `
  --output-dir ".\runs\hard_threshold_decoder_smoke"

if ($LASTEXITCODE -ne 0) {
    throw "Hard Threshold smoke test failed"
}
```

For encoder-decoder LoRA, change:

```powershell
--lora encoder_decoder
```

and use a different, unused output directory.

The above commands are intended for the next GPU verification step. They have not yet been executed against the November 24 sampling implementation.

### E. Check Training Artifacts

```powershell
$run = ".\runs\hard_threshold_decoder_smoke"

Test-Path "$run\adapter-final\adapter_config.json"

Get-Content "$run\gradient_check.json"

Get-Content "$run\gpu_usage.json"
```

Verify that:

- The expected sampling diagnostics are printed.
- Both optimizer steps complete.
- The intended LoRA groups receive nonzero gradients.
- The final adapter is saved successfully.
- GPU memory usage is reasonable.

A two-step smoke test verifies basic training integration, not model quality or convergence.

---

## 10. Evaluation

Continue using the team's existing `run_asr_baselines.py`.

The evaluator expects the original PD-format manifest containing unprefixed reference transcripts.

**Do not pass the Qwen-format validation manifest to the evaluator.**

If audio paths have changed, create a private local copy of the original validation manifest with corrected audio paths while preserving the original split and reference transcripts.

Example:

```powershell
$evalManifest = "<PRIVATE_VALIDATION_MANIFEST_WITH_VALID_AUDIO_PATHS>"

$adapter = ".\runs\hard_threshold_decoder_smoke\adapter-final"

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
  --out ".\runs\hard_threshold_smoke_eval"
```

This is an evaluation smoke configuration, not a full validation experiment.

For a controlled zero-shot versus fine-tuned comparison, use identical evaluation manifests, backend, model family, decoding settings, duration policy, and seed.

Run the zero-shot model without `--qwen-adapter` and the fine-tuned model with the corresponding adapter.

Check prediction coverage and compare matched recordings. Do not directly compare WER/CER calculated on different subsets.

---

## 11. Shared Integration API

The original sampler interface is retained.

```python
from sampling import (
    load_speaker_wer_csv,
    load_hard_labels_csv,
    build_sampler,
    summarize_sampling,
)

strategy = "hard_threshold"  # or "continuous"

speaker_wer = load_speaker_wer_csv(TRAIN_ZERO_SHOT_CSV)

hard_labels = (
    load_hard_labels_csv(TRAIN_ZERO_SHOT_CSV)
    if strategy == "hard_threshold"
    else None
)

sampler = build_sampler(
    train_rows=rows,
    speaker_wer=speaker_wer,
    strategy=strategy,
    category_probs=shared_policy,
    alpha=1.0,
    epsilon=0.02,
    max_wer=0.30,
    seed=42,
    num_samples=len(rows),
    hard_labels=hard_labels,
)

summary = summarize_sampling(
    rows,
    sampler.weights.tolist(),
    hard_labels=hard_labels,
)
```

The returned sampler remains compatible with the shared Trainer integration.

Do not reorder the training dataset independently of the sampler weights.

---

## 12. Experiment Safeguards

### Privacy

Never commit:

- SAP audio.
- Private training or evaluation manifests.
- Transcripts or real speaker identifiers.
- Speaker difficulty CSVs.
- Zero-shot predictions.
- LoRA adapters and checkpoints.
- `runs/` directories.
- Private paths or run metadata.

Only source code, synthetic tests, documentation, and non-sensitive configuration should be tracked.

### Speaker Isolation

All sampling difficulty information must originate from TRAIN speakers.

Validation remains unbalanced and is used for model selection. Test data remains held out until the agreed final evaluation.

### Controlled Comparisons

When comparing Continuous and Hard Threshold Sampling, hold the following constant:

- Training and validation datasets.
- Duration filtering.
- Category probabilities.
- Base model and backend.
- LoRA configuration.
- Optimizer and learning rate.
- Optimization budget.
- Draws per epoch.
- Seed policy.
- Evaluation settings and subsets.

When comparing LoRA configurations, keep sampling strategy and other experimental settings fixed.

### Reproducibility

Preserve the Git commit, dependency versions, training configuration, data fingerprints, sampling diagnostics, and evaluation configuration.

The sampler uses a seeded random generator, but exact sampler RNG-state restoration after checkpoint resume is not implemented.

### Output Directories

Use a new output directory for every training run. The existing Trainer rejects nonempty output directories.

### Long Audio

The original GPU smoke tests used short recordings. Longer training recordings require separate memory and stability validation.

---

## 13. Remaining Team Tasks

The sampling implementation and local CPU testing are complete.

The remaining work is:

- [ ] Run GPU smoke tests for the updated Continuous and Hard Threshold implementations.
- [ ] Confirm the agreed experimental configurations and run full-scale controlled training.
- [ ] Evaluate the resulting adapters using the shared WER/CER evaluator and compare results.

No additional sampler implementation is required unless GPU integration reveals an issue.

---

## 14. Git and Team Handoff

The November 24 sampling update was developed on:

```text
category-stratified-sampling
```

It builds on the shared fine-tuning pipeline.

After pulling the updated branch, run the CPU tests before launching GPU training.

For this sampling update, the intended source files are:

```text
fine-tuning-pipeline/sampling.py
fine-tuning-pipeline/train.py
fine-tuning-pipeline/tests/test_sampling.py
fine-tuning-pipeline/README.md
```

Inspect staged files before committing:

```bash
git diff --cached --name-only
```

A successful Git pull transfers source code, not private SAP data, difficulty metadata, or model artifacts.

---

## Development History

**November 23, 2026 — Original Shared Pipeline**

- Established the shared data preparation, sampling, training, and Qwen backend interfaces.
- Implemented category-controlled recording-level Continuous Sampling.
- Implemented both LoRA configurations.
- Completed short-audio GPU smoke tests and adapter reload verification.

**November 24, 2026 — Category-Stratified Sampling Update**

- Converted Continuous Sampling to speaker-level difficulty weighting within each category.
- Implemented Hard Threshold Sampling with category-specific 50/50 hard/other probability.
- Added precomputed hard-label loading and shared metadata validation.
- Preserved the existing training architecture and backend.
- Integrated privacy-conscious sampling diagnostics.
- Added and passed 17 CPU-based unit and integration tests.

**Current status:** Sampling development is complete. Updated GPU smoke testing and formal experiments are pending.