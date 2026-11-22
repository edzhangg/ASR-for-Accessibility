# Category-First E3 Sampling — Implementation and Handoff Notes

**Project:** Speech Recognition for Accessible Communication / ASR-for-Accessibility  
**Workstream:** Hard-speaker oversampling for Qwen3-ASR fine-tuning  
**Working branch:** `category-stratified-sampling`  
**Status:** Implementation and unit tests completed; real-data simulation and GPU training pending  
**Last updated:** 2026-11-22

## 1. Executive summary

I implemented a configurable **category-first version of E3 hard-speaker sampling** for the accessible speech recognition project. The sampler controls the training distribution along two dimensions:

1. **Speech category:** 50% Digital Assistant Commands, 30% Novel Sentences, 20% Spontaneous Speech Prompts.
2. **Speaker difficulty:** within **each category**, 50% of sampling probability goes to hard speakers and 50% to non-hard speakers.

The sampling process is:

**Category → Hard/non-hard group → Speaker → Recording**

This extends the project's existing E3 hard-speaker oversampling method while retaining the original sampling implementation for comparison. The new approach is intended to prevent category imbalance from confounding comparisons of speaker-difficulty sampling strategies.

**Current evidence:** the local test suite completed with `Ran 63 tests ... OK (skipped=1)`. The implementation has been committed locally on `category-stratified-sampling` but has **not** been pushed to GitHub. A real-data sampling simulation and model training have **not** yet been confirmed.

---

## 2. How this fits into the overall project

### Overall research objective

The project investigates automatic speech recognition (ASR) for speakers whose speech may be difficult for general-purpose models to transcribe, including speakers with speech impairments. The broader workflow involves preparing speech data, evaluating baseline ASR performance, fine-tuning a model, and comparing approaches using metrics such as **word error rate (WER)** and **character error rate (CER)**.

Our model workstream uses **Qwen3-ASR-1.7B** with **LoRA fine-tuning**. My assigned experimental contribution is **hard-speaker oversampling with decoder-only LoRA**.

### Why sampling matters

Uniform sampling over recordings can overrepresent speakers or speech categories that happen to have more recordings. Conversely, oversampling difficult speakers without controlling category composition can change the mix of speech tasks seen during training.

The new category-first E3 sampler explicitly controls both factors. This helps us answer a cleaner experimental question:

> Does emphasizing difficult speakers during fine-tuning improve recognition performance, when the speech-category mixture is held fixed?

The sampler is **not** itself an ASR model and does not change model architecture or the loss function. It changes **which training recordings are drawn, and how often**. Whether this improves WER/CER must be established experimentally.

---

## 3. Sampling strategies and design decisions

The existing project supports several sampling strategies (E1–E4). The current work specifically extends **E3**, the hard-speaker 50/50 strategy.

### Original E3 versus category-first E3

| Aspect | Existing E3 | New category-first E3 |
|---|---|---|
| Main priority | Hard/non-hard speaker sampling | Fixed category mix, then hard/non-hard split |
| Category proportions | Not explicitly fixed to 50/30/20 | Explicitly fixed to 50/30/20 |
| Hard-speaker probability | 50% overall | 50% **within each category** |
| Speaker selection | Based on existing E3 weights | Uniform among eligible speakers in each category/group |
| Recording selection | Based on existing E3 weighting | Uniform among that speaker's recordings in the selected category |
| Reproducibility | Existing seeded sampler | Seeded `WeightedRandomSampler` |
| Intended role | Existing baseline/comparator | Controlled experimental variant |

A previous **speaker-first category-stratified** option also exists in the training workflow (`--category-stratified`, `--category-balance`). That approach is distinct from the new **category-first** design: it does not guarantee the same exact global 50/30/20 proportions. We have kept the two options separate rather than silently changing the old method.

### Fixed target distribution

| Category | Target sampling probability | Hard | Non-hard |
|---|---:|---:|---:|
| Digital Assistant Commands | 50% | 25% | 25% |
| Novel Sentences | 30% | 15% | 15% |
| Spontaneous Speech Prompts | 20% | 10% | 10% |
| **Total** | **100%** | **50%** | **50%** |

**Non-spontaneous Speech Prompts** is excluded from this sampling configuration; rows in categories absent from the configuration receive weight zero.

The design assumes that every included category contains at least one hard and one non-hard speaker. The implementation raises an error if this assumption is violated.

### Definition of a hard speaker

The existing weights CSV supplies the `hard_top25` classification, based on speaker-level baseline recognition difficulty (zero-shot WER). The new sampler **reuses** this classification; it does not recompute WER or redefine the hard-speaker threshold.

---

## 4. Exact sampling probability

For recording \(i\), belonging to speaker \(s\), category \(c\), and difficulty group \(g\):

\[
P(i)
=
q_c
\cdot
p_g
\cdot
\frac{1}{N_{c,g}}
\cdot
\frac{1}{n_{s,c}}
\]

where:

- \(q_c\): configured category probability (0.50, 0.30, or 0.20).
- \(p_g\): probability of the speaker-difficulty group (0.50 for hard, 0.50 for non-hard).
- \(N_{c,g}\): number of distinct eligible speakers in category \(c\) and group \(g\).
- \(n_{s,c}\): number of recordings belonging to speaker \(s\) in category \(c\).

The four factors correspond exactly to the four-stage sampling procedure:

1. Select a category according to \(q_c\).
2. Select hard or non-hard with equal probability.
3. Select a speaker uniformly within the chosen category/group.
4. Select one of that speaker's recordings uniformly within the category.

This construction ensures that a speaker with many recordings does not automatically receive more probability mass than another speaker in the same category/group.

The implementation checks that:

- All recording probabilities sum to 1.
- Each included category's probability mass equals its configured target.
- The overall hard-speaker probability mass equals the configured target.

These are **theoretical probability checks**. Finite random draws will fluctuate around the targets.

---

## 5. Files and changes

### `category_first_sampler.py` — new

The core implementation is CPU-compatible and contains:

- `load_category_config(path)`: loads and validates the JSON configuration; checks positive finite category probabilities, sum-to-one, and a valid hard-group probability.
- `build_category_first_weights(train_rows, speaker_info, config)`: computes one probability per training-manifest row in the original row order; assigns zero probability to excluded categories; validates probability totals.
- `make_category_first_sampler(train_rows, speaker_info, config, *, num_samples=None, seed=42)`: constructs a seeded PyTorch `WeightedRandomSampler` with replacement.

**Why it matters:** isolates the experimental sampling policy from the ASR model, making it testable without loading Qwen or using a GPU.

### `configs/category_sampling.json` — new

Configuration for the experimental distribution:

```json
{
  "category_proportions": {
    "Digital Assistant Commands": 0.5,
    "Novel Sentences": 0.3,
    "Spontaneous Speech Prompts": 0.2
  },
  "hard_group_probability": 0.5
}
```

**Why it matters:** the sampling targets can be changed without modifying the implementation, making experiments easier to reproduce.

### `tests/test_category_first_sampler.py` — new

Two tests inspect the probability calculations on a small synthetic dataset:

- `test_marginals`: checks that weights sum to 1, category totals match the targets, hard-speaker probability is 50%, excluded rows get zero weight, and selected recording weights are correct.
- `test_missing_group`: verifies that the sampler raises a clear error if a configured category lacks non-hard speakers.

**Why it matters:** catches errors in the probability distribution before expensive GPU experiments.

### `train_qwen_lora.py` — training integration

The local training script recognizes:

- `--category-config PATH`
- `--sampling e3`
- `--weights-csv PATH`

It also validates that:

- `--category-config` is used only with `--sampling e3`.
- `--category-config` and the older `--category-stratified` mode cannot be used together.

**Why it matters:** the new sampling policy can be selected as an experiment configuration without replacing the existing E3 behavior.

**Integration caveat:** command-line argument recognition has been confirmed locally, but a full GPU training run using this option has not yet been performed.

### `hard_speaker_sampler.py` — existing dependency

The new sampler uses `load_speaker_weights(..., strategy="e3")` to read speaker metadata and hard/non-hard labels from the existing sampling-weights CSV. The existing sampling strategies remain available.

### `parkinsons_eda.ipynb` — separate untracked notebook

This notebook appeared as an untracked file in `git status`. It was intentionally **not** included in the category-first sampling commit. Its contents and analysis results are not documented here.

---

## 6. Work completed

| Item | Status | Evidence / notes |
|---|---|---|
| Define category-first E3 design | **Done** | Category → group → speaker → recording |
| Agree on category proportions | **Done** | 50% / 30% / 20% |
| Define hard/non-hard split | **Done** | 50/50 within each category |
| Exclude non-spontaneous category | **Done** | Zero sampling weight |
| Make proportions configurable | **Done** | `configs/category_sampling.json` |
| Implement probability calculations | **Done** | `build_category_first_weights()` |
| Implement seeded PyTorch sampler | **Done** | `make_category_first_sampler()` |
| Add synthetic unit tests | **Done** | Two tests in `test_category_first_sampler.py` |
| Run local unit-test suite | **Done** | `Ran 63 tests in 1.316s; OK (skipped=1)` |
| Confirm training CLI recognizes option | **Done** | `--category-config` and incompatibility checks found |
| Commit changes locally | **Done** | Committed on `category-stratified-sampling`; commit hash not recorded here |
| Push branch to GitHub | **Not done** | Intentionally deferred |
| Validate real training manifest | **Pending** | Must check actual file paths and fields |
| Run 100,000-draw CPU simulation | **Pending** | No simulation output yet |
| Verify experiment launcher forwards config | **Pending** | `run_experiment.py` integration not confirmed |
| Run GPU fine-tuning | **Pending** | No training results yet |
| Evaluate WER/CER and compare experiments | **Pending** | No outcome claims yet |

---

## 7. Next steps (recommended order)

### Step 1 — Validate against the real training data

Confirm the actual Qwen training manifest contains both `speaker_id` and `category` for every training row. Confirm the sampling-weights CSV exists and includes all referenced speakers and their `hard_top25` labels.

The expected data sources in the repository workflow are:

- `data/qwen_finetuning/train.jsonl`
- The generated `training_sampling_weights.csv` (verify its actual local path)
- `configs/category_sampling.json`

The new sampler relies on the category values matching the JSON configuration **exactly**.

### Step 2 — Run a CPU-only sampling simulation

Use the real manifest, the real weights CSV, and the configured sampler to draw **100,000 indices** with a fixed seed.

Report:

1. Theoretical probability mass for each category.
2. Simulated frequency of each category.
3. Theoretical and simulated hard/non-hard proportions **within each category**.
4. Number of excluded-category samples (expected: zero).
5. Optionally, per-speaker sample frequencies and concentration.

**Expected simulated frequencies:** approximately 50% / 30% / 20%, and approximately 50/50 within each category. Small deviations are normal due to randomness.

The existing unit tests already validate the mathematical construction on synthetic data; this step tests the **actual data and metadata integration**.

### Step 3 — Verify the end-to-end experiment launcher

Inspect `run_experiment.py` and confirm it forwards `--category-config` to `train_qwen_lora.py`, if experiments are normally launched through that script.

If it does not, add the argument in a backward-compatible way. The training command should specify E3, the weights CSV, and the JSON configuration together.

### Step 4 — Prepare the GPU training run

After the CPU checks pass:

- Use the existing Qwen3-ASR training pipeline and decoder-only LoRA configuration.
- Use the same train/validation/test split as the comparison experiments.
- Keep model, LoRA settings, optimizer, training budget, random seed(s), and evaluation procedure as consistent as possible.
- Save the configuration, seed, dataset version, and training logs for reproducibility.
- Coordinate GPU use with teammates before launching jobs on the shared VM.

**Important:** an experiment can use category-first E3 without changing the Qwen model architecture.

### Step 5 — Evaluate recognition performance

Compare the new category-first E3 run with relevant existing baselines and sampling strategies.

Recommended reporting:

- Overall **WER** and **CER**.
- WER/CER for **hard speakers** versus **non-hard speakers**.
- WER/CER **by speech category**.
- Per-speaker results or distributions, to avoid hiding improvements/regressions behind an overall mean.
- Training stability, time, and any notable failure cases.

**Research interpretation:** improvements should be attributed cautiously. Category-first E3 differs from existing sampling in its category distribution and within-category speaker allocation. A fair comparison should control these factors where possible, especially when comparing with E4.

### Step 6 — Document, commit, and share

Once validated:

- Add the simulation command and its measured results to the repository documentation.
- Record any changes as a new commit.
- Push `category-stratified-sampling` when ready to share.
- Open a pull request for teammate review rather than merging directly into the shared branch.
- Keep raw speech data, private manifests, large checkpoints, and sensitive paths out of Git.

---

## 8. How to run the checks

From the repository root on macOS:

```bash
# Run the local unit tests (CPU only)
python3 -m unittest discover -s tests -v

# Confirm the active branch
git branch --show-current

# Inspect local changes
git status
```

The category-first sampler is a **module**, not a command-line application. Running `python3 category_first_sampler.py --help` is not a meaningful validation command because it does not expose a CLI.

To run the real-data simulation, use a separate validation script that imports:

```python
from category_first_sampler import (
    load_category_config,
    build_category_first_weights,
    make_category_first_sampler,
)
from hard_speaker_sampler import load_speaker_weights
```

**Do not start a GPU run merely because unit tests pass.** First validate the real manifest and sampling distribution.

---

## 9. Deliverables and research contribution

**Engineering deliverable:** a reusable, configurable, deterministic-seed-compatible category-first sampler integrated into the E3 training path, with synthetic tests.

**Methodological contribution:** a way to hold the speech-category distribution constant while deliberately allocating half the training probability mass to hard speakers **inside each category**.

**Value to the broader project:**

- Improves control over the training-data distribution.
- Reduces the risk that category imbalance obscures the effect of hard-speaker oversampling.
- Makes E3 easier to compare with other strategies using shared category targets.
- Supports category-level and difficulty-level evaluation.
- Makes the sampling method reproducible and easier for teammates to inspect.

**What has not been demonstrated yet:** improved ASR accuracy, reduced hard-speaker WER, successful GPU fine-tuning with this configuration, or superiority to E1/E2/E4. Those are experimental questions for the next stage.

---

## 10. Handoff summary for teammates

The category-first E3 sampling implementation is ready for **real-data validation**. It uses the existing E3 hard-speaker labels, configurable 50/30/20 speech-category targets, and a 50/50 hard/non-hard split inside each category. The code has passed the local unit-test suite and is committed on the `category-stratified-sampling` branch, but has not been pushed.

**Immediate next action:** run the 100,000-draw CPU simulation on the real training manifest and confirm the experiment launcher passes the new configuration correctly. Only then schedule GPU fine-tuning and WER/CER evaluation.
