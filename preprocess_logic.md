# Parkinson's ASR — Data Preprocessing, Splitting & Evaluation

## 1. Dataset Preprocessing

We use the original SAP dataset and filter recordings where `Etiology = Parkinson's Disease`.

### Transcript Cleaning Rules

| Markup | Rule |
|---|---|
| `[...]` | Remove the brackets and all enclosed text (prompt questions). |
| `(...)` | Remove only the parentheses; preserve the spoken content, including fillers and repetitions. |
| `{...}` | Exclude the entire recording due to uncertain annotations. |
| Empty transcript | Exclude the recording. |
| Unbalanced brackets | Exclude the recording for manual inspection. |

Additional rules:
- Use `Prompt.Transcript`, not `Prompt.Prompt Text`.


### Cleaning Results

| Metric | Recordings |
|---|---:|
| Original PD Recordings | 116,721 |
| Excluded: Curly Braces | 389 |
| Excluded: Unbalanced Brackets | 77 |
| **Final Usable Recordings** | **116,255** |

## 2. Train / Validation / Test Split

We use a **speaker-independent split**, ensuring that no speaker appears in multiple subsets.

| Split | Speakers | Recordings | Source | Purpose |
|---|---:|---:|---|---|
| Train | 221 | 91,035 | Original Train | Fine-tuning |
| Validation | 25 | 10,733 | Original Train | Hyperparameter tuning and checkpoint selection |
| Test | 35 | 14,487 | Original Dev | Final evaluation |

### Splitting Logic


- Keep all recordings from the same speaker in one split.
- Save speaker assignments and dataset manifests for reuse across experiments.
- Speaker overlap across Train, Validation, and Test: **0**.

## 3. WER / CER Text Normalization

Dataset cleaning and evaluation normalization are separate steps.

During evaluation, we apply the **same English text normalizer to both reference and prediction** to avoid counting differences in writing style as recognition errors.

We plan to use `EnglishTextNormalizer` from `whisper-normalizer`.

### Normalization Rules

| Difference | Handling |
|---|---|
| Uppercase vs. lowercase | Normalize case |
| Punctuation | Remove or standardize |
| `I'm` vs. `I am` | Normalize contractions where supported |
| `twenty one` vs. `21` | Normalize number formats where supported |
| Fillers (`um`, `uh`) | Preserve for primary evaluation |
| Repeated words | Preserve |
| Partial words (e.g., `s-`) | Requires a consistent policy; verify normalizer behavior |

The exact behavior of the normalizer will be verified with unit tests before final evaluation.

### Evaluation Metrics

**Primary: Normalized Verbatim WER / CER**
- Preserves spoken words, fillers, and repetitions.
- Normalizes orthographic differences.
- Measures how accurately the model transcribes the actual speech.

**Secondary (Optional): Clean WER**
- May ignore selected disfluencies under a predefined policy.
- Provides an additional measure of readable transcription quality.

## 4. Overall Pipeline

```text
Original SAP WAV + JSON
          |
          v
Filter Parkinson's Disease
          |
          v
Transcript Cleaning
(Remove [], unwrap (), exclude {})
          |
          v
Speaker-level Split (Seed = 42)
          |
          v
train.jsonl / val.jsonl / test.jsonl
          |
          v
Pretrained ASR / Fine-Tuned ASR
          |
          v
Predicted Transcripts
          |
          v
Normalize Reference & Prediction
          |
          v
WER / CER Evaluation
```

### Evaluation Consistency

- Use identical datasets and text normalization across models.
- Report overall, per-category, and per-speaker WER/CER.
- Use Validation for development and model selection.
- Reserve Test for final evaluation.
- Store raw predictions and normalized texts so metrics can be recalculated without rerunning inference.