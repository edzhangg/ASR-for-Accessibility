# E3: Hard-Speaker Oversampling — Implementation Specification

## Objective

Oversample speakers with high baseline WER while maintaining speaker-balanced sampling within the hard and non-hard groups.

## Sampling Strategy

1. **Hard-speaker definition:** Top 25% of training speakers ranked by baseline WER (`hard_top25 == 1`).
2. **Group probabilities:**
   - Hard speakers: 50% of total sampling probability.
   - Remaining speakers: 50% of total sampling probability.
3. **Within-group sampling:** Uniform across speakers.
4. **Within-speaker sampling:** Uniform across recordings.

For recording $i$ belonging to speaker $s$:

$$
P(x_{s,i}) =
\begin{cases}
\frac{0.5}{N_H n_s}, & s \in H \\
\frac{0.5}{N_O n_s}, & s \notin H
\end{cases}
$$

where $N_H$ and $N_O$ are the numbers of hard and other speakers, and $n_s$ is the number of training recordings for speaker $s$.

## Implementation

**Input files:**
- `train.jsonl` — training manifest.
- `runs/qwen_train_analysis/training_sampling_weights.csv` — precomputed sampling weights.

Use the CSV column `e3_per_recording_probability` directly. Do not recompute speaker difficulty or sampling weights independently.

**PyTorch configuration:**

```python
sampler = WeightedRandomSampler(
    weights=recording_weights,
    num_samples=len(train_dataset),
    replacement=True,
    generator=torch.Generator().manual_seed(42),
)
```

The recording weights must follow the exact order of `train_dataset`.

## Consistency Requirements

Both E3 implementations should use:

- Identical training manifest and sampling weights CSV.
- Identical hard-speaker classification.
- Sampling **with replacement**.
- `num_samples = len(train_dataset)` per epoch.
- Random seed `42` and consistent RNG handling.

The expected hard-speaker sampling proportion is 50%, not necessarily exactly 50% in each batch or epoch.

## Validation

Before training, verify:

- All training speakers match the weights CSV.
- Recording counts per speaker match `n_manifest`.
- Recording probabilities sum to 1.
- Hard-speaker probability mass sums to 0.5.
- Simulated sampling produces approximately 50% hard-speaker recordings.

**Current status:** Synthetic sampler unit tests passed. Real-data sampling validation and training integration are still pending.