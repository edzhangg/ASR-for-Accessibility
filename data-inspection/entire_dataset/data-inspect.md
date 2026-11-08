# SAP Dataset Inventory Results

## 1. Dataset Overview

| Metric | Value |
|---|---:|
| Unique Speakers | 899 |
| WAV References | 346,034 |
| Disease Categories | 5 |

### Train / Dev Distribution

| Split | Speakers | WAV References |
|---|---:|---:|
| Train | 775 | 297,785 |
| Dev | 124 | 48,249 |
| **Total** | **899** | **346,034** |

### Data Quality

- Duplicate recording keys: **0**
- Missing WAV references: **0**
- JSON parsing errors: **0**
- Train/Dev shared speakers: **0**
- Empty transcripts: **1**

Note: These checks cover JSON metadata and WAV file existence, not audio decoding or transcript accuracy.

## 2. Disease Distribution

| Etiology | Train Speakers | Dev Speakers | Total Speakers | Recordings |
|---|---:|---:|---:|---:|
| Parkinson's Disease | 246 | 35 | 281 | 116,721 |
| Cerebral Palsy | 162 | 28 | 190 | 74,898 |
| ALS | 148 | 29 | 177 | 63,778 |
| Down Syndrome | 137 | 20 | 157 | 57,969 |
| Stroke | 82 | 12 | 94 | 32,668 |
| **Total** | **775** | **124** | **899** | **346,034** |

### Parkinson's Disease Subset

- **281** unique speakers
- **116,721** recording references
- **33.7%** of all recordings
- Approximately **415 recordings per speaker** on average
- Largest disease category by both speaker count and recording count

## 3. Parkinson's Disease Train / Dev Split

| Split | Speakers | Recording References |
|---|---:|---:|
| Train | 246 | 102,177 |
| Dev | 35 | 14,544 |
| **Total** | **281** | **116,721** |

**Train/Dev Speaker Overlap: 0**

The official Train and Dev splits have no overlapping Contributor IDs.


Whole dataset audio duration = 761.299 hours





# SAP Dataset Inventory — Additional Findings

## 1. Audio Characteristics

**Total Audio Duration:** 761.299 hours

**Audio Format:**
- Sample Rate: 16,000 Hz (all recordings)
- Channels: 1 (Mono, all recordings)

### Audio Duration Distribution

| Statistic | Duration |
|---|---:|
| Minimum | 0.40 s |
| 25th Percentile | 3.93 s |
| Median | 5.58 s |
| 75th Percentile | 8.49 s |
| 95th Percentile | 20.69 s |
| Maximum | 180.21 s |
| Mean | 7.92 s |

**Observations:**
- Most recordings are relatively short: 75% are under 8.49 seconds, and 95% are under 20.69 seconds.
- Some recordings are significantly longer (up to 180.21 seconds), which may require special handling or segmentation during ASR training.

## 2. Speech Category Distribution

| Category | Train | Dev | Total |
|---|---:|---:|---:|
| Digital Assistant Commands | 220,430 | 35,749 | 256,179 |
| Novel Sentences | 32,716 | 5,206 | 37,922 |
| Spontaneous Speech Prompts | 36,770 | 6,158 | 42,928 |
| Non-spontaneous Speech Prompts | 7,869 | 1,136 | 9,005 |

**Observations:**
- Digital Assistant Commands account for approximately **74% of all recordings**.
- If we train directly on the entire dataset without adjusting sampling, the model will be exposed to a large proportion of short digital assistant commands (e.g., controlling smart home devices or playing music).
- However, the Care2.0 project aims not only to recognize short commands but also to improve natural communication for people with speech disabilities.
- Therefore, we may consider **Category-balanced Sampling** during training instead of sampling strictly according to the original category distribution.
- The category distribution within the Parkinson's-specific subset still needs to be examined.

## 3. Transcript Characteristics

### Transcript Word Count

| Statistic | Word Count |
|---|---:|
| Minimum | 0 |
| Median | 6 |
| 75th Percentile | 9 |
| 95th Percentile | 34 |
| Maximum | 395 |
| Mean | 10.16 |

### Prompt Text vs. Transcript

| Comparison | Recordings |
|---|---:|
| Prompt Text = Transcript | 235,912 |
| Prompt Text ≠ ≠ Transcript

110,122

Approximately 31.8% of recordings have different Prompt Text and Transcript values.

For ASR training and evaluation, the Transcript field should be used as the reference rather than Prompt Text, subject to annotation quality checks.

Transcript Markup

Markup Type

Recordings Containing It

Square brackets [...]

43,245

Parentheses (...)

44,944

Curly braces {...}

1,659

Observations:

The dataset contains special transcription annotations whose meanings need to be confirmed before preprocessing.

These annotations may represent prompt text, disfluencies, corrections, unintelligible speech, or other metadata.

We should determine which annotations correspond to actual spoken words and establish consistent text normalization rules for ASR training and WER/CER evaluation.