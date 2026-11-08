# Parkinson's Disease Dataset — EDA Results

## 1. Dataset Overview

| Metric | Value |
|---|---:|
| Total Speakers | 281 |
| Total Recordings | 116,721 |
| Total Audio Duration | 267.187 hours |
| Sample Rate | 16,000 Hz |
| Audio Channels | 1 (Mono) |
| Train/Dev Speaker Overlap | 0 |

### Train / Dev Distribution

| Split | Speakers | Recordings |
|---|---:|---:|
| Train | 246 | 102,177 |
| Dev | 35 | 14,544 |
| **Total** | **281** | **116,721** |

**Finding:** The original Train and Dev splits have no overlapping Contributor IDs.

## 2. Audio Characteristics

### Audio Duration Distribution

| Statistic | Duration |
|---|---:|
| Minimum | 0.399 s |
| 25th Percentile | 3.82 s |
| Median | 5.31 s |
| 75th Percentile | 8.13 s |
| 95th Percentile | 24.77 s |
| Maximum | 180.14 s |
| Mean | 8.24 s |

**Findings:**
- Most recordings are relatively short: 75% are under 8.13 seconds, and 95% are under 24.77 seconds.
- Some recordings are significantly longer (up to 180 seconds), which may require segmentation or special handling during ASR training.
- All recordings use 16 kHz mono audio, providing a consistent input format.

## 3. Speech Category Distribution

| Category | Train | Dev | Total | Percentage |
|---|---:|---:|---:|---:|
| Digital Assistant Commands | 68,207 | 9,691 | 77,898 | 66.7% |
| Novel Sentences | 22,641 | 3,247 | 25,888 | 22.2% |
| Spontaneous Speech Prompts | 11,240 | 1,606 | 12,846 | 11.0% |
| Non-spontaneous Speech Prompts | 89 | 0 | 89 | 0.1% |
| **Total** | **102,177** | **14,544** | **116,721** | **100%** |

**Findings:**
- Digital Assistant Commands account for approximately **66.7% of Parkinson's recordings**.
- Training with the original data distribution would expose the model to substantially more short digital assistant commands than spontaneous speech.
- The Care2.0 project aims to improve natural communication for people with speech disabilities, not only short command recognition.
- **Category-balanced Sampling** could be considered during training to reduce the dominance of Digital Assistant Commands.
- WER/CER should also be evaluated separately for different speech categories.

## 4. Transcript Characteristics

### Transcript Word Count

| Statistic | Word Count |
|---|---:|
| Minimum | 1 |
| 25th Percentile | 4 |
| Median | 7 |
| 75th Percentile | 12 |
| 95th Percentile | 51 |
| Maximum | 346 |
| Mean | 13.15 |

### Prompt Text vs. Transcript

| Comparison | Recordings |
|---|---:|
| Prompt Text = Transcript | 87,564 |
| Prompt Text ≠ Transcript | 29,157 |

Approximately **25.0% of recordings** have different Prompt Text and Transcript values.

### Transcript Markup

| Markup Type | Recordings |
|---|---:|
| Square brackets `[...]` | 12,853 |
| Parentheses `(...)` | 12,451 |
| Curly braces `{...}` | 384 |

**Findings:**
- Transcripts contain special annotations for prompts, disfluencies, repetitions, and potentially unintelligible speech.
- Annotation meanings need to be confirmed before establishing text preprocessing and WER/CER normalization rules.
- Original transcripts should be preserved until a consistent normalization policy is established.

## 5. Speech Feature Ratings

The dataset contains ratings for specific speech characteristics associated with Parkinson's Disease.

### Rating Availability

| Metric | Recordings |
|---|---:|
| Recordings with Ratings | 6,919 |
| Recordings without Ratings | 109,802 |
| Percentage with Ratings | 5.9% |
| Unique Rating Dimensions | 46 |

### Major Parkinson's-specific Rating Dimensions

| Speech Feature | Rating Objects |
|---|---:|
| Breathy voice (continuous) | 6,874 |
| Harsh voice | 6,874 |
| Imprecise consonants | 6,874 |
| Monoloudness | 6,874 |
| Pitch level | 6,874 |
| Short rushes of speech | 6,874 |
| Variable rate | 6,874 |
| Inappropriate silences | 6,873 |
| Intelligbility | 6,873 |
| Reduced stress | 6,870 |
| Naturalness | 6,865 |
| Monopitch | 6,857 |

*Counts refer to rating objects, not unique speakers. Dimension names preserve the dataset's original spelling.*

### Example: Inappropriate Silences Ratings

| Level | Rating Objects |
|---|---:|
| 1 | 6,271 |
| 2 | 276 |
| 3 | 220 |
| 4 | 80 |
| 5 | 11 |
| 6 | 5 |
| 7 | 3 |
| Missing / NA | 8 |

**Findings:**
- The dataset includes annotations for specific speech characteristics, including **Inappropriate silences**, **Variable rate**, **Short rushes of speech**, and **Imprecise consonants**.
- These dimensions may support feature-specific ASR error analysis, such as examining the relationship between speech characteristics and WER.
- Ratings are available for only **5.9% of PD recordings**, so coverage across speakers and speech categories needs further investigation.
- Ratings are highly concentrated at Level 1 for several dimensions, including Inappropriate silences.
- The meaning and direction of the rating scale (Levels 1–7) have not yet been confirmed.
- The unit of rating (individual recording vs. speaker-level assessment) also needs to be verified.
- The `Inappropriate silences` dimension could potentially support pause-related analysis, but it is not necessarily equivalent to measured pause duration.

## 6. Key Findings

1. The Parkinson's subset contains **281 speakers, 116,721 recordings, and 267.2 hours of audio**, with no speaker overlap between the original Train and Dev sets.
2. Most recordings are short, and all audio is stored in a consistent 16 kHz mono format.
3. **Digital Assistant Commands dominate the dataset (66.7%)**, while Spontaneous Speech accounts for approximately 11.0%.
4. Around 25% of recordings have differences between Prompt Text and Transcript.
5. Special transcript markup requires a consistent interpretation and normalization policy before formal ASR training and evaluation.
6. **Speech Feature Ratings are available**, including dimensions relevant to pauses, speaking rate, articulation, and intelligibility.
7. Ratings cover only a small subset of recordings, and their coverage, granularity, and scoring scale still need to be confirmed.

*Note: All results are based on the Parkinson's Disease subset of the original SAP Train/Dev data. Audio statistics were obtained from WAV headers; full waveform decoding and transcript accuracy validation have not yet been performed.*