#!/usr/bin/env python3
"""CPU-only, paired per-hard-speaker WER comparison of saved predictions."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from compare_runs import by_speaker, read_scored, score_rows


def compare_hard(baseline_path, model_paths, fraction=0.25):
    if not 0 < fraction < 1:
        raise ValueError('hard fraction must be between 0 and 1')
    baseline = read_scored(baseline_path)
    for row in baseline.values():
        if not isinstance(row['speaker_id'], str) or not row['speaker_id'].strip():
            raise ValueError('Baseline has an empty or invalid speaker_id')
        if not row['ref_norm'].strip():
            raise ValueError('Baseline has an empty normalized reference')
    # Freeze membership using ALL baseline rows, independent of model failures.
    ranking = sorted(
        ((speaker, score_rows(rows)['wer']) for speaker, rows in by_speaker(baseline.values()).items()),
        key=lambda pair: (-pair[1], pair[0]),
    )
    hard = dict(ranking[:math.ceil(fraction * len(ranking))])
    models = {name: read_scored(path) for name, path in model_paths.items()}
    if not models:
        raise ValueError('At least one fine-tuned model is required')
    common = set(baseline).intersection(*(set(rows) for rows in models.values()))
    if not common:
        raise ValueError('No common audio_path values; check manifests and path conventions')
    for name, rows in models.items():
        for key in common:
            for field in ('speaker_id', 'ref_norm'):
                if rows[key][field] != baseline[key][field]:
                    raise ValueError(f'{name}: {field} differs from baseline for a common utterance')
    paired = by_speaker([baseline[key] for key in sorted(common)])
    missing = set(hard) - set(paired)
    if missing:
        raise ValueError(f'{len(missing)} fixed hard speakers have no common recordings; cannot compare')
    records, summaries = [], []
    all_groups = by_speaker(baseline.values())
    for name, predictions in models.items():
        deltas = []
        for speaker, selection_wer in hard.items():
            base = score_rows(paired[speaker])
            tuned = score_rows([predictions[r['audio_path']] for r in paired[speaker]])
            delta = tuned['wer'] - base['wer']
            deltas.append(delta)
            records.append(dict(
                speaker_id=speaker, model=name,
                baseline_selection_wer=selection_wer,
                baseline_n_utterances=len(all_groups[speaker]),
                n_common_utterances=base['n'], ref_words=base['ref_words'],
                baseline_wer=base['wer'], finetuned_wer=tuned['wer'], delta_pp=delta,
                relative_wer_reduction_pct=(-100 * delta / base['wer'] if base['wer'] else None),
                outcome='improved' if delta < 0 else 'worsened' if delta > 0 else 'unchanged',
            ))
        summaries.append(dict(model=name, n_hard=len(hard),
                              improved=sum(d < 0 for d in deltas),
                              worsened=sum(d > 0 for d in deltas),
                              unchanged=sum(d == 0 for d in deltas),
                              median_delta_pp=statistics.median(deltas)))
    return dict(hard_fraction=fraction, hard_speakers=list(hard),
                n_common_utterances=len(common),
                dropped_by_intersection={'baseline': len(baseline) - len(common),
                                         **{n: len(r) - len(common) for n, r in models.items()}},
                sources={'baseline': str(baseline_path),
                         'models': {n: str(p) for n, p in model_paths.items()}},
                per_speaker=records, summary=summaries)


def write_report(result, out):
    out.mkdir(parents=True, exist_ok=True)
    for key, filename in [('per_speaker', 'hard_speaker_comparison.csv'),
                          ('summary', 'hard_speaker_summary.csv')]:
        with (out / filename).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    (out / 'hard_speaker_comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Per-hard-speaker WER comparison', '',
             'Hard membership is fixed from the full supplied zero-shot output. WER uses the common recordings across all supplied models.',
             'WER is in percent; delta = fine-tuned minus baseline (percentage points); negative is better.',
             'These results describe the supplied evaluation subset, not necessarily the full validation population.', '',
             f"Common recordings: {result['n_common_utterances']}. Dropped: {result['dropped_by_intersection']}", '',
             '| Speaker | Model | Recordings | Reference words | Baseline WER | Tuned WER | Delta pp | Outcome |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for r in result['per_speaker']:
        speaker = r['speaker_id'].replace('|', '\\|').replace('\n', ' ')
        model = r['model'].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {speaker} | {model} | {r['n_common_utterances']} | {r['ref_words']} | {r['baseline_wer']:.4f} | {r['finetuned_wer']:.4f} | {r['delta_pp']:+.4f} | {r['outcome']} |")
    (out / 'hard_speaker_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path, help='Zero-shot scored_predictions.jsonl')
    parser.add_argument('--model', action='append', required=True, metavar='NAME=PATH', help='Repeat for each checkpoint scored_predictions.jsonl')
    parser.add_argument('--hard-fraction', type=float, default=0.25)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    try:
        models = {}
        for spec in args.model:
            name, sep, path = spec.partition('=')
            if not sep or not name or not path or name == 'baseline' or name in models:
                raise ValueError('Each --model must have a unique NAME=PATH; baseline is reserved')
            models[name] = Path(path)
        result = compare_hard(args.baseline, models, args.hard_fraction)
        write_report(result, args.out)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Cannot compare saved outputs: {exc}\nNo training or model inference was launched.\n')
    print(f"Wrote {len(result['hard_speakers'])} hard speakers to {args.out}")
    print(f"Dropped by intersection: {result['dropped_by_intersection']}")
    for row in result['summary']:
        print(f"{row['model']}: {row['improved']}/{row['n_hard']} improved; {row['worsened']} worsened; median delta {row['median_delta_pp']:+.4f} pp")


if __name__ == '__main__':
    main()
