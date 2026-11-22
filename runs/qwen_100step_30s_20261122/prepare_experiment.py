"""CPU-only preparation for the authorized 100-step grid; no model loading."""
import collections
import csv
import hashlib
import json
import math
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

OUT = pathlib.Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = OUT / 'source'
DATA = OUT / 'prepared'
sys.path.insert(0, str(SOURCE))


def read(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    if (OUT / 'preparation.json').exists():
        raise SystemExit('Preparation already recorded; refusing to change frozen inputs.')
    if DATA.exists() and any(DATA.iterdir()):
        raise SystemExit('Partial preparation exists; inspect before retrying.')
    DATA.mkdir(exist_ok=True)
    train_source = ROOT / 'data/pd_prepared/train.jsonl'
    val_source = ROOT / 'data/pd_prepared/val.jsonl'
    csv_source = ROOT / 'runs/qwen_train_analysis/training_sampling_weights.csv'
    baseline = ROOT / 'runs/qwen_full_baseline/train'
    baseline_config = json.loads((baseline / 'run_config.json').read_text())
    assert sha(train_source) == baseline_config['manifest_sha256'], 'Training baseline provenance mismatch'
    train = read(train_source)
    predictions = read(baseline / 'predictions.jsonl')
    scored = read(baseline / 'scored_predictions.jsonl')
    key = lambda row: (row['speaker_id'], row['audio_path'])
    assert {key(x) for x in train} == {key(x) for x in predictions}
    counts = collections.Counter(x['speaker_id'] for x in train)
    totals = collections.defaultdict(lambda: [0, 0])
    for row in scored:
        totals[row['speaker_id']][0] += row['word_errors']
        totals[row['speaker_id']][1] += row['ref_words']
    from make_sampling_weights import read_input
    speakers, _ = read_input(csv_source)
    assert set(counts) == {x['speaker_id'] for x in speakers}
    for row in speakers:
        sid = row['speaker_id']
        assert counts[sid] == row['n']
        assert math.isclose(row['wer'], 100 * totals[sid][0] / totals[sid][1], abs_tol=0.00011)
    local_sources = DATA / 'source_manifests'
    local_sources.mkdir()
    audio_root = pathlib.Path('C:/SAP_downsampled_database')
    relocated = {}
    for split, source_path in [('train', train_source), ('val', val_source)]:
        relocated[split] = local_sources / (split + '.jsonl')
        with relocated[split].open('w', encoding='utf-8', newline='\n') as stream:
            for row in read(source_path):
                new = dict(row)
                new['original_audio_path'] = row['audio_path']
                new['audio_path'] = str(audio_root / row['audio_relpath'])
                stream.write(json.dumps(new, ensure_ascii=False) + '\n')
    print('Verified existing training baseline and all 221 speaker scores. Reading relocated audio headers on CPU...', flush=True)
    subprocess.run([sys.executable, str(SOURCE / 'prepare_qwen_finetuning.py'),
                    '--train', str(relocated['train']), '--val', str(relocated['val']), '--out', str(DATA),
                    '--max-duration', '30', '--verify-audio'], check=True)
    weights = DATA / 'training_sampling_weights_v2.csv'
    subprocess.run([sys.executable, str(SOURCE / 'make_sampling_weights.py'),
                    '--in', str(csv_source), '--out', str(weights), '--verify'], check=True)
    from hard_speaker_sampler import build_sampling_weights, load_speaker_weights
    prepared = read(DATA / 'train.jsonl')
    checks = {}
    for strategy in ('e3', 'e4'):
        info = load_speaker_weights(weights, strategy=strategy)
        w = build_sampling_weights(prepared, info, check_hard_mass=strategy == 'e3')
        checks[strategy] = {'recordings': len(w), 'speakers': len(info), 'mass': sum(w),
                            'hard_speaker_mass': sum(p for row, p in zip(prepared, w) if info[row['speaker_id']]['hard'])}
        print(strategy, checks[strategy], flush=True)
    import soundfile as sf
    from run_asr_baselines import select_rows
    val = read(relocated['val'])
    eligible = []
    for row in val:
        header = sf.info(row['audio_path'])
        if header.frames / header.samplerate <= 30:
            eligible.append(row)
    frozen = select_rows(eligible, 1000, 42)
    assert len(frozen) == 1000
    assert {x['speaker_id'] for x in prepared}.isdisjoint(x['speaker_id'] for x in frozen)
    frozen_path = DATA / 'val_30s_seed42_1000.jsonl'
    with frozen_path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in frozen:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    cfg = json.loads((SOURCE / 'configs/shared.json').read_text())
    cfg['data']['train_file'] = str(DATA / 'train.jsonl')
    cfg['data']['weights_csv'] = str(weights)
    cfg['data']['val_manifest'] = str(frozen_path)
    cfg['data']['test_manifest'] = str(ROOT / 'data/pd_prepared/test.jsonl')
    cfg['training'].update(max_steps=100, save_steps=100, seed=42)
    cfg['eval'].update(max_utts=0, max_duration=30.0, seed=42)
    cfg['_notes'].append('100-step pilot: difficulty scores reused from original full training split; frozen 1000-row validation subset; source commit in source_provenance.json.')
    write(OUT / 'experiment_config.json', cfg)
    missing_scores = {key(x) for x in train} - {key(x) for x in scored}
    assert len(missing_scores) == 2, 'Unexpected baseline scoring gaps'
    meta = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_rows': len(prepared), 'training_speakers': len({x['speaker_id'] for x in prepared}),
        'train_max_duration_seconds': 30, 'sampling_validation': checks,
        'audio_path_relocation': {'original_root': 'C:/SAP downsampled database', 'current_root': str(audio_root),
            'method': 'Reconstruct audio_path from the unchanged audio_relpath; preserve original_audio_path. Original source manifests untouched.'},
        'difficulty_provenance': {'source_csv': str(csv_source), 'csv_sha256': sha(csv_source),
            'baseline_directory': str(baseline), 'baseline_config': baseline_config,
            'original_training_rows': len(train), 'predicted_rows': len(predictions), 'scored_rows': len(scored),
            'interpretation': 'Per-speaker WER measured on original full training split, including long audio; scores retained and speaker mass redistributed across retained <=30s recordings.',
            'empty_normalized_reference_policy': 'Two original training recordings have predictions but no scored WER. They remain eligible training rows; speaker WER uses scored references only, as in existing verified CSV. No fresh inference.'},
        'validation': {'original_rows': len(val), 'eligible_rows': len(eligible), 'frozen_rows': len(frozen),
            'selection_seed': 42, 'selector': 'shared-finetuning run_asr_baselines.select_rows; duration filter before category-balanced selection',
            'categories': dict(collections.Counter(x['category'] for x in frozen)),
            'speakers': len({x['speaker_id'] for x in frozen}),
            'speaker_counts': dict(sorted(collections.Counter(x['speaker_id'] for x in frozen).items())),
            'representativeness': 'Category-balanced development subset, not the original validation category distribution.'},
        'input_hashes': {str(p): sha(p) for p in [train_source, val_source, csv_source]},
        'frozen_hashes': {str(p): sha(p) for p in [DATA / 'train.jsonl', weights, frozen_path, OUT / 'experiment_config.json']},
        'gpu_stages': 'pending; E4_L1 train+eval, matching original model eval, E4_L2 train+eval, E3_L1 train+eval, E3_L2 train+eval',
    }
    write(OUT / 'preparation.json', meta)
    print(json.dumps({k: meta[k] for k in ('training_rows', 'training_speakers', 'sampling_validation', 'validation')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
