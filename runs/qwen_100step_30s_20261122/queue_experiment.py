"""Sequential GPU-gated execution and reporting for this prepared experiment."""
import argparse
import csv
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

OUT = pathlib.Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = OUT / 'source'
CONFIG = OUT / 'experiment_config.json'
ARMS = [('e4', 'L1'), ('e4', 'L2'), ('e3', 'L1'), ('e3', 'L2')]
STATUS = OUT / 'queue_status.json'
STATE = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def status(**fields):
    STATE.update(fields, updated_at_utc=now())
    save_json(STATUS, STATE)


def check_inputs():
    prep = json.loads((OUT / 'preparation.json').read_text())
    for path, expected in prep['frozen_hashes'].items():
        assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == expected, 'Frozen input changed: ' + path
    config = json.loads(CONFIG.read_text())
    assert config['training']['max_steps'] == 100
    assert config['training']['batch_size'] == 1 and config['training']['grad_acc'] == 8
    assert config['eval']['max_utts'] == 0 and config['eval']['max_duration'] == 30
    provenance = json.loads((OUT / 'source_provenance.json').read_text())
    for relative, digest in provenance.get('source_sha256', {}).items():
        assert hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest() == digest, 'Source snapshot changed: ' + relative
    return prep


def gpu_reading():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    memory, utilization = [int(x.strip()) for x in result.stdout.strip().splitlines()[0].split(',')]
    processes = subprocess.run(['tasklist', '/FI', 'PID eq 11200', '/FO', 'CSV', '/NH'],
                               capture_output=True, text=True, check=True).stdout
    blocker_alive = any(len(row) > 1 and row[1] == '11200' for row in csv.reader(processes.splitlines()))
    return {'memory_used_mib': memory, 'utilization_percent': utilization, 'existing_training_pid_11200_alive': blocker_alive}


def wait_gpu(stage):
    stable = 0
    while stable < 3:
        if (OUT / 'STOP_QUEUE').exists():
            raise RuntimeError('STOP_QUEUE exists; queue stopped between GPU stages.')
        if STATE.get('start_now', False):
            return
        reading = gpu_reading()
        available = (reading['memory_used_mib'] <= 2000 and reading['utilization_percent'] <= 10
                     and not reading['existing_training_pid_11200_alive'])
        stable = stable + 1 if available else 0
        status(state='waiting_for_gpu', next_stage=stage, gpu=reading, consecutive_idle_readings=stable)
        if stable == 0:
            report()
        if not available or stable == 1:
            print(now(), 'Waiting for exclusive GPU availability:', stage, reading, flush=True)
        if stable < 3:
            time.sleep(20)


def command(stages, sampling=None, lora=None):
    cmd = [sys.executable, '-u', str(SOURCE / 'run_experiment.py'), '--config', str(CONFIG),
           '--runs-dir', str(OUT), '--stages', *stages]
    if sampling:
        cmd += ['--sampling', sampling, '--lora', lora]
    return cmd


def execute(label, cmd):
    check_inputs()
    wait_gpu(label)
    status(state='running', current_stage=label, next_stage=None)
    print(now(), label, subprocess.list2cmdline(cmd), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_CACHE=str(OUT / 'cache/datasets'))
    started = time.time()
    with (OUT / 'execution.log').open('a', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
        status(child_pid=proc.pid)
        code = proc.wait()
    seconds = round(time.time() - started, 2)
    STATE.setdefault('stage_timings', {})[label] = {'seconds': seconds, 'exit_code': code}
    status(child_pid=None)
    if code:
        raise RuntimeError(f'{label} failed with exit code {code}; inspect execution.log and stage logs.')
    return seconds


def annotate_source(run_dir):
    path = run_dir / 'run_meta.json'
    if path.exists():
        meta = json.loads(path.read_text())
        meta['workspace_git_commit_reported_by_runner'] = meta.get('git_commit')
        meta['git_commit'] = json.loads((OUT / 'source_provenance.json').read_text())['source_commit']
        meta['source_branch'] = 'shared-finetuning'
        meta['source_snapshot'] = str(SOURCE)
        save_json(path, meta)


def validate_training(name):
    folder = OUT / name
    state = json.loads((folder / 'checkpoint-100/trainer_state.json').read_text())
    assert state['global_step'] == 100 and state['max_steps'] == 100, name + ' did not finish exactly 100 steps'
    assert (folder / 'adapter-final/adapter_model.safetensors').is_file()
    final = hashlib.sha256((folder / 'adapter-final/adapter_model.safetensors').read_bytes()).hexdigest()
    checkpoint = hashlib.sha256((folder / 'checkpoint-100/adapter_model.safetensors').read_bytes()).hexdigest()
    assert final == checkpoint, name + ' final weights differ from checkpoint 100'
    gradients = json.loads((folder / 'gradient_check.json').read_text())
    assert gradients, 'Missing gradient checks'
    config = json.loads((folder / 'training_config.json').read_text())
    assert config['train_rows'] == json.loads((OUT / 'preparation.json').read_text())['training_rows']


def validate_eval(name):
    folder = OUT / name / 'eval_val/qwen'
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['n_selected'] == 1000 and summary['n_predicted'] == 1000, name + ' evaluation has missing predictions'
    assert summary['overall']['n'] == 1000, name + ' has unscored recordings'
    config = json.loads((folder / 'run_config.json').read_text())
    shared = json.loads(CONFIG.read_text())
    assert config['manifest'] == shared['data']['val_manifest']
    assert config['qwen_backend'] == shared['eval']['backend']
    assert config['max_duration'] == 30 and config['max_new_tokens'] == shared['eval']['max_new_tokens']


def report():
    prep = json.loads((OUT / 'preparation.json').read_text())
    current = json.loads(STATUS.read_text()) if STATUS.exists() else {'state': 'prepared; GPU stages pending'}
    provenance = json.loads((OUT / 'source_provenance.json').read_text())
    lines = ['# Qwen 100-step grid results and status', '', 'Updated: ' + now(), '',
             '**State:** ' + current.get('state', 'unknown'), '',
             f"Source: shared-finetuning `{provenance['source_commit']}`; isolated snapshot, existing checkout preserved.", '',
             f"Training: {prep['training_rows']:,} recordings, {prep['training_speakers']} speakers; <=30 seconds; exactly 100 optimizer steps per adapter; batch 1, accumulation 8; approximately 800 weighted draws with replacement per arm.", '',
             'Difficulty: verified scores from the original full training split, retained unchanged and redistributed over eligible recordings. No new training inference.', '',
             'Validation: frozen category-balanced 1,000-row <=30s subset, seed 42, all 25 validation speakers; max_utts=0. This is a development subset, not a full-distribution validation estimate.', '',
             'Category counts: ' + json.dumps(prep['validation']['categories']) + '.', '',
             '| Model | Scored / selected | Overall WER % | CER % | Mean speaker WER % | Train minutes | Eval minutes |',
             '|---|---:|---:|---:|---:|---:|---:|']
    category_rows = []
    summary_rows = []
    for name in ['zero_shot', 'E4_L1', 'E4_L2', 'E3_L1', 'E3_L2']:
        folder = OUT / name
        p = folder / 'eval_val/qwen/summary.json'
        if not p.exists():
            lines.append(f'| {name} | pending | — | — | — | — | — |')
            continue
        s = json.loads(p.read_text())
        meta_path = folder / 'run_meta.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        stages = meta.get('stages', {})
        train_minutes = stages.get('train', {}).get('seconds', 0) / 60
        eval_minutes = stages.get('eval_val', {}).get('seconds', 0) / 60
        overall = s['overall']
        lines.append(f"| {name} | {overall['n']} / {s['n_selected']} | {overall['wer']} | {overall['cer']} | {s.get('mean_speaker_wer')} | {train_minutes:.2f} | {eval_minutes:.2f} |")
        summary_rows.append({'model': name, 'scored': overall['n'], 'selected': s['n_selected'],
                             'wer_percent': overall['wer'], 'cer_percent': overall['cer'],
                             'train_minutes': train_minutes, 'eval_minutes': eval_minutes,
                             'unpredicted': s['n_unpredicted'], 'failures_this_run': s.get('failures_this_run', {})})
        for category, scores in s['by_category'].items():
            category_rows.append({'model': name, 'category': category, 'n': scores['n'], 'wer_percent': scores['wer'], 'cer_percent': scores['cer']})
    lines += ['', '| Model | Category | Scored | WER % | CER % |', '|---|---|---:|---:|---:|']
    lines += [f"| {x['model']} | {x['category']} | {x['n']} | {x['wer_percent']} | {x['cer_percent']} |" for x in category_rows]
    lines += ['', '## Timing and limitations', '',
              'GPU stages run sequentially. Stage durations include subprocess startup and model loading. GPU availability checks are skipped for launches using --start-now.', '',
              'Initial reservation: approximately 2–3.5 hours for four arms, plus approximately 15–25 minutes for the matching original-model baseline. Unmeasured L2 speed remains a source of uncertainty.', '',
              'Coverage and exceptions: ' + json.dumps([{x['model']: {'unpredicted': x['unpredicted'], 'failures': x['failures_this_run']}} for x in summary_rows]), '',
              'The original-model baseline uses the same qwen-asr backend, model, token limit, normalizer and frozen manifest as the adapters; older HF baseline metrics are not substituted.', '',
              'Inputs and hashes: preparation.json. Source provenance: source_provenance.json. Effective configuration: experiment_config.json. Queue state and revised estimates: queue_status.json. Logs: execution.log and each model folder/logs/.', '']
    if current.get('first_arm_update'):
        lines += ['## First-arm timing update', '', json.dumps(current['first_arm_update'], indent=2), '']
    if current.get('error'):
        lines += ['## Execution issue', '', current['error'], '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(OUT / 'results.json', {'updated_at_utc': now(), 'status': current, 'overall': summary_rows, 'categories': category_rows})
    for filename, values in [('results_overall.csv', summary_rows), ('results_categories.csv', category_rows)]:
        if values:
            with (OUT / filename).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)


def run_queue(start_now=False):
    global STATE
    lock = OUT / 'queue.lock'
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit('queue.lock exists. Check the recorded PID before considering a restart.')
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    STATE = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    status(state='starting', start_now=start_now, queue_pid=os.getpid(), queue_started_at_utc=now())
    try:
        check_inputs()
        report()
        completed = set(STATE.get('completed_models', []))
        for sampling, lora in ARMS:
            name = sampling.upper() + '_' + lora
            if name in completed:
                validate_training(name)
                validate_eval(name)
                continue
            if (OUT / name / 'adapter-final').exists() or (OUT / name / 'training_config.json').exists():
                raise RuntimeError(name + ' has existing partial outputs. Inspect manually; automatic retraining is disabled.')
            started = time.time()
            train_seconds = execute(name + ':train', command(['train'], sampling, lora))
            annotate_source(OUT / name)
            validate_training(name)
            eval_seconds = execute(name + ':eval', command(['eval'], sampling, lora))
            annotate_source(OUT / name)
            validate_eval(name)
            completed.add(name)
            status(completed_models=sorted(completed))
            if name == 'E4_L1':
                low = 3 * train_seconds + 4 * eval_seconds
                high = 5 * train_seconds + 5 * eval_seconds
                update = {'measured_at_utc': now(), 'train_seconds': train_seconds, 'eval_seconds': eval_seconds,
                          'arm_wall_seconds_including_gpu_gates': round(time.time() - started, 1),
                          'remaining_hours_low': round(low / 3600, 2), 'remaining_hours_high': round(high / 3600, 2),
                          'finish_utc_low_if_gpu_stays_available': (datetime.now(timezone.utc) + timedelta(seconds=low)).isoformat(),
                          'finish_utc_high_if_gpu_stays_available': (datetime.now(timezone.utc) + timedelta(seconds=high)).isoformat(),
                          'basis': 'Three remaining adapter evaluations plus one baseline; L2 training provisionally 1–2x measured L1; overhead allowance in upper estimate.'}
                status(first_arm_update=update)
                print('E4_L1 completed. Updated finish estimate:', json.dumps(update), flush=True)
            report()
            if name == 'E4_L1' and 'zero_shot' not in completed:
                execute('zero_shot:eval', command(['zero-shot']))
                annotate_source(OUT / 'zero_shot')
                validate_eval('zero_shot')
                completed.add('zero_shot')
                status(completed_models=sorted(completed))
                report()
        # Covers an explicitly resumed queue whose first arm and baseline were already recorded.
        if 'zero_shot' not in completed:
            execute('zero_shot:eval', command(['zero-shot']))
            annotate_source(OUT / 'zero_shot')
            validate_eval('zero_shot')
            completed.add('zero_shot')
        # Comparison reads saved predictions on CPU; it does not invoke inference.
        cmd = command(['compare'])
        with (OUT / 'execution.log').open('a', encoding='utf-8') as log:
            subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
        status(state='complete', current_stage=None, completed_models=sorted(completed), finished_at_utc=now())
        report()
    except Exception as exc:
        status(state='failed', error=str(exc))
        report()
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['queue', 'report', 'dry-run'])
    parser.add_argument('--start-now', action='store_true', help='queue only: skip GPU and old PID availability checks before every GPU stage')
    args = parser.parse_args()
    action = args.action
    if args.start_now and action != 'queue':
        parser.error('--start-now requires the queue action')
    if action == 'queue':
        run_queue(args.start_now)
    elif action == 'report':
        check_inputs()
        report()
    else:
        check_inputs()
        for sampling, lora in ARMS:
            subprocess.run(command(['train', 'eval'], sampling, lora) + ['--dry-run'], check=True)
        subprocess.run(command(['zero-shot', 'compare']) + ['--dry-run'], check=True)
