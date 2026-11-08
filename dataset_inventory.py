#!/usr/bin/env python3
"""Read-only inventory of original SAP Train/Dev audio and participant JSONs.

Python standard library only. No original file is modified. Output is aggregate
statistics; no transcripts, participant IDs, or absolute paths are exported.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import wave


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def stats_summary(xs):
    if not xs:
        return {'n': 0}
    a = sorted(xs)
    def percentile(p):
        pos = (len(a) - 1) * p
        low = int(pos)
        high = min(low + 1, len(a) - 1)
        return round(a[low] + (a[high] - a[low]) * (pos - low), 3)
    return {'n':len(a),'min':round(a[0],3),'p25':percentile(.25),
            'median':percentile(.5),'p75':percentile(.75),'p95':percentile(.95),
            'max':round(a[-1],3),'mean':round(sum(a)/len(a),3)}


def is_original_shard(path, split):
    return path.is_dir() and bool(re.fullmatch(r'SpeechAccessibility_\d{4}-\d{2}-\d{2}_\d+', path.name, flags=re.I))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(r'C:\SAP downsampled database'))
    ap.add_argument('--out', type=Path, default=Path('inventory_results'))
    ap.add_argument('--audio-headers', action='store_true', help='Read WAV headers for durations/sample rates/channels; slower')
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    if not root.is_dir():
        ap.error(f'Dataset root does not exist: {root}')
    if out == root or root in out.parents:
        ap.error('Output directory must be outside the dataset root (avoid modifying SAP).')
    out.mkdir(parents=True, exist_ok=True)

    summary = {'dataset_root_name': root.name, 'source_splits': {}, 'excluded_directories': {},
               'json_files_scanned':0, 'json_files_with_files_array':0, 'metadata_references':0,
               'unique_record_keys':0, 'recordings_missing_audio':0, 'recordings_missing_transcript':0,
               'json_parse_errors':0, 'audio_header_errors':0,
               'notes':['Only original shard dirs named SpeechAccessibility_YYYY-MM-DD_NNN are scanned.',
                        'PD subsets, ZIPs and Only_Json directories are excluded.',
                        'Speaker identity comes from Contributor ID; same ID may occur in multiple blocks.',
                        'Audio statistics from headers are optional; no audio waveforms decoded.']}
    errors=[]
    split_etiology_speakers=defaultdict(set)
    split_etiology_records=Counter()
    split_category=Counter()
    etiology_speakers=defaultdict(set)
    speaker_splits=defaultdict(set)
    speaker_blocks=defaultdict(set)
    speaker_labels=defaultdict(set)
    categories=Counter(); subcategories=Counter()
    prompt_differences=Counter(); transcript_markups=Counter()
    transcript_lens=[]; audio_lengths=[]
    sample_rates=Counter(); channels=Counter(); durations=[]
    duplicate_keys=Counter(); paths_seen=set(); ref_name_occurrences=Counter()
    file_metadata_fields=Counter(); participant_metadata_fields=Counter(); prompt_fields=Counter(); rating_fields=Counter()
    rating_presence=Counter(); audio_count_by_split=Counter(); json_count_by_split=Counter()
    known_wav_names=set()

    for split in ('Train','Dev'):
        split_dir = root / split
        if not split_dir.is_dir():
            errors.append(f'Missing split directory: {split}')
            continue
        original_shards = [p for p in split_dir.iterdir() if is_original_shard(p, split)]
        excluded = [p.name for p in split_dir.iterdir() if p.is_dir() and p not in original_shards]
        summary['excluded_directories'][split] = excluded
        summary['source_splits'][split] = {'original_shards':len(original_shards), 'shard_names': sorted(p.name for p in original_shards)}
        print(f'[{split}] original shards: {len(original_shards)}; ignored other dirs: {len(excluded)}', flush=True)
        for shard in sorted(original_shards):
            for jp in shard.rglob('*.json'):
                summary['json_files_scanned']+=1
                json_count_by_split[split]+=1
                try:
                    meta=json.loads(jp.read_text(encoding='utf-8-sig'))
                    if not isinstance(meta,dict):
                        errors.append(f'Unexpected JSON shape in {split}/{shard.name}'); continue
                except (OSError,UnicodeError,ValueError) as e:
                    summary['json_parse_errors']+=1
                    errors.append(f'JSON read/parse error in {split}/{shard.name}: {type(e).__name__}: {str(e)[:120]}')
                    continue
                participant_metadata_fields.update(meta.keys())
                items=meta.get('Files')
                if not isinstance(items,list):
                    continue
                summary['json_files_with_files_array']+=1
                sid=str(meta.get('Contributor ID') or '').strip()
                if not sid:
                    errors.append(f'Missing Contributor ID in {split}/{shard.name}')
                    continue
                etiology=str(meta.get('Etiology') or 'MISSING').strip() or 'MISSING'
                etiology_speakers[etiology].add(sid)
                split_etiology_speakers[(split,etiology)].add(sid)
                speaker_splits[sid].add(split)
                speaker_labels[sid].add(etiology)
                speaker_blocks[sid].add((split,str(meta.get('BlockNumber')),shard.name))
                for item in items:
                    if not isinstance(item,dict): continue
                    file_metadata_fields.update(item.keys())
                    prompt=item.get('Prompt') or {}
                    if not isinstance(prompt,dict):prompt={}
                    prompt_fields.update(prompt.keys())
                    rating=item.get('Ratings')
                    rating_presence['nonempty' if rating else 'empty_or_missing']+=1
                    if isinstance(rating,list):
                        for el in rating:
                            if isinstance(el,dict):rating_fields.update(el.keys())
                    fn=item.get('Filename')
                    if not isinstance(fn,str) or not fn.strip():
                        errors.append(f'Missing filename in {split}/{shard.name}'); continue
                    summary['metadata_references']+=1
                    key=(split,sid,fn)
                    duplicate_keys[key]+=1
                    ref_name_occurrences[(split,fn)]+=1
                    split_etiology_records[(split,etiology)]+=1
                    cat=str(prompt.get('Category Description') or 'MISSING')
                    subcat=str(prompt.get('Sub Category Description') or 'MISSING')
                    categories[cat]+=1; subcategories[subcat]+=1
                    split_category[(split,cat)]+=1
                    ref=prompt.get('Transcript')
                    text=ref if isinstance(ref,str) else ''
                    if not text.strip(): summary['recordings_missing_transcript']+=1
                    transcript_lens.append(len(text.split()))
                    ptxt=prompt.get('Prompt Text')
                    if isinstance(ptxt,str):
                        prompt_differences['different' if ptxt!=text else 'same']+=1
                    if '[' in text or ']' in text:transcript_markups['square_brackets']+=1
                    if '(' in text or ')' in text:transcript_markups['parentheses']+=1
                    if '{' in text or '}' in text:transcript_markups['curly_braces']+=1
                    wav=jp.parent / fn
                    if not wav.is_file():
                        summary['recordings_missing_audio']+=1
                        if len(errors)<200: errors.append(f'Missing WAV for JSON in {split}/{shard.name}: {fn}')
                        continue
                    audio_count_by_split[split]+=1
                    if args.audio_headers:
                        try:
                            with wave.open(str(wav),'rb') as wf:
                                dur=wf.getnframes()/wf.getframerate()
                                durations.append(dur)
                                audio_lengths.append(dur)
                                sample_rates[wf.getframerate()]+=1
                                channels[wf.getnchannels()]+=1
                        except (wave.Error,OSError,EOFError,ZeroDivisionError) as e:
                            summary['audio_header_errors']+=1
                            if len(errors)<200:errors.append(f'WAV header error in {split}/{shard.name}: {fn}: {type(e).__name__}')
        print(f'  JSON scanned: {json_count_by_split[split]:,}; referenced WAV found: {audio_count_by_split[split]:,}',flush=True)

    summary['unique_record_keys']=len(duplicate_keys)
    summary['duplicate_record_keys']=sum(v-1 for v in duplicate_keys.values() if v>1)
    summary['duplicate_record_key_groups']=sum(v>1 for v in duplicate_keys.values())
    summary['unique_speakers']=len(speaker_splits)
    summary['speakers_in_both_train_and_dev']=sum(len(v)>1 for v in speaker_splits.values())
    summary['speakers_with_multiple_etiologies']=sum(len(v)>1 for v in speaker_labels.values())
    summary['json_count_by_split']=dict(json_count_by_split)
    summary['matched_audio_references_by_split']=dict(audio_count_by_split)
    summary['transcript_word_count']=stats_summary(transcript_lens)
    summary['prompt_vs_transcript']=dict(prompt_differences)
    summary['transcript_markup_counts']=dict(transcript_markups)
    summary['ratings_presence']=dict(rating_presence)
    summary['sample_rates']=dict(sample_rates)
    summary['channels']=dict(channels)
    summary['audio_duration_seconds']=stats_summary(durations)
    summary['total_hours_from_audio_headers']=round(sum(durations)/3600,3) if args.audio_headers else None
    summary['audio_headers_enabled']=args.audio_headers
    summary['metadata_fields']={'participant':dict(participant_metadata_fields),'file':dict(file_metadata_fields),
                                'prompt':dict(prompt_fields),'rating':dict(rating_fields)}
    summary['warnings_sample']=errors[:50]

    etiologies=sorted(set(etiology_speakers)|{k[1] for k in split_etiology_records})
    erows=[]
    for et in etiologies:
        erows.append({'etiology':et,'train_speakers':len(split_etiology_speakers[('Train',et)]),
                      'dev_speakers':len(split_etiology_speakers[('Dev',et)]),
                      'unique_speakers_all':len(etiology_speakers[et]),
                      'train_recording_references':split_etiology_records[('Train',et)],
                      'dev_recording_references':split_etiology_records[('Dev',et)]})
    write_csv(out/'etiology_distribution.csv',erows,['etiology','train_speakers','dev_speakers','unique_speakers_all','train_recording_references','dev_recording_references'])
    write_csv(out/'category_distribution.csv',
              [{'split':sp,'category':ca,'recording_references':n} for (sp,ca),n in sorted(split_category.items())],
              ['split','category','recording_references'])
    write_csv(out/'subcategory_distribution.csv',
              [{'subcategory':ca,'recording_references':n} for ca,n in sorted(subcategories.items())],
              ['subcategory','recording_references'])
    summary['speaker_block_count_distribution']=stats_summary([len(v) for v in speaker_blocks.values()])
    summary['speaker_overlap_detected']=summary['speakers_in_both_train_and_dev']>0
    summary['possible_duplicate_metadata_warning']='Review duplicate record counts before building training manifests.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n=== INVENTORY SUMMARY ===')
    print('Unique contributors:',summary['unique_speakers'])
    print('Train/Dev shared contributors:',summary['speakers_in_both_train_and_dev'])
    print('Record references:',summary['metadata_references'])
    print('Duplicate record keys:',summary['duplicate_record_keys'])
    print('Unmatched WAV references:',summary['recordings_missing_audio'])
    print('Empty transcript references:',summary['recordings_missing_transcript'])
    print('JSON parse errors:',summary['json_parse_errors'])
    if args.audio_headers:print('Total matched audio hours:',summary['total_hours_from_audio_headers'])
    print('\nEtiology summary (includes speaker counts):')
    for row in erows: print(row)
    print('\nFiles saved to:', out)
    print('Caution: This inventories WAVs referenced by JSON; it does not search for unreferenced WAVs.')

if __name__=='__main__':main()
