"""Align reviewed recording features to assessment windows; never expose text."""
from __future__ import annotations
import numpy as np
import pandas as pd


def pool_to_visits(visits, recordings, window=14, allow_same_day=True):
    required = {'recording_id', 'subject_id', 'recording_date', 'task'}
    if not required.issubset(recordings):
        raise ValueError('recording_feature_schema_missing')
    if recordings.recording_id.duplicated().any():
        raise ValueError('duplicate_recording_features')
    columns = [c for c in recordings if c.startswith(('audio__', 'text__'))]
    if not columns or recordings.empty:
        return pd.DataFrame({'sample_id': visits.sample_id}), {'status': 'no_reviewed_recording_features', 'matched_visits': 0}
    if not recordings.task.isin(['free_speech', 'fixed_reading']).all():
        raise ValueError('unverified_task_in_recording_feature_table')
    from .speech import eligibility
    for record in recordings.to_dict('records'):
        reference = record.get('reviewer_reference')
        if not eligibility(record)[0] or not isinstance(reference, str) or not reference.strip() or reference.lower() in {'nan', 'unknown'}:
            raise ValueError('recording_metadata_evidence_not_verified_at_pooling')
        has_text = record['task'] == 'free_speech' and any(pd.notna(record.get(col)) for col in columns if col.startswith('text__'))
        if has_text and (not eligibility(record, semantic=True)[0] or record.get('semantic_transcript_status') != 'approved'):
            raise ValueError('semantic_transcript_evidence_not_verified_at_pooling')
    recordings = recordings.copy()
    recordings['recording_date'] = pd.to_datetime(recordings.recording_date).dt.normalize()
    groups = {key: group for key, group in recordings.groupby('subject_id', sort=False)}
    result, matched, same_day = [], 0, 0
    for row in visits.itertuples(index=False):
        out = {'sample_id': row.sample_id}
        group = groups.get(row.subject_id)
        if group is None:
            result.append(out)
            continue
        lower = row.index_date - pd.Timedelta(days=window)
        upper = row.index_date if allow_same_day else row.index_date - pd.Timedelta(days=1)
        selected = group.loc[group.recording_date.between(lower, upper)]
        if selected.empty:
            result.append(out)
            continue
        matched += 1
        same_day += int(selected.recording_date.eq(row.index_date).any())
        # Keep elicitation tasks distinct so fixed text cannot become semantic signal.
        for task, task_rows in selected.groupby('task'):
            for col in columns:
                if task == 'fixed_reading' and col.startswith('text__'):
                    continue
                prefix, name = col.split('__', 1)
                vals = pd.to_numeric(task_rows[col], errors='coerce')
                out[f'{prefix}__{task}__{name}'] = float(vals.mean()) if vals.notna().any() else np.nan
        out['quality__recording_count'] = len(selected)
        out['quality__audio_same_day_order_unknown'] = int(selected.recording_date.eq(row.index_date).any())
        result.append(out)
    return pd.DataFrame(result), {'status': 'pooled_reviewed_features', 'matched_visits': matched,
                                  'metadata_gates_verified': True,
                                  'same_day_concurrent_visits': same_day,
                                  'prediction_interpretation': 'concurrent_screening; same_day_order_not_inferred'}
