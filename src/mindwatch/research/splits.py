"""Immutable enrollment/time split, with all future outcomes purged."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .privacy import write_json, write_parquet


def make_temporal_split(visits, test_fraction=.20):
    if not 0 < test_fraction < 1:
        raise ValueError('invalid_test_fraction')
    people = visits.groupby('subject_id').enrolled_date.min().sort_values()
    cutoff = people.iloc[min(len(people) - 1, int(len(people) * (1 - test_fraction)))]
    test_ids = set(people[people >= cutoff].index)
    partitions = np.where(visits.subject_id.isin(test_ids), 'temporal_test',
                          np.where(visits.index_date < cutoff, 'development', 'excluded_after_cutoff'))
    out = visits[['sample_id', 'subject_id', 'index_date']].copy()
    out['partition'] = partitions
    if set(out.loc[out.partition.eq('development'), 'subject_id']) & test_ids:
        raise ValueError('subject_leakage')
    canonical = out.sort_values('sample_id').to_csv(index=False, date_format='%Y-%m-%d').encode()
    spec = {'cutoff': str(pd.Timestamp(cutoff).date()), 'requested_test_fraction': test_fraction,
            'realized_test_subjects': len(test_ids), 'development_subjects': len(people) - len(test_ids),
            'tie_rule': 'all_people_at_cutoff_are_test',
            'partition_counts': out.partition.value_counts().to_dict(),
            'digest': hashlib.sha256(canonical).hexdigest(),
            'enrollment_basis': 'first_observed_survey_proxy; registration_semantics_need_verification'}
    return out, spec


def lock_split(visits, output_dir, test_fraction=.20):
    manifest, spec = make_temporal_split(visits, test_fraction)
    path = Path(output_dir) / 'split.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['digest'] != spec['digest']:
            raise ValueError('locked_split_changed_create_new_version_instead')
        return pd.read_parquet(Path(output_dir) / 'split_manifest.parquet'), old
    write_parquet(Path(output_dir) / 'split_manifest.parquet', manifest)
    write_json(path, spec)
    return manifest, spec


def prognosis_split(prognosis, manifest, cutoff):
    out = prognosis.merge(manifest[['sample_id', 'partition']], on='sample_id', validate='one_to_one')
    # A target occurring beyond training cutoff cannot be used for development.
    purge = out.partition.eq('development') & (out.target_date >= pd.Timestamp(cutoff))
    out.loc[purge, 'partition'] = 'excluded_future_outcome_after_cutoff'
    return out
