"""Privacy-safe structural metadata checks: never return unknown row strings."""
import numpy as np
import pandas as pd


def csv_layout(first_row,known_subject_ids=()):
    safe={'id','timestamp','date','regdt','participant_id','userid','phq9_score','-'}
    values=[str(v).strip().lstrip('\ufeff') for v in first_row]
    recognized=[v.lower() for v in values if v.lower() in safe]
    participant_in_first_cell=bool(values and values[0] in set(known_subject_ids))
    return {'columns':len(values),'has_recognized_header':bool(recognized) and not participant_in_first_cell,
            'first_row_is_participant_record':participant_in_first_cell,
            'recognized_header_tokens':sorted(set(recognized)) if not participant_in_first_cell else [],
            'raw_strings_released':False}


def timestamp_grid(values):
    raw=pd.Series(values).astype('string')
    extracted=raw.str.extract(r'((?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?)',expand=False)
    extracted=extracted.str.replace('/','-',regex=False).str.replace('.','-',regex=False)
    # Retain wall-clock timestamps for structural export cadence; never silently
    # turn a mixed UTC/local export into a claimed verified recording date.
    dt=pd.to_datetime(extracted,format='mixed',errors='coerce')
    unique=dt.dropna().drop_duplicates().sort_values()
    delta=unique.diff().dropna().dt.total_seconds()
    offsets=raw.str.extract(r'(Z|[+-]\d{2}:?\d{2})\s*$',expand=False)
    offset_n=int(offsets.nunique(dropna=True))
    return {'rows':len(raw),'valid_timestamps':int(dt.notna().sum()),'unique_timestamps':len(unique),
        'duplicate_timestamps':int(dt.notna().sum()-len(unique)),
        'min_date':str(unique.min().date()) if len(unique) else None,'max_date':str(unique.max().date()) if len(unique) else None,
        'median_wall_clock_step_seconds':float(delta.median()) if len(delta) else None,
        'most_common_wall_clock_step_seconds':float(delta.mode().iloc[0]) if len(delta) else None,
        'explicit_timezone_variants':offset_n,'rows_without_explicit_timezone':int(offsets.isna().sum()),
        'timezone_semantics':'mixed_or_unverified' if offset_n>1 or (offset_n and offsets.isna().any()) else 'not_independently_verified',
        'cadence_interpretation':'export_grid_only; not per-person wear or true acquisition schedule'}
