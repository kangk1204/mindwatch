"""Validated item totals, cohort tables, and future outcomes."""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd
from .privacy import token
from .sources import normalized
from .demographics import age_band_series

SCALES = {
    'phq9': ([f'PHQ9_{i}' for i in range(1, 10)], 'PHQ9_Score', 3, 10),
    'gad7': ([f'GAD7_{i}' for i in range(1, 8)], 'GAD7_Score', 3, 10),
    'isi': (['ISI_1_1', 'ISI_1_2', 'ISI_1_3', 'ISI_2', 'ISI_3', 'ISI_4', 'ISI_5'], 'ISI_Score', 4, 15),
}


def validate_scale(df, columns, provided_column, maximum):
    if not set(columns).issubset(df):
        return pd.Series(np.nan, index=df.index), {'valid': 0, 'missing_columns': True}
    items = df[columns].apply(pd.to_numeric, errors='coerce')
    legal = items.notna().all(axis=1) & items.ge(0).all(axis=1) & items.le(maximum).all(axis=1)
    legal &= items.eq(items.round()).all(axis=1)
    totals = items.sum(axis=1, min_count=len(columns))
    provided = pd.to_numeric(df.get(provided_column, pd.Series(np.nan, index=df.index)), errors='coerce')
    matched = legal & provided.notna() & totals.eq(provided)
    return totals.where(matched), {'valid': int(matched.sum()), 'invalid_items': int((~legal).sum()),
                                  'provided_missing': int(provided.isna().sum()),
                                  'provided_mismatch': int((legal & provided.notna() & ~totals.eq(provided)).sum())}


def build_visits(raw_tables, key):
    tables, audits, raw_mapping = [], [], {}
    for wave, (source, raw) in enumerate(raw_tables, 1):
        d = pd.DataFrame(index=raw.index)
        rawid = raw['_raw_id'].map(normalized)
        raw_mapping.update({x: token(key, x) for x in rawid if x})
        d['subject_id'] = rawid.map(lambda x: token(key, x) if x else '')
        d['index_date'] = raw['_date']
        d['wave'] = wave
        d['sample_id'] = [token(key, f'{p}|{dt}|{wave}', 'sample') for p, dt in zip(d.subject_id, d.index_date)]
        d['age_group'],age_format = age_band_series(raw.get('Age', pd.Series('', index=raw.index)).map(normalized),d.index_date)
        d['sex'] = raw.get('Sex', pd.Series('', index=raw.index)).map(normalized)
        # Geography is never mislabelled as recruitment institution.
        d['region_group'] = raw.get('area', pd.Series('', index=raw.index)).map(
            lambda x: token(key, normalized(x), 'geography') if normalized(x) else '')
        audit = {'wave': wave, 'source_id': source.source_id, 'rows': len(d), 'scales': {}}
        audit['age_input_formats']=age_format.value_counts().to_dict()
        audit['age_band_unknown']=int(d.age_group.eq('unknown').sum())
        for scale, (cols, provided, upper, threshold) in SCALES.items():
            d[scale], audit['scales'][scale] = validate_scale(raw, cols, provided, upper)
            d[f'y_{scale}'] = (d[scale] >= threshold).astype(float).where(d[scale].notna())
        for col, lower, upper in [('Height', 100, 230), ('Weight', 25, 300)]:
            vals = pd.to_numeric(raw.get(col, pd.Series(np.nan, index=raw.index)), errors='coerce')
            audit[f'{col.lower()}_outside_qc_range'] = int((vals.notna() & ~vals.between(lower, upper)).sum())
        audit['empty_id'] = int(rawid.eq('').sum())
        audit['invalid_date'] = int(d.index_date.isna().sum())
        audits.append(audit)
        tables.append(d.loc[rawid.ne('') & d.index_date.notna()])
    visits = pd.concat(tables, ignore_index=True).sort_values(['subject_id', 'index_date', 'wave'])
    if visits.duplicated(['subject_id', 'index_date']).any():
        raise ValueError('duplicate_subject_assessment_date_requires_resolution')
    if visits.sample_id.duplicated().any():
        raise ValueError('duplicate_sample_id')
    visits['enrolled_date'] = visits.groupby('subject_id').index_date.transform('min')
    history = pd.DataFrame({'sample_id': visits.sample_id})
    for scale in SCALES:
        history[f'history__previous_{scale}'] = visits.groupby('subject_id')[scale].shift(1)
    prior = visits.groupby('subject_id').index_date.shift(1)
    history['history__days_since_previous'] = (visits.index_date - prior).dt.days
    prognosis = visits[['sample_id', 'subject_id', 'index_date', 'phq9']].copy()
    prognosis['target_date'] = visits.groupby('subject_id').index_date.shift(-1)
    prognosis['future_phq9'] = visits.groupby('subject_id').phq9.shift(-1)
    prognosis['horizon_days'] = (prognosis.target_date - prognosis.index_date).dt.days
    eligible = prognosis.horizon_days.between(90, 180) & prognosis.phq9.notna() & prognosis.future_phq9.notna()
    prognosis['worsening_ge5'] = (prognosis.future_phq9 - prognosis.phq9 >= 5).astype(float).where(eligible)
    prognosis['onset_ge10'] = (prognosis.future_phq9 >= 10).astype(float).where(eligible & prognosis.phq9.lt(10))
    prognosis['future_phq9'] = prognosis.future_phq9.where(eligible)
    report = {'subjects': visits.subject_id.nunique(), 'visits': len(visits), 'waves': audits,
              'enrollment_date_basis': 'first_observed_survey_proxy_not_verified_registration',
              'assessment_timestamp_semantics': 'export_timestamp_not_independently_verified',
              'prognosis_pairs_90_180_days': int(eligible.sum()),
              'age_handling': 'DOB_to_age_at_assessment_to_fixed_decade; explicit_age_bands_preserved; DOB_not_serialized',
              'height_weight_handling': 'QC_counts_only_not_primary_predictors'}
    return visits.reset_index(drop=True), history.reset_index(drop=True), prognosis.reset_index(drop=True), raw_mapping, report
