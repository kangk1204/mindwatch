"""Past-only daily windows. No synthetic minute-resolution actigraphy."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from .sources import DAILY, date_series, normalized

CLOCKS = {'sleep_onset', 'wake_time'}


def clock_hours(values: pd.Series) -> pd.Series:
    from .parsing import robust_clock
    return pd.Series(robust_clock(values), index=values.index)


def aggregate_daily(sources, raw_mapping, registry=None, exploratory=True):
    registry = registry or {}
    records, audit = [], []
    for source in sources:
        modality = source.modality
        daily = modality in DAILY or modality.startswith('ema_')
        rule = registry.get(modality, {})
        approved = rule.get('status') == 'verified' and bool(rule.get('unit')) and bool(rule.get('evidence'))
        if source.kind != 'sensor' or not daily:
            continue
        if not exploratory and not approved:
            audit.append({'source': source.source_id, 'modality': modality, 'status': 'held_unverified_metadata'})
            continue
        before = (source.path.stat().st_size, source.path.stat().st_mtime_ns)
        with source.open() as f:
            header = pd.read_csv(f, nrows=0).columns.tolist()
        idcols = {col: raw_mapping[normalized(col)] for col in header[1:] if normalized(col) in raw_mapping}
        if not idcols:
            continue
        if len(set(idcols.values())) != len(idcols):
            raise ValueError('duplicate_subject_columns_need_resolution')
        consumed, negative = 0, 0
        seen_dates = set()
        with source.open() as f:
            for chunk in pd.read_csv(f, usecols=[header[0], *idcols], dtype=str, chunksize=128):
                dates = date_series(chunk[header[0]])
                # Verify these are genuinely daily exports, not a minute signal.
                counts = dates.value_counts()
                current_dates = set(dates.dropna())
                if (counts > 1).any() or seen_dates.intersection(current_dates):
                    raise ValueError('daily_modality_has_multiple_rows_per_day')
                seen_dates.update(current_dates)
                dat = chunk[list(idcols)].copy()
                if modality in CLOCKS:
                    for c in dat:
                        dat[c] = clock_hours(dat[c])
                else:
                    from .parsing import robust_numeric
                    for c in dat:
                        dat[c] = robust_numeric(dat[c])
                arr = dat.to_numpy(dtype=float, na_value=np.nan, copy=True)
                negative += int((arr < 0).sum())
                arr[arr < 0] = np.nan
                if approved and rule.get('valid_range'):
                    lo, hi = rule['valid_range']
                    arr[(arr < lo) | (arr > hi)] = np.nan
                ri, ci = np.where(np.isfinite(arr) & dates.notna().to_numpy()[:, None])
                consumed += len(ri)
                records.append(pd.DataFrame({'subject_id': np.asarray(list(idcols.values()))[ci],
                                             'date': dates.to_numpy()[ri], 'modality': modality,
                                             'value': arr[ri, ci], 'priority': source.export_date or '',
                                             'metadata_verified': approved}))
        if before != (source.path.stat().st_size, source.path.stat().st_mtime_ns):
            raise ValueError('source_changed_during_daily_read')
        audit.append({'source': source.source_id, 'modality': modality, 'observed_cells': consumed,
                      'negative_values_excluded': negative, 'metadata_verified': approved,
                      'status': 'processed_verified' if approved else 'processed_exploratory_export_units'})
    if not records:
        return pd.DataFrame(columns=['subject_id', 'date']), {'sources': audit, 'conflicts_excluded': 0}
    long = pd.concat(records, ignore_index=True)
    keys = ['subject_id', 'date', 'modality']
    # Same date/variable/person is one observation. Highest documented export date wins;
    # contradictory values at the SAME highest priority are excluded, never averaged.
    top = long.groupby(keys, sort=False).priority.transform('max')
    top_rows = long.loc[long.priority.eq(top)].copy()
    nvals = top_rows.groupby(keys, sort=False).value.transform('nunique')
    conflicts = top_rows.loc[nvals > 1, keys].drop_duplicates().shape[0]
    canonical = top_rows.loc[nvals.eq(1)].drop_duplicates(keys)
    wide = canonical.pivot(index=['subject_id', 'date'], columns='modality', values='value').reset_index()
    wide.columns.name = None
    # Circular encodings avoid an artificial discontinuity between 23:59 and 00:01.
    for name in CLOCKS:
        if name in wide:
            radians = wide.pop(name) * 2 * np.pi / 24
            wide[name + '_sin'] = np.sin(radians)
            wide[name + '_cos'] = np.cos(radians)
    return wide.sort_values(['subject_id', 'date']), {
        'sources': audit, 'conflicts_excluded': int(conflicts),
        'duplicate_export_observations_removed': int(len(long) - len(canonical)),
        'canonical_rule': 'latest_documented_export_date; conflicting_same_priority_excluded',
        'zero_policy': 'preserved; no assumption zero means nonwear',
        'high_frequency_status': 'held_until_sampling_units_and_sensor_validity_verified',
    }


def elapsed_delta(mask):
    mask = np.asarray(mask, dtype=bool)
    delta = np.ones(mask.shape, dtype=np.float32)
    for i in range(1, mask.shape[0]):
        delta[i] = 1 + (~mask[i - 1]) * delta[i - 1]
    return delta


def demo_features(visits):
    result = pd.DataFrame({'sample_id': visits.sample_id})
    from .demographics import age_band_series
    # Fixed vocabulary; DOB age is computed at each visit, never from today's date.
    bands,_ = age_band_series(visits.age_group,visits.index_date)
    decade=pd.to_numeric(bands.str.extract(r'^(\d+)-',expand=False),errors='coerce')
    for d in range(0, 110, 10):
        result[f'demo__age_decade_{d}'] = decade.eq(d).fillna(False).astype(float)
    result['demo__age_unknown'] = decade.isna().astype(float)
    for code in ['0', '1', '2']:
        result[f'demo__sex_code_{code}'] = visits.sex.astype(str).eq(code).astype(float)
    result['demo__sex_other_or_unknown'] = (~visits.sex.astype(str).isin(['0', '1', '2'])).astype(float)
    return result


def window_features(visits, daily, window=14, min_days=4):
    sensor_cols = [c for c in daily if c not in {'subject_id', 'date'} and not c.startswith('ema_')]
    ema_cols = [c for c in daily if c.startswith('ema_')]
    groups = {k: g.set_index('date') for k, g in daily.groupby('subject_id', sort=False)}
    values = np.full((len(visits), window, len(sensor_cols)), np.nan, dtype=np.float32)
    rows = []
    for index, visit in enumerate(visits.itertuples(index=False)):
        # Current date excluded even when a full timestamp might exist.
        dates = pd.date_range(visit.index_date - pd.Timedelta(days=window), periods=window, freq='D')
        subject = groups.get(visit.subject_id)
        if subject is None:
            block = pd.DataFrame(np.nan, index=dates, columns=[*sensor_cols, *ema_cols])
        else:
            block = subject.reindex(dates)
        vals = block[sensor_cols].to_numpy(dtype=float)
        values[index] = vals
        seen_days = int(np.isfinite(vals).any(axis=1).sum())
        row = {'sample_id': visit.sample_id, 'quality__sensor_observed_days': seen_days,
               'quality__sensor_eligible': int(seen_days >= min_days)}
        for col in [*sensor_cols, *ema_cols]:
            a = block[col].to_numpy(dtype=float)
            ok = np.isfinite(a)
            prefix = 'ema__' if col.startswith('ema_') else 'sensor__'
            name = prefix + col
            row[name + '__coverage'] = float(ok.mean())
            row[name + '__mean'] = float(a[ok].mean()) if ok.any() else np.nan
            row[name + '__std'] = float(a[ok].std()) if ok.sum() >= 2 else np.nan
            row[name + '__last'] = float(a[ok][-1]) if ok.any() else np.nan
            row[name + '__min'] = float(a[ok].min()) if ok.any() else np.nan
            row[name + '__max'] = float(a[ok].max()) if ok.any() else np.nan
            row[name + '__slope'] = float(np.polyfit(np.where(ok)[0], a[ok], 1)[0]) if ok.sum() >= 2 else np.nan
        rows.append(row)
    features = pd.DataFrame(rows).merge(demo_features(visits), on='sample_id', validate='one_to_one')
    mask = np.isfinite(values)
    delta = np.stack([elapsed_delta(m) for m in mask]) if len(mask) else np.empty_like(values)
    arrays = {'sample_ids': visits.sample_id.to_numpy(dtype=str), 'values': values,
              'mask': mask.astype(np.float32), 'delta': delta,
              'feature_names': np.asarray(sensor_cols, dtype=str)}
    return features, arrays


def personal_change(daily, min_history=14, history_days=28):
    """Past-only robust z; no clinical-event label is invented."""
    outputs = []
    columns = [c for c in daily if c not in {'subject_id', 'date'} and not c.startswith('ema_')]
    for pid, group in daily.groupby('subject_id', sort=False):
        g = group.sort_values('date').set_index('date')[columns]
        median = g.rolling(f'{history_days}D', closed='left', min_periods=min_history).median()
        q1 = g.rolling(f'{history_days}D', closed='left', min_periods=min_history).quantile(.25)
        q3 = g.rolling(f'{history_days}D', closed='left', min_periods=min_history).quantile(.75)
        scale = (q3 - q1) / 1.349
        z = (g - median) / scale.where(scale > 1e-6)
        result = z.add_prefix('robust_z__').reset_index()
        result['subject_id'] = pid
        result['change_score'] = np.sqrt(z.pow(2).mean(axis=1)).to_numpy()
        result['clinical_validation_status'] = 'exploratory_unvalidated_change_signal'
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
