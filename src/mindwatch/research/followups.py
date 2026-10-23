"""Exploratory follow-ups restricted to the locked development population."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from . import analyses
from .privacy import write_json


def run_followups(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if (output / 'summary.json').exists():
        raise FileExistsError('Follow-up version exists; use a new output directory.')
    visits = pd.read_parquet(dataset / 'visits.parquet')
    features = pd.read_parquet(dataset / 'features_14d.parquet')
    partition = pd.read_parquet(dataset / 'split_manifest.parquet')
    split = json.loads((dataset / 'split.json').read_text())
    # Partition before accessing outcomes; no temporal test result is analyzed.
    ids = set(partition.loc[partition.partition.eq('development'), 'sample_id'])
    dev = visits.loc[visits.sample_id.isin(ids)].merge(features, on='sample_id', validate='one_to_one')
    dev['partition'] = 'development'
    dev['sensor_window_end'] = dev.index_date - pd.Timedelta(days=1)
    sensor = [c for c in features if c.startswith('sensor__') and not c.endswith('__coverage')]
    core = [f'sensor__{s}__mean' for s in ['steps','screen_time','distance','sleep_total'] if f'sensor__{s}__mean' in sensor]
    eligible = dev.loc[dev.quality__sensor_eligible.eq(1)].copy()
    as_of = visits.index_date.max()
    retention = analyses.retention_report(visits, as_of=as_of, output_dir=output / 'retention')
    prognosis = pd.read_parquet(dataset / 'prognosis.parquet')
    # Already-purged future outcomes are never brought into development again.
    prognosis = prognosis.loc[prognosis.partition.eq('development') & prognosis.sample_id.isin(set(eligible.sample_id))]
    needed = ['sample_id', 'target_date', 'future_phq9']
    prognostic = eligible.merge(prognosis[needed], on='sample_id', how='inner', validate='one_to_one')
    prog_report = analyses.run_prognosis_regression(
        prognostic, core, cutoff=split['cutoff'],
        provenance={'past_only': True, 'extraction_version': json.loads((dataset/'data_manifest.json').read_text())['data_digest'],
                    'window_end_column': 'sensor_window_end'}, output_dir=output/'prognosis')
    geo_report = analyses.run_geography_validation(eligible, [*core, *[c for c in dev if c.startswith('demo__')]],
                                                   target_column='phq9', output_dir=output/'geography')
    mixed_report = analyses.longitudinal_associations(eligible, core, output_dir=output/'longitudinal')
    summary = {'status': 'exploratory_followups', 'clinical_validation': False, 'temporal_test_evaluated': False,
               'retention': retention.get('status'), 'retention_as_of': str(as_of.date()),
               'prognosis': prog_report.get('status'), 'geography': geo_report.get('status'),
               'longitudinal': mixed_report.get('status')}
    write_json(output/'summary.json', summary)
    return summary
