"""Aggregate-only reports with an independent recomputation from private OOFs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from .privacy import secure_dir, write_json, sha256


def cluster_auc_interval(table, n_boot=2000, seed=42):
    subjects = table.subject_id.to_numpy()
    unique, inv, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    y, p = table.target.to_numpy(), table.probability.to_numpy()
    rng, result = np.random.default_rng(seed), []
    for _ in range(n_boot):
        sampled = rng.integers(0,len(unique),len(unique))
        multiplicity = np.bincount(sampled,minlength=len(unique))
        weight = multiplicity[inv]/counts[inv]
        present = weight>0
        if np.unique(y[present]).size == 2:
            result.append(roc_auc_score(y[present],p[present],sample_weight=weight[present]))
    return ([float(v) for v in np.quantile(result,[.025,.975])] if result else [None,None]), len(result)


def summarize_benchmarks(project, output):
    project, output = Path(project), secure_dir(Path(output))
    rows, sources = [], []
    for run, family in [('sensor_phq_v1','tabular'),('sequences_phq_v1','sequence')]:
        directory = project/'artifacts/private'/run
        report = json.loads((directory/'development_report.json').read_text())
        oof = pd.read_parquet(directory/'oof_predictions.parquet')
        sources.append({'run':run,'oof_sha256':sha256(directory/'oof_predictions.parquet'),
                        'report_sha256':sha256(directory/'development_report.json'),
                        'analysis_type':report['analysis_type'],'temporal_test_evaluated':False})
        for record in report['models']:
            if record['status'] != 'ok':
                continue
            if family=='sequence' and record['model'] in {'prevalence','demographic'}:
                continue
            table = oof.loc[oof.model.eq(record['model'])].copy()
            if table.sample_id.duplicated().any():
                raise ValueError('duplicate_oof_sample_in_summary')
            _, inverse, counts=np.unique(table.subject_id,return_inverse=True,return_counts=True)
            weight=1/counts[inverse]
            auc=roc_auc_score(table.target,table.probability,sample_weight=weight)
            ap=average_precision_score(table.target,table.probability,sample_weight=weight)
            brier=brier_score_loss(table.target,table.probability,sample_weight=weight)
            for name, value in [('auroc',auc),('auprc',ap),('brier',brier)]:
                if not np.isclose(value,record['pooled_oof_metrics'][name],atol=1e-10):
                    raise ValueError('independent_oof_metric_recalculation_mismatch')
            ci,valid=cluster_auc_interval(table)
            rows.append({'family':family,'model':record['model'],'auroc':auc,'ci_lower':ci[0],'ci_upper':ci[1],
                         'auprc':ap,'brier':brier,'subjects':table.subject_id.nunique(),'visits':len(table),
                         'positive_visits':int(table.target.sum()),'bootstrap_valid':valid})
    summary=pd.DataFrame(rows)
    summary.to_csv(output/'benchmark_summary.csv',index=False)
    (output/'benchmark_summary.csv').chmod(0o600)
    write_json(output/'benchmark_provenance.json',{'sources':sources,'seed':42,'n_bootstrap':2000,
        'independent_recalculation':'sklearn_metrics_with_explicit_equal_participant_weights',
        'population':'strict_development_only; 14d sensor coverage>=4 days; provisional sensor units',
        'claim':'exploratory_software_and_cohort_feasibility_only; no clinical superiority claim'})
    return summary
