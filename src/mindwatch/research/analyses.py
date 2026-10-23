"""Prespecified secondary analyses; all individual results remain local/private.

These functions print neither participant records nor subgroup category values.
They never read files themselves and require callers to supply prepared tables.
Development-only analyses filter partition membership before touching outcomes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .benchmarks import _logistic_regularization, participant_calibration_split
from .evaluation import binary_metrics, holm_adjust, paired_cluster_bootstrap, participant_weights, select_sensitivity_threshold
from .privacy import write_json, write_parquet


def _save(report, output_dir, basename, tables=None):
    if output_dir is not None:
        directory = Path(output_dir)
        write_json(directory / f"{basename}.json", report)
        for name, table in (tables or {}).items():
            write_parquet(directory / f"{name}.parquet", table)
    return report


def _development(frame: pd.DataFrame) -> pd.DataFrame:
    if "partition" not in frame:
        raise ValueError("Explicit partition membership is required")
    # Outcomes of excluded/test rows are never accessed in this function.
    result = frame.loc[frame.partition.eq("development")].copy()
    if not {"subject_id", "sample_id"}.issubset(result):
        raise ValueError("Sample and participant identifiers are required")
    if result.sample_id.isna().any() or result.sample_id.duplicated().any() or result.subject_id.isna().any():
        raise ValueError("Unique samples and present participant identifiers required")
    test_subjects = set(frame.loc[frame.partition.eq("temporal_test"), "subject_id"])
    if set(result.subject_id) & test_subjects:
        raise ValueError("Participants cannot overlap development and temporal test")
    return result.reset_index(drop=True)


def _binary_target(values: pd.Series, column: str) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if column in {"phq9", "gad7", "isi"}:
        upper, threshold = {"phq9": (27, 10), "gad7": (21, 10), "isi": (28, 15)}[column]
        valid = np.isfinite(values) & (values >= 0) & (values <= upper) & (values == np.floor(values))
        return np.where(valid, (values >= threshold).astype(float), np.nan)
    if np.any(np.isfinite(values) & ~np.isin(values, [0, 1])):
        raise ValueError("Outcome must be binary or a recognized scale")
    return values


def _predictor_columns(columns: Sequence[str], prefixes=("sensor__", "audio__", "text__", "ema__", "history__", "demo__", "quality__")):
    if not columns or len(set(columns)) != len(columns) or any(not col.startswith(prefixes) for col in columns):
        raise ValueError("Unique predictors with approved prefixes are required")


def paired_modality_comparisons(oof_tables: Mapping[str, pd.DataFrame], comparisons: Sequence[tuple[str, str]],
                                output_dir=None, n_bootstrap: int = 2000, seed: int = 42,
                                metric: str = "auroc") -> dict:
    """Paired augmented-minus-baseline contrasts on identical common OOF samples.

    Each table is one model, with sample_id, subject_id, target, probability and
    outer_fold. Comparisons form a single prespecified Holm family; unavailable
    contrasts remain in the multiplicity family rather than being dropped.
    """
    if len(set(comparisons)) != len(comparisons) or not comparisons:
        raise ValueError("A nonempty unique prespecified comparison family is required")
    required = {"sample_id", "subject_id", "target", "probability", "outer_fold"}
    for table in oof_tables.values():
        if not required.issubset(table) or table.sample_id.duplicated().any() or table.sample_id.isna().any():
            raise ValueError("One unique OOF prediction per sample is required")
        if "partition" in table and not table.partition.eq("development").all():
            raise ValueError("Modality OOF comparisons must contain development rows only")
    results = []
    for index, (augmented, baseline) in enumerate(comparisons):
        name = f"comparison_{index+1:03d}"
        record = {"comparison": name, "augmented": augmented, "baseline": baseline}
        if augmented not in oof_tables or baseline not in oof_tables:
            results.append({**record, "status": "oof_table_unavailable", "p_value": None})
            continue
        left, right = oof_tables[augmented], oof_tables[baseline]
        common = left[list(required)].merge(right[list(required)], on="sample_id", suffixes=("_a", "_b"), validate="one_to_one")
        if not common.subject_id_a.equals(common.subject_id_b):
            raise ValueError("Paired samples have inconsistent participant identity")
        if not np.array_equal(common.outer_fold_a, common.outer_fold_b):
            raise ValueError("Paired OOF predictions require identical outer folds")
        yy, zz = common.target_a.to_numpy(dtype=float), common.target_b.to_numpy(dtype=float)
        if not np.array_equal(yy, zz, equal_nan=True):
            raise ValueError("Paired OOF predictions have inconsistent targets")
        valid = np.isfinite(yy) & np.isfinite(common.probability_a) & np.isfinite(common.probability_b)
        paired = common.loc[valid]
        inference = paired_cluster_bootstrap(paired.target_a, paired.probability_a, paired.probability_b,
                                             paired.subject_id_a, metric=metric, n_bootstrap=n_bootstrap, seed=seed+index)
        results.append({**record, **inference, "n_common_samples": len(common), "n_paired_predicted_samples": len(paired),
                        "n_augmented_samples": len(left), "n_baseline_samples": len(right),
                        "estimand": "performance_difference_on_identical_labelled_predicted_samples"})
    adjusted = holm_adjust({record["comparison"]: record.get("p_value") for record in results})
    for record in results:
        record["p_holm"] = adjusted[record["comparison"]]
    return _save({"status": "ok", "metric": metric, "holm_family_size": len(comparisons), "comparisons": results,
                  "selection_scope": "development_oof_only", "p_values": "approximate_centred_cluster_bootstrap"},
                 output_dir, "modality_comparisons")


def subgroup_reports(frame: pd.DataFrame, target_column: str = "target", probability_column: str = "probability",
                     coverage_column: str | None = None, output_dir=None, min_subjects: int = 30,
                     min_events: int = 10, group_columns=("age_group", "sex", "region_group")) -> dict:
    """Descriptive subgroup discrimination/calibration, with small-N status.

    Supply all labelled development rows including NaN predictions for coverage.
    Group names in the returned report are opaque; private subgroup_key.parquet
    maps them back to the supplied categories. No category is printed.
    """
    if min_subjects < 1 or min_events < 1:
        raise ValueError("Positive subgroup support thresholds required")
    data = _development(frame)
    y = _binary_target(data[target_column], target_column)
    p = pd.to_numeric(data[probability_column], errors="coerce").to_numpy(dtype=float)
    groups = list(group_columns)
    if coverage_column is not None:
        coverage = pd.to_numeric(data[coverage_column], errors="coerce")
        if (coverage.notna() & ~coverage.between(0, 1)).any():
            raise ValueError("Observation coverage must lie between zero and one")
        data["_coverage_group"] = pd.cut(coverage, [-np.inf, .25, .5, .75, np.inf],
                                          labels=["[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1]"], right=False).astype("string").fillna("missing")
        groups.append("_coverage_group")
    rows, key = [], []
    for grouping in groups:
        if grouping not in data:
            rows.append({"grouping": grouping, "status": "metadata_unavailable"})
            continue
        values = data[grouping].fillna("missing").astype(str).replace("", "missing")
        for index, value in enumerate(sorted(values.unique())):
            opaque = f"group_{index+1:03d}"
            keep = values.eq(value).to_numpy()
            metrics = binary_metrics(y[keep], p[keep], data.loc[keep, "subject_id"], threshold=None)
            labelled = keep & np.isfinite(y)
            pos = data.loc[labelled & (y == 1), "subject_id"].nunique()
            neg = data.loc[labelled & (y == 0), "subject_id"].nunique()
            small = metrics["n_labelled_participants"] < min_subjects or min(pos, neg) < min_events
            if small:
                for field in ("calibration_intercept", "calibration_slope"):
                    metrics[field] = None
                metrics["calibration_status"] = "suppressed_small_n"
            rows.append({"grouping": "coverage" if grouping == "_coverage_group" else grouping,
                         "group": opaque, **metrics, "analysis_status": "small_n_descriptive_only" if small else "descriptive",
                         "positive_participants": int(pos), "negative_participants": int(neg)})
            key.append({"grouping": grouping, "group": opaque, "category": value})
    return _save({"status": "ok", "groups": rows, "min_subjects": min_subjects, "min_events_per_class": min_events,
                  "inference": "descriptive_no_subgroup_significance_tests", "category_key": "private_subgroup_key.parquet"},
                 output_dir, "subgroups", {"private_subgroup_key": pd.DataFrame(key)})


def run_geography_validation(frame: pd.DataFrame, feature_columns: Sequence[str], target_column: str = "phq9",
                             output_dir=None, seed: int = 42, calibration_fraction: float = .2,
                             target_sensitivity: float = .8, min_test_subjects: int = 10) -> dict:
    """Leave-one-geography-out development validation, fixed elastic net.

    This is geographic transportability, never institution/external validation.
    All participants from the held geography are absent from fit and calibration.
    Region-changing subjects are excluded because their domain is ambiguous.
    """
    _predictor_columns(feature_columns)
    if not 0 < calibration_fraction < .5 or not 0 < target_sensitivity <= 1 or min_test_subjects < 1:
        raise ValueError("Invalid calibration or geographic support settings")
    data = _development(frame)
    if "region_group" not in data:
        return _save({"status": "geography_metadata_unavailable"}, output_dir, "geography_validation")
    data["region_group"] = data.region_group.fillna("").astype(str)
    conflicting = set(data.groupby("subject_id").region_group.nunique().loc[lambda values: values > 1].index)
    excluded = data.subject_id.isin(conflicting) | data.region_group.eq("")
    n_excluded = int(data.loc[excluded, "subject_id"].nunique())
    data = data.loc[~excluded].reset_index(drop=True)
    y = _binary_target(data[target_column], target_column)
    observed = np.isfinite(data[list(feature_columns)].to_numpy(dtype=float)).any(1)
    regions = sorted(data.region_group.unique())
    if len(regions) < 2:
        return _save({"status": "insufficient_geographies", "n_geographies": len(regions)}, output_dir, "geography_validation")
    reports, predictions, roles, keys = [], [], [], []
    for number, region in enumerate(regions):
        domain = f"geography_{number+1:03d}"
        keys.append({"geography": domain, "region_group": region})
        held = np.where(data.region_group.eq(region) & np.isfinite(y))[0]
        candidates = np.where(data.region_group.ne(region) & np.isfinite(y) & observed)[0]
        record = {"geography": domain, "n_test_subjects": int(data.iloc[held].subject_id.nunique())}
        if record["n_test_subjects"] < min_test_subjects:
            reports.append({**record, "status": "small_test_geography"})
            continue
        split = participant_calibration_split(y[candidates], data.iloc[candidates].subject_id, calibration_fraction, seed+number)
        if split is None:
            reports.append({**record, "status": "insufficient_fit_calibration_class_support"})
            continue
        fit, calibration = candidates[split[0]], candidates[split[1]]
        X = data[list(feature_columns)].to_numpy(dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                              LogisticRegression(solver="saga", **_logistic_regularization(.5), C=.1, max_iter=3000, random_state=seed))
        try:
            model.fit(X[fit], y[fit], logisticregression__sample_weight=participant_weights(data.iloc[fit].subject_id))
            raw_cal = model.predict_proba(X[calibration])[:, 1]
            calibrator = LogisticRegression(C=1e6, max_iter=1000)
            calibrator.fit(logit(np.clip(raw_cal, 1e-7, 1-1e-7)).reshape(-1, 1), y[calibration],
                           sample_weight=participant_weights(data.iloc[calibration].subject_id))
            calibrated = lambda raw: calibrator.predict_proba(logit(np.clip(raw, 1e-7, 1-1e-7)).reshape(-1, 1))[:, 1]
            cal_p = calibrated(raw_cal)
            operating = select_sensitivity_threshold(y[calibration], cal_p, data.iloc[calibration].subject_id, target_sensitivity)
            probabilities = np.full(len(held), np.nan)
            if observed[held].any():
                probabilities[observed[held]] = calibrated(model.predict_proba(X[held[observed[held]]])[:, 1])
            metrics = binary_metrics(y[held], probabilities, data.iloc[held].subject_id, operating["threshold"])
            reports.append({**record, "status": "ok", "metrics": metrics, "operating_point": operating,
                            "n_fit_subjects": int(data.iloc[fit].subject_id.nunique()),
                            "n_calibration_subjects": int(data.iloc[calibration].subject_id.nunique())})
            prediction = data.iloc[held][["sample_id", "subject_id"]].copy()
            prediction["geography"], prediction["target"], prediction["probability"] = domain, y[held], probabilities
            predictions.append(prediction)
            for role, indices in (("fit", fit), ("calibration", calibration), ("held_geography", held)):
                audit = data.iloc[indices][["sample_id", "subject_id"]].copy()
                audit["geography"], audit["role"] = domain, role
                roles.append(audit)
        except Exception as error:
            reports.append({**record, "status": "fit_failed", "exception_type": type(error).__name__})
    report = {"status": "ok" if any(row["status"] == "ok" for row in reports) else "no_evaluable_geography",
              "validation_type": "development_internal_external_geography_not_institution", "model": "fixed_elasticnet_C0.1_l1ratio0.5",
              "n_ambiguous_or_missing_geography_subjects_excluded": n_excluded, "geographies": reports}
    return _save(report, output_dir, "geography_validation", {
        "geography_predictions": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        "geography_roles": pd.concat(roles, ignore_index=True) if roles else pd.DataFrame(),
        "private_geography_key": pd.DataFrame(keys),
    })


def _bh_adjust(pvalues: Sequence[float | None]) -> list[float | None]:
    order = sorted([i for i, p in enumerate(pvalues) if p is not None and np.isfinite(p)], key=lambda i: pvalues[i])
    adjusted = [None]*len(pvalues)
    running = 1.0
    for rank in range(len(order)-1, -1, -1):
        index = order[rank]
        running = min(running, pvalues[index]*len(pvalues)/(rank+1))
        adjusted[index] = float(min(1.0, running))
    return adjusted


def longitudinal_associations(frame: pd.DataFrame, feature_columns: Sequence[str], outcome_column: str = "phq9",
                              covariate_columns: Sequence[str] = (), output_dir=None, min_subjects: int = 10) -> dict:
    """Exploratory random-intercept association: between mean + within deviation.

    Continuous scores are modeled, not diagnostic outcomes. BH spans both terms
    across all declared features. This is observational association, not a causal
    or out-of-sample predictive analysis. Test outcomes remain untouched.
    """
    _predictor_columns(feature_columns)
    if outcome_column in covariate_columns or len(set(covariate_columns)) != len(covariate_columns):
        raise ValueError("Unique covariates must exclude the outcome")
    data = _development(frame)
    if importlib.util.find_spec("statsmodels") is None:
        return _save({"status": "optional_dependency_unavailable", "dependency": "statsmodels", "exploratory": True},
                     output_dir, "longitudinal_associations")
    import statsmodels.api as sm
    results = []
    for feature in feature_columns:
        terms = [{"feature": feature, "term": term, "status": "not_fitted", "p_value": None} for term in ("between_subject_mean", "within_subject_deviation")]
        columns = ["subject_id", "index_date", outcome_column, feature, *covariate_columns]
        analysis = data[list(dict.fromkeys(columns))].copy()
        numeric = [outcome_column, feature, *covariate_columns]
        analysis[numeric] = analysis[numeric].apply(pd.to_numeric, errors="coerce")
        analysis = analysis.replace([np.inf, -np.inf], np.nan).dropna(subset=[*numeric, "index_date"])
        counts = analysis.groupby("subject_id").size()
        if len(counts) < min_subjects or counts.ge(2).sum() < 5:
            for term in terms:
                term["status"] = "insufficient_repeated_subjects"
            results.extend(terms)
            continue
        means = analysis.groupby("subject_id")[feature].transform("mean")
        design = pd.DataFrame({"intercept": 1., "between_subject_mean": means,
                               "within_subject_deviation": analysis[feature]-means}, index=analysis.index)
        if np.linalg.matrix_rank(design) < 3:
            for term in terms:
                term["status"] = "insufficient_between_or_within_variation"
            results.extend(terms)
            continue
        dates = pd.to_datetime(analysis.index_date)
        elapsed = (dates-analysis.assign(_date=dates).groupby("subject_id")._date.transform("min")).dt.days/365.25
        candidates = {"followup_years": elapsed, "season_sin": np.sin(2*np.pi*dates.dt.dayofyear/365.25),
                      "season_cos": np.cos(2*np.pi*dates.dt.dayofyear/365.25),
                      **{f"covariate_{i}": analysis[col] for i, col in enumerate(covariate_columns)}}
        # Drop rank-redundant nuisance terms rather than producing spurious fits.
        for name, value in candidates.items():
            augmented = design.assign(**{name: value})
            if np.linalg.matrix_rank(augmented) == len(augmented.columns):
                design = augmented
        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                fitted = sm.MixedLM(analysis[outcome_column].to_numpy(), design.to_numpy(), groups=analysis.subject_id).fit(reml=True, method="lbfgs", disp=False)
            converged = bool(fitted.converged)
            for position, term in enumerate(terms, start=1):
                interval = np.asarray(fitted.conf_int())[position]
                term.update(status="ok" if converged else "not_converged", estimate=float(fitted.fe_params[position]),
                            ci_lower=float(interval[0]), ci_upper=float(interval[1]),
                            p_value=float(fitted.pvalues[position]) if converged and np.isfinite(fitted.pvalues[position]) else None,
                            n_subjects=len(counts), n_visits=len(analysis), n_repeated_subjects=int(counts.ge(2).sum()),
                            warnings=len(recorded), n_adjustment_columns=len(design.columns)-3)
        except Exception as error:
            for term in terms:
                term.update(status="fit_failed", exception_type=type(error).__name__)
        results.extend(terms)
    for record, qvalue in zip(results, _bh_adjust([row["p_value"] for row in results])):
        record["q_bh_exploratory"] = qvalue
    fitted_count = sum(row["status"] == "ok" for row in results)
    status = "ok" if results and fitted_count == len(results) else "partially_estimable" if fitted_count else "no_estimable_associations"
    return _save({"status": status, "n_estimable_terms": fitted_count, "exploratory": True, "model": "continuous_score_random_intercept_mixed_model",
                  "fdr_family_size": len(results), "causal_interpretation": False, "results": results},
                 output_dir, "longitudinal_associations")


def retention_report(visits: pd.DataFrame, as_of, expected_interval_days: int = 124, grace_days: int = 30,
                     output_dir=None, max_waves: int = 5) -> dict:
    """Scheduled follow-up availability proxy, never inferred actual dropout.

    as_of is an explicit administrative extract date supplied by the caller.
    Eligibility dates are enrollment proxies + fixed expected interval; withdrawal,
    device enrollment and genuine missed/terminated follow-up remain unknown.
    """
    if expected_interval_days < 1 or grace_days < 0 or max_waves < 1:
        raise ValueError("Positive schedule and nonnegative grace required")
    as_of = pd.Timestamp(as_of)
    if pd.isna(as_of):
        raise ValueError("Explicit administrative extract date required")
    data = visits.loc[pd.to_datetime(visits.index_date) <= as_of].copy()
    data["index_date"] = pd.to_datetime(data.index_date)
    data["enrolled_date"] = pd.to_datetime(data.enrolled_date)
    if data.subject_id.isna().any() or data.enrolled_date.isna().any() or (data.enrolled_date > data.index_date).any():
        raise ValueError("Present subject and valid enrollment proxy dates required")
    enrollment = data.groupby("subject_id").enrolled_date.min()
    rows, private = [], []
    for wave in range(1, max_waves+1):
        observed = set(data.loc[data.wave.eq(wave), "subject_id"])
        due = enrollment+pd.to_timedelta((wave-1)*expected_interval_days, unit="D")
        missing = ~enrollment.index.isin(observed)
        counts = {"wave": wave, "n_enrolled": len(enrollment), "n_observed": len(observed),
                  "n_not_yet_due_proxy": int((missing & (due > as_of)).sum()),
                  "n_due_within_grace_missing_proxy": int((missing & (due <= as_of) & (due+pd.Timedelta(days=grace_days) >= as_of)).sum()),
                  "n_overdue_missing_status_unknown": int((missing & (due+pd.Timedelta(days=grace_days) < as_of)).sum())}
        rows.append(counts)
        for subject, date in due.items():
            state = "observed" if subject in observed else "not_yet_due_proxy" if date > as_of else "due_within_grace_missing_proxy" if date+pd.Timedelta(days=grace_days) >= as_of else "overdue_missing_status_unknown"
            private.append({"subject_id": subject, "wave": wave, "due_date_proxy": date, "state": state})
    report = {"status": "ok", "as_of": as_of, "expected_interval_days": expected_interval_days, "grace_days": grace_days,
              "enrollment_basis": "supplied_enrolled_date_first_observed_survey_proxy", "schedule_semantics": "proxy_not_verified_appointments",
              "withdrawal_information": "unknown", "actual_dropout_estimated": False, "waves": rows}
    return _save(report, output_dir, "retention", {"private_retention_states": pd.DataFrame(private)})


def regression_metrics(y, prediction, subjects) -> dict:
    """Participant-weighted errors and descriptive linear calibration."""
    y, prediction, subjects = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float), np.asarray(subjects)
    if not (y.shape == prediction.shape == subjects.shape) or y.ndim != 1:
        raise ValueError("Aligned regression vectors required")
    labelled = np.isfinite(y)
    valid = labelled & np.isfinite(prediction)
    all_w = participant_weights(subjects[labelled]) if labelled.any() else np.array([])
    report = {"n_labelled_visits": int(labelled.sum()), "n_predicted_visits": int(valid.sum()),
              "n_predicted_participants": int(pd.Series(subjects[valid]).nunique()),
              "coverage": float(all_w[np.isfinite(prediction[labelled])].sum()/all_w.sum()) if len(all_w) else None,
              "mae": None, "rmse": None, "calibration_intercept": None, "calibration_slope": None}
    if not valid.any():
        return {**report, "status": "no_predictions"}
    yy, pp = y[valid], prediction[valid]
    w = participant_weights(subjects[valid])
    report.update(status="ok", mae=float(np.average(np.abs(yy-pp), weights=w)), rmse=float(np.sqrt(np.average((yy-pp)**2, weights=w))))
    design = np.column_stack([np.ones(len(pp)), pp])
    if np.linalg.matrix_rank(design) < 2:
        report["calibration_status"] = "constant_predictions"
    else:
        beta = np.linalg.lstsq(design*np.sqrt(w[:, None]), yy*np.sqrt(w), rcond=None)[0]
        report.update(calibration_intercept=float(beta[0]), calibration_slope=float(beta[1]), calibration_status="descriptive_weighted_linear_fit")
    return report


def run_prognosis_regression(frame: pd.DataFrame, sensor_columns: Sequence[str], cutoff, provenance: Mapping,
                             output_dir=None, outer_folds: int = 5, seed: int = 42) -> dict:
    """Grouped development OOF 90–180-day PHQ prognosis versus current-PHQ baseline.

    Provenance requires past_only=True, extraction_version, window_end_column.
    Every retained row must have sensor window end strictly before index_date.
    Index and future target dates must both precede the locked training cutoff.
    """
    _predictor_columns(sensor_columns, prefixes=("sensor__",))
    data = _development(frame)
    end_column = provenance.get("window_end_column")
    if provenance.get("past_only") is not True or not provenance.get("extraction_version") or end_column not in data:
        return _save({"status": "past_only_provenance_unavailable", "required": ["past_only", "extraction_version", "window_end_column"]}, output_dir, "prognosis_regression")
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff):
        raise ValueError("Explicit locked cutoff required")
    index_date, target_date = pd.to_datetime(data.index_date), pd.to_datetime(data.target_date)
    end = pd.to_datetime(data[end_column], errors="coerce")
    days = (target_date-index_date).dt.total_seconds()/86400
    before = len(data)
    temporal = days.between(90, 180) & (index_date < cutoff) & (target_date < cutoff) & end.notna()
    if (temporal & (end >= index_date)).any():
        raise ValueError("Prognosis sensors must end strictly before the index date")
    # Purge future outcomes before accessing score values, including development
    # participants whose subsequent visit occurs after the training cutoff.
    data = data.loc[temporal].copy()
    current, future = pd.to_numeric(data.phq9, errors="coerce"), pd.to_numeric(data.future_phq9, errors="coerce")
    eligible = current.between(0, 27) & future.between(0, 27) & current.eq(current.round()) & future.eq(future.round())
    data = data.loc[eligible].reset_index(drop=True)
    groups = data.subject_id.to_numpy()
    subjects = np.array(sorted(data.subject_id.unique()))
    if outer_folds < 2 or len(subjects) < 4:
        return _save({"status": "insufficient_prognosis_participants", "n_subjects": len(subjects), "n_eligible_pairs": len(data)}, output_dir, "prognosis_regression")
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    folds = np.array_split(subjects, min(outer_folds, len(subjects)))
    y, current = data.future_phq9.to_numpy(dtype=float), data.phq9.to_numpy(dtype=float)
    X = data[list(sensor_columns)].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)
    observed = np.isfinite(X).any(1)
    predictions = []
    for fold, validation_subjects in enumerate(folds):
        valid = np.isin(groups, validation_subjects)
        train = ~valid
        for model_name in ("current_phq_persistence", "current_phq_regression", "current_phq_plus_sensor"):
            p = np.full(int(valid.sum()), np.nan)
            if model_name == "current_phq_persistence":
                p[:] = current[valid]
            else:
                augmented = model_name == "current_phq_plus_sensor"
                values = np.column_stack([current, X]) if augmented else current[:, None]
                model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                      ElasticNet(alpha=.1, l1_ratio=.5, max_iter=5000, random_state=seed) if augmented else Ridge(alpha=1.0))
                name = "elasticnet" if augmented else "ridge"
                fit = train & observed if augmented else train
                if fit.any():
                    model.fit(values[fit], y[fit], **{f"{name}__sample_weight": participant_weights(groups[fit])})
                    can_predict = observed[valid] if augmented else np.ones(valid.sum(), dtype=bool)
                    if can_predict.any():
                        p[can_predict] = model.predict(values[valid][can_predict])
            table = data.loc[valid, ["sample_id", "subject_id", "index_date", "target_date"]].copy()
            table["model"], table["outer_fold"], table["target"], table["prediction"] = model_name, fold, y[valid], p
            predictions.append(table)
    oof = pd.concat(predictions, ignore_index=True)
    reports = []
    for model_name, table in oof.groupby("model"):
        reports.append({"model": model_name, **regression_metrics(table.target, table.prediction, table.subject_id)})
    # Augmented and baseline errors on identical predictable visits are separate
    # from each model's full-population availability report.
    wide = oof.pivot(index=["sample_id", "subject_id"], columns="model", values="prediction").reset_index()
    targets = data.set_index("sample_id").future_phq9
    common = wide.current_phq_plus_sensor.notna() if "current_phq_plus_sensor" in wide else np.zeros(len(wide), dtype=bool)
    paired = []
    for model_name in ("current_phq_regression", "current_phq_plus_sensor"):
        if model_name in wide:
            subset = wide.loc[common]
            paired.append({"model": model_name, **regression_metrics(subset.sample_id.map(targets), subset[model_name], subset.subject_id)})
    return _save({"status": "ok", "n_eligible_pairs": len(data), "n_excluded_development_pairs": before-len(data),
                  "n_subjects": len(subjects), "outer_folds": len(folds), "target_horizon_days": [90, 180],
                  "cutoff": cutoff, "feature_provenance": dict(provenance), "models": reports,
                  "paired_common_sample_metrics": paired, "model_selection": "fixed_models_no_test_tuning",
                  "calibration": "descriptive_OOF_linear_calibration_not_posthoc_test_correction"},
                 output_dir, "prognosis_regression", {"prognosis_oof_predictions": oof})
