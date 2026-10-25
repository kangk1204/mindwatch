"""Independent frozen risk and score endpoints with caller-selected policies.

This module supplies no fitted model, selected setting, membership or study
result. Select settings using development data and fit calibration to the
classifier's complete routing policy before constructing a frozen release.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import lightweight as core
from . import lightweight_policy as history


def _membership(values):
    if isinstance(values, (str, bytes)):
        raise ValueError('Excluded membership must be a collection of keys')
    people = frozenset(values)
    if not all(isinstance(value, str) and bool(value.strip()) for value in people):
        raise ValueError('Nonempty pseudonymous membership keys required')
    return people


def guard_queries(queries, excluded_people):
    """Check admission before feature construction or numeric model loading.

    This helper checks participant membership only. Endpoint-specific schema,
    chronology and target exclusions remain in the prediction methods.
    """
    people = _membership(excluded_people)
    if not isinstance(queries, pd.DataFrame) or not queries.columns.is_unique or 'subject_id' not in queries:
        raise ValueError('Unique participant metadata required')
    if not queries.subject_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Nonempty pseudonymous participant keys required')
    if queries.subject_id.isin(people).any():
        raise ValueError('Excluded participant query rejected before processing')


@dataclass(frozen=True)
class FrozenEndpointPolicy:
    """Keep risk unchanged while separately controlling PHQ score changes.

    With any recent EMA observation, shrink the base regressor's already
    clipped total toward the previous PHQ. Without recent EMA, the required
    ``no_ema_score_route`` selects either ``persistence`` or ``base_regressor``.
    The risk probability and its threshold are never derived from this total.

    Settings and version strings are required caller inputs; there is no
    default selected model. Exclusion membership is private release metadata.
    Source strings are descriptive provenance, not authenticity guarantees.
    """
    classifier: history.FrozenHistoryPolicy
    ema_change_weight: float
    no_ema_score_route: str
    classification_model_version: str
    regression_model_version: str
    excluded_people: frozenset[str] = frozenset()
    classification_source: str = 'not_provided'
    regression_source: str = 'not_provided'

    def __post_init__(self):
        if not isinstance(self.classifier, history.FrozenHistoryPolicy):
            raise ValueError('Frozen history classifier policy required')
        weight = self.ema_change_weight
        if (not isinstance(weight, (int, float, np.integer, np.floating))
                or isinstance(weight, (bool, np.bool_)) or not np.isfinite(weight)
                or not 0 <= weight <= 1):
            raise ValueError('Finite real EMA change weight between zero and one required')
        object.__setattr__(self, 'ema_change_weight', float(weight))
        if self.no_ema_score_route not in ('persistence', 'base_regressor'):
            raise ValueError('Explicit no-EMA score route required')
        for field in ('classification_model_version', 'regression_model_version',
                      'classification_source', 'regression_source'):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Nonempty endpoint version and provenance strings required')
        object.__setattr__(self, 'excluded_people', _membership(self.excluded_people))

    def predict(self, queries, features, policy='auto'):
        """Predict from ordered, aligned, target-free causal prepared features.

        Prepared arrays cannot establish their own chronology. Their input
        provenance remains unavailable; prefer the raw-history method.
        """
        guard_queries(queries, self.excluded_people)
        out = self.classifier.predict(queries, features, policy)
        prior = features.h_previous_phq9.to_numpy(float)
        valid = np.isfinite(prior)
        has_ema = features.loc[:, list(core.FEATURES[1:])].notna().any(axis=1).to_numpy()
        # FrozenHistoryPolicy already clipped each base total to the scale.
        base = out.predicted_phq9.to_numpy(float)
        weight = np.where(has_ema, self.ema_change_weight,
                          0. if self.no_ema_score_route == 'persistence' else 1.)
        total = np.where(weight == 0, prior, np.where(weight == 1, base,
                                                    prior + weight*(base-prior)))
        out['predicted_phq9'] = np.where(valid, total, np.nan)
        out['predicted_phq9_change'] = out.predicted_phq9-prior
        out['classification_route'] = out.inference_route
        out['classification_fallback_used'] = out.history_fallback_used
        out['classification_fallback_reason'] = out.fallback_reason
        out['fallback_reason_scope'] = 'classification'
        out['regression_route'] = np.where(~valid, 'none', np.where(weight == 0,
            'persistence', np.where(has_ema, 'shrunken_base_change', 'base_regressor')))
        out['regression_change_weight'] = np.where(valid, weight, np.nan)
        out['regression_fallback_reason'] = np.where(~valid, 'missing_prior_phq',
            np.where(~has_ema & (weight == 0), 'no_recent_ema_use_previous_phq', ''))
        observed = features.loc[:, list(core.FEATURES[1:])].notna().to_numpy()
        def retained_mask(head):
            retained = {name for name, seen in zip(head.features, head.seen) if seen}
            return np.array([name in retained for name in core.FEATURES[1:]], dtype=bool)
        active_primary = (valid & has_ema)[:, None]
        class_retained = active_primary & retained_mask(self.classifier.primary.classifier)
        reg_retained = active_primary & (weight > 0)[:, None] & retained_mask(self.classifier.primary.change_regressor)
        class_effective, reg_effective = observed & class_retained, observed & reg_retained
        class_uses, reg_uses = class_effective.any(axis=1), reg_effective.any(axis=1)
        for prefix, retained, effective in (('classification', class_retained, class_effective),
                                             ('regression', reg_retained, reg_effective)):
            out[prefix+'_ema_retained_features'] = retained.sum(axis=1)
            out[prefix+'_ema_used_features'] = effective.sum(axis=1)
            out[prefix+'_ema_used'] = effective.any(axis=1)
        out['classification_used_modalities'] = np.where(~valid, 'none', np.where(~has_ema,
            'past_surveys', np.where(class_uses, 'past_phq+ema', 'past_phq')))
        out['regression_used_modalities'] = np.where(~valid, 'none', np.where(reg_uses,
            'past_phq+ema', np.where(~has_ema & (weight > 0), 'past_surveys', 'past_phq')))
        # The underlying policy describes both original heads. Recompute usage
        # when this endpoint disables a regressor, without changing predictions.
        retained, effective = class_retained | reg_retained, class_effective | reg_effective
        out['ema_retained_features'] = retained.sum(axis=1)
        out['ema_used_features'] = effective.sum(axis=1)
        out['ema_used'] = effective.any(axis=1)
        out['used_modalities'] = np.where(~valid, 'none', np.where(~has_ema, 'past_surveys',
            np.where(out.ema_used, 'past_phq+ema', 'past_phq')))
        class_imputed = class_retained.any(axis=1) & ~class_uses
        out['classification_fallback_reason'] = np.where(valid & has_ema,
            np.where(class_imputed, 'ema_missing_frozen_training_imputation', ''),
            out.classification_fallback_reason)
        out['fallback_reason'] = out.classification_fallback_reason
        out['classification_status'] = np.where(~valid, 'insufficient_data',
            np.where(~has_ema, 'ok_history_fallback', np.where(class_imputed, 'ok_ema_missing_imputed', 'ok')))
        out['status'] = np.where(~valid, 'insufficient_data', np.where(~has_ema, 'ok_history_fallback',
            np.where(retained.any(axis=1) & ~effective.any(axis=1), 'ok_ema_missing_imputed', 'ok')))
        out['classification_model_version'] = self.classification_model_version
        out['regression_model_version'] = self.regression_model_version
        out['classification_source'] = self.classification_source
        out['regression_source'] = self.regression_source
        return out

    def predict_from_history(self, queries, surveys, ema, policy='auto'):
        """Use only own observations strictly before each assessment day."""
        guard_queries(queries, self.excluded_people)
        meta, features, provenance = history.features_from_history(queries, surveys, ema)
        out = self.predict(meta, features, policy)
        for column in provenance:
            out[column] = provenance[column]
        return out


_SPEC_FIELDS = ('ema_change_weight', 'no_ema_score_route', 'classification_model_version',
                'regression_model_version', 'classification_source', 'regression_source')
_FILES = frozenset(('endpoint.json', 'classifier/manifest.json', 'classifier/policy.json',
                   'classifier/history_weights.npz', 'classifier/primary/manifest.json',
                   'classifier/primary/model.json', 'classifier/primary/weights.npz'))


def _private_directory(directory):
    from .pipeline import private_output
    requested = Path(directory).expanduser().absolute()
    if any(path.is_symlink() for path in (requested, *requested.parents)):
        raise ValueError('Use a canonical private directory without symlinks')
    return private_output(requested)


def _sources():
    folder = Path(__file__).parent
    return {name: core._digest(folder/name) for name in
            ('endpoint_policy.py', 'lightweight_policy.py', 'lightweight.py',
             'features.py', 'pipeline.py', 'models.py', 'validation.py',
             'research/__init__.py', 'research/evaluation.py')}


def save_release(model, directory):
    """Save only to a new private directory, never inside a Git checkout."""
    if not isinstance(model, FrozenEndpointPolicy):
        raise TypeError('FrozenEndpointPolicy required')
    folder = _private_directory(directory)
    folder.mkdir(parents=True, exist_ok=False, mode=0o700)
    folder.chmod(0o700)
    history.save_release(model.classifier, folder/'classifier')
    core._write_json(folder/'endpoint.json', {'format_version': 1,
        **{name: getattr(model, name) for name in _SPEC_FIELDS},
        'excluded_people': sorted(model.excluded_people)})
    core._write_json(folder/'manifest.json', {'format_version': 1, 'sources': _sources(),
        'files': {name: core._digest(folder/name) for name in sorted(_FILES)}})


def load_release(directory, *, queries=None):
    """Restore a source/checksum-checked numeric release, without pickle.

    Pass queries to reject excluded people before loading any numeric arrays.
    Checksums detect changes; they are not signatures. Use trusted releases.
    """
    folder = _private_directory(directory)
    manifest_path = folder/'manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('Endpoint manifest symlink forbidden')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('format_version') != 1 or set(manifest.get('files', {})) != _FILES
            or manifest.get('sources') != _sources()):
        raise ValueError('Endpoint source or release member schema changed')
    for name in _FILES:
        path = folder/name
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                   if parent != folder and folder in parent.parents):
            raise ValueError('Endpoint member symlink forbidden')
    spec_path = folder/'endpoint.json'
    if core._digest(spec_path) != manifest['files']['endpoint.json']:
        raise ValueError('Endpoint policy checksum changed')
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    if set(spec) != {'format_version', 'excluded_people', *_SPEC_FIELDS} or spec['format_version'] != 1:
        raise ValueError('Unexpected endpoint policy contract')
    excluded = _membership(spec['excluded_people'])
    if queries is not None:
        guard_queries(queries, excluded)
    for name, digest in manifest['files'].items():
        if core._digest(folder/name) != digest:
            raise ValueError('Endpoint release checksum changed')
    classifier = history.load_release(folder/'classifier')
    return FrozenEndpointPolicy(classifier, excluded_people=excluded,
                                **{name: spec[name] for name in _SPEC_FIELDS})
