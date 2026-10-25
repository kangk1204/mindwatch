"""Metadata gates for a prospectively locked, date-only evaluation batch.

These checks do not establish that records were never inspected. Maintain an
access ledger separately and declare prior outcome access honestly. No outcome
values are needed to check eligibility; no fitting or file IO is performed.
"""
from dataclasses import dataclass
import re

import pandas as pd


def _date(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
        raise ValueError("Use a nonmissing, resolved local calendar date")
    return value


@dataclass(frozen=True)
class FutureValidationLock:
    model_sha256: str
    protocol_sha256: str
    freeze_date: str
    first_assessment_date: str
    previously_accessed_samples: frozenset[str] = frozenset()

    def __post_init__(self):
        for value in (self.model_sha256, self.protocol_sha256):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Frozen model and protocol SHA-256 required")
        if _date(self.first_assessment_date) <= _date(self.freeze_date):
            raise ValueError("Assessment collection must start after the freeze day")
        ids = self.previously_accessed_samples
        if isinstance(ids, (str, bytes)) or not all(isinstance(s, str) and s.strip() for s in ids):
            raise ValueError("Prior-access ledger requires pseudonymous sample keys")
        object.__setattr__(self, "previously_accessed_samples", frozenset(ids))

    def check(self, frame, *, model_sha256, protocol_sha256):
        """Require a frozen model, later assessments and already saved predictions.

        Without event times, predictions must be saved before the assessment
        day and inputs must precede the prediction day. This intentionally
        stricter protocol avoids assuming within-day event order. Own earlier
        history may be used for both
        population-seen and unseen people; this gate does not split participants.
        """
        if model_sha256 != self.model_sha256 or protocol_sha256 != self.protocol_sha256:
            raise ValueError("Frozen model or protocol changed")
        required = {"sample_id", "subject_id", "index_date", "prediction_recorded_date",
                    "label_available_date", "latest_input_date"}
        if not frame.columns.is_unique or not required <= set(frame.columns):
            raise ValueError("Complete, uniquely named prospective manifest required")
        for key in ("sample_id", "subject_id"):
            if not frame[key].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
                raise ValueError("Pseudonymous string sample and participant keys required")
        if frame.sample_id.duplicated().any() or frame.duplicated(["subject_id", "index_date"]).any():
            raise ValueError("Repeated assessment keys in evaluation manifest")
        if frame.sample_id.isin(self.previously_accessed_samples).any():
            raise ValueError("Previously accessed outcomes cannot be called new validation")
        dates = {name: pd.Series([_date(v) for v in frame[name]], index=frame.index,
                                dtype="datetime64[ns]") for name in required if name.endswith("date")}
        if pd.MultiIndex.from_arrays([frame.subject_id, dates["index_date"]]).duplicated().any():
            raise ValueError("Repeated participant assessment date after date normalization")
        anchor = dates["index_date"]
        valid = (anchor.ge(_date(self.first_assessment_date))
                 & dates["prediction_recorded_date"].ge(_date(self.freeze_date))
                 & dates["prediction_recorded_date"].lt(anchor)
                 & dates["label_available_date"].gt(anchor)
                 & dates["latest_input_date"].lt(anchor)
                 & dates["latest_input_date"].lt(dates["prediction_recorded_date"]))
        if not valid.all():
            raise ValueError("Prospective date ordering failed")
        return {"rows": len(frame), "people": int(frame.subject_id.nunique()),
                "date_contract_passed": True, "prior_access_ledger_checked": True,
                "independence_established_by_metadata_alone": False}
