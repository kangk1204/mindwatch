"""Explicit table/channel contracts shared by portable feature builders."""
import re
import pandas as pd


def calendar_table(frame, date):
    if not {"subject_id", date}.issubset(frame) or frame.columns.duplicated().any():
        raise ValueError("Unique columns and participant/date fields required")
    out=frame.copy().reset_index(drop=True)
    if out.subject_id.isna().any() or out.subject_id.astype(str).str.strip().eq("").any():
        raise ValueError("Missing participant key")
    if out.subject_id.nunique()!=out.subject_id.astype(str).nunique():
        raise ValueError("Participant key conversion collision")
    out["subject_id"]=out.subject_id.astype(str)
    out[date]=pd.to_datetime(out[date],errors="raise")
    if out[date].isna().any() or out[date].dt.tz is not None:
        raise ValueError("Timezone-resolved local calendar required")
    if out.duplicated(["subject_id",date]).any():
        raise ValueError("Resolve duplicate participant/date keys first")
    return out


def sensor_channels(channels):
    if not channels or len(set(channels))!=len(channels):
        raise ValueError("Unique nonempty sensor channel names required")
    for name in channels:
        if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",name):
            raise ValueError("Unsafe channel name")
        if re.search(r"(?:^|_)(?:phq9|gad7|isi|target|label|subject|sample|id|ema)(?:_|$)",name,re.I):
            raise ValueError("Survey outcomes, EMA and identifiers are not sensor channels")
    return tuple(channels)
