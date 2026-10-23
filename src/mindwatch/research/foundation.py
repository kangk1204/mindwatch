"""Local frozen EMA forecasting; no inference service or automatic downloads."""
from pathlib import Path
import numpy as np
import pandas as pd
from mindwatch.features import DOMAINS,validate_ema
from .contracts import calendar_table


def contexts(anchors,daily,length=112):
    anchors=calendar_table(anchors,"index_date");daily=validate_ema(daily)
    if not anchors.index_date.eq(anchors.index_date.dt.normalize()).all():raise ValueError("Date-only EMA forecast anchors required")
    result=np.full((len(anchors),len(DOMAINS),length),np.nan,np.float32)
    groups={key:g.set_index("date") for key,g in daily.groupby("subject_id")}
    for i,row in enumerate(anchors.itertuples()):
        if row.subject_id not in groups:continue
        dates=pd.date_range(row.index_date-pd.Timedelta(days=length),periods=length)
        result[i]=groups[row.subject_id].reindex(dates)[list(DOMAINS)].to_numpy(float).T
    return result


def load_local(kind,directory):
    directory=Path(directory).expanduser().resolve()
    if not directory.is_dir():raise ValueError("An explicitly prepared local public-model directory is required")
    if kind=="chronos2":
        from chronos import Chronos2Pipeline
        return Chronos2Pipeline.from_pretrained(str(directory),device_map="cpu",local_files_only=True)
    if kind=="timestral5m":
        from timestral import TimestralForecaster
        return TimestralForecaster.from_pretrained(str(directory),device="cpu",local_files_only=True)
    raise ValueError("Unknown frozen EMA forecaster")


def forecast(kind,model,history,batch_size=64):
    history=np.asarray(history,np.float32)
    if history.ndim!=3 or history.shape[1]!=len(DOMAINS) or np.isinf(history).any():raise ValueError("Expected observed/missing N by domain by time array")
    result=np.full(history.shape[:2],np.nan)
    eligible=np.flatnonzero(np.isfinite(history).any((1,2)))
    for begin in range(0,len(eligible),batch_size):
        rows=eligible[begin:begin+batch_size]
        if kind=="chronos2":
            predictions=model.predict(history[rows],prediction_length=1,batch_size=batch_size,cross_learning=False)
            median=int(np.argmin(abs(np.asarray(model.quantiles)-.5)))
            for i,item in enumerate(predictions):result[rows[i]]=item.detach().cpu().numpy()[:,median,0]
        elif kind=="timestral5m":
            result[rows]=model.forecast(history[rows],prediction_length=1,symmetric_averaging=False,clip_nonnegative=False).median[:,:,0]
        else:raise ValueError("Unknown frozen EMA forecaster")
    result=np.clip(result,0,10);result[~np.isfinite(history).any(2)]=np.nan
    return result
