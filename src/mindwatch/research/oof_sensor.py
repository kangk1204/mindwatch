"""Time-OOF sensor correction with a fixed parent and exact missing fallback."""
from dataclasses import dataclass
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from mindwatch.models import person_weights
from .monitoring_models import _names
from .temporal_splits import _date_series, _days, _positions


def time_oof(frame,features,folds,make_parent,policy="known",gap_days=14):
    if policy not in ("known","new"):raise ValueError("Unknown population policy")
    gap = _days(gap_days, "gap_days")
    if len(frame)!=len(features) or not frame.index.equals(features.index):raise ValueError("Aligned raw feature rows required")
    if not {"sample_id","subject_id","index_date","label_available_at","target"}<=set(frame):raise ValueError("Outcome availability metadata required")
    if (frame.columns.duplicated().any() or frame.sample_id.isna().any() or
            frame.sample_id.astype(str).duplicated().any() or frame.subject_id.isna().any() or
            frame.subject_id.astype(str).str.strip().eq("").any() or
            frame.sample_id.astype(str).str.strip().eq("").any()):raise ValueError("Unique sample and nonmissing participant keys required")
    if frame.subject_id.nunique()!=frame.subject_id.astype(str).nunique():raise ValueError("Participant key conversion collision")
    if "partition" in frame and not frame.partition.eq("development").all():raise ValueError("OOF fitting accepts development rows only")
    index_dates = _date_series(frame.index_date)
    release_dates = _date_series(frame.label_available_at)
    if (release_dates < index_dates).any():raise ValueError("Label availability cannot precede the index date")
    if "target_date" in frame:
        targets = _date_series(frame.target_date)
        if (targets < index_dates).any() or (release_dates < targets).any():raise ValueError("Invalid future outcome availability")
    names=[]
    for name in features:
        if not isinstance(name,str):raise ValueError("Predictor names must be strings")
        if re.search(r"(?:^|_)(?:current|future)(?:_|$)",name.lower()):raise ValueError("Current and future outcome features are forbidden")
        names.append(next((prefix+name[len(short):] for short,prefix in (("h_","history__"),("e_","ema__"),("s_","sensor__")) if name.startswith(short)),name))
    _names(names)
    predictions=np.full(len(frame),np.nan);trace=[];assigned=np.zeros(len(frame),bool)
    for train,validate in folds:
        train=_positions(train,len(frame));validate=_positions(validate,len(frame))
        if not len(train) or not len(validate):continue
        if np.intersect1d(train,validate).size or len(np.unique(train))!=len(train) or len(np.unique(validate))!=len(validate):raise ValueError("Overlapping or duplicate fold rows")
        if (np.r_[train,validate]<0).any() or (np.r_[train,validate]>=len(frame)).any():raise ValueError("Invalid fold index")
        if assigned[validate].any():raise ValueError("Each OOF row must be predicted once")
        tf,vf=frame.iloc[train],frame.iloc[validate]
        if not release_dates.iloc[train].lt(index_dates.iloc[validate].min()-gap).all():raise ValueError("Future or unavailable parent labels in OOF fit")
        if policy=="new" and set(tf.subject_id)&set(vf.subject_id):raise ValueError("New-person OOF contains population overlap")
        target=tf.target.to_numpy(float)
        if not np.isin(target,[0,1]).all():raise ValueError("Binary parent training labels required")
        parent=make_parent();parent.fit(features.iloc[train].copy(),target,sample_weight=person_weights(tf.subject_id))
        probabilities=np.asarray(parent.predict_proba(features.iloc[validate].copy()),float)
        classes=np.asarray(getattr(parent,"classes_",[0,1]))
        positive=np.flatnonzero(classes==1)
        if probabilities.shape!=(len(validate),len(classes)) or len(positive)!=1:raise ValueError("Aligned parent class probabilities required")
        result=probabilities[:,positive[0]]
        if not np.isfinite(result).all() or ((result<0)|(result>1)).any():raise ValueError("Finite parent OOF probabilities required")
        predictions[validate]=result;assigned[validate]=True
        trace.append({"training_rows":len(train),"validation_rows":len(validate),"latest_parent_label":str(pd.to_datetime(tf.label_available_at).max()),"validation_start":str(pd.to_datetime(vf.index_date).min()),"policy":policy})
    return predictions,trace


@dataclass
class SensorOffset:
    names: tuple
    seen: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    gamma: float=.5

    def predict(self,base,x,quality):
        a=_sensor_matrix(x)
        if tuple(x)!=self.names:raise ValueError("Sensor schema changed")
        gamma=_nonnegative(self.gamma,"gamma")
        base=_aligned_vector(base,x,"baseline");q=_aligned_vector(quality,x,"quality")
        _probabilities(base,q)
        a[:,~self.seen]=np.nan
        present=np.isfinite(a).any(1)&(q>0);z=np.clip((np.where(np.isfinite(a),a,self.median)-self.mean)/self.scale,-12,12)
        result=expit(logit(np.clip(base,1e-7,1-1e-7))+gamma*q*(z@self.coefficients))
        return np.where(present,result,base)


def _sensor_matrix(x):
    if not isinstance(x,pd.DataFrame) or not len(x.columns) or x.columns.duplicated().any():raise ValueError("Nonempty unique sensor schema required")
    names=[]
    for name in x:
        if not isinstance(name,str) or not name.startswith(("s_","sensor__")):raise ValueError("Sensor predictors only")
        if re.search(r"(?:^|_)(?:current|future)(?:_|$)",name.lower()):raise ValueError("Current and future outcome features are forbidden")
        names.append("sensor__"+name[2:] if name.startswith("s_") else name)
    _names(names,("sensor__",))
    a=x.to_numpy(float,copy=True)
    if np.isinf(a).any():raise ValueError("Infinite sensor values are not missing observations")
    return a


def _aligned_vector(value,x,name,numeric=True):
    if isinstance(value,pd.Series) and not value.index.equals(x.index):raise ValueError(f"{name} row index is misaligned")
    a=np.asarray(value,float) if numeric else np.asarray(value)
    if a.shape!=(len(x),):raise ValueError(f"Aligned {name} vector required")
    return a


def _nonnegative(value,name):
    if isinstance(value,(bool,np.bool_)) or not np.isscalar(value):raise ValueError(f"Finite nonnegative {name} required")
    try:value=float(value)
    except (TypeError,ValueError):raise ValueError(f"Finite nonnegative {name} required") from None
    if not np.isfinite(value) or value<0:raise ValueError(f"Finite nonnegative {name} required")
    return value


def _probabilities(base,quality):
    # Missing base forecasts remain abstentions. Quality is always explicit.
    if np.isinf(base).any() or np.any(np.isfinite(base)&((base<0)|(base>1))):raise ValueError("Baseline probabilities must be in 0 to 1 or missing")
    if not np.isfinite(quality).all() or ((quality<0)|(quality>1)).any():raise ValueError("Observed quality must be finite in 0 to 1")


def fit_offset(x,y,ids,base_oof,quality,penalty=1.,gamma=.5):
    matrix=_sensor_matrix(x)
    penalty,gamma=_nonnegative(penalty,"penalty"),_nonnegative(gamma,"gamma")
    y=_aligned_vector(y,x,"targets");base=_aligned_vector(base_oof,x,"baseline");q=_aligned_vector(quality,x,"quality")
    ids=_aligned_vector(ids,x,"participants",numeric=False)
    if pd.isna(ids).any() or pd.Series(ids).astype(str).str.strip().eq("").any():raise ValueError("Nonmissing participant keys required")
    if pd.Series(ids).nunique()!=pd.Series(ids).astype(str).nunique():raise ValueError("Participant key conversion collision")
    _probabilities(base,q)
    if np.isinf(y).any() or np.any(np.isfinite(y)&~np.isin(y,[0,1])):raise ValueError("Binary targets or missing labels required")
    good=np.isfinite(y)&np.isfinite(base)
    if not good.any():raise ValueError("No aligned labelled OOF rows")
    a=matrix[good];seen=np.isfinite(a).any(0)
    effective_quality=q[good]*np.isfinite(a).any(1)
    median=np.array([np.nanmedian(a[:,j]) if seen[j] else 0. for j in range(a.shape[1])]);a=np.where(np.isfinite(a),a,median)
    w=person_weights(ids[good]);mean=np.average(a,axis=0,weights=w)
    with np.errstate(over="ignore",invalid="ignore"):
        scale=np.sqrt(np.average((a-mean)**2,axis=0,weights=w))
    scale[scale<1e-6]=1
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():raise FloatingPointError("Sensor normalization is nonfinite")
    z=np.clip((a-mean)/scale,-12,12)*effective_quality[:,None];offset=logit(np.clip(base[good],1e-7,1-1e-7));w=w/w.sum()
    def objective(beta):
        eta=offset+z@beta
        return float(np.sum(w*(np.logaddexp(0,eta)-y[good]*eta))+.5*penalty*np.dot(beta,beta)),z.T@(w*(expit(eta)-y[good]))+penalty*beta
    fit=minimize(objective,np.zeros(a.shape[1]),jac=True,method="L-BFGS-B")
    if not fit.success or not np.isfinite(fit.x).all():raise RuntimeError("Sensor correction did not converge")
    return SensorOffset(tuple(x),seen,median,mean,scale,fit.x,gamma)
