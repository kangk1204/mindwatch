"""Time-OOF sensor correction with a fixed parent and exact missing fallback."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from mindwatch.models import person_weights


def time_oof(frame,features,folds,make_parent,policy="known",gap_days=14):
    if policy not in ("known","new"):raise ValueError("Unknown population policy")
    if len(frame)!=len(features) or not frame.index.equals(features.index):raise ValueError("Aligned raw feature rows required")
    if not {"sample_id","subject_id","index_date","label_available_at","target"}<=set(frame):raise ValueError("Outcome availability metadata required")
    if frame.sample_id.duplicated().any():raise ValueError("Unique sample keys required")
    if any(not name.startswith(("h_","e_","s_","history__","ema__","sensor__","demo__","quality__")) for name in features):raise ValueError("Current labels or metadata are not predictor columns")
    predictions=np.full(len(frame),np.nan);trace=[]
    for train,validate in folds:
        train=np.asarray(train,int);validate=np.asarray(validate,int)
        if not len(train) or not len(validate):continue
        if np.intersect1d(train,validate).size or len(np.unique(train))!=len(train) or len(np.unique(validate))!=len(validate):raise ValueError("Overlapping or duplicate fold rows")
        if (np.r_[train,validate]<0).any() or (np.r_[train,validate]>=len(frame)).any():raise ValueError("Invalid fold index")
        if np.isfinite(predictions[validate]).any():raise ValueError("Each OOF row must be predicted once")
        tf,vf=frame.iloc[train],frame.iloc[validate]
        if not pd.to_datetime(tf.label_available_at).lt(pd.to_datetime(vf.index_date).min()-pd.Timedelta(days=gap_days)).all():raise ValueError("Future or unavailable parent labels in OOF fit")
        if policy=="new" and set(tf.subject_id)&set(vf.subject_id):raise ValueError("New-person OOF contains population overlap")
        parent=make_parent();parent.fit(features.iloc[train].copy(),tf.target.to_numpy(float),sample_weight=person_weights(tf.subject_id))
        predictions[validate]=parent.predict_proba(features.iloc[validate].copy())[:,1]
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
        if tuple(x)!=self.names:raise ValueError("Sensor schema changed")
        base=np.asarray(base,float);q=np.asarray(quality,float)
        if base.shape!=(len(x),) or q.shape!=base.shape or not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError("Aligned baseline and observed quality in 0 to 1 required")
        a=x.to_numpy(float,copy=True);a[:,~self.seen]=np.nan
        present=np.isfinite(a).any(1)&(q>0);z=np.clip((np.where(np.isfinite(a),a,self.median)-self.mean)/self.scale,-12,12)
        result=expit(logit(np.clip(base,1e-7,1-1e-7))+self.gamma*q*(z@self.coefficients))
        return np.where(present,result,base)


def fit_offset(x,y,ids,base_oof,quality,penalty=1.,gamma=.5):
    if not len(x.columns) or any(not name.startswith(("s_","sensor__")) for name in x):raise ValueError("Sensor predictors only")
    y,base,q=np.asarray(y,float),np.asarray(base_oof,float),np.asarray(quality,float)
    good=np.isfinite(y)&np.isfinite(base)&np.isfinite(q)
    if not good.any() or ((y[good]!=0)&(y[good]!=1)).any() or ((q[good]<0)|(q[good]>1)).any() or ((base[good]<0)|(base[good]>1)).any():raise ValueError("Valid binary targets, OOF probabilities and quality required")
    a=x.loc[good].to_numpy(float,copy=True);seen=np.isfinite(a).any(0)
    median=np.array([np.nanmedian(a[:,j]) if seen[j] else 0. for j in range(a.shape[1])]);a=np.where(np.isfinite(a),a,median)
    w=person_weights(np.asarray(ids)[good]);mean=np.average(a,axis=0,weights=w);scale=np.sqrt(np.average((a-mean)**2,axis=0,weights=w));scale[scale<1e-6]=1
    z=np.clip((a-mean)/scale,-12,12)*q[good,None];offset=logit(np.clip(base[good],1e-7,1-1e-7));w=w/w.sum()
    def objective(beta):
        eta=offset+z@beta
        return float(np.sum(w*(np.logaddexp(0,eta)-y[good]*eta))+.5*penalty*np.dot(beta,beta)),z.T@(w*(expit(eta)-y[good]))+penalty*beta
    fit=minimize(objective,np.zeros(a.shape[1]),jac=True,method="L-BFGS-B")
    if not fit.success:raise RuntimeError("Sensor correction did not converge")
    return SensorOffset(tuple(x),seen,median,mean,scale,fit.x,gamma)
