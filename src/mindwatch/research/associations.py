"""Exploratory participant-weighted ranks and strictly past personal deviations."""
import numpy as np
import pandas as pd
from mindwatch.models import person_weights
from .contracts import calendar_table

def weighted_rank(v,w):
 order=np.argsort(v,kind='stable');values=v[order];ww=w[order];begin=np.r_[0,np.flatnonzero(np.diff(values)!=0)+1];end=np.r_[begin[1:],len(v)];rank=np.empty(len(v));acc=0.
 for b,z in zip(begin,end):total=ww[b:z].sum();rank[order[b:z]]=acc+.5*total;acc+=total
 return rank

def rho(a,b,ids):
 good=np.isfinite(a)&np.isfinite(b);a=np.asarray(a)[good];b=np.asarray(b)[good];ids=np.asarray(ids)[good]
 if len(a)<30 or np.std(a)==0 or np.std(b)==0:return np.nan,len(a),len(set(ids))
 w=person_weights(ids);x=weighted_rank(a,w);y=weighted_rank(b,w);xm=np.average(x,weights=w);ym=np.average(y,weights=w);r=np.average((x-xm)*(y-ym),weights=w)/np.sqrt(np.average((x-xm)**2,weights=w)*np.average((y-ym)**2,weights=w));return float(r),len(a),len(set(ids))

def between_within(frame,feature,target,minimum_history=3):
    """Return descriptive correlations; selection and multiple testing stay external."""
    data=calendar_table(frame,"index_date",date_only=True).sort_values(["subject_id","index_date"])
    if feature==target or minimum_history<1:raise ValueError("Distinct variables and positive history required")
    means=data.groupby("subject_id")[[feature,target]].mean()
    between=rho(means[feature].to_numpy(),means[target].to_numpy(),means.index.to_numpy())
    past=data.groupby("subject_id")[[feature,target]].transform(lambda series:series.shift().expanding(min_periods=minimum_history).mean())
    within=rho((data[feature]-past[feature]).to_numpy(),(data[target]-past[target]).to_numpy(),data.subject_id.to_numpy())
    return {"between":between,"within_strict_past":within}
