"""Actual response history and fixed causal EMA expert averaging."""
import numpy as np
import pandas as pd
from mindwatch.features import DOMAINS

def event_history(anchors,daily):
 from .contracts import calendar_table
 from mindwatch.features import validate_ema
 anchors=calendar_table(anchors,'index_date',date_only=True);daily=validate_ema(daily)
 groups={sid:g.sort_values('date') for sid,g in daily.groupby('subject_id')};rows=[]
 for r in anchors.itertuples():
  row={};g=groups.get(r.subject_id)
  for dom in DOMAINS:
   past=g.loc[g.date.lt(r.index_date),['date',dom]].dropna().tail(8) if g is not None else pd.DataFrame(columns=['date',dom]);a=past[dom].to_numpy(float);n=len(a);pre='e_event8_'+dom+'__'
   row[pre+'n']=n;row[pre+'last']=float(a[-1]) if n else np.nan;row[pre+'median3']=float(np.median(a[-3:])) if n else np.nan;row[pre+'median8']=float(np.median(a)) if n else np.nan;row[pre+'iqr8']=float(np.quantile(a,.75)-np.quantile(a,.25)) if n>=3 else np.nan;row[pre+'change']=float(a[-1]-a[-2]) if n>=2 else np.nan;row[pre+'gap_days']=float((r.index_date-past.date.iloc[-1]).days) if n else np.nan
   row[pre+'slope']=float((a[-1]-a[0])/(past.date.iloc[-1]-past.date.iloc[0]).days) if n>=2 and past.date.iloc[-1]>past.date.iloc[0] else np.nan
  rows.append(row)
 return pd.DataFrame(rows)

def adaptive_one(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if ((x<0)|(x>10)).any():raise ValueError('EMA must use the harmonized 0 to 10 scale')
    if not len(x):return np.nan
    def experts(y):return np.array([y[-1],np.median(y[-3:]),np.median(y[-5:]),.5*(y[-1]+np.median(y[-3:]))])
    errors=[]
    for k in range(max(1,len(x)-4),len(x)):errors.append(abs(experts(x[:k])-x[k]))
    w=np.array([2.,1.,1.,1.]);w*=np.exp(-np.mean(errors,axis=0)) if errors else 1
    return float(np.average(experts(x),weights=w))
