"""Corrected export parsing without raw source names or unit guesses."""
import numpy as np
import pandas as pd
from .sources import date_series

def robust_numeric(s):
    s=s.astype('string').str.strip();grouped=s.str.match(r'^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$',na=False)
    s=s.where(~grouped,s.str.replace(',','',regex=False))
    return pd.to_numeric(s,errors='coerce').to_numpy(dtype=float,na_value=np.nan)

def robust_clock(s):
    ss=s.astype('string').str.strip()
    # Match the complete field. Both '12:30AM' and Korean markers are explicit;
    # embedded digits, contradictory markers and arbitrary suffixes are invalid.
    p=ss.str.extract(r'(?i)^(?:(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}[ T])?(?:(AM|PM|오전|오후)\s*)?(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?\s*(AM|PM|오전|오후)?$')
    h=pd.to_numeric(p[1],errors='coerce');mi=pd.to_numeric(p[2],errors='coerce');se=pd.to_numeric(p[3],errors='coerce').fillna(0)
    marker=p[0].fillna(p[4]).str.upper()
    am=marker.isin(['AM','오전']);pm=marker.isin(['PM','오후'])
    marked=am|pm;valid=(~marked & h.between(0,23))|(marked & h.between(1,12))
    valid &= ~(p[0].notna() & p[4].notna())
    h=h.where(~marked,(h%12)+12*pm.astype(int))
    value=h+mi/60+se/3600
    return value.where(valid&mi.between(0,59)&se.between(0,59.999999)).to_numpy(dtype=float,na_value=np.nan)

def robust_dates(s):
    # One parser contract; never re-accept a partial field rejected upstream.
    return date_series(s)

def canonical(records):
    a=pd.concat(records,ignore_index=True);keys=['subject_id','date','modality']
    latest=a.groupby(keys,sort=False).priority.transform('max');top=a.loc[a.priority.eq(latest)].copy()
    conflict=top.groupby(keys,sort=False).value.transform('nunique')>1
    count=top.loc[conflict,keys].drop_duplicates().shape[0]
    chosen=top.loc[~conflict].drop_duplicates(keys)
    out=chosen.pivot(index=['subject_id','date'],columns='modality',values='value').reset_index();out.columns.name=None
    for c in ['sleep_onset','wake_time']:
        if c in out:
            angle=out.pop(c)*2*np.pi/24;out[c+'_sin']=np.sin(angle);out[c+'_cos']=np.cos(angle)
    return out.sort_values(['subject_id','date']).reset_index(drop=True),{'input_records':len(a),'canonical_records':len(chosen),'conflicting_keys_excluded':count,'duplicate_records_removed':len(a)-len(chosen)-int(conflict.sum())}

def compare_daily(a,b,columns,label):
    z=a[['subject_id','date']+columns].merge(b[['subject_id','date']+columns],on=['subject_id','date'],how='outer',suffixes=('_a','_b'),validate='one_to_one')
    rows=[]
    for c in columns:
        aa=z[c+'_a'].to_numpy(float);bb=z[c+'_b'].to_numpy(float);both=np.isfinite(aa)&np.isfinite(bb)
        equal=np.isclose(aa,bb,atol=1e-12,rtol=1e-12,equal_nan=True)
        rows.append({'comparison':label,'channel':c,'both_observed':int(both.sum()),'different_shared_values':int((both&~equal).sum()),'new_observations':int((~np.isfinite(aa)&np.isfinite(bb)).sum()),'lost_observations':int((np.isfinite(aa)&~np.isfinite(bb)).sum()),'changed_cells':int((~equal).sum())})
    return rows
