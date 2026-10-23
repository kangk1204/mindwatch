"""Past-only daily sensor features; fit selection inside training folds only."""
import numpy as np
import pandas as pd

def sensor_features(meta,daily,channels):
    from .contracts import calendar_table,sensor_channels
    meta=calendar_table(meta,'index_date');daily=calendar_table(daily,'date');channels=list(sensor_channels(channels))
    if not daily.date.eq(daily.date.dt.normalize()).all() or not meta.index_date.eq(meta.index_date.dt.normalize()).all():raise ValueError('Daily calendar dates required')
    if daily.duplicated(['subject_id','date']).any():raise ValueError('Daily keys must be unique')
    groups={s:g.sort_values('date').set_index('date') for s,g in daily.groupby('subject_id',sort=False)}
    rows=[]
    for r in meta.itertuples(index=False):
        t=r.index_date.normalize();g=groups.get(r.subject_id)
        if g is None:g=daily.iloc[:0].set_index('date')
        recent=g.loc[(g.index>=t-pd.Timedelta(days=14))&(g.index<t),channels]
        baseline=g.loc[(g.index>=t-pd.Timedelta(days=42))&(g.index<t-pd.Timedelta(days=14)),channels]
        observed=np.isfinite(recent.to_numpy(float));usable=False;row={}
        for channel in channels:
            s=recent[channel].dropna();old=baseline[channel].dropna();pre='s_fe_'+channel+'__'
            values=s.to_numpy(float);past=old.to_numpy(float);circular=channel.endswith(('_sin','_cos'))
            if not circular:values=np.sign(values)*np.log1p(np.abs(values));past=np.sign(past)*np.log1p(np.abs(past))
            n=len(values);valid=n>=4;usable|=valid;median=float(np.median(values)) if valid else np.nan
            newer=values[s.index>=t-pd.Timedelta(days=7)];older=values[s.index<t-pd.Timedelta(days=7)]
            adj=np.diff(s.index.to_numpy()).astype('timedelta64[D]').astype(int)==1
            lag=float(pd.Series(values[:-1][adj]).corr(pd.Series(values[1:][adj]))) if adj.sum()>=4 and np.std(values[:-1][adj])>0 and np.std(values[1:][adj])>0 else np.nan
            wk=values[s.index.dayofweek<5];we=values[s.index.dayofweek>=5]
            bmed=float(np.median(past)) if len(past)>=14 else np.nan;biqr=float(np.subtract(*np.quantile(past,[.75,.25]))) if len(past)>=14 else np.nan
            row.update({pre+'coverage14':n/14,pre+'median':median,pre+'mad':float(np.median(abs(values-median))) if valid else np.nan,
                pre+'iqr':float(np.subtract(*np.quantile(values,[.75,.25]))) if valid else np.nan,
                pre+'last':float(values[-1]) if valid else np.nan,
                pre+'change7':float(np.median(newer)-np.median(older)) if len(newer)>=3 and len(older)>=3 else np.nan,
                pre+'slope':float(np.polyfit((s.index-t).days,values,1)[0]) if n>=7 else np.nan,
                pre+'personal_delta':median-bmed,pre+'personal_z':(median-bmed)/(biqr/1.349) if valid and len(past)>=14 and biqr>0 else np.nan,
                pre+'adjacent_day_correlation':lag,pre+'weekend_difference':float(np.median(we)-np.median(wk)) if len(we)>=2 and len(wk)>=3 else np.nan})
        for clock in ['sleep_onset','wake_time']:
            names=[clock+'_sin',clock+'_cos']
            if not set(names)<=set(channels):continue
            z=recent[names].dropna();b=baseline[names].dropna()
            sn,cs=z.mean().to_numpy() if len(z)>=4 else (np.nan,np.nan)
            bn,bc=b.mean().to_numpy() if len(b)>=14 else (np.nan,np.nan)
            phase=np.arctan2(sn,cs);bp=np.arctan2(bn,bc)
            row['s_fe_'+clock+'__circular_dispersion']=1-np.hypot(sn,cs)
            row['s_fe_'+clock+'__personal_phase_hours']=np.arctan2(np.sin(phase-bp),np.cos(phase-bp))*12/np.pi
        row['s_available']=bool(observed.any(axis=1).sum()>=4 and usable)
        row['s_quality']=float(observed.sum()/(14*len(channels)))
        rows.append(row)
    return pd.DataFrame(rows)

def select_columns(x,y,k):
    candidates=[c for c in x if c.startswith('s_fe_') and x[c].notna().mean()>=.3 and x[c].nunique()>1]
    if not candidates:raise ValueError('No supported sensor features')
    if k==0:return tuple(candidates)
    a=x[candidates];corr=a.corr(method='spearman',min_periods=30).abs().fillna(0)
    y=pd.Series(np.asarray(y),index=a.index);relevance={c:abs(a[c].corr(y,method='spearman')) for c in candidates}
    relevance={c:(float(v) if np.isfinite(v) else 0.) for c,v in relevance.items()};chosen=[]
    while len(chosen)<k:
        available=[c for c in candidates if c not in chosen and (not chosen or corr.loc[c,chosen].max()<.9)]
        if not available:break
        score=lambda c:relevance[c]-.5*(corr.loc[c,chosen].max() if chosen else 0.)
        chosen.append(sorted(available,key=lambda c:(-score(c),c))[0])
    return tuple(chosen)
