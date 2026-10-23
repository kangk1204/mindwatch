"""Observed hourly patterns only; callers supply reconciled hourly tables."""
import numpy as np
import pandas as pd

def hourly_view(anchors,hourly,channels):
 from .contracts import calendar_table,sensor_channels
 anchors=calendar_table(anchors,'index_date');channels=list(sensor_channels(channels))
 if hourly is None:raise ValueError('Explicit reconciled hourly tables required')
 if not anchors.index_date.eq(anchors.index_date.dt.normalize()).all():raise ValueError('Date-only anchors required')
 hourly={ch:calendar_table(hourly[ch],'hour') for ch in channels}
 for table in hourly.values():
  if not table.hour.eq(table.hour.dt.floor('h')).all():raise ValueError('Hourly aggregates required')
  if not {'median','n'}<=set(table) or table.n.le(0).any():raise ValueError('Actual hourly observation counts required')
 n=len(anchors);v=np.full((n,336,len(channels)),np.nan,np.float32);counts=np.zeros_like(v);records=[{} for _ in range(n)]
 by={sid:ix.to_numpy() for sid,ix in anchors.groupby('subject_id').groups.items()}
 for j,ch in enumerate(channels):
  a=hourly[ch];groups={sid:g for sid,g in a.groupby('subject_id',sort=False)}
  for sid,ii in by.items():
   if sid not in groups:continue
   g=groups[sid].sort_values('hour');times=g.hour.to_numpy('datetime64[h]').astype(np.int64)
   vals=np.sign(g['median'].to_numpy())*np.log1p(abs(g['median'].to_numpy()));ns=g.n.to_numpy()
   for i in ii:
    end=int(anchors.index_date.iloc[i].to_datetime64().astype('datetime64[h]').astype(np.int64));lo=np.searchsorted(times,end-336);hi=np.searchsorted(times,end)
    tt=times[lo:hi];x=vals[lo:hi];num=ns[lo:hi];ix=tt-(end-336)
    v[i,ix,j]=x;counts[i,ix,j]=num
    day=tt//24;days=len(np.unique(day));r={};prefix='s_hour_'+ch+'__'
    r['coverage']=len(x)/336;r['observations']=float(num.sum());r['observed_days']=days
    if days>=4:
     r['median']=float(np.median(x));r['iqr']=float(np.quantile(x,.75)-np.quantile(x,.25));r['last']=float(x[-1]);r['hour_variation']=float(np.std(x))
     night=(tt%24<6);daytime=(tt%24>=8)&(tt%24<20)
     if np.unique(day[night]).size>=4 and np.unique(day[daytime]).size>=4:r['night_day']=float(np.median(x[night])-np.median(x[daytime]))
    if days>=7 and len(x)>2:r['slope']=float(np.polyfit((tt-tt[0])/24,x,1)[0])
    gooddays=[d for d in np.unique(day) if (day==d).sum()>=6];ok=np.isin(day,gooddays)
    if len(gooddays)>=7:
     angle=2*np.pi*(tt[ok]%24)/24;design=np.column_stack([np.ones(ok.sum()),np.sin(angle),np.cos(angle)]);b=np.linalg.lstsq(design,x[ok],rcond=None)[0]
     r['circadian_amplitude']=float(np.hypot(b[1],b[2]));phase=np.arctan2(b[1],b[2]);r['phase_sin']=float(np.sin(phase));r['phase_cos']=float(np.cos(phase))
    b0=np.searchsorted(times,end-1008);b1=np.searchsorted(times,end-336);bt=times[b0:b1];bx=vals[b0:b1]
    if days>=4 and np.unique(bt//24).size>=14:
     med=np.median(bx);iqr=np.quantile(bx,.75)-np.quantile(bx,.25);r['personal_delta']=float(np.median(x)-med)
     if iqr>0:r['personal_z']=float((np.median(x)-med)/iqr)
     daily_meds=np.array([np.median(x[day==d]) for d in sorted(np.unique(day))]);r['change_persistent_days']=float(np.sum(abs(daily_meds-med)>(iqr if iqr>0 else 0)))
    records[i].update({prefix+k:val for k,val in r.items()})

 # True common-hour association; no alignment through imputed observations.
 for j in range(len(channels)):
  for k in range(j+1,len(channels)):
   name='s_cross_'+channels[j]+'_'+channels[k]
   for i in range(n):
    x,y=v[i,:,j],v[i,:,k];good=np.isfinite(x)&np.isfinite(y)
    if good.sum()>=24 and np.std(x[good])>0 and np.std(y[good])>0:records[i][name]=float(np.corrcoef(x[good],y[good])[0,1])
    good=np.isfinite(x[:-24])&np.isfinite(y[24:])
    if good.sum()>=24 and np.std(x[:-24][good])>0 and np.std(y[24:][good])>0:records[i][name+'_lag1d']=float(np.corrcoef(x[:-24][good],y[24:][good])[0,1])
 return pd.DataFrame(records),v,counts
