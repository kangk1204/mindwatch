"""Pure missing-data stress transforms; no bundled public or private records."""
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
SEED=42
SCENARIOS=("retain75","retain50","retain25","missing3","missing7")

def features(windows):
    """No interpolation. Slope uses actual day positions; minimum7 observations."""
    x=np.asarray(windows,float);out=np.full((len(x),x.shape[2]*3),np.nan)
    for i in range(len(x)):
        for c in range(x.shape[2]):
            good=np.isfinite(x[i,:,c]);v=x[i,good,c]
            if len(v)>=4:
                out[i,3*c]=np.median(v);out[i,3*c+1]=np.subtract(*np.percentile(v,[75,25]))
            if len(v)>=7:out[i,3*c+2]=theilslopes(v,np.flatnonzero(good)).slope
    return out

def mask_windows(windows,scenario,seed=SEED):
    if scenario not in SCENARIOS:raise ValueError('Unknown missingness scenario')
    x=np.array(windows,copy=True,dtype=float);rng=np.random.default_rng(seed)
    for i in range(len(x)):
        days=np.flatnonzero(np.isfinite(x[i]).any(axis=1))
        if scenario.startswith('retain'):
            n=int(np.floor(len(days)*int(scenario[6:])/100));keep=rng.choice(days,n,replace=False) if n else []
            remove=np.ones(len(x[i]),bool);remove[keep]=False;x[i,remove]=np.nan
        else:
            n=min(int(scenario[7:]),len(x[i]));start=rng.integers(0,len(x[i])-n+1);x[i,start:start+n]=np.nan
    return x

def past_window(daily,anchor,length=14):
    """Date-only anchor excluded; exact day grid, never future interpolation."""
    anchor=pd.Timestamp(anchor)
    if pd.isna(anchor) or anchor.tz is not None or anchor!=anchor.normalize():
        raise ValueError('Date-only local anchor required')
    return daily.reindex(pd.date_range(anchor-pd.Timedelta(days=length),periods=length)).to_numpy(float)

def cluster_change_ci(people,change,n_boot=500):
    # Equal-person mean prediction perturbation; resample whole people.
    g=np.unique(people);means=np.array([np.mean(change[people==v]) for v in g]);rng=np.random.default_rng(SEED)
    return np.percentile([rng.choice(means,len(means),replace=True).mean() for _ in range(n_boot)],[2.5,97.5]).tolist()
