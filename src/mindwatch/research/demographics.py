"""Convert documented birth dates to visit-time age bands without exposing DOB."""
import numpy as np
import pandas as pd


def age_band_series(values,index_dates):
    raw=pd.Series(values,index=values.index).astype('string').str.strip()
    dates=pd.to_datetime(pd.Series(index_dates,index=raw.index),errors='coerce')
    dob_like=raw.str.match(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T].*)?$').fillna(False)
    birthday=pd.to_datetime(raw.where(dob_like),errors='coerce',format='mixed')
    birthday_not_reached=(dates.dt.month<birthday.dt.month)|((dates.dt.month==birthday.dt.month)&(dates.dt.day<birthday.dt.day))
    years=dates.dt.year-birthday.dt.year-birthday_not_reached.astype(float)
    years=years.where(dob_like&birthday.notna()&dates.notna()&birthday.le(dates))
    # Preserve explicit age bands or numeric ages from other documented exports.
    band_number=pd.to_numeric(raw.str.extract(r'^(\d{1,3})(?:\s*(?:대|~|-|–)|\s*$)',expand=False),errors='coerce')
    years=years.where(dob_like,band_number)
    decade=(years//10*10).where(years.between(0,109))
    bands=pd.Series('unknown',index=raw.index,dtype='string')
    for lower in range(0,110,10):
        bands.loc[decade.eq(lower)]=f'{lower}-{lower+9}'
    formats=pd.Series(np.where(dob_like,'birth_date',np.where(band_number.notna(),'explicit_band_or_age','unknown')),index=raw.index)
    return bands,formats
