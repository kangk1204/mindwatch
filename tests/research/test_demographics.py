import pandas as pd
from mindwatch.research.demographics import age_band_series
from mindwatch.research.features import demo_features


def test_birthdays_use_each_assessment_date_and_only_bands_are_returned():
    v=pd.DataFrame({'sample_id':['a','b','c','d'],'age_group':['1995-06-15']*4,
        'index_date':pd.to_datetime(['2025-06-14','2025-06-15','2026-06-14','1990-01-01']),'sex':['1']*4})
    bands,formats=age_band_series(v.age_group,v.index_date)
    assert bands.tolist()==['20-29','30-39','30-39','unknown']
    assert formats.eq('birth_date').all()
    encoded=demo_features(v)
    assert encoded.demo__age_decade_20.tolist()==[1,0,0,0]
    assert encoded.demo__age_decade_30.tolist()==[0,1,1,0]
    assert encoded.demo__age_unknown.tolist()==[0,0,0,1]
    assert not any('1995' in str(value) for value in encoded.to_numpy().ravel())


def test_explicit_age_bands_work_and_invalid_dates_do_not_become_ages():
    values=pd.Series(['20대','30~39','40-49','51','1900-01-01','2020-02-31','unavailable'])
    dates=pd.Series(pd.to_datetime(['2025-01-01']*len(values)))
    bands,_=age_band_series(values,dates)
    assert bands.tolist()==['20-29','30-39','40-49','50-59','unknown','unknown','unknown']
