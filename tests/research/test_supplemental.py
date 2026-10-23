import pandas as pd
import pytest
from mindwatch.research.supplemental import supplemental_membership,combine_supplemental_test


def test_old_test_people_never_enter_amended_development_even_for_early_dates():
    visits=pd.DataFrame({'sample_id':['a','b','c','d'],'subject_id':['old_train','old_train','reserved','reserved'],
        'index_date':pd.to_datetime(['2025-03-01','2026-02-01','2025-09-01','2026-02-01'])})
    original=visits[['sample_id','subject_id']].copy()
    original['partition']=['development','excluded_after_cutoff','temporal_test','temporal_test']
    result,reserved=supplemental_membership(visits,original,'2026-01-01')
    assert reserved=={'reserved'}
    assert set(result.subject_id)=={'old_train'}
    assert result.partition.tolist()==['development','excluded_after_supplemental_cutoff']
    assert not result.partition.eq('temporal_test').any()
    assert original.partition.tolist()==['development','excluded_after_cutoff','temporal_test','temporal_test']


def test_missing_original_membership_is_never_silently_admitted():
    visits=pd.DataFrame({'sample_id':['x'],'subject_id':['new_person'],'index_date':[pd.Timestamp('2025-01-01')]})
    original=pd.DataFrame({'sample_id':['old'],'subject_id':['old_person'],'partition':['development']})
    with pytest.raises(ValueError,match='subject_missing'):
        supplemental_membership(visits,original)


def test_supplemental_test_requires_both_participant_and_calendar_disjointness():
    dev=pd.DataFrame({'sample_id':['a'],'subject_id':['train'],'index_date':[pd.Timestamp('2025-12-31')],'partition':['development']})
    test=pd.DataFrame({'sample_id':['b','c'],'subject_id':['reserved']*2,'index_date':pd.to_datetime(['2025-12-31','2026-01-01']),'partition':['temporal_test']*2})
    result=combine_supplemental_test(dev,test)
    assert result.sample_id.tolist()==['a','c']
    overlap=test.copy();overlap['subject_id']='train'
    with pytest.raises(ValueError,match='participant_overlap'):combine_supplemental_test(dev,overlap)
    late=dev.copy();late['index_date']=pd.Timestamp('2026-01-01')
    with pytest.raises(ValueError,match='calendar_overlap'):combine_supplemental_test(late,test)
