"""Synthetic regression checks for date-only history and elapsed-day encoders."""
import numpy as np
import pandas as pd
import pytest

from mindwatch.research.ema_personalization import event_history
from mindwatch.research.associations import between_within
from mindwatch.research.missingness_stress import past_window


def test_event_history_rejects_intraday_anchor_and_excludes_current_day():
    anchors=pd.DataFrame({'subject_id':['synthetic'], 'index_date':['2024-01-03']})
    daily=pd.DataFrame({'subject_id':['synthetic']*3,
                        'date':pd.date_range('2024-01-02',periods=3),
                        **{domain:[1.,9.,8.] for domain in ['ema_depression','ema_anxiety','ema_sleep','ema_stress']}})
    result=event_history(anchors,daily)
    assert result.e_event8_ema_depression__last.iloc[0]==1
    assert result.e_event8_ema_depression__gap_days.iloc[0]==1
    anchors['index_date']='2024-01-03 15:00'
    with pytest.raises(ValueError,match='Date-only'):
        event_history(anchors,daily)


def test_other_daily_helpers_reject_intraday_anchors():
    frame=pd.DataFrame({'subject_id':['synthetic'], 'index_date':['2024-01-03 15:00'], 'x':[1.], 'y':[2.]})
    with pytest.raises(ValueError,match='Date-only'):
        between_within(frame,'x','y')
    with pytest.raises(ValueError,match='Date-only'):
        past_window(pd.Series([1.],index=pd.to_datetime(['2024-01-02'])),'2024-01-03 15:00')


@pytest.mark.parametrize('family',['classifier','joint'])
def test_encoder_elapsed_days_and_legacy_checkpoint_compatibility(family):
    pytest.importorskip('torch')
    from mindwatch.research.neural_comparators import NeuralModel
    from mindwatch.research.ema_joint import JointModel
    from mindwatch.research.sensor_models import Head
    from mindwatch.research.representations import gaps
    values=np.array([[[1.,np.nan],[np.nan,np.nan],[np.nan,np.nan],[2.,np.nan],[np.nan,np.nan]]],np.float32)
    spec={'sequence_length':5,'delta_units':'days'}
    seen=np.array([True,False]);mean=np.zeros(2);scale=np.ones(2)
    if family=='joint':
        model=JointModel(spec,mean,scale,seen,[],[],'ema_depression',np.ones(4,bool))
        arrays=lambda:model.arrays(values)
    else:
        model=NeuralModel('ema',spec,Head(tuple(),np.array([],bool),None,0,'ema'),mean,scale,seen,np.empty(0),np.empty(0),[],[])
        arrays=lambda:model.arrays(pd.DataFrame(index=[0]),values)
    _,mask,delta,*_=arrays()
    np.testing.assert_array_equal(delta[0,:,0],[0,1,2,3,1])
    np.testing.assert_array_equal(delta[:,:,1],0)
    model.spec.pop('delta_units')
    expected=gaps(mask.astype(bool));expected[:,:,1]=0
    np.testing.assert_array_equal(arrays()[2],expected)


def test_new_fits_record_day_units():
    pytest.importorskip('torch')
    from mindwatch.research import neural_comparators,ema_joint
    rng=np.random.default_rng(7);v=rng.normal(size=(6,5,2)).astype(np.float32)
    ids=np.array(['synthetic-'+str(i) for i in range(6)])
    f=pd.DataFrame({'subject_id':ids});x=pd.DataFrame({'e_history':np.arange(6,dtype=float)})
    model=neural_comparators.fit(f,x,v,np.array([0,1,0,1,0,1]),'phq','phq9','grud',epochs=1,seeds=(7,))
    assert model.spec['delta_units']=='days'
    joint=ema_joint.fit(ids,v,np.ones((6,4)),np.zeros((6,4)),'ema_depression',epochs=1,seeds=(7,))
    assert joint.spec['delta_units']=='days'
