import pytest
pytest.importorskip("torch")
import numpy as np
import pandas as pd
from mindwatch.research.controls import permute_trajectories, shuffle_time_order
from mindwatch.research.neural import flatten_sequence


def test_permutation_preserves_whole_trajectories_and_strata():
    frame = pd.DataFrame({'subject_id':np.repeat(['a','b','c','d'],2), 'index_date':list(pd.date_range('2026-01-01', periods=2))*4,
        'wave':[1,2]*4, 'partition':['development']*8, 'y':[0,1,1,1,0,0,1,0]})
    result,audit = permute_trajectories(frame,'y')
    assert sorted(tuple(g.y) for _,g in result.groupby('subject_id')) == sorted(tuple(g.y) for _,g in frame.groupby('subject_id'))
    assert audit['permutable_subjects'] == 4


def test_time_shuffle_preserves_observation_bag():
    values = np.arange(28.).reshape(2,14,1);mask=np.ones_like(values)
    flat=flatten_sequence(values,mask,np.ones_like(values));columns=[f'sensor__seq_{i}' for i in range(flat.shape[1])]
    result=shuffle_time_order(pd.DataFrame(flat,columns=columns),columns)
    assert np.array_equal(np.sort(result[columns[:14]].to_numpy(),axis=1),np.sort(values[:,:,0],axis=1))
    assert not np.array_equal(result[columns[:14]].to_numpy(),values[:,:,0])
