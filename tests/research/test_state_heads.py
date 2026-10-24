import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from mindwatch.research.state_heads import absolute_objective, fit, weights


def test_absolute_gradient():
    r=np.random.default_rng(42);x=np.column_stack([np.ones(30),r.normal(size=(30,4))])
    y=r.normal(size=30);w=np.ones(30)/30;t=r.normal(size=5)
    error=check_grad(lambda a:absolute_objective(a,x,y,w)[0],lambda a:absolute_objective(a,x,y,w)[1],t)
    assert error<1e-6


def test_affine_state_duplicate():
    x=pd.DataFrame({'base':np.arange(30.),'ss_a_level':np.sin(np.arange(30.)),
                    'ss_a_deviation':np.sin(np.arange(30.))-3})
    h=fit(x,np.arange(30)%2,np.arange(30),list(x),'class')
    assert h.diagnostic['affine_duplicates_removed']==['ss_a_deviation']


def test_unobserved_feature_is_blocked():
    x=pd.DataFrame({'base':np.arange(20.),'ss_b_level':np.nan})
    h=fit(x,np.arange(20)%2,np.arange(20),list(x),'class');z=x.copy();z['ss_b_level']=999
    np.testing.assert_array_equal(h.predict(x),h.predict(z))


def test_constant_absolute_is_not_mean():
    x=pd.DataFrame({'base':np.zeros(10)});y=np.r_[np.zeros(9),20.]
    h=fit(x,y,np.arange(10),list(x),'absolute',np.zeros(10))
    assert h.predict(x,np.zeros(10)).max()<.1


@pytest.mark.parametrize('invalid',['empty','nonbinary','ids','inf','previous'])
def test_contract_rejections(invalid):
    x=pd.DataFrame({'base':np.arange(10.)});y=np.arange(10)%2;ids=np.arange(10)
    task='class';previous=None
    if invalid=='empty':x=x.iloc[:0];y=y[:0];ids=ids[:0]
    if invalid=='nonbinary':y=np.arange(10)
    if invalid=='ids':ids=ids[:-1]
    if invalid=='inf':x.loc[0,'base']=np.inf
    if invalid=='previous':task='ridge';previous=np.ones(2)
    with pytest.raises(ValueError):fit(x,y,ids,list(x),task,previous)


def test_person_weights_and_mixed_key_types():
    w = weights([1, 1, '1'])
    assert w[0] + w[1] == pytest.approx(w[2])
    assert w.mean() == pytest.approx(1)
    with pytest.raises(ValueError):
        weights(['synthetic', None])


def test_predict_does_not_fit_or_accept_infinity():
    x = pd.DataFrame({'base': np.arange(20.)})
    head = fit(x, np.arange(20) % 2, np.arange(20), list(x), 'class')
    before = head.mean.copy()
    head.predict(x * 100)
    np.testing.assert_array_equal(before, head.mean)
    with pytest.raises(ValueError, match='Infinite'):
        head.predict(x.assign(base=np.inf))


def test_duplicate_columns_rejected():
    x = pd.DataFrame({'base': np.arange(10.)})
    with pytest.raises(ValueError, match='Unique'):
        fit(x, np.arange(10) % 2, np.arange(10), ['base', 'base'], 'class')
