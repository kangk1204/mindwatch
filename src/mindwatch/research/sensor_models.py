"""Small fit-only tabular/representation heads, without persisted study artifacts."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression,QuantileRegressor
from mindwatch.models import person_weights

def select(x,y,k):
 cols=[n for n in x if x[n].notna().mean()>=.3 and x[n].nunique()>1]
 if not k:return tuple(cols)
 a=x[cols];corr=a.corr(method='spearman',min_periods=30).abs().fillna(0);yy=pd.Series(np.asarray(y),index=x.index)
 rel={n:abs(a[n].corr(yy,method='spearman')) for n in cols};rel={n:float(v) if np.isfinite(v) else 0 for n,v in rel.items()};chosen=[]
 while len(chosen)<k:
  eligible=[n for n in cols if n not in chosen and (not chosen or corr.loc[n,chosen].max()<.9)]
  if not eligible:break
  chosen.append(min(eligible,key=lambda n:(-(rel[n]-.5*(corr.loc[n,chosen].max() if chosen else 0)),n)))
 return tuple(chosen)

@dataclass
class Head:
 cols:tuple
 seen:np.ndarray
 pipeline:object
 constant:float
 task:str
 def predict(self,x):
  if tuple(x.columns)!=self.cols:raise ValueError('Input schema includes unknown/current predictors')
  if self.pipeline is None:return np.full(len(x),self.constant)
  a=x.to_numpy(float).copy();a[:,~self.seen]=np.nan
  return self.pipeline.predict_proba(a)[:,1] if self.task=='phq' else self.pipeline.predict(a)

def fit_head(x,y,ids,task,spec='linear',C=.01):
 y=np.asarray(y,float);valid=np.isfinite(y);x=x.iloc[np.flatnonzero(valid)];ids=np.asarray(ids)[valid];y=y[valid]
 a=x.to_numpy(float);seen=np.isfinite(a).any(axis=0);w=person_weights(ids);constant=float(np.average(y,weights=w))
 if not seen.any() or (task=='phq' and np.unique(y).size<2):return Head(tuple(x),seen,None,constant,task)
 if spec=='catboost':
  from catboost import CatBoostClassifier,CatBoostRegressor
  est=CatBoostClassifier(iterations=300,depth=2,learning_rate=.03,l2_leaf_reg=20,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False) if task=='phq' else CatBoostRegressor(loss_function='MAE',iterations=300,depth=2,learning_rate=.03,l2_leaf_reg=20,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
 else:est=(LogisticRegression(C=C,solver='lbfgs',max_iter=4000,random_state=42) if C==.01 else LogisticRegression(C=C,solver='saga',penalty='elasticnet',l1_ratio=.5,max_iter=4000,random_state=42)) if task=='phq' else QuantileRegressor(quantile=.5,alpha=.1 if C==.01 else .01,solver='highs')
 pipe=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),est)
 pipe.fit(a,y,**{pipe.steps[-1][0]+'__sample_weight':w});return Head(tuple(x),seen,pipe,constant,task)

@dataclass
class SensorModel:
 task:str
 spec:dict
 sensor_cols:tuple
 base:Head
 joint:Head
 representation:object=None
 calibrator:object=None
 threshold:float=.5
 projection:object=None
 observed_channels:object=None
 daily_parent:object=None
 daily_columns:tuple=()
 def design(self,x,values):
  if self.representation is None:return x[list(self.sensor_cols)].copy()
  z=self.representation.transform(values)
  if self.projection is not None:z=self.projection.transform(z)
  out=pd.DataFrame(z,columns=['s_rep_'+str(i) for i in range(z.shape[1])],index=x.index)
  return pd.concat([x[list(self.daily_columns)],out],axis=1) if self.daily_columns else out
 def raw(self,x,values,p0=None):
  if any(not n.startswith(('h_','e_','s_')) for n in x):raise ValueError('Current targets or metadata passed to model')
  basecols=list(self.base.cols);base=self.base.predict(x[basecols]) if self.task=='phq' else np.asarray(p0,float).copy()
  sx=self.design(x,values);joint=pd.concat([x[basecols],sx],axis=1) if self.task=='phq' else sx
  pred=self.joint.predict(joint);support=np.isfinite(values).any(axis=(1,2)) if self.representation else x[list(self.sensor_cols)].notna().any(axis=1).to_numpy()
  if self.representation:support=np.isfinite(values[:,:,self.representation.seen]).any(axis=(1,2)) if self.representation.seen.any() else np.zeros(len(x),bool)
  vv=values[:,:,self.representation.seen] if self.representation else values[:,:,self.observed_channels]
  quality=np.isfinite(vv).mean(axis=(1,2)) if vv.shape[-1] else np.zeros(len(x));quality=np.minimum(1,quality*4);alpha=self.spec.get('alpha',.5)
  p=base+support*quality*(alpha*(pred-base) if self.task=='phq' else alpha*pred)
  if self.daily_parent is not None:
   dp=self.daily_parent.predict(pd.concat([x[basecols],x[list(self.daily_columns)]],axis=1) if self.task=='phq' else x[list(self.daily_columns)])
   meaningful=[n for n in self.daily_columns if 'coverage' not in n];ds=x[meaningful].notna().any(axis=1).to_numpy() if meaningful else np.zeros(len(x),bool)
   qc=[n for n in x if n.startswith('s_fe_') and n.endswith('coverage14') and any(k.split('__')[0]==n.split('__')[0] for k in meaningful)]
   dq=np.clip(x[qc].mean(axis=1).to_numpy(float),0,1) if qc else np.zeros(len(x));daily=base+ds*dq*alpha*(dp-base if self.task=='phq' else dp)
   child=base+alpha*(pred-base if self.task=='phq' else pred);p=daily+support*quality*(child-daily)
  clipped=np.clip(p,1e-7,1-1e-7) if self.task=='phq' else np.clip(p,0,10)
  seen=self.representation.seen if self.representation is not None else self.observed_channels
  physical=np.isfinite(values[:,:,seen]).any((1,2)) if seen is not None and seen.any() else np.zeros(len(x),bool)
  meaningful=[name for name in self.daily_columns if 'coverage' not in name]
  daily=x[meaningful].notna().any(axis=1).to_numpy() if self.daily_parent is not None and meaningful else np.zeros(len(x),bool)
  return np.where(physical|daily,clipped,base)

def fit_sensor(x,values,y,ids,task,spec,basecols,p0=None):
 y=np.asarray(y,float);residual=y if task=='phq' else y-np.asarray(p0)
 rep=None;projection=None
 if spec['kind'] in ['minirocket','ssl','random']:
  from .representations import fit_representation
  rep=fit_representation(values,ids,spec['kind'],source_dir=spec.get('source_dir'),epochs=spec.get('epochs',40));z=rep.transform(values)
  if task=='ema' and z.shape[1]>64:
   from sklearn.decomposition import PCA
   projection=PCA(n_components=min(16,z.shape[0],z.shape[1]),svd_solver='randomized',random_state=42);z=projection.fit_transform(z)
  sx=pd.DataFrame(z,columns=['s_rep_'+str(i) for i in range(z.shape[1])],index=x.index);cols=tuple(sx)
 else:
  cols=select(x[[n for n in x if n.startswith('s_')]],residual,spec.get('k',8));sx=x[list(cols)]
 daily_cols=tuple();daily_parent=None
 if spec.get('view')=='hourly':
  daily_cols=select(x[[n for n in x if n.startswith('s_fe_')]],residual,8)
  if rep is not None:sx=pd.concat([x[list(daily_cols)],sx],axis=1)
 base=fit_head(x[list(basecols)],y,ids,'phq','linear',.01) if task=='phq' else Head(tuple(),np.array([],bool),None,0,'ema')
 joint=pd.concat([x[list(basecols)],sx],axis=1) if task=='phq' else sx
 head=fit_head(joint,residual,ids,task,'catboost' if spec['kind']=='catboost' else 'linear',spec.get('C',.01))
 if daily_cols:daily_parent=fit_head(pd.concat([x[list(basecols)],x[list(daily_cols)]],axis=1) if task=='phq' else x[list(daily_cols)],residual,ids,task,'linear',.01)
 return SensorModel(task,spec,cols,base,head,rep,projection=projection,observed_channels=np.isfinite(values).any(axis=(0,1)),daily_parent=daily_parent,daily_columns=daily_cols)
