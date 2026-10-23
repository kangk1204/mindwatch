"""Existing GRU-D/TCN and verified official MTM, new matched daily inputs."""
from dataclasses import dataclass
import numpy as np,pandas as pd,torch
from torch import nn
from . import sensor_models as m
from .representations import gaps
from mindwatch.models import person_weights
from mindwatch.research.neural import GRUDEncoder,MaskAwareTCN
from mindwatch.research.mtm_adapter import load_official_mtm,verify_mtm_source

class Core(nn.Module):
 def __init__(self,arch,nf,ns,source_dir=None):
  super().__init__();self.arch=arch
  if arch=='mtm':
   verify_mtm_source(source_dir);cls=load_official_mtm(source_dir);self.encoder=cls(num_chn=nf,d_static=ns,num_cls=1,ratios=[2,2],d_model=64,r_hid=2,drop=.2,norm_first=True,down_mode='concat')
  else:self.encoder=GRUDEncoder(nf,64) if arch=='grud' else MaskAwareTCN(nf,64,.2);self.head=nn.Sequential(nn.Linear(64+ns,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
 def forward(self,x,mask,delta,static):
  if self.arch=='mtm':
   times=torch.arange(x.shape[1],device=x.device).expand(len(x),-1);return self.encoder(x,~mask.bool(),times,static).reshape(-1)
  return self.head(torch.cat([self.encoder(x,mask,delta),static],dim=1)).reshape(-1)

@dataclass
class NeuralModel:
 task:str
 spec:dict
 base:m.Head
 mean:np.ndarray
 scale:np.ndarray
 seen:np.ndarray
 static_mean:np.ndarray
 static_scale:np.ndarray
 states:list
 trace:list
 sensor_cols:tuple=()
 def arrays(self,x,v):
  mask=np.isfinite(v)&self.seen[None,None,:];z=np.where(mask,(v-self.mean)/self.scale,0).astype(np.float32);delta=gaps(mask);delta[:,:,~self.seen]=0
  if self.task=='phq':
   a=x[list(self.base.cols)].to_numpy(dtype=float,copy=True);a[:,~self.base.seen]=np.nan;static=np.nan_to_num((a-self.static_mean)/self.static_scale).astype(np.float32)
  else:static=np.empty((len(x),0),np.float32)
  return z,mask.astype(np.float32),delta,static
 def raw(self,x,v,p0=None):
  if any(not n.startswith(('h_','e_','s_')) for n in x):raise ValueError('Current targets or metadata passed to neural model')
  z,mask,delta,static=self.arrays(x,v);pred=[]
  for state in self.states:
   net=Core(self.spec['architecture'],v.shape[-1],static.shape[-1],self.spec.get('mtm_source_dir'));net.load_state_dict(state);net.eval();parts=[]
   with torch.no_grad():
    for start in range(0,len(v),256):parts.append(net(*[torch.tensor(a[start:start+256]) for a in [z,mask,delta,static]]).numpy())
   q=np.concatenate(parts);pred.append(1/(1+np.exp(-np.clip(q,-30,30))) if self.task=='phq' else q)
  p=np.mean(pred,axis=0) if self.task=='phq' else np.median(pred,axis=0)
  base=self.base.predict(x[list(self.base.cols)]) if self.task=='phq' else np.asarray(p0,float);quality=np.minimum(1,mask.mean(axis=(1,2))*4);available=mask.any(axis=(1,2))
  q=base+np.where(available,quality*.5*(p-base if self.task=='phq' else p),0)
  clipped=np.clip(q,1e-7,1-1e-7) if self.task=='phq' else np.clip(q,0,10)
  return np.where(available,clipped,base)

def fit(f,x,v,y,task,domain,arch,*,p0_values=None,base_columns=None,mtm_source_dir=None,epochs=40,seeds=(42,2026,2027)):
 if task not in ['phq','ema'] or arch not in ['grud','tcn','mtm']:raise ValueError('Unknown neural task/architecture')
 if task=='ema' and p0_values is None:raise ValueError('Explicit strictly prior P0 forecasts required')
 if base_columns is None:base_columns=tuple(n for n in x if n.startswith('e_'))
 y=np.asarray(y,float);good=np.isfinite(y)
 if task=='ema':good&=np.isfinite(np.asarray(p0_values,float))
 f=f.loc[good].reset_index(drop=True);x=x.loc[good].reset_index(drop=True);v=v[good];y=y[good];base=m.fit_head(x[list(base_columns)],y,f.subject_id,'phq') if task=='phq' else m.Head(tuple(),np.array([],bool),None,0,'ema')
 mask=np.isfinite(v);w=person_weights(f.subject_id);den=(mask*w[:,None,None]).sum((0,1));mean=(np.nan_to_num(v)*w[:,None,None]).sum((0,1))/np.maximum(den,1);scale=np.sqrt((np.where(mask,(v-mean)**2,0)*w[:,None,None]).sum((0,1))/np.maximum(den,1));scale[scale<1e-6]=1
 if task=='phq':
  s=x[list(base.cols)].to_numpy(float);sm=np.nanmedian(s,axis=0);sm[~np.isfinite(sm)]=0;ss=np.nanstd(s,axis=0);ss[~np.isfinite(ss)|(ss<1e-6)]=1
 else:sm=ss=np.empty(0)
 result=NeuralModel(task,{'kind':'neural','architecture':arch,'view':'daily','alpha':.5,'mtm_source_dir':mtm_source_dir},base,mean,scale,den>0,sm,ss,[],[])
 arrays=result.arrays(x,v);target=y if task=='phq' else y-np.asarray(p0_values,float)[good];eligible=np.flatnonzero(mask.any((1,2)));rng=None
 if not len(eligible):raise ValueError('No supported training sensor window')
 for seed in seeds:
  torch.manual_seed(seed);rng=np.random.default_rng(seed);net=Core(arch,v.shape[-1],len(sm),mtm_source_dir);
  if arch!='mtm':net.encoder.feature_observed.fill_(True)
  opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
  for epoch in range(epochs):
   order=rng.permutation(eligible);net.train();losses=[]
   for start in range(0,len(order),64):
    ix=order[start:start+64];p=net(*[torch.tensor(a[ix]) for a in arrays]);yy=torch.tensor(target[ix],dtype=torch.float32);ww=torch.tensor(w[ix],dtype=torch.float32)
    err=nn.functional.binary_cross_entropy_with_logits(p,yy,reduction='none') if task=='phq' else abs(p-yy);loss=(err*ww).sum()/ww.sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(float(loss.detach()))
   if epoch in [0,epochs-1]:result.trace.append({'seed':seed,'epoch':epoch+1,'loss':float(np.mean(losses))})
  result.states.append({k:a.detach().cpu() for k,a in net.state_dict().items()})
 return result
