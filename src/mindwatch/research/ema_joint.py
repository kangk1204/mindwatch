"""Four-domain EMA sensor residual heads; no private cache or fitted states bundled.

Callers supply training-only targets and strictly prior P0 predictions explicitly.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from mindwatch.models import person_weights
from .neural import GRUDEncoder, MaskAwareTCN
from .mtm_adapter import load_official_mtm, verify_mtm_source
from .representations import gaps

DOMAINS=("ema_depression","ema_anxiety","ema_sleep","ema_stress")

class JointCore(nn.Module):
 def __init__(self,arch,nf,source_dir=None):
  super().__init__();self.arch=arch
  if arch=='mtm':
   verify_mtm_source(source_dir);cls=load_official_mtm(source_dir);self.encoder=cls(num_chn=nf,d_static=0,num_cls=4,ratios=[2,2],d_model=64,r_hid=2,drop=.2,norm_first=True,down_mode='concat')
  else:
   self.encoder=GRUDEncoder(nf,64) if arch=='grud' else MaskAwareTCN(nf,64,.2);self.encoder.feature_observed.fill_(True);self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,4))
 def forward(self,x,mask,delta):
  if self.arch=='mtm':
   with torch.sparse.check_sparse_tensor_invariants(enable=True):return self.encoder(x,~mask.bool(),torch.arange(x.shape[1]).expand(len(x),-1),x.new_empty(len(x),0))
  return self.head(self.encoder(x,mask,delta))

@dataclass
class JointModel:
 spec:dict
 mean:np.ndarray
 scale:np.ndarray
 seen:np.ndarray
 states:list
 trace:list
 domain:str
 task:str='ema'
 sensor_cols:tuple=()
 def arrays(self,v):
  mask=np.isfinite(v)&self.seen[None,None,:];a=np.where(mask,(v-self.mean)/self.scale,0).astype(np.float32);delta=gaps(mask);delta[:,:,~self.seen]=0;return a,mask.astype(np.float32),delta
 def raw(self,x,v,p0=None):
  if any(not n.startswith(('h_','e_','s_')) for n in x):raise ValueError('Current targets forbidden')
  arrays=self.arrays(v);pred=[]
  for state in self.states:
   net=JointCore(self.spec['architecture'],v.shape[-1],self.spec.get('mtm_source_dir'));net.load_state_dict(state);net.eval();parts=[]
   with torch.no_grad():
    for start in range(0,len(v),256):parts.append(net(*[torch.tensor(a[start:start+256]) for a in arrays]).numpy())
   pred.append(np.concatenate(parts)[:,DOMAINS.index(self.domain)])
  correction=np.median(pred,axis=0);mask=arrays[1];quality=np.minimum(1,mask.mean((1,2))*4);base=np.asarray(p0,float);available=mask.any((1,2));return np.where(available,np.clip(base+np.where(available,.5*quality*correction,0),0,10),base)

def fit(groups,values,targets,p0_values,domain,architecture="grud",*,epochs=40,seeds=(42,2026,2027),mtm_source_dir=None):
    if domain not in DOMAINS or architecture not in ("grud","tcn","mtm"):
        raise ValueError("Unknown EMA domain or architecture")
    values=np.asarray(values,np.float32);targets=np.asarray(targets,float);p0_values=np.asarray(p0_values,float);groups=np.asarray(groups)
    if values.ndim!=3 or targets.shape!=(len(values),4) or p0_values.shape!=targets.shape or len(groups)!=len(values) or np.isinf(values).any():
        raise ValueError("Aligned four-domain labels, prior forecasts and sensor sequences required")
    y=targets-p0_values;valid=np.isfinite(y);mask=np.isfinite(values);weights=person_weights(groups)[:,None,None]
    den=(mask*weights).sum((0,1));mean=(np.nan_to_num(values)*weights).sum((0,1))/np.maximum(den,1)
    scale=np.sqrt((np.where(mask,(values-mean)**2,0)*weights).sum((0,1))/np.maximum(den,1));scale[scale<1e-6]=1
    result=JointModel({"architecture":architecture,"mtm_source_dir":mtm_source_dir},mean,scale,den>0,[],[],domain)
    arrays=result.arrays(values);weight=np.zeros(y.shape,np.float32)
    for j in range(4):
        if valid[:,j].any():weight[valid[:,j],j]=person_weights(groups[valid[:,j]])
    fitrows=np.flatnonzero(valid.any(1)&mask.any((1,2)))
    if not len(fitrows):raise ValueError("No observed training window with a target")
    if epochs<1 or not seeds:raise ValueError("Positive epochs and fixed seeds required")
    for seed in seeds:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=JointCore(architecture,values.shape[-1],mtm_source_dir)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        for epoch in range(epochs):
            net.train();order=rng.permutation(fitrows);losses=[]
            for start in range(0,len(order),64):
                ix=order[start:start+64];prediction=net(*[torch.tensor(a[ix]) for a in arrays])
                target=torch.tensor(np.nan_to_num(y[ix]),dtype=torch.float32);w=torch.tensor(weight[ix])
                loss=(abs(prediction-target)*w).sum()/w.sum().clamp(min=1)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
            if epoch in (0,epochs-1):result.trace.append({"seed":seed,"epoch":epoch+1,"L1_loss":float(np.mean(losses))})
        result.states.append({key:value.detach().cpu() for key,value in net.state_dict().items()})
    return result
