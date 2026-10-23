"""Small masked wearable representations; fit only supplied training windows."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from mindwatch.models import person_weights

def gaps(mask):
 n,t,f=mask.shape;d=np.zeros(mask.shape,np.float32);last=np.full((n,f),-1)
 for k in range(t):d[:,k]=np.where(last>=0,k-last,t+1);last=np.where(mask[:,k],k,last)
 return np.asarray(np.log1p(d)/np.log1p(t+1),dtype=np.float32)

class PatchMAE(nn.Module):
 def __init__(self,f,patch):
  super().__init__();self.f=f;self.patch=patch;self.project=nn.Linear(f*patch*3,64)
  layer=nn.TransformerEncoderLayer(64,4,128,.2,batch_first=True,norm_first=True)
  self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.embedding=nn.Linear(64,16);self.decoder=nn.Linear(16,f*patch)
 def encode(self,a):
  z=self.project(a);pos=torch.arange(z.shape[1],device=z.device,dtype=z.dtype)[None,:,None];freq=torch.arange(64,device=z.device,dtype=z.dtype)[None,None,:]
  z=z+.05*torch.sin(pos/(1+freq));return self.embedding(self.encoder(z))
 def forward(self,a):return self.decoder(self.encode(a))


@dataclass
class MaskedRepresentation:
    seen: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    states: list
    patch: int
    shape: tuple
    trace: list

    def inputs(self,values):
        a=np.asarray(values,np.float32).copy()
        if a.ndim!=3 or a.shape[1:]!=self.shape or np.isinf(a).any():raise ValueError("Sequence shape changed")
        a[:,:,~self.seen]=np.nan;mask=np.isfinite(a)
        x=np.where(mask,(a-self.mean)/self.scale,0).astype(np.float32)
        delta=gaps(mask);delta[:,:,~self.seen]=0
        return np.concatenate([x,mask.astype(np.float32),delta],2),mask

    def transform(self,values):
        z,_=self.inputs(values)
        if not len(z):return np.empty((0,16))
        result=[]
        for state in self.states:
            net=PatchMAE(len(self.seen),self.patch);net.load_state_dict(state);net.eval();parts=[]
            with torch.no_grad():
                for start in range(0,len(z),256):
                    a=z[start:start+256];parts.append(net.encode(torch.tensor(a.reshape(len(a),a.shape[1]//self.patch,-1))).mean(1).numpy())
            result.append(np.concatenate(parts))
        return np.mean(result,0)


def fit_representation(values,ids,kind="ssl",source_dir=None,epochs=40,seeds=(42,2026,2027)):
    if kind=="minirocket":
        from .minirocket_adapter import RocketTransform
        return RocketTransform(source_dir).fit(values,ids)
    if kind not in ("ssl","random") or epochs<1 or not seeds:raise ValueError("Unknown representation or training schedule")
    a=np.asarray(values,np.float32);ids=np.asarray(ids)
    if a.ndim!=3 or a.shape[0]!=len(ids) or not len(a) or np.isinf(a).any():raise ValueError("Aligned finite-or-missing training windows required")
    patch=24 if a.shape[1]>=336 else 1
    if a.shape[1]%patch:raise ValueError("Sequence length must be divisible by patch length")
    obs=np.isfinite(a);w=person_weights(ids)[:,None,None];den=(obs*w).sum((0,1))
    mean=(np.nan_to_num(a)*w).sum((0,1))/np.maximum(den,1)
    scale=np.sqrt((np.where(obs,(a-mean)**2,0)*w).sum((0,1))/np.maximum(den,1));scale[scale<1e-6]=1
    rep=MaskedRepresentation(den>0,mean,scale,[],patch,a.shape[1:],[]);z,observed=rep.inputs(a)
    valid=observed.any((1,2));groups=[np.flatnonzero((ids==sid)&valid) for sid in np.unique(ids[valid])]
    if not groups:raise ValueError("No observed fitting sequence")
    for seed in seeds:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=PatchMAE(a.shape[-1],patch)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        if kind=="ssl":
            for epoch in range(epochs):
                # Rotate one allowed historical window/person each epoch.
                order=rng.permutation([int(rng.choice(ix)) for ix in groups]);losses=[];net.train()
                for start in range(0,len(order),64):
                    ix=order[start:start+64];xx=z[ix].copy();mask=observed[ix]
                    hide=(rng.random(mask.shape)<.25)&mask
                    if epoch%2==0:
                        width=min(a.shape[1],(3 if epoch%4==0 else 7)*patch)
                        begin=int(rng.integers(0,a.shape[1]-width+1));hide[:,begin:begin+width]|=mask[:,begin:begin+width]
                    xx[:,:,:a.shape[-1]][hide]=0;xx[:,:,a.shape[-1]:2*a.shape[-1]][hide]=0
                    delta=gaps(mask&~hide);delta[:,:,~rep.seen]=0;xx[:,:,2*a.shape[-1]:]=delta
                    hm=torch.tensor(hide.reshape(len(ix),-1,a.shape[-1]*patch))
                    if not hm.any():continue
                    target=torch.tensor(z[ix,:,:a.shape[-1]].reshape(len(ix),-1,a.shape[-1]*patch))
                    pred=net(torch.tensor(xx.reshape(len(ix),a.shape[1]//patch,-1)))
                    loss=(torch.where(hm,(pred-target)**2,0).sum((1,2))/hm.sum((1,2)).clamp(min=1)).mean()
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
                rep.trace.append({"seed":seed,"epoch":epoch+1,"loss":float(np.mean(losses)) if losses else None,"one_window_per_person":True})
        rep.states.append({name:t.detach().cpu() for name,t in net.state_dict().items()})
    return rep
