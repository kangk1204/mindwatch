"""Pinned external MiniRocket adapter; upstream GPL source is not redistributed."""
from pathlib import Path
import hashlib
import importlib.util
import sys
import urllib.request
import numpy as np
from mindwatch.models import person_weights
from .privacy import secure_dir

REVISION="0b1c245d9c9dbc50886f28bc7b32d5d45b5663d6"
UPSTREAM="https://github.com/angus924/minirocket"
MEMBERS={"code/minirocket_multivariate.py":"43d1cdcf045e013adf10af3591fa47f0f3efcae9","LICENSE":"f288702d2fa16d3cdf0035b15a9fcbc552cd88e7"}


def _blob(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def prepare_source(directory):
    """Explicitly download public source and its license to a private external directory."""
    directory=secure_dir(directory)
    for name,expected in MEMBERS.items():
        target=directory/name;secure_dir(target.parent)
        if target.exists():
            if target.is_symlink() or _blob(target.read_bytes())!=expected:raise ValueError("Existing upstream source differs")
            continue
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/angus924/minirocket/{REVISION}/{name}",timeout=30) as response:
            data=response.read(1_000_001)
        if len(data)>1_000_000 or _blob(data)!=expected:raise ValueError("Upstream source verification failed")
        with target.open("xb") as stream:stream.write(data)
        target.chmod(0o600)
    return directory


def load_source(directory):
    if directory is None:raise ValueError("Prepare and supply external MiniRocket source explicitly")
    directory=Path(directory).resolve()
    for name,expected in MEMBERS.items():
        path=directory/name
        if path.is_symlink() or not path.is_file() or _blob(path.read_bytes())!=expected:raise ValueError("Source and GPL license must match the pinned upstream revision")
    name="_mindwatch_external_minirocket_"+REVISION[:12]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,directory/"code/minirocket_multivariate.py")
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name,None);raise
    return sys.modules[name]


class RocketTransform:
    def __init__(self,source_dir,num_features=2016):
        self.source_dir=source_dir;self.num_features=num_features

    def _inputs(self,values):
        a=np.asarray(values,np.float32).copy()
        if a.ndim!=3 or a.shape[1:]!=self.shape or np.isinf(a).any():raise ValueError("Sequence schema changed")
        a[:,:,~self.seen]=np.nan;mask=np.isfinite(a)
        x=np.where(mask,(a-self.mean)/self.scale,0)
        delta=np.zeros_like(a);last=np.full((len(a),a.shape[2]),-1)
        for t in range(a.shape[1]):
            delta[:,t]=np.where(last>=0,t-last,a.shape[1]+1);last=np.where(mask[:,t],t,last)
        delta=np.log1p(delta)/np.log1p(a.shape[1]+1);delta[:,:,~self.seen]=0
        return np.ascontiguousarray(np.concatenate([x,mask.astype(np.float32),delta],2).transpose(0,2,1), dtype=np.float32),mask

    def fit(self,values,groups):
        a=np.asarray(values,np.float32)
        if a.ndim!=3 or a.shape[1]<9 or a.shape[2]<1 or not len(a) or np.asarray(groups).shape!=(len(a),) or np.isinf(a).any():raise ValueError("Real sequences of at least nine time bins required")
        self.shape=a.shape[1:];w=person_weights(groups)[:,None,None];mask=np.isfinite(a)
        den=(mask*w).sum((0,1));self.seen=den>0
        self.mean=(np.nan_to_num(a)*w).sum((0,1))/np.maximum(den,1e-12)
        self.scale=np.sqrt((np.where(mask,(a-self.mean)**2,0)*w).sum((0,1))/np.maximum(den,1e-12));self.scale[self.scale<1e-6]=1
        x,obs=self._inputs(a);good=obs.any((1,2))
        if not good.any():raise ValueError("No observed fitting sequence")
        module=load_source(self.source_dir)
        from numba import njit
        @njit
        def initialize():np.random.seed(42)
        initialize();self.state=module.fit(x[good],num_features=self.num_features)
        self.n_features_out_=int(module.transform(x[good][:1],self.state).shape[1])
        return self

    def transform(self,values):
        x,_=self._inputs(values)
        if not len(x):return np.empty((0,self.n_features_out_),np.float32)
        return load_source(self.source_dir).transform(x,self.state)
