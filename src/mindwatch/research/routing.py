"""Known/new-person routing without bundled identities, choices or weights."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PopulationRouter:
    known_model: object
    new_model: object
    fit_people: tuple
    calibration_people: tuple=()
    version: str="unversioned_research"

    def predict(self,metadata,features,policy="auto"):
        if policy not in ("auto","known","new"):raise ValueError("Unknown population policy")
        if not {"sample_id","subject_id","index_date"}<=set(metadata):raise ValueError("Participant/sample/date metadata required")
        forbidden={"phq9","gad7","isi","y_phq9","y_gad7","y_isi","target"}
        if any(n in forbidden or n.startswith("target_") for n in metadata):raise ValueError("Current outcomes forbidden")
        if len(metadata)!=len(features) or not metadata.index.equals(features.index) or metadata.sample_id.duplicated().any():raise ValueError("Aligned unique rows required")
        if any(not n.startswith(("h_","e_","s_","history__","ema__","sensor__","demo__","quality__")) for n in features):raise ValueError("Only declared predictor fields allowed")
        people=metadata.subject_id.astype(str);fit=set(map(str,self.fit_people));cal=set(map(str,self.calibration_people))
        if people.isin(cal-fit).any():raise ValueError("Calibration-only people have no validated known/new route")
        seen=people.isin(fit).to_numpy()
        if policy=="new" and seen.any():raise ValueError("Population-seen person cannot be declared new")
        if policy=="known" and not seen.all():raise ValueError("Unseen person cannot be declared known")
        modes=np.where(seen,"known","new") if policy=="auto" else np.full(len(features),policy)
        out=metadata[["sample_id","subject_id","index_date"]].copy();out["policy"]=modes;out["prediction"]=np.nan
        for mode,model in (("known",self.known_model),("new",self.new_model)):
            rows=np.flatnonzero(modes==mode)
            if not len(rows):continue
            x=features.iloc[rows]
            p=np.asarray(model.predict_proba(x)[:,1] if hasattr(model,"predict_proba") else model.predict(x),float)
            if p.shape!=(len(rows),) or np.isinf(p).any() or ((p[np.isfinite(p)]<0)|(p[np.isfinite(p)]>1)).any():raise ValueError("Risk predictor must return aligned probabilities")
            if "h_previous_phq9" in x:p=np.where(x.h_previous_phq9.notna(),p,np.nan)
            out.iloc[rows,out.columns.get_loc("prediction")]=p
        out["status"]=np.where(np.isfinite(out.prediction),"ok","insufficient_data")
        out["model_version"]=self.version
        return out
