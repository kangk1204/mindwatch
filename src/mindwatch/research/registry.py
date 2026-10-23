"""Private frozen registry interface; no fitted registries are distributed."""
from pathlib import Path
import json
import os
import joblib
import numpy as np
import pandas as pd
from mindwatch.pipeline import private_output,source_digest,sha256
from .privacy import secure_dir,write_json,write_parquet


def freeze_router(router,feature_names,output,data_version,metadata=None,features=None):
    output=private_output(output)
    if output.exists():raise FileExistsError("Use a new private registry directory")
    if len(set(feature_names))!=len(feature_names):raise ValueError("Unique feature schema required")
    if (metadata is None)!=(features is None):raise ValueError("Prepared-row contract requires both metadata and features")
    if metadata is not None:
        if len(metadata)!=len(features) or not metadata.index.equals(features.index) or tuple(features)!=tuple(feature_names):raise ValueError("Aligned prepared contract required")
        if not {"sample_id","subject_id","index_date"}<=set(metadata) or metadata.sample_id.duplicated().any():raise ValueError("Unique prepared sample metadata required")
    secure_dir(output);path=output/"router.joblib"
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as stream:joblib.dump(router,stream,compress=3)
    lock={"model_sha256":sha256(path),"source_sha256":source_digest(),"feature_names":list(feature_names),"data_version":str(data_version),"contract_sha256":None}
    if metadata is not None:
        contract=metadata[["sample_id","subject_id","index_date"]].copy();contract["feature_hash"]=pd.util.hash_pandas_object(features,index=False).to_numpy()
        write_parquet(output/"contract.parquet",contract);lock["contract_sha256"]=sha256(output/"contract.parquet")
    write_json(output/"freeze.json",lock)


def predict_registry(directory,metadata,features,policy="auto"):
    """Load only your own trusted local joblib files, whose content can execute code."""
    directory=private_output(directory);lock=json.loads((directory/"freeze.json").read_text())
    if sha256(directory/"router.joblib")!=lock["model_sha256"] or source_digest()!=lock["source_sha256"]:raise ValueError("Frozen model/source changed")
    if tuple(features)!=tuple(lock["feature_names"]):raise ValueError("Frozen feature order changed")
    if lock["contract_sha256"] is not None:
        if sha256(directory/"contract.parquet")!=lock["contract_sha256"]:raise ValueError("Prepared contract changed")
        reference=pd.read_parquet(directory/"contract.parquet").set_index("sample_id")
        if not metadata.sample_id.isin(reference.index).all():raise ValueError("Unknown prepared sample")
        expected=reference.loc[metadata.sample_id]
        for key in ("subject_id","index_date"):
            if not np.array_equal(metadata[key].to_numpy(),expected[key].to_numpy()):raise ValueError("Prepared participant/date mismatch")
        if not np.array_equal(pd.util.hash_pandas_object(features,index=False).to_numpy(),expected.feature_hash.to_numpy()):raise ValueError("Prepared features changed or rows reordered")
    router=joblib.load(directory/"router.joblib");out=router.predict(metadata,features,policy)
    out["data_version"]=lock["data_version"];out["model_sha256"]=lock["model_sha256"]
    return out
