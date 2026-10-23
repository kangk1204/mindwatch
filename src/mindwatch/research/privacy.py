"""Restricted local artifacts; no logging of patient material."""
from __future__ import annotations
import hashlib
import hmac
import json
import math
import os
import secrets
import tempfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd


def secure_dir(path: Path) -> Path:
    from mindwatch.pipeline import private_output
    path = private_output(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


@contextmanager
def atomic_private_path(path: Path, suffix: str = ''):
    """Create private bytes first; replace only after a successful complete write."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('artifact_symlink_not_allowed')
    secure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.write-', suffix=suffix)
    os.close(fd)
    try:
        yield Path(tmp)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: Path, obj) -> None:
    with atomic_private_path(path) as tmp:
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(clean_json(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    with atomic_private_path(path, '.parquet') as tmp:
        frame.to_parquet(tmp, index=False)


def write_joblib(path: Path, model) -> None:
    import joblib
    with atomic_private_path(path, '.joblib') as tmp:
        joblib.dump(model, tmp)


def get_key(private_dir: Path) -> bytes:
    path = secure_dir(private_dir) / 'identity.key'
    if path.is_symlink():
        raise ValueError('identity_key_symlink_not_allowed')
    if not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'wb') as f:
                f.write(secrets.token_bytes(32))
    if path.stat().st_mode & 0o077:
        raise ValueError('identity_key_permissions_are_not_private')
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError('invalid_identity_key')
    return key


def token(key: bytes, value: str, namespace: str = 'subject') -> str:
    return hmac.new(key, (namespace + '\0' + str(value)).encode(), hashlib.sha256).hexdigest()[:32]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def artifact_digest(*paths: Path) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(Path(path).name.encode())
        h.update(sha256(Path(path)).encode())
    return h.hexdigest()
