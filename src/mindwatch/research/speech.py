"""Local-only speech representations with explicit provenance and review gates.

No patient payload is passed to a network client. Model fetch is a separate public
weight-only command; every inference loader requires an existing local directory.
File permissions do not make embeddings or transcripts anonymous.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aif", ".aiff"}
PUBLIC_MODELS = {
    "wavlm": "microsoft/wavlm-base-plus",
    "whisper": "openai/whisper-large-v3-turbo",
    "klue": "klue/roberta-base",
}
REVIEW_FIELDS = (
    "consent_status", "task", "speaker_status", "timestamp_status",
    "transcript_review", "deidentification_review", "leakage_review",
)
MAX_NESTED_ARCHIVE_BYTES = 512 * 1024**2


def _private_path(path):
    """Validate every file, even when a caller bypasses a directory helper."""
    from mindwatch.pipeline import private_output
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Private output files cannot be symlinks')
    return private_output(path)


def _child(parent, name):
    parent = Path(parent).resolve()
    if not isinstance(name, str) or Path(name).name != name or name in {'', '.', '..'}:
        raise ValueError('Invalid private artifact name')
    path = parent / name
    if path.is_symlink() or path.resolve().parent != parent:
        raise ValueError('Private artifact must stay inside its declared directory')
    return _private_path(path)


def _source_fingerprint(path):
    from .privacy import sha256
    path = Path(path)
    if not path.is_file():
        raise ValueError('Regular audio source file required')
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Audio source changed while fingerprinting')
    return {'fingerprint_version': 1, 'path': str(path.resolve()),
            'size_bytes': after.st_size, 'sha256': digest}


def _member_fingerprint(info, prefix):
    return {prefix+'_name_sha256': hashlib.sha256(info.filename.encode('utf-8')).hexdigest(),
            prefix+'_crc32': int(info.CRC), prefix+'_size': int(info.file_size)}


def _bounded_copy(source, destination, limit):
    total = 0
    while True:
        block = source.read(min(1024**2, limit - total + 1))
        if not block:
            return total
        total += len(block)
        if total > limit:
            raise ValueError('Nested audio archive exceeds configured byte limit')
        destination.write(block)


def _private_dir(path: str | Path) -> Path:
    from mindwatch.pipeline import private_output
    path = private_output(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def _private_file(path: str | Path) -> Path:
    path = _private_path(path)
    path.chmod(0o600)
    return path


def _json_private(path: str | Path, value: object) -> None:
    path = _private_path(path)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0), 0o600), "w") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    _private_file(path)


def _parquet_private(frame: pd.DataFrame, path: str | Path) -> None:
    path = _private_path(path)
    # Create with restrictive mode before a writer has a chance to persist data.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    os.fchmod(fd, 0o600)
    os.close(fd)
    frame.to_parquet(path, index=False)
    _private_file(path)


def _safe_error(error: Exception) -> str:
    """Return only exception class: library messages can contain private paths."""
    return type(error).__name__


def subject_matcher(known_ids: Mapping[str, str]):
    """Exact Unicode tokens, with a bounded fallback for punctuated IDs."""
    import unicodedata
    known = {unicodedata.normalize('NFC', str(k)): v for k, v in known_ids.items()}
    punctuation = [(re.compile(r'(?<![^\W_])' + re.escape(k) + r'(?![^\W_])'), v)
                   for k, v in known.items() if re.search(r'[/\\_.()\s-]', k)]
    def match(filename):
        # Participant tokens may reside in an archive directory, while the
        # recording basename contains only a timestamp. Keep the whole member
        # path in memory for provisional exact-token linkage, never serialize it.
        stem = unicodedata.normalize('NFC', str(filename))
        tokens = set(re.split(r'[/\\_.()\s-]+', stem))
        matches = {known[t] for t in tokens if t in known}
        matches.update(value for pattern, value in punctuation if pattern.search(stem))
        return next(iter(matches)) if len(matches) == 1 else None
    return match


def match_subject(filename: str, known_ids: Mapping[str, str]) -> str | None:
    """Raw identifiers stay in memory; ambiguous or substring matches are rejected."""
    return subject_matcher(known_ids)(filename)


def filename_datetime(filename: str) -> tuple[pd.Timestamp | None, str]:
    """Extract date/time as provisional filename metadata, never verified timing."""
    stem = Path(filename).stem
    candidates = []
    patterns = (
        (r"(?<!\d)(20\d{12})(?!\d)", "%Y%m%d%H%M%S"),
        (r"(?<!\d)(20\d{6})[_-](\d{6})(?!\d)", "%Y%m%d%H%M%S"),
        (r"(?<!\d)(20\d{2}[-_]\d{2}[-_]\d{2})[T_ -](\d{2}[-_:]\d{2}[-_:]\d{2})(?!\d)", "%Y%m%d%H%M%S"),
        (r"(?<!\d)(20\d{6})(?!\d)", "%Y%m%d"),
        (r"(?<!\d)(20\d{2}[-_]\d{2}[-_]\d{2})(?!\d)", "%Y%m%d"),
    )
    for pattern, fmt in patterns:
        for match in re.finditer(pattern, stem):
            raw = re.sub(r"\D", "", "".join(match.groups()))
            try:
                candidates.append(pd.Timestamp(dt.datetime.strptime(raw, fmt)))
            except ValueError:
                continue
        if candidates:
            unique = set(candidates)
            return (next(iter(unique)), "PROVISIONAL_FILENAME") if len(unique) == 1 else (None, "AMBIGUOUS_FILENAME")
    return None, "UNKNOWN"


def inventory_audio(original_dir: str | Path, known_ids: Mapping[str, str], output_dir: str | Path,
                    max_nested_bytes: int = MAX_NESTED_ARCHIVE_BYTES) -> dict:
    """Inventory loose audio and audio in ZIPs, including one nested ZIP level.

    Does not decode, transcribe, or infer consent/task/speaker from filenames. Raw
    source paths live only in private_sources.json (0600). Public-facing return
    value consists solely of counts and pseudonymous output paths.
    """
    root, out = Path(original_dir), _private_dir(output_dir)
    if (out / 'inventory_summary.json').exists():
        raise FileExistsError('Inventory version exists; use a new output directory.')
    if not root.is_dir():
        raise ValueError("Original directory is unavailable")
    if not isinstance(max_nested_bytes, int) or isinstance(max_nested_bytes, bool) or max_nested_bytes < 1:
        raise ValueError('Positive nested archive byte limit required')
    rows, sources, failures = [], {}, []
    matcher = subject_matcher(known_ids)
    from .sources import decoded_name

    def add(source_id, name, member_index=-1, inner_member_index=-1, size=0, member=None, child=None):
        stamp, provenance = filename_datetime(name)
        locator = f"{source_id}:{member_index}:{inner_member_index}"
        row = {
            "recording_id": hashlib.sha256(locator.encode()).hexdigest()[:24],
            "subject_id": matcher(name), "recording_date": stamp,
            "timestamp_provenance": provenance, "source_id": source_id,
            "member_index": member_index, "inner_member_index": inner_member_index,
            "byte_size": size, "consent_status": "unknown", "task": "unknown",
            "speaker_status": "unknown", "timestamp_status": "unknown",
            "transcript_review": "unknown", "deidentification_review": "unknown",
            "leakage_review": "unknown",
            'fingerprint_version': 1, 'source_sha256': sources[source_id]['sha256'],
            **{prefix+suffix: ('' if suffix=='_name_sha256' else -1)
               for prefix in ('member', 'child') for suffix in ('_name_sha256','_crc32','_size')},
        }
        if member is not None:row.update(_member_fingerprint(member, 'member'))
        if child is not None:row.update(_member_fingerprint(child, 'child'))
        rows.append(row)

    for source in sorted(root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in AUDIO_SUFFIXES | {".zip"}:
            continue
        source_id = hashlib.sha256(str(source.relative_to(root)).encode()).hexdigest()[:24]
        sources[source_id] = _source_fingerprint(source)
        if source.suffix.lower() in AUDIO_SUFFIXES:
            add(source_id, source.name, size=source.stat().st_size)
            continue
        try:
            with zipfile.ZipFile(source) as archive:
                for i, info in enumerate(archive.infolist()):
                    if info.is_dir():
                        continue
                    suffix = Path(info.filename).suffix.lower()
                    if suffix in AUDIO_SUFFIXES:
                        add(source_id, decoded_name(info), i, size=info.file_size, member=info)
                    elif suffix == ".zip":
                        try:
                            if info.file_size > max_nested_bytes:
                                raise ValueError('Nested archive exceeds configured byte limit')
                            # Avoid keeping large nested archives wholly in RAM.
                            with tempfile.SpooledTemporaryFile(max_size=64 * 1024**2) as spool:
                                with archive.open(info) as nested:
                                    _bounded_copy(nested, spool, max_nested_bytes)
                                spool.seek(0)
                                with zipfile.ZipFile(spool) as inner:
                                    for j, child in enumerate(inner.infolist()):
                                        if not child.is_dir() and Path(child.filename).suffix.lower() in AUDIO_SUFFIXES:
                                            add(source_id, decoded_name(child), i, j, child.file_size, member=info, child=child)
                                        elif not child.is_dir() and Path(child.filename).suffix.lower() == ".zip":
                                            failures.append({"source_id": source_id, "member_index": i,
                                                             "inner_member_index": j, "reason": "deeper_nesting_unsupported"})
                        except Exception as error:
                            failures.append({"source_id": source_id, "member_index": i, "reason": _safe_error(error)})
        except Exception as error:
            failures.append({"source_id": source_id, "reason": _safe_error(error)})
    columns = ["recording_id", "subject_id", "recording_date", "timestamp_provenance", "source_id",
               "member_index", "inner_member_index", "byte_size", *REVIEW_FIELDS,
               'fingerprint_version', 'source_sha256',
               *[prefix+suffix for prefix in ('member','child') for suffix in ('_name_sha256','_crc32','_size')]]
    frame = pd.DataFrame(rows, columns=columns)
    frame["recording_date"] = pd.to_datetime(frame["recording_date"])
    _parquet_private(frame, out / "audio_inventory.parquet")
    _json_private(out / "private_sources.json", sources)
    _json_private(out / "inventory_failures.json", failures)
    review = frame[["recording_id", "recording_date", *REVIEW_FIELDS]].copy()
    review["reviewer_reference"] = ""
    _parquet_private(review, out / "recording_review_template.parquet")
    summary = {"recordings": len(frame), "matched_recordings": int(frame.subject_id.notna().sum()),
               "matched_subjects": int(frame.subject_id.nunique()), "failures": len(failures),
               "main_processing_eligible": 0, "timestamp_status": "all_unverified",
               "inventory_path": str(out / "audio_inventory.parquet")}
    _json_private(out / "inventory_summary.json", summary)
    return summary


def read_recording_bytes(record: Mapping, sources: Mapping, max_bytes: int = 512 * 1024**2,
                         max_nested_bytes: int = MAX_NESTED_ARCHIVE_BYTES, *, _verified_sources=None) -> bytes:
    """Reopen a private archive locator without extracting patient files to disk."""
    source = sources[str(record['source_id'])]
    if not isinstance(source, Mapping) or source.get('fingerprint_version') != 1 or record.get('fingerprint_version') != 1:
        raise ValueError('Legacy audio locator lacks fingerprints; regenerate the inventory')
    if source.get('sha256') != record.get('source_sha256') or not re.fullmatch(r'[0-9a-f]{64}', str(source.get('sha256', ''))):
        raise ValueError('Audio source fingerprint mismatch; regenerate the inventory')
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (max_bytes, max_nested_bytes)):
        raise ValueError('Positive audio byte limits required')
    path = Path(source['path'])
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    key = (str(path.resolve()), source['sha256'])
    cache = {} if _verified_sources is None else _verified_sources
    if cache.get(key) != signature:
        actual = _source_fingerprint(path)
        if actual['sha256'] != source['sha256'] or actual['size_bytes'] != source.get('size_bytes'):
            raise ValueError('Audio source changed since inventory; regenerate the inventory')
        cache[key] = signature
    def finish(data):
        end = path.stat()
        if (end.st_dev,end.st_ino,end.st_size,end.st_mtime_ns,end.st_ctime_ns) != signature:
            raise ValueError('Audio source changed during reading')
        if len(data) > max_bytes or len(data) != int(record['byte_size']):
            raise ValueError('Recording size differs from reviewed inventory')
        return data
    def verify_member(info, prefix):
        for name, value in _member_fingerprint(info, prefix).items():
            if record.get(name) != value:
                raise ValueError('Archive member changed since inventory; regenerate the inventory')
    i, j = int(record["member_index"]), int(record["inner_member_index"])
    if i == -1 and j == -1:
        if path.stat().st_size > max_bytes:
            raise ValueError("Recording exceeds configured byte limit")
        with path.open('rb') as stream:
            return finish(stream.read(max_bytes + 1))
    if i < 0 or j < -1:
        raise ValueError('Invalid audio archive locator')
    with zipfile.ZipFile(path) as archive:
        if i >= len(archive.infolist()):raise ValueError('Audio archive member is unavailable')
        info = archive.infolist()[i]
        verify_member(info, 'member')
        if j < 0:
            if info.file_size > max_bytes:
                raise ValueError("Recording exceeds configured byte limit")
            with archive.open(info) as stream:
                return finish(stream.read(max_bytes + 1))
        if info.file_size > max_nested_bytes:
            raise ValueError('Nested archive exceeds configured byte limit')
        with tempfile.SpooledTemporaryFile(max_size=64 * 1024**2) as spool:
            with archive.open(info) as stream:
                _bounded_copy(stream, spool, max_nested_bytes)
            spool.seek(0)
            with zipfile.ZipFile(spool) as inner:
                if j >= len(inner.infolist()):raise ValueError('Nested audio member is unavailable')
                child = inner.infolist()[j]
                verify_member(child, 'child')
                if child.file_size > max_bytes:
                    raise ValueError("Recording exceeds configured byte limit")
                with inner.open(child) as stream:
                    return finish(stream.read(max_bytes + 1))


def eligibility(record: Mapping, semantic: bool = False) -> tuple[bool, list[str]]:
    """Fail closed on unverified metadata; semantic release needs human review."""
    reasons = []
    required = {"consent_status": "approved", "speaker_status": "participant_verified", "timestamp_status": "verified"}
    for name, expected in required.items():
        if record.get(name) != expected:
            reasons.append(f"{name}_unverified")
    if record.get("task") not in {"free_speech", "fixed_reading"}:
        reasons.append("task_unverified_or_ineligible")
    subject = record.get("subject_id")
    if subject is None or pd.isna(subject) or str(subject).strip() == "":
        reasons.append("subject_unmatched")
    if pd.isna(pd.to_datetime(record.get("recording_date"), errors="coerce")):
        reasons.append("recording_date_missing")
    if semantic:
        if record.get("task") != "free_speech":
            reasons.append("semantic_requires_free_speech")
        for field in ("transcript_review", "deidentification_review", "leakage_review"):
            if record.get(field) != "approved":
                reasons.append(f"{field}_required")
    return not reasons, reasons


def decode_audio(data: bytes, sample_rate: int = 16000, max_seconds: int = 3600) -> tuple[np.ndarray, int]:
    """Decode with local PyAV, downmix/resample to mono; retain original duration."""
    import av
    chunks = []
    total = 0
    try:
        with av.open(io.BytesIO(data)) as container:
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ValueError("No audio stream")
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunk = converted.to_ndarray().reshape(-1).astype(np.float32)
                    total += chunk.size
                    if total > sample_rate * max_seconds:
                        raise ValueError("Recording exceeds duration limit")
                    chunks.append(chunk)
            for converted in resampler.resample(None):
                chunk = converted.to_ndarray().reshape(-1).astype(np.float32)
                total += chunk.size
                if total > sample_rate * max_seconds:
                    raise ValueError('Recording exceeds duration limit')
                chunks.append(chunk)
    except Exception as error:
        raise ValueError(f"Audio decoding failed ({_safe_error(error)})") from None
    if not chunks:
        raise ValueError("Empty audio")
    audio = np.concatenate(chunks)
    if not np.isfinite(audio).all():
        raise ValueError("Nonfinite decoded samples")
    return audio, sample_rate


def energy_vad(audio: np.ndarray, sample_rate: int = 16000, frame_ms: int = 20,
               hop_ms: int = 10) -> dict:
    """Simple energy VAD for reproducible QC, not speaker or speech recognition.

    Frame coordinates always refer to the untrimmed recording. The returned mask
    is an energy-active mask; calling it speech requires additional validation.
    """
    x = np.asarray(audio, dtype=float)
    n, hop = max(1, sample_rate * frame_ms // 1000), max(1, sample_rate * hop_ms // 1000)
    starts = np.arange(0, max(len(x), 1), hop)
    rms = np.array([np.sqrt(np.mean(x[start:min(start+n, len(x))] ** 2)) if start < len(x) else 0.0 for start in starts])
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    threshold = max(-45.0, min(float(np.percentile(db, 20)) + 10.0, float(np.max(db)) - 6.0))
    active = db > threshold
    transitions = np.diff(np.r_[False, active, False].astype(int))
    segments = [(float(starts[a] / sample_rate), float(min(starts[b-1] + n, len(x)) / sample_rate))
                for a, b in zip(np.where(transitions == 1)[0], np.where(transitions == -1)[0])]
    return {"frame_start_seconds": starts / sample_rate, "rms": rms, "active": active,
            "segments_seconds": segments, "threshold_dbfs": threshold, "hop_seconds": hop / sample_rate}


def handcrafted_audio_features(audio: np.ndarray, sample_rate: int = 16000) -> dict[str, float]:
    """Energy/pause/pitch features; pauses use original coordinates, never VAD trim."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty mono audio")
    vad = energy_vad(x, sample_rate)
    active = vad["active"]
    duration = len(x) / sample_rate
    segments = vad["segments_seconds"]
    pauses = [max(0.0, b[0] - a[1]) for a, b in zip(segments[:-1], segments[1:])]
    zcr = float(np.mean(x[1:] * x[:-1] < 0)) if len(x) > 1 else 0.0
    features = {
        "audio__duration_seconds": duration, "audio__energy_active_fraction": float(active.mean()),
        "audio__rms_mean": float(vad["rms"].mean()), "audio__rms_std": float(vad["rms"].std()),
        "audio__rms_p95": float(np.percentile(vad["rms"], 95)), "audio__zero_crossing_rate": zcr,
        "audio__clipping_fraction": float(np.mean(np.abs(x) >= 0.999)),
        "audio__active_segment_count": float(len(segments)),
        "audio__internal_pause_seconds": float(sum(pauses)),
        "audio__internal_pause_mean_seconds": float(np.mean(pauses)) if pauses else 0.0,
        "audio__internal_pause_rate_per_minute": len(pauses) / duration * 60,
        "audio__f0_median_hz": math.nan, "audio__f0_iqr_hz": math.nan,
    }
    # Frame-local autocorrelation; energy and periodicity filters reject silence.
    pitches = []
    frame, hop = int(sample_rate * .04), int(sample_rate * .02)
    lo, hi = sample_rate // 400, sample_rate // 60
    for start in range(0, max(0, len(x)-frame+1), hop):
        window = x[start:start+frame].astype(float)
        if np.sqrt(np.mean(window**2)) < 10 ** (vad["threshold_dbfs"] / 20):
            continue
        window = (window-window.mean()) * np.hanning(frame)
        corr = np.correlate(window, window, mode="full")[frame-1:]
        if corr[0] <= 1e-10:
            continue
        lag = lo + int(np.argmax(corr[lo:hi+1]))
        if corr[lag] / corr[0] > .6:
            pitches.append(sample_rate / lag)
    if pitches:
        features["audio__f0_median_hz"] = float(np.median(pitches))
        features["audio__f0_iqr_hz"] = float(np.percentile(pitches, 75)-np.percentile(pitches, 25))
    return features


def _offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"


def _local_directory(path: str | Path) -> str:
    _offline()
    path = Path(path)
    if not path.is_dir() or not (path / "config.json").exists():
        raise ValueError("Explicit local model snapshot with config.json required")
    return str(path)


def download_public_models(cache_dir: str | Path, models: Sequence[str] = ("wavlm", "whisper", "klue")) -> dict:
    """Fetch whitelisted public weights only. No data input or arbitrary repository."""
    if not set(models).issubset(PUBLIC_MODELS):
        raise ValueError("Only the three declared public model identifiers are allowed")
    out = _private_dir(cache_dir)
    revisions = {}
    prior = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    try:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        from huggingface_hub import snapshot_download
        # Explicit False prevents an ambient Hugging Face token being attached.
        for name in models:
            snapshot = snapshot_download(repo_id=PUBLIC_MODELS[name], cache_dir=str(out), token=False, local_files_only=False,
                                         ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.ot", "*.tflite"])
            revisions[name] = {"repository": PUBLIC_MODELS[name], "revision": Path(snapshot).name,
                               "local_snapshot": str(Path(snapshot).resolve())}
        _json_private(out / "public_model_versions.json", revisions)
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return revisions


class FrozenWavLM:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        path = _local_directory(model_dir)
        self.device, self.torch = device, torch
        self.extractor = AutoFeatureExtractor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False).to(device).eval()
        self.model.requires_grad_(False)

    def encode(self, audio: np.ndarray, sample_rate: int = 16000, chunk_seconds: int = 20) -> np.ndarray:
        if sample_rate != 16000 or len(audio) < 400:
            raise ValueError("WavLM requires >=25 ms mono 16-kHz audio")
        sums, squares, count = None, None, 0
        with self.torch.inference_mode():
            for start in range(0, len(audio), sample_rate * chunk_seconds):
                chunk = audio[start:start+sample_rate*chunk_seconds]
                if len(chunk) < 400:
                    continue
                inputs = self.extractor(chunk, sampling_rate=sample_rate, return_tensors="pt")
                hidden = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).last_hidden_state[0]
                values = hidden.float().cpu().numpy().astype(np.float64)
                sums = values.sum(0) if sums is None else sums + values.sum(0)
                squares = (values**2).sum(0) if squares is None else squares + (values**2).sum(0)
                count += len(values)
        mean = sums / count
        return np.r_[mean, np.sqrt(np.maximum(squares/count-mean**2, 0))].astype(np.float32)


class LocalWhisper:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        path = _local_directory(model_dir)
        self.device, self.torch = device, torch
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(path, local_files_only=True, trust_remote_code=False).to(device).eval()
        self.model.requires_grad_(False)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        if sample_rate != 16000:
            raise ValueError("Whisper requires 16-kHz audio")
        segments = []
        with self.torch.inference_mode():
            for start in range(0, len(audio), 30*sample_rate):
                chunk = audio[start:start+30*sample_rate]
                if not len(chunk) or np.sqrt(np.mean(chunk**2)) < 1e-5:
                    continue
                inputs = self.processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
                generated = self.model.generate(input_features=inputs.input_features.to(self.device),
                                                language="korean", task="transcribe", do_sample=False,
                                                max_new_tokens=448, return_timestamps=False)
                text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
                tokens = text.split()
                repetition = bool(len(tokens) >= 12 and len(set(tokens))/len(tokens) < .35)
                segments.append({"start_seconds": start/sample_rate, "end_seconds": (start+len(chunk))/sample_rate,
                                 "text": text, "speaker": "unreviewed", "repetition_flag": repetition})
        return {"segments": segments, "status": "needs_review", "timestamp_precision": "30s_chunk_not_word_alignment",
                "language": "ko", "model": PUBLIC_MODELS["whisper"]}


PII_PATTERNS = (
    ("EMAIL", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+82[- .]?)?0?1[016789][- .]?\d{3,4}[- .]?\d{4}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)0\d{1,2}[- .]\d{3,4}[- .]\d{4}(?!\d)")),
    ("RESIDENT_ID", re.compile(r"(?<!\d)\d{6}[- ]?[1-8]\d{6}(?!\d)")),
    ("BIRTHDATE", re.compile(r"(?:생년월일|생일|출생일)\s*(?:은|는|:)?\s*(?:\d{2,4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|\d{2,4}[-./]\d{1,2}[-./]\d{1,2})")),
    ("NAME", re.compile(r"(?:제?\s*이름(?:은|:)?|성명\s*[:은]?)\s*[가-힣]{2,5}")),
    ("ADDRESS", re.compile(r"[가-힣]{2,}(?:로|길)\s*\d+(?:[- ]\d+)?(?:\s*\d+동\s*\d+호)?")),
)
LEAK_PATTERNS = (
    ("assessment_name", re.compile(r"PHQ\s*[- ]?\s*9|GAD\s*[- ]?\s*7|\bISI\b|우울\s*척도|불안\s*척도|불면\s*척도", re.I)),
    ("assessment_score", re.compile(r"(?:총점|합계\s*점수|설문\s*점수|평가\s*점수)|\d+\s*점(?:입니다|이에요|으로|이라고|이야)?")),
    ("assessment_qa", re.compile(r"(?:몇\s*번\s*문항|[0-9]+\s*번\s*문항|질문에\s*(?:답|응답)|지난\s*2\s*주.*(?:전혀|며칠|거의\s*매일))")),
)


def redact_pii(text: str, local_ner: Callable[[str], Iterable[tuple[int, int, str]]] | None = None) -> dict:
    """Regex candidate redaction + optional local NER; human review always required.

    NER is an in-process callable supplied by the caller, never a remote client.
    Failed NER raises rather than returning apparently cleared text.
    """
    spans = []
    for label, pattern in PII_PATTERNS:
        spans.extend((match.start(), match.end(), label) for match in pattern.finditer(text))
    if local_ner is not None:
        try:
            for start, end, label in local_ner(text):
                if not (0 <= start < end <= len(text)):
                    raise ValueError("NER span is outside text")
                spans.append((start, end, str(label)))
        except Exception:
            raise ValueError("Local NER failed; deidentification remains blocked") from None
    # Merge overlaps without accidentally restoring part of a longer identifier.
    merged = []
    for start, end, label in sorted(spans):
        if merged and start < merged[-1][1]:
            previous = merged[-1]
            merged[-1] = (previous[0], max(end, previous[1]), "PII")
        else:
            merged.append((start, end, label))
    redacted = text
    for start, end, label in reversed(merged):
        redacted = redacted[:start] + f"[{label}]" + redacted[end:]
    return {"text": redacted, "redaction_count": len(merged), "status": "needs_human_review",
            "ner_status": "executed_local" if local_ner is not None else "unavailable",
            "privacy_guarantee": False}


def leakage_reasons(text: str, remove_symptom_mentions: bool = False) -> list[str]:
    reasons = [name for name, pattern in LEAK_PATTERNS if pattern.search(text)]
    if remove_symptom_mentions and re.search(r"우울|불안|불면|자살|depress|anxiet|insomnia", text, re.I):
        reasons.append("direct_symptom_mention")
    return reasons


def sanitize_transcript(transcript: Mapping, remove_symptom_mentions: bool = False,
                        local_ner: Callable | None = None) -> dict:
    segments = []
    for segment in transcript.get("segments", []):
        raw_text = str(segment.get("text", ""))
        reasons = leakage_reasons(raw_text, remove_symptom_mentions)
        if segment.get("speaker") != "participant":
            reasons.append("speaker_not_reviewed_participant")
        if segment.get("repetition_flag"):
            reasons.append("asr_repetition")
        redacted = redact_pii(raw_text, local_ner)
        segments.append({"start_seconds": segment.get("start_seconds"), "end_seconds": segment.get("end_seconds"),
                         "text": "" if reasons else redacted["text"], "excluded_reasons": reasons,
                         "redaction_count": redacted["redaction_count"]})
    return {"segments": segments, "status": "needs_human_review", "semantic_ready": False,
            "ner_status": "executed_local" if local_ner is not None else "unavailable"}


def semantic_text(record: Mapping, reviewed_transcript: Mapping) -> str:
    valid, _ = eligibility(record, semantic=True)
    if not valid or reviewed_transcript.get("status") != "approved":
        return ""
    # Record-level approval cannot override a segment explicitly excluded for leak.
    text = " ".join(str(segment.get("text", "")) for segment in reviewed_transcript.get("segments", [])
                    if not segment.get("excluded_reasons"))
    return "" if leakage_reasons(text) else text.strip()


class FrozenKLUE:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        path = _local_directory(model_dir)
        self.device, self.torch = device, torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False).to(device).eval()
        self.model.requires_grad_(False)

    def encode(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Approved nonempty semantic text required")
        # Overflow chunks retain all tokens. Using tokenizer() also preserves the
        # tokenizer's native special-token processing across fast/slow backends.
        limit = min(512, int(self.tokenizer.model_max_length))
        encoded = self.tokenizer(text, truncation=True, max_length=limit,
                                 return_overflowing_tokens=True, return_special_tokens_mask=True,
                                 padding=True, return_tensors="pt")
        total, count = None, 0
        with self.torch.inference_mode():
            for chunk in range(encoded["input_ids"].shape[0]):
                special = encoded["special_tokens_mask"][chunk:chunk+1].to(self.device)
                inputs = {k: v[chunk:chunk+1].to(self.device) for k, v in encoded.items()
                          if k in self.tokenizer.model_input_names}
                output = self.model(**inputs).last_hidden_state[0]
                mask = inputs["attention_mask"][0].bool() & ~special[0].bool()
                value = output[mask].float().sum(0).cpu().numpy()
                total = value if total is None else total+value
                count += int(mask.sum())
        if not count:
            raise ValueError("No semantic tokens")
        return np.asarray(total/count, dtype=np.float32)


def make_tfidf_pipeline(C: float = 1.0):
    """Unfitted pipeline: call fit only on the current training partition.

    Korean-compatible character n-grams are a transparent, morphology-independent
    baseline. No vocabulary/IDF is learned until the pipeline's fit call.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=20000, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=C, solver="liblinear", max_iter=1000, random_state=42)),
    ])


def pool_recordings(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Equal-recording mean/std; recordings never become independent samples."""
    if not embeddings:
        raise ValueError("At least one recording embedding is required")
    matrix = np.stack(embeddings)
    return np.r_[matrix.mean(0), matrix.std(0)].astype(np.float32)


def asr_qc_manifest(inventory: pd.DataFrame, split_manifest: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Select a subject-balanced development-only ASR QC list, without labels.

    One recording per subject first; round-robin date/task/size strata then sample
    remaining development recordings if fewer than n distinct subjects exist.
    """
    if n < 1:
        raise ValueError("QC sample count must be positive")
    dev = set(split_manifest.loc[split_manifest.partition == "development", "subject_id"].dropna())
    forbidden = set(split_manifest.loc[split_manifest.partition != "development", "subject_id"].dropna())
    if dev & forbidden:
        raise ValueError("QC split contains subjects across partitions")
    eligible = inventory[inventory.subject_id.isin(dev)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["recording_id", "subject_id", "qc_status", "cer", "wer", "reviewer_reference"])
    rng = np.random.default_rng(seed)
    eligible["_random"] = rng.random(len(eligible))
    dates = pd.to_datetime(eligible.recording_date, errors="coerce")
    eligible["_stratum"] = (eligible.task.fillna("unknown").astype(str) + "|" + dates.dt.year.fillna(0).astype(int).astype(str)
                            + "|" + pd.cut(eligible.byte_size, [-1, 100000, 1000000, np.inf], labels=["small", "medium", "large"]).astype(str))
    first = eligible.sort_values("_random").drop_duplicates("subject_id")
    groups = [group.sort_values("_random") for _, group in first.groupby("_stratum", sort=True)]
    selected = []
    for pos in range(max(len(group) for group in groups)):
        for group in groups:
            if pos < len(group):
                selected.append(group.iloc[pos].recording_id)
    selected = selected[:n]
    remaining = eligible[~eligible.recording_id.isin(selected)].sort_values("_random").recording_id.tolist()
    selected += remaining[:max(0, n-len(selected))]
    result = eligible.set_index("recording_id").loc[selected].reset_index()[["recording_id", "subject_id"]]
    result["qc_status"] = "pending_reference_transcription"
    result["cer"], result["wer"] = np.nan, np.nan
    result["reviewer_reference"] = ""
    return result


def asr_error_rates(reference: str, hypothesis: str) -> dict[str, float]:
    """CER/WER against a human reference, with no text in returned metrics."""
    def distance(a, b):
        previous = list(range(len(b)+1))
        for i, left in enumerate(a, start=1):
            current = [i]
            for j, right in enumerate(b, start=1):
                current.append(min(current[j-1]+1, previous[j]+1, previous[j-1]+(left != right)))
            previous = current
        return previous[-1]
    ref_chars, hyp_chars = "".join(reference.split()), "".join(hypothesis.split())
    ref_words, hyp_words = reference.split(), hypothesis.split()
    if not ref_chars or not ref_words:
        raise ValueError("Nonempty human reference is required")
    return {"cer": distance(ref_chars, hyp_chars)/len(ref_chars), "wer": distance(ref_words, hyp_words)/len(ref_words)}


def process_audio_inventory(inventory_path: str | Path, review_path: str | Path, output_dir: str | Path,
                            wavlm_dir: str | Path | None = None, whisper_dir: str | Path | None = None,
                            klue_dir: str | Path | None = None, device: str = "cpu", limit: int | None = None) -> dict:
    """Process only explicitly reviewed recordings; default template processes zero.

    A fresh ASR transcript is always pending. Semantic encoders consume only a
    separately reviewed <output>/transcripts/<recording_id>.reviewed.json file.
    Human transcript approvals are never inferred from ASR confidence or regexes.
    """
    _offline()
    out = _private_dir(output_dir)
    transcript_dir = _private_dir(_child(out, 'transcripts'))
    embedding_dir = _private_dir(_child(out, 'embeddings'))
    inventory = pd.read_parquet(inventory_path)
    reviews = pd.read_csv(review_path, keep_default_na=False, dtype={'recording_id':'string'}) if Path(review_path).suffix.lower() == ".csv" else pd.read_parquet(review_path)
    for table in (inventory, reviews):
        if 'recording_id' not in table or table.recording_id.isna().any() or not table.recording_id.astype(str).str.fullmatch(r'[0-9a-f]{24}').all():
            raise ValueError('Inventory and review require opaque recording identifiers')
        if table.recording_id.duplicated().any():
            raise ValueError('Duplicate recording inventory or review rows')
    if not set(reviews.recording_id) <= set(inventory.recording_id):
        raise ValueError('Recording review does not match the supplied inventory')
    if 'fingerprint_version' not in inventory or not inventory.fingerprint_version.eq(1).all():
        raise ValueError('Legacy audio locator lacks fingerprints; regenerate the inventory')
    review_map = reviews.set_index("recording_id").to_dict("index")
    with open(Path(inventory_path).parent / "private_sources.json") as handle:
        sources = json.load(handle)
    wavlm = whisper = klue = None
    processed, deferred = [], []
    verified_sources = {}
    for row in inventory.to_dict("records"):
        review = review_map.get(row["recording_id"], {})
        for field in (*REVIEW_FIELDS, "recording_date"):
            if field in review:
                row[field] = review[field]
        allowed, reasons = eligibility(row)
        if not allowed:
            deferred.append({"recording_id": row["recording_id"], "reasons": reasons})
            continue
        if limit is not None and len(processed) >= limit:
            break
        try:
            waveform, rate = decode_audio(read_recording_bytes(row, sources, _verified_sources=verified_sources))
            features = handcrafted_audio_features(waveform, rate)
            result = {"recording_id": row["recording_id"], "subject_id": row["subject_id"],
                      "recording_date": pd.to_datetime(row["recording_date"]), "task": row["task"], **features}
            result.update({name: row.get(name, 'unknown') for name in REVIEW_FIELDS})
            result['reviewer_reference'] = review.get('reviewer_reference', '')
            if wavlm_dir:
                wavlm = wavlm or FrozenWavLM(wavlm_dir, device)
                vector = wavlm.encode(waveform, rate)
                np.save(_private_file_empty(_child(embedding_dir, f"{row['recording_id']}.wavlm.npy")), vector)
                result.update({f"audio__wavlm_{i:04d}": float(v) for i, v in enumerate(vector)})
            if whisper_dir:
                whisper = whisper or LocalWhisper(whisper_dir, device)
                transcript = whisper.transcribe(waveform, rate)
                _json_private(_child(transcript_dir, f"{row['recording_id']}.restricted_asr.json"), transcript)
                _json_private(_child(transcript_dir, f"{row['recording_id']}.review_candidate.json"), sanitize_transcript(transcript))
            reviewed = _child(transcript_dir, f"{row['recording_id']}.reviewed.json")
            if klue_dir and reviewed.exists():
                with reviewed.open() as handle:
                    text = semantic_text(row, json.load(handle))
                if text:
                    klue = klue or FrozenKLUE(klue_dir, device)
                    vector = klue.encode(text)
                    np.save(_private_file_empty(_child(embedding_dir, f"{row['recording_id']}.klue.npy")), vector)
                    result.update({f"text__klue_{i:04d}": float(v) for i, v in enumerate(vector)})
                    result['semantic_transcript_status'] = 'approved'
            processed.append(result)
        except Exception as error:
            deferred.append({"recording_id": row["recording_id"], "reasons": ["processing_error", _safe_error(error)]})
    columns = ["recording_id", "subject_id", "recording_date", "task"]
    _parquet_private(pd.DataFrame(processed) if processed else pd.DataFrame(columns=columns), out / "recording_features.parquet")
    _json_private(out / "deferred_recordings.json", deferred)
    summary = {"processed_recordings": len(processed), "deferred_recordings": len(deferred),
               "main_study_feature_table": "requires_visit_window_pooling_by_parent",
               "clinical_validation": False, "external_inference_calls": 0}
    _json_private(out / "processing_summary.json", summary)
    return summary


def _private_file_empty(path: Path) -> Path:
    path = _private_path(path)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    os.fchmod(fd, 0o600)
    os.close(fd)
    return _private_file(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone CLI also callable by the package command dispatcher."""
    from mindwatch.arguments import PrivateArgumentParser
    parser = PrivateArgumentParser(description="Local-only reviewed speech pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-models", help="Public model weights only; no patient inputs")
    download.add_argument("--cache-dir", required=True)
    download.add_argument("--models", nargs="+", choices=sorted(PUBLIC_MODELS), default=list(PUBLIC_MODELS))
    process = commands.add_parser("process", help="Unknown review metadata defers every recording")
    process.add_argument("--inventory", required=True)
    process.add_argument("--review", required=True)
    process.add_argument("--output", required=True)
    process.add_argument("--wavlm-dir")
    process.add_argument("--whisper-dir")
    process.add_argument("--klue-dir")
    process.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    process.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.command == "download-models":
        result = download_public_models(args.cache_dir, args.models)
        print(json.dumps({name: {"repository": value["repository"], "revision": value["revision"]} for name, value in result.items()}))
    else:
        print(json.dumps(process_audio_inventory(args.inventory, args.review, args.output, args.wavlm_dir,
                                               args.whisper_dir, args.klue_dir, args.device, args.limit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
