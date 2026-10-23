"""Read archives in place. Patient names/IDs never form public aliases."""
from __future__ import annotations
import csv
import io
import re
import unicodedata
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

MODALITIES = {
    'SCREEN_TIME': 'screen_time', '일어난시간': 'wake_time', '램수면': 'sleep_rem',
    'EMA_ANXIETY': 'ema_anxiety', 'EMA_DEPRESSION': 'ema_depression',
    'EMA_SLEEP': 'ema_sleep', 'EMA_STRESS': 'ema_stress',
    '걸음수': 'steps', '스크린타임': 'screen_time', '이동거리': 'distance',
    '총수면시간': 'sleep_total', '깊은수면시간': 'sleep_deep',
    '얕은수면시간': 'sleep_light', '렘수면시간': 'sleep_rem',
    '잠든시간': 'sleep_onset', '기상시간': 'wake_time',
    'HRV': 'hrv', '심박수': 'heart_rate', '조도센서': 'light',
    '근접센서': 'proximity', '피부체온': 'skin_temperature',
    '심부체온': 'core_temperature', '혈당': 'glucose', '혈압': 'blood_pressure',
    '호흡수': 'respiration', '산소포화도': 'spo2',
}
DAILY = {'steps', 'screen_time', 'distance', 'sleep_total', 'sleep_deep',
         'sleep_light', 'sleep_rem', 'sleep_onset', 'wake_time'}


def normalized(value) -> str:
    s = unicodedata.normalize('NFC', str(value)).strip().lstrip('\ufeff')
    return '' if s.lower() in {'nan', 'none', 'null', '', 'nat'} else s


def decoded_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode('cp437').decode('cp949')
        except UnicodeError:
            pass
    return unicodedata.normalize('NFC', name)


def date_series(s: pd.Series) -> pd.Series:
    # Preserve local calendar dates; no unsupported timezone conversion.
    d = s.astype(str).str.extract(r'((?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2})', expand=False)
    d = d.str.replace('/', '-', regex=False).str.replace('.', '-', regex=False)
    parsed = pd.to_datetime(d, errors='coerce', format='mixed').dt.normalize()
    compact = s.astype('string').str.extract(r'^((?:19|20)\d{6})(?:$|[ T])', expand=False)
    return parsed.fillna(pd.to_datetime(compact, format='%Y%m%d', errors='coerce'))


@dataclass(frozen=True)
class Source:
    source_id: str
    path: Path
    member_index: int
    modality: str
    export_date: str | None
    bytes: int
    crc32: int
    kind: str

    @contextmanager
    def open(self):
        with zipfile.ZipFile(self.path) as z:
            with z.open(z.infolist()[self.member_index]) as f:
                yield f

    def frame(self, **kwargs):
        with self.open() as f:
            return pd.read_csv(f, encoding='utf-8-sig', **kwargs)


def discover(original: Path) -> list[Source]:
    sources = []
    for index, path in enumerate(sorted(Path(original).glob('*.zip')), 1):
        m = re.search(r'(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})', unicodedata.normalize('NFC', path.name))
        stamp = '-'.join(m.groups()) if m else None
        with zipfile.ZipFile(path) as z:
            for j, info in enumerate(z.infolist()):
                if info.is_dir() or not info.filename.lower().endswith('.csv'):
                    continue
                name = decoded_name(info)
                modality = next((v for k, v in MODALITIES.items() if k in name.upper()), 'unknown')
                kind = 'sensor'
                if modality == 'unknown':
                    with z.open(info) as f:
                        h = next(csv.reader(io.TextIOWrapper(f, encoding='utf-8-sig')), [])
                    kind = 'clinical' if 'PHQ9_Score' in h else 'unstructured'
                    if kind == 'clinical':
                        modality = 'survey'
                sources.append(Source(f'A{index:02d}', path.resolve(), j, modality, stamp,
                                      info.file_size, info.CRC, kind))
    return sources


def load_clinical_raw(sources: list[Source]):
    """Only used in-process; raw identifiers are never serialized by this helper."""
    result = []
    for source in sources:
        if source.kind != 'clinical':
            continue
        df = source.frame(dtype=str)
        idcol = next((c for c in df if c.lower() in {'id', 'participant_id', 'userid'}), None)
        dtcol = next((c for c in df if c.lower() in {'timestamp', 'date', 'regdt'}), None)
        if idcol is None or dtcol is None:
            raise ValueError('clinical_identifier_or_date_schema_missing')
        df['_raw_id'] = df[idcol].map(normalized)
        df['_date'] = date_series(df[dtcol])
        result.append((source, df))
    if not result:
        raise ValueError('no_structured_clinical_sources')
    result.sort(key=lambda x: x[1]['_date'].min())
    return result
