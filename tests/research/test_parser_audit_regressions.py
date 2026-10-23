"""Bounded synthetic parser/privacy regressions; never read cohort material."""
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.parsing import robust_clock, robust_dates
from mindwatch.research.sources import Source
from mindwatch.research.features import aggregate_daily
from mindwatch.research.metadata_audit import timestamp_grid
from mindwatch.research.speech import (
    _bounded_copy, _child, _private_file_empty, inventory_audio,
    process_audio_inventory, read_recording_bytes,
)


@pytest.fixture(scope='module')
def hourly_binary(tmp_path_factory):
    compiler = shutil.which('clang++') or shutil.which('g++')
    if compiler is None:
        pytest.skip('C++17 compiler unavailable')
    from mindwatch.research import raw_hourly
    output = tmp_path_factory.mktemp('synthetic-hourly-parser') / 'parser'
    result = subprocess.run([compiler, '-std=c++17', '-O1',
                             str(Path(raw_hourly.__file__).with_name('hourly_stream.cpp')),
                             '-o', str(output)], capture_output=True, timeout=120)
    assert result.returncode == 0
    return output


def run_parser(binary, tmp_path, content, n=1, bp=False):
    output = tmp_path / 'hours.csv'
    result = subprocess.run([str(binary), str(n), str(int(bp)), str(output)],
                            input=content.encode(), capture_output=True, timeout=10)
    return result, output


def test_hourly_whitespace_is_missing_and_true_zero_remains(hourly_binary, tmp_path):
    result, output = run_parser(hourly_binary, tmp_path,
        'timestamp,synthetic\n2020-01-01 00:00:00,  \n2020-01-01 00:01:00,0\n2020-01-01 00:02:00,2\n')
    assert result.returncode == 0
    audit = json.loads(result.stderr)
    data = pd.read_csv(output)
    assert audit['observed'] == 2
    assert data.n.tolist() == [2] and data.zero.tolist() == [1]
    assert data['median'].tolist() == [1.0]


@pytest.mark.parametrize('content', [
    '', '2020-01-01 00:00:00,1\n',
    'time,a,a\n2020-01-01 00:00:00,1,2\n',
    'time,a\n2020-01-01 00:00:00,2,9\n',
    'time,a\n2020-01-01 00:00:00\n',
    'time,a\n2020-01-01 00:00:00,"2',
    'time,a\n2020-01-01 00:00:00,"2"garbage\n',
    'time,a\n2020-02-30 00:00:00,2\n',
    'time,a\n2021-02-29 00:00:00,2\n',
    'time,a\n2020-01-01 00:00:nan,2\n',
    'time,a\n2020-01-01 00:00:00garbage,2\n',
    'time,a\n2020-01-01 00:00:00+09:00,2\n',
    'time,a\n2020-01-01 00:00:00Z,2\n',
    'time,a\n2020-01-01 00:00:00,2junk\n',
    'time,a\n2020-01-01 00:00:00,inf\n',
    'time,a\n2020-01-01 00:00:00,1e999\n',
    'time,a\n2020-01-01 00:00:00,2\n2020-01-01 00:00:00,3\n',
    'time,a\n2020-01-01 00:01:00,2\n2020-01-01 00:00:00,3\n',
    'time,a\n2020-01-01 00:00:00,'+'1' * 1024+'\n',
])
def test_hourly_rejects_malformed_input(hourly_binary, tmp_path, content):
    result, _ = run_parser(hourly_binary, tmp_path, content)
    assert result.returncode != 0
    assert result.stderr == b'hourly_parser_invalid_input_or_io_failure\n'


def test_hourly_subsecond_leapday_pressure_and_crlf(hourly_binary, tmp_path):
    result, output = run_parser(hourly_binary, tmp_path,
        '\ufefftimestamp,synthetic\r\n2020-02-29 00:00:00.1,"120/80"\r\n2020-02-29 00:00:00.2,100/70\r\n', bp=True)
    assert result.returncode == 0
    data = pd.read_csv(output)
    assert data.component.tolist() == [0, 1]
    assert data['median'].tolist() == [110.0, 75.0]
    assert data.first_time.eq('2020-02-29 00:00:00.1').all()


def test_hourly_output_open_and_dimension_failures(hourly_binary, tmp_path):
    content=b'time,a\n2020-01-01 00:00:00,2\n'
    for n,output in [('1',tmp_path/'missing'/'out.csv'), ('99999999999999999999',tmp_path/'out.csv'), ('-1',tmp_path/'out.csv')]:
        result=subprocess.run([str(hourly_binary),n,'0',str(output)],input=content,capture_output=True,timeout=10)
        assert result.returncode != 0


def test_wrapper_does_not_publish_partial_output(tmp_path):
    if not (shutil.which('clang++') or shutil.which('g++')):
        pytest.skip('C++17 compiler unavailable')
    from mindwatch.research.raw_hourly import aggregate_csv
    source=tmp_path/'input.csv';output=tmp_path/'output.csv'
    source.write_text('time,a\n2020-01-01 00:00:00,2\n2020-01-01 00:00:00,3\n')
    with pytest.raises(ValueError,match='Hourly parsing failed'):
        aggregate_csv(source,output,1)
    assert not output.exists()
    assert source.read_text().endswith(',3\n')


def test_clock_anchoring_and_adjacent_meridiem():
    values=['12:30AM','1:30PM','오전 12:30','오후 1:30','2020-01-01 23:30:00.5',
            '123:00','noise 23:00tail','12:30 AM PM','AM 12:30 PM','24:00','12:60']
    expected=[.5,13.5,.5,13.5,23.5+.5/3600,*([np.nan]*6)]
    np.testing.assert_allclose(robust_clock(pd.Series(values)),expected,equal_nan=True)


def test_dates_are_complete_local_fields():
    values=pd.Series(['20200101','2020/01/02','2020.01.03 12:30:00.1','2020-01-04T23:30',
                      '2020-01-013','word2020-01-01foo','2020-01-01 25:00:00','2020-01-01T00:00:00Z',
                      '2020-01-01 00:00:00+09:00','2020-02-30','20200101 garbage'])
    result=robust_dates(values)
    assert result.iloc[:4].dt.day.tolist()==[1,2,3,4]
    assert result.iloc[4:].isna().all()


def test_metadata_fractional_seconds_preserved_without_timezone_conversion():
    result=timestamp_grid(['2020.01.01 00:00:00.1','2020.01.01 00:00:00.2'])
    assert result['valid_timestamps']==2 and result['explicit_timezone_variants']==0
    assert result['median_wall_clock_step_seconds']==pytest.approx(.1)
    mixed=timestamp_grid(['2020-01-01 00:00:00+09:00','2020-01-01 00:01:00Z'])
    assert mixed['explicit_timezone_variants']==2 and mixed['median_wall_clock_step_seconds']==60


def test_daily_raw_header_duplicates_rejected_before_pandas_mangling(tmp_path):
    path=tmp_path/'source.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('steps.csv','date,synthetic-key,synthetic-key\n2020-01-01,1,9\n')
    source=Source('A01',path,0,'steps','2020-01-02',0,0,'sensor')
    with pytest.raises(ValueError,match='duplicate_or_empty_source_headers'):
        aggregate_daily([source],{'synthetic-key':'subject-a'})


def audio_inventory(tmp_path, nested=False):
    original=tmp_path/'original';original.mkdir()
    archive=original/'synthetic.zip'
    data=[('SYNTHETIC1_20200101.wav',b'a'),('SYNTHETIC2_20200101.wav',b'b')]
    with zipfile.ZipFile(archive,'w') as z:
        if nested:
            content=io.BytesIO()
            with zipfile.ZipFile(content,'w') as inner:
                for name,value in data:inner.writestr(name,value)
            z.writestr('nested.zip',content.getvalue())
        else:
            for name,value in data:z.writestr(name,value)
    output=tmp_path/'inventory'
    inventory_audio(original,{'SYNTHETIC1':'subject-a','SYNTHETIC2':'subject-b'},output)
    frame=pd.read_parquet(output/'audio_inventory.parquet')
    sources=json.loads((output/'private_sources.json').read_text())
    return archive,output,frame,sources


def test_audio_reordered_archive_rejected_and_legacy_locator_closed(tmp_path):
    archive,_,frame,sources=audio_inventory(tmp_path)
    row=frame.iloc[0].to_dict()
    assert read_recording_bytes(row,sources)==b'a'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('SYNTHETIC2_20200101.wav',b'b');z.writestr('SYNTHETIC1_20200101.wav',b'a')
    with pytest.raises(ValueError,match='changed since inventory'):
        read_recording_bytes(row,sources)
    with pytest.raises(ValueError,match='regenerate'):
        read_recording_bytes(row,{row['source_id']:str(archive)})


def test_audio_nested_member_fingerprint_and_byte_limit(tmp_path):
    _,_,frame,sources=audio_inventory(tmp_path,nested=True)
    row=frame.iloc[0].to_dict()
    assert read_recording_bytes(row,sources)==b'a'
    bad=dict(row);bad['child_crc32']+=1
    with pytest.raises(ValueError,match='member changed'):
        read_recording_bytes(bad,sources)
    with pytest.raises(ValueError,match='Nested archive'):
        read_recording_bytes(row,sources,max_nested_bytes=1)
    with pytest.raises(ValueError,match='byte limit'):
        _bounded_copy(io.BytesIO(b'abc'),io.BytesIO(),2)


@pytest.mark.parametrize('bad_id',['../../escaped','/tmp/escaped','bad-id','a'*24])
def test_audio_id_validation_and_duplicates_precede_processing(tmp_path,bad_id):
    _,folder,frame,_=audio_inventory(tmp_path)
    frame.loc[0,'recording_id']=bad_id
    if bad_id=='a'*24:frame.loc[1,'recording_id']=bad_id
    frame.to_parquet(folder/'audio_inventory.parquet',index=False)
    reviews=frame[['recording_id']];reviews.to_csv(folder/'review.csv',index=False)
    with pytest.raises(ValueError,match='identifiers|Duplicate'):
        process_audio_inventory(folder/'audio_inventory.parquet',folder/'review.csv',tmp_path/'processed')
    assert not (tmp_path/'escaped.wavlm.npy').exists()


def test_audio_child_and_file_helpers_block_symlink_and_git(tmp_path):
    output=tmp_path/'output';output.mkdir()
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
    (output/'embeddings').symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError,match='inside'):_child(output,'embeddings')
    (elsewhere/'.git').mkdir()
    with pytest.raises(ValueError,match='checkout'):_private_file_empty(elsewhere/'synthetic.npy')
