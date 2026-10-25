"""Prospective metadata checks use generated synthetic identities only."""
from dataclasses import replace
import pandas as pd
import pytest

from mindwatch.research.future_validation import FutureValidationLock


def example():
    lock = FutureValidationLock('a'*64, 'b'*64, '2020-01-01', '2020-01-02')
    frame = pd.DataFrame({'sample_id': ['synthetic_sample'], 'subject_id': ['synthetic_person'],
        'index_date': ['2020-01-03'], 'prediction_recorded_date': ['2020-01-02'],
        'label_available_date': ['2020-01-04'], 'latest_input_date': ['2020-01-01']})
    return lock, frame


def check(lock, frame):
    return lock.check(frame, model_sha256='a'*64, protocol_sha256='b'*64)


def test_valid_metadata_and_empty_batch():
    lock, frame = example()
    assert check(lock, frame)['rows'] == 1
    assert check(lock, frame.iloc[:0])['rows'] == 0
    assert not check(lock, frame)['independence_established_by_metadata_alone']


@pytest.mark.parametrize('field,value', [
    ('index_date', '2019-12-31'), ('prediction_recorded_date', '2020-01-04'),
    ('prediction_recorded_date', '2020-01-03'), ('latest_input_date', '2020-01-02'),
    ('prediction_recorded_date', '2019-12-31'), ('latest_input_date', '2020-01-03'),
    ('label_available_date', '2020-01-03'), ('index_date', '2020-01-03 15:00'),
    ('index_date', '2020-01-03T00:00:00+09:00'), ('index_date', None),
    ('subject_id', None), ('sample_id', '')])
def test_invalid_inputs_fail_closed(field, value):
    lock, frame = example()
    with pytest.raises(ValueError):check(lock, frame.assign(**{field:value}))


def test_prior_access_and_frozen_artifact_changes_rejected():
    lock, frame = example()
    with pytest.raises(ValueError):
        check(replace(lock, previously_accessed_samples=frozenset(frame.sample_id)), frame)
    with pytest.raises(ValueError):lock.check(frame, model_sha256='c'*64, protocol_sha256='b'*64)
    with pytest.raises(ValueError):lock.check(frame, model_sha256='a'*64, protocol_sha256='c'*64)
    with pytest.raises(ValueError):check(lock, pd.concat([frame, frame]))
    with pytest.raises(ValueError):replace(lock, first_assessment_date=lock.freeze_date)


def test_inputs_cannot_postdate_recorded_prediction():
    lock, frame = example()
    frame['prediction_recorded_date'] = '2020-01-01'
    with pytest.raises(ValueError):check(lock, frame)


def test_equivalent_date_spellings_do_not_create_distinct_assessments():
    lock, frame = example()
    duplicate = frame.assign(sample_id='second_synthetic_sample', index_date='2020/01/03')
    with pytest.raises(ValueError):check(lock, pd.concat([frame, duplicate], ignore_index=True))
