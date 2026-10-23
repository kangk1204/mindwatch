import json
from mindwatch.research.metadata_audit import csv_layout,timestamp_grid


def test_headerless_record_never_releases_any_original_cell():
    result=csv_layout(['SECRET-PERSON','PRIVATE-RESPONSE','1950-02-03'],['SECRET-PERSON'])
    assert result['first_row_is_participant_record']
    assert not result['has_recognized_header']
    assert 'SECRET' not in json.dumps(result) and 'PRIVATE' not in json.dumps(result) and '1950' not in json.dumps(result)
    assert csv_layout(['ID','Timestamp','PHQ9_Score'])['has_recognized_header']


def test_mixed_timezone_exports_do_not_crash_or_claim_verified_acquisition():
    result=timestamp_grid(['2025-01-01 00:00:00+09:00','2025-01-01 00:05:00','2025-01-01 00:10:00Z'])
    assert result['valid_timestamps']==3
    assert result['most_common_wall_clock_step_seconds']==300
    assert result['timezone_semantics']=='mixed_or_unverified'


def test_invalid_and_duplicate_timestamps_are_counted_separately():
    result=timestamp_grid(['not_a_date','2025-01-01','2025-01-01','2025-01-02'])
    assert result['valid_timestamps']==3 and result['duplicate_timestamps']==1
    assert result['most_common_wall_clock_step_seconds']==86400
