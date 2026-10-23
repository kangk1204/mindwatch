import pandas as pd
from mindwatch.research.multimodal import pool_to_visits


def test_no_future_audio_or_fixed_reading_semantics():
    visits = pd.DataFrame({'sample_id':['v'], 'subject_id':['p'], 'index_date':pd.to_datetime(['2026-04-20'])})
    recordings = pd.DataFrame({'recording_id':['r1','r2','r3'], 'subject_id':['p']*3,
        'recording_date':pd.to_datetime(['2026-04-19','2026-04-20','2026-04-21']),
        'task':['fixed_reading','free_speech','free_speech'], 'audio__f0':[100.,200.,1e8], 'text__embedding':[3.,4.,1e8]})
    for key, value in {'consent_status':'approved','speaker_status':'participant_verified',
                       'timestamp_status':'verified','reviewer_reference':'synthetic_evidence',
                       'transcript_review':'approved','deidentification_review':'approved',
                       'leakage_review':'approved','semantic_transcript_status':'approved'}.items():
        recordings[key] = value
    out, audit = pool_to_visits(visits, recordings)
    assert out['audio__free_speech__f0'].iloc[0] == 200
    assert 'text__fixed_reading__embedding' not in out
    assert out['text__free_speech__embedding'].iloc[0] == 4
    assert audit['same_day_concurrent_visits'] == 1


def test_unknown_task_is_not_silently_included():
    import pytest
    visits = pd.DataFrame({'sample_id':['v'], 'subject_id':['p'], 'index_date':pd.to_datetime(['2026-04-20'])})
    recordings = pd.DataFrame({'recording_id':['r'], 'subject_id':['p'], 'recording_date':pd.to_datetime(['2026-04-19']), 'task':['unknown'],'audio__f0':[100.]})
    with pytest.raises(ValueError):
        pool_to_visits(visits, recordings)


def test_forged_task_alone_does_not_bypass_review_gates():
    import pytest
    visits = pd.DataFrame({'sample_id':['v'], 'subject_id':['p'], 'index_date':pd.to_datetime(['2026-04-20'])})
    recordings = pd.DataFrame({'recording_id':['r'], 'subject_id':['p'], 'recording_date':pd.to_datetime(['2026-04-19']), 'task':['free_speech'],'audio__f0':[100.]})
    with pytest.raises(ValueError):
        pool_to_visits(visits, recordings)
