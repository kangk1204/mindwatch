"""Synthetic software checks; no patient recordings or texts are used."""
import io
import json
from pathlib import Path
import stat
import wave
import zipfile

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.speech import (
    asr_error_rates, asr_qc_manifest, decode_audio, eligibility, energy_vad,
    filename_datetime, handcrafted_audio_features, inventory_audio, leakage_reasons,
    make_tfidf_pipeline, match_subject, pool_recordings, process_audio_inventory, FrozenKLUE, FrozenWavLM,
    read_recording_bytes, redact_pii, sanitize_transcript, semantic_text,
)


def synthetic_wave(rate=8000, stereo=False):
    t = np.arange(rate)/rate
    tone = (np.sin(2*np.pi*180*t)*.3*32767).astype("<i2")
    if stereo:
        tone = np.stack([tone, tone], axis=1)
    handle = io.BytesIO()
    with wave.open(handle, "wb") as wav:
        wav.setnchannels(2 if stereo else 1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(tone.tobytes())
    return handle.getvalue()


def approved_record():
    return {"subject_id": "synthetic-hmac-1", "recording_date": "2025-02-03",
            "consent_status": "approved", "speaker_status": "participant_verified",
            "timestamp_status": "verified", "task": "free_speech",
            "transcript_review": "approved", "deidentification_review": "approved", "leakage_review": "approved"}


def test_exact_token_matching_and_ambiguity():
    known = {"P001": "hmac-1", "P002": "hmac-2"}
    assert match_subject("P001_20250101.wav", known) == "hmac-1"
    assert match_subject("XP001_20250101.wav", known) is None
    assert match_subject("P0019_20250101.wav", known) is None
    assert match_subject("P001_P002.wav", known) is None
    assert match_subject("folder/P001/20250101.wav", known) == "hmac-1"


def test_unicode_token_and_legacy_zip_name_decoding():
    from mindwatch.research.sources import decoded_name
    known = {'테스트001': 'hmac-1'}
    name = '테스트001_20250101.wav'
    assert match_subject(name, known) == 'hmac-1'
    assert match_subject('앞테스트001_20250101.wav', known) is None
    info = zipfile.ZipInfo(name.encode('cp949').decode('cp437'))
    assert match_subject(decoded_name(info), known) == 'hmac-1'


def test_timestamp_is_provisional_and_not_gate_approval():
    stamp, source = filename_datetime("P001_20250102_123456.wav")
    assert stamp == pd.Timestamp("2025-01-02 12:34:56")
    assert source == "PROVISIONAL_FILENAME"
    assert filename_datetime("P001_20250230.wav") == (None, "UNKNOWN")
    record = approved_record()
    record["timestamp_status"] = "PROVISIONAL_FILENAME"
    assert not eligibility(record)[0]


def test_nested_inventory_private_paths_and_default_zero_processing(tmp_path):
    original, output = tmp_path / "original", tmp_path / "output"
    original.mkdir()
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("folder/P001_20250101_123456.wav", synthetic_wave())
        archive.writestr("folder/XP001_20250101.wav", synthetic_wave())
    with zipfile.ZipFile(original / "synthetic_source.zip", "w") as archive:
        archive.writestr("nested_archive.zip", inner.getvalue())
        archive.writestr("P002_20250103.wav", synthetic_wave())
    summary = inventory_audio(original, {"P001": "hmac-1", "P002": "hmac-2"}, output)
    frame = pd.read_parquet(output / "audio_inventory.parquet")
    assert summary["recordings"] == 3 and summary["matched_recordings"] == 2
    assert frame.task.eq("unknown").all()
    assert "synthetic_source" not in frame.to_json(date_format="iso")
    assert "P001" not in frame.to_json(date_format="iso")
    assert stat.S_IMODE((output / "private_sources.json").stat().st_mode) == 0o600
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    sources = json.loads((output / "private_sources.json").read_text())
    nested = frame.loc[frame.inner_member_index >= 0].iloc[0].to_dict()
    assert read_recording_bytes(nested, sources)[:4] == b"RIFF"
    result = process_audio_inventory(output / "audio_inventory.parquet", output / "recording_review_template.parquet", output / "processed")
    assert result["processed_recordings"] == 0
    assert result["deferred_recordings"] == 3
    assert not list((output / "processed" / "transcripts").glob("*.json"))


@pytest.mark.parametrize("field", ["consent_status", "speaker_status", "timestamp_status", "task"])
def test_unverified_metadata_blocks_each_modality(field):
    record = approved_record()
    assert eligibility(record)[0]
    record[field] = "unknown"
    assert not eligibility(record)[0]
    assert not eligibility(record, semantic=True)[0]


def test_missing_subject_and_semantic_review_are_fail_closed():
    record = approved_record()
    record["subject_id"] = pd.NA
    assert not eligibility(record)[0]
    record = approved_record()
    record["deidentification_review"] = "unknown"
    assert eligibility(record)[0]
    assert not eligibility(record, semantic=True)[0]


def test_pyav_decodes_mono16k_preserving_duration():
    audio, rate = decode_audio(synthetic_wave(stereo=True))
    assert rate == 16000 and audio.ndim == 1
    assert abs(len(audio)/rate-1.0) < .005
    assert np.isfinite(audio).all()
    with pytest.raises(ValueError, match="Audio decoding failed"):
        decode_audio(b"this is not audio")


def test_original_timeline_and_silence_prosody_features():
    rate = 16000
    t = np.arange(rate)/rate
    tone = np.sin(2*np.pi*180*t).astype(np.float32)*.25
    audio = np.r_[tone, np.zeros(rate), tone]
    vad = energy_vad(audio, rate)
    assert vad["segments_seconds"][1][0] >= 1.9
    features = handcrafted_audio_features(audio, rate)
    assert features["audio__duration_seconds"] == 3
    assert .9 < features["audio__internal_pause_seconds"] < 1.1
    assert 170 < features["audio__f0_median_hz"] < 190
    silence = handcrafted_audio_features(np.zeros(rate, dtype=np.float32), rate)
    assert silence["audio__energy_active_fraction"] == 0
    assert np.isnan(silence["audio__f0_median_hz"])


def test_pii_redaction_is_candidate_and_never_automatic_clearance():
    source = "제 이름은 가나다 연락처 010-0000-0000 이메일 example@example.test 생일은 2000년 1월 1일"
    result = redact_pii(source)
    assert "010-0000-0000" not in result["text"]
    assert "example@example.test" not in result["text"]
    assert "가나다" not in result["text"]
    assert "2000년 1월 1일" not in result["text"]
    assert result["status"] == "needs_human_review"
    assert not result["privacy_guarantee"]
    assert result["ner_status"] == "unavailable"
    with pytest.raises(ValueError, match="Local NER failed"):
        redact_pii("합성 문장", local_ner=lambda _: [(-1, 2, "NAME")])


def test_assessment_and_evaluator_segments_removed():
    transcript = {"segments": [
        {"text": "PHQ-9 총점 12점입니다", "speaker": "participant"},
        {"text": "합성 질문", "speaker": "evaluator"},
        {"text": "동네 산책을 했습니다", "speaker": "participant"},
    ]}
    result = sanitize_transcript(transcript)
    assert not result["semantic_ready"]
    assert result["segments"][0]["text"] == ""
    assert result["segments"][1]["text"] == ""
    assert result["segments"][2]["text"] == "동네 산책을 했습니다"
    assert leakage_reasons("요즘 우울한 기분", remove_symptom_mentions=True) == ["direct_symptom_mention"]
    assert semantic_text(approved_record(), result) == ""
    result["status"] = "approved"
    assert semantic_text(approved_record(), result) == "동네 산책을 했습니다"
    fixed = approved_record()
    fixed["task"] = "fixed_reading"
    assert eligibility(fixed)[0]
    assert semantic_text(fixed, result) == ""


def test_qc_selection_uses_development_subjects_only():
    inventory = pd.DataFrame({"recording_id": [f"r{i}" for i in range(12)],
                              "subject_id": [f"s{i//2}" for i in range(12)],
                              "recording_date": pd.date_range("2025-01-01", periods=12),
                              "task": ["unknown"]*12, "byte_size": [10000]*12})
    split = pd.DataFrame({"subject_id": ["s0", "s1", "s2", "s3", "s4", "s5"],
                          "partition": ["development"]*4+["temporal_test"]*2})
    result = asr_qc_manifest(inventory, split, n=4)
    assert len(result) == 4 and result.subject_id.nunique() == 4
    assert set(result.subject_id) <= {"s0", "s1", "s2", "s3"}
    bad = pd.concat([split, pd.DataFrame({"subject_id": ["s0"], "partition": ["temporal_test"]})])
    with pytest.raises(ValueError, match="across partitions"):
        asr_qc_manifest(inventory, bad)


def test_cer_wer_and_pooling_do_not_emit_input_text():
    rates = asr_error_rates("가나 다라", "가나 다마")
    assert rates == {"cer": .25, "wer": .5}
    assert np.allclose(pool_recordings([np.array([1., 3.]), np.array([3., 7.])]), [2., 5., 1., 2.])


def test_tfidf_vocabulary_learned_only_on_fit_partition():
    pipeline = make_tfidf_pipeline()
    assert not hasattr(pipeline["tfidf"], "vocabulary_")
    pipeline.fit(["가나다 산책", "가나다 산책", "라마바 운동", "라마바 운동"], [0, 0, 1, 1])
    assert "보류" not in pipeline["tfidf"].vocabulary_
    assert pipeline.predict_proba(["보류 시험 문장"]).shape == (1, 2)


def test_local_frozen_encoders_with_random_tiny_models_and_network_disabled(tmp_path, monkeypatch):
    """Real loader/forward paths with random synthetic weights, not clinical models."""
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    import socket
    from transformers import (BertConfig, BertModel, BertTokenizer, WavLMConfig,
                              WavLMModel, Wav2Vec2FeatureExtractor)
    text_dir, audio_dir = tmp_path / "text", tmp_path / "audio"
    tokenizer = BertTokenizer(vocab={"[PAD]": 0, "[UNK]": 1, "[CLS]": 2,
                                    "[SEP]": 3, "[MASK]": 4, "가나다": 5, "산책": 6},
                              model_max_length=16)
    tokenizer.save_pretrained(text_dir)
    BertModel(BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=16)).save_pretrained(text_dir)
    Wav2Vec2FeatureExtractor(sampling_rate=16000).save_pretrained(audio_dir)
    WavLMModel(WavLMConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                          intermediate_size=16, conv_dim=(8, 8), conv_stride=(5, 2), conv_kernel=(10, 3),
                          num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
                          num_buckets=8, max_bucket_distance=16)).save_pretrained(audio_dir)
    def no_network(*args, **kwargs):
        raise AssertionError("Network calls must not occur during inference")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    text_encoder = FrozenKLUE(text_dir)
    text_vector = text_encoder.encode("가나다 산책 "*40)
    assert text_vector.shape == (8,) and np.isfinite(text_vector).all()
    assert all(not parameter.requires_grad for parameter in text_encoder.model.parameters())
    audio_encoder = FrozenWavLM(audio_dir)
    waveform = np.sin(np.arange(8000)*2*np.pi*180/16000).astype(np.float32)*.2
    audio_vector = audio_encoder.encode(waveform)
    assert audio_vector.shape == (16,) and np.isfinite(audio_vector).all()
    assert all(not parameter.requires_grad for parameter in audio_encoder.model.parameters())
    with pytest.raises(ValueError, match="Explicit local model snapshot"):
        FrozenKLUE("klue/roberta-base")
