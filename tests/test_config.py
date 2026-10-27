from dataclasses import asdict, replace
import json

import pytest

from mouse_sleep.config import Settings


@pytest.mark.parametrize("wrapped", [False, True])
def test_settings_round_trip_preserves_resume_identity(tmp_path, wrapped):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"metadata fixture")
    settings = Settings(video=str(video), output=str(tmp_path / "out"), roi=(5, 5, 80, 54))
    data = asdict(settings)
    if wrapped:
        data = {"settings": data, "identity": {"settings": settings.identity()}}
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8-sig")
    loaded = Settings.load(path)
    assert loaded == settings
    assert loaded.identity() == json.loads(json.dumps(settings.identity()))
    assert replace(loaded, roi=(6, 5, 80, 54)).identity() != settings.identity()


def test_identity_detects_source_and_timing_metadata_changes(tmp_path):
    video, timing = tmp_path / "source.mp4", tmp_path / "timing.csv"
    video.write_bytes(b"video")
    timing.write_bytes(b"timing")
    settings = Settings(video=str(video), timing_csv=str(timing), output=str(tmp_path / "out"))
    original = settings.identity()
    timing.write_bytes(b"changed timing")
    assert settings.identity() != original
    original = settings.identity()
    video.write_bytes(b"changed video")
    assert settings.identity() != original
