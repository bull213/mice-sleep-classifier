from dataclasses import replace
import pytest

from mouse_sleep.cli import main
from mouse_sleep.config import Cancelled, Settings


def test_cli_overrides_saved_settings_and_keeps_resume(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"metadata fixture")
    settings = Settings(video=str(video), output=str(tmp_path / "out"), masking_mode="legacy", roi=(1, 2, 30, 40))
    path = tmp_path / "settings.json"
    settings.save(path)
    calls = []

    def analyze(actual, **kwargs):
        calls.append((actual, kwargs))
        return {"status": "complete", "summary": {}}

    monkeypatch.setattr("mouse_sleep.pipeline.analyze", analyze)
    assert main(["analyze", "--config", str(path), "--appearance", "white", "--batch-size", "7", "--resume"]) == 0
    actual, options = calls[0]
    assert actual == replace(settings, appearance="white", batch_size=7)
    assert options["resume"] is True


@pytest.mark.parametrize("result,expected", [("complete", 0), ("cancelled", 130), (ValueError("bad input"), 1), (Cancelled(), 130), (KeyboardInterrupt(), 130)])
def test_cli_analysis_exit_codes(tmp_path, monkeypatch, result, expected):
    video = tmp_path / "video.mp4"
    video.touch()

    def analyze(*args, **kwargs):
        if isinstance(result, BaseException):
            raise result
        return {"status": result, "summary": {}}

    monkeypatch.setattr("mouse_sleep.pipeline.analyze", analyze)
    assert main(["analyze", "--video", str(video), "--output", str(tmp_path / "out")]) == expected
