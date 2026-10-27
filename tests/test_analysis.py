from dataclasses import replace
from pathlib import Path
import threading
import json
import sqlite3
import pytest
from mouse_sleep.config import Settings
from mouse_sleep.pipeline import analyze
from mouse_sleep.locking import OutputLock
from tests.helpers import make_video, FakeClassifier, FakeSegmenter


@pytest.fixture
def analysis_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr("mouse_sleep.pipeline.sha256", lambda path: "test-segmentation")
    video = make_video(tmp_path / "mouse.mp4", count=615)
    return Settings(video=str(video), output=str(tmp_path / "run"), masking_mode="legacy", refinement="off", batch_size=7, roi=(0, 0, 96, 64))


def run_fake(settings, **kwargs):
    kwargs.setdefault("segmenter_factory", FakeSegmenter)
    return analyze(settings, classifier_factory=FakeClassifier, **kwargs)


def test_legacy_roi_run_resumes_without_migration(analysis_inputs, monkeypatch):
    settings = analysis_inputs
    cancel = threading.Event()
    identity = Settings.identity

    def legacy_identity(self):
        result = identity(self)
        result["roi"] = self.roi  # The pre-refactor representation, serialized to a JSON array.
        return result

    with monkeypatch.context() as legacy:
        legacy.setattr(Settings, "identity", legacy_identity)
        result = run_fake(settings, cancel=cancel.is_set,
                          progress=lambda event: cancel.set() if event.get("seconds", 0) >= 10 else None)
    assert result["status"] == "cancelled"
    # A real pre-mode manifest had none of the newly introduced masking fields.
    from mouse_sleep.config import MASKING_FIELDS
    path = Path(settings.output) / "run.json"
    old_manifest = json.loads(path.read_text())
    for key in MASKING_FIELDS:
        old_manifest["settings"].pop(key, None)
    path.write_text(json.dumps(old_manifest))
    loaded = Settings.load(Path(settings.output) / "run.json")
    assert loaded.masking_mode == "legacy"
    result = run_fake(loaded, resume=True)
    assert result["status"] == "complete"
    assert result["completed_epochs"] == 3
    full = replace(settings, output=str(Path(settings.output).parent / "full"))
    run_fake(full)
    assert (Path(settings.output) / "epochs.csv").read_bytes() == (Path(full.output) / "epochs.csv").read_bytes()
    with pytest.raises(ValueError, match="Resume requires"):
        run_fake(replace(settings, roi=(1, 0, 95, 64)), resume=True)


def test_inference_failure_closes_resources_and_can_resume(analysis_inputs):
    closed = []

    class BrokenSegmenter(FakeSegmenter):
        def predict(self, images):
            raise RuntimeError("inference failed")

        def __exit__(self, *args):
            closed.append(True)

    with pytest.raises(RuntimeError, match="inference failed"):
        run_fake(analysis_inputs, segmenter_factory=BrokenSegmenter)
    assert closed == [True]
    folder = Path(analysis_inputs.output)
    manifest = json.loads((folder / "run.json").read_text())
    assert manifest["status"] == "error" and manifest["completed_epochs"] == 0
    with OutputLock(folder):
        pass
    assert run_fake(analysis_inputs, resume=True)["status"] == "complete"


def test_report_failure_keeps_committed_epochs(analysis_inputs, monkeypatch):
    def fail(*args):
        raise OSError("report failed")

    with monkeypatch.context() as failure:
        failure.setattr("mouse_sleep.pipeline.export_records", fail)
        with pytest.raises(OSError, match="report failed"):
            run_fake(analysis_inputs)
    folder = Path(analysis_inputs.output)
    manifest = json.loads((folder / "run.json").read_text())
    assert manifest["status"] == "error" and manifest["completed_epochs"] == 3
    with sqlite3.connect(folder / "analysis.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM epochs").fetchone()[0] == 3
    result = run_fake(analysis_inputs, resume=True, segmenter_factory=lambda *args: pytest.fail("Reprocessed committed epochs"))
    assert result["status"] == "complete"


def test_cancellation_before_first_frame_exports_empty_results(analysis_inputs):
    result = run_fake(analysis_inputs, cancel=lambda: True)
    assert result["status"] == "cancelled" and result["completed_epochs"] == 0
    assert result["summary"]["analyzed_seconds"] == 0
    assert result["summary"]["unprocessed_seconds"] == pytest.approx(20.5)


def test_pipeline_partial_batch_cancellation_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("mouse_sleep.pipeline.sha256", lambda path: "test-segmentation")
    video = make_video(tmp_path / "mouse.mp4", count=675)  # 22.5 s; incomplete last epoch and partial batch
    settings = Settings(video=str(video), output=str(tmp_path / "run"), masking_mode="legacy", refinement="off", batch_size=7)
    cancel = threading.Event()

    def progress(event):
        if event.get("seconds", 0) >= 10:
            cancel.set()

    result = analyze(settings, progress=progress, cancel=cancel.is_set, segmenter_factory=FakeSegmenter, classifier_factory=FakeClassifier)
    assert result["status"] == "cancelled" and result["completed_epochs"] == 1
    cancel.clear()
    result = analyze(settings, resume=True, segmenter_factory=FakeSegmenter, classifier_factory=FakeClassifier)
    assert result["status"] == "complete" and result["completed_epochs"] == 3
    uninterrupted = replace(settings, output=str(tmp_path / "full"))
    analyze(uninterrupted, segmenter_factory=FakeSegmenter, classifier_factory=FakeClassifier)
    assert (Path(settings.output) / "epochs.csv").read_text() == (Path(uninterrupted.output) / "epochs.csv").read_text()
    assert result["summary"]["unknown"] == pytest.approx(2.5)
    assert result["summary"]["sleep"] == 20
    with pytest.raises(ValueError, match="Resume requires"):
        analyze(replace(settings, threshold=31), resume=True, segmenter_factory=FakeSegmenter, classifier_factory=FakeClassifier)


def test_enhanced_resume_preserves_masks_and_lighting_state(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    monkeypatch.setattr("mouse_sleep.pipeline.sha256", lambda path: "test-segmentation")
    video = tmp_path / "light_dark.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (96, 64))
    assert writer.isOpened()
    for i in range(675):
        frame = np.full((64, 96, 3), 180 if i < 360 else 100, np.uint8)
        cv2.ellipse(frame, (40 + int(10 * np.sin(i / 80)), 32), (10, 7), 0, 0, 360, (10, 10, 10), -1)
        writer.write(frame)
    writer.release()
    settings = Settings(video=str(video), output=str(tmp_path / "resumed"), batch_size=7)
    cancel = threading.Event()
    first = run_fake(settings, cancel=cancel.is_set,
                     progress=lambda event: cancel.set() if event.get("seconds", 0) >= 10 else None)
    assert first["completed_epochs"] == 1 and first["status"] == "cancelled"
    resumed = run_fake(Settings.load(Path(settings.output) / "run.json"), resume=True)
    full = replace(settings, output=str(tmp_path / "full"))
    uninterrupted = run_fake(full)
    assert resumed["summary"] == uninterrupted["summary"]
    assert resumed["summary"]["sleep"] > 0
    for name in ("epochs.csv", "summary.json", "masking_diagnostics.csv"):
        assert (Path(settings.output) / name).read_bytes() == (Path(full.output) / name).read_bytes()
    with sqlite3.connect(Path(settings.output) / "analysis.sqlite3") as a, sqlite3.connect(Path(full.output) / "analysis.sqlite3") as b:
        for table in ("epochs", "masking_epochs"):
            assert a.execute(f"SELECT * FROM {table} ORDER BY epoch").fetchall() == b.execute(f"SELECT * FROM {table} ORDER BY epoch").fetchall()
    diagnostics = json.loads((Path(full.output) / "masking_summary.json").read_text())
    assert diagnostics["lighting_conditions"] == 2 and diagnostics["counts"]["lighting_changes"] == 1
    for changed in (replace(settings, masking_mode="legacy"), replace(settings, background_fraction=0.4)):
        with pytest.raises(ValueError, match="Resume requires"):
            run_fake(changed, resume=True)
    manifest_path = Path(settings.output) / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["settings"]["masking_version"] -= 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume requires"):
        run_fake(settings, resume=True)
