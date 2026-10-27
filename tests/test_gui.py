import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from dataclasses import asdict
import csv
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from mouse_sleep.config import Settings, atomic_json
from mouse_sleep.gui import Window
from mouse_sleep.reports import export_records
from mouse_sleep.tasks import check_cancelled
from tests.helpers import FakeClassifier, make_video


def wait_until(app, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            pytest.fail("GUI task did not finish")
        time.sleep(0.005)
    app.processEvents()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    view = Window(load_defaults=False)
    view.show()
    yield view
    view.tasks.cancel()
    wait_until(app, lambda: view.worker is None)
    view.close()


def test_gui_settings_and_preview_round_trip(window, app, tmp_path):
    video = make_video(tmp_path / "preview.mp4", count=90)
    settings = Settings(video=str(video), output=str(tmp_path / "out"), appearance="white", masking_mode="legacy",
                        roi=(5, 5, 80, 54), batch_size=7, threshold=31, max_seconds=120)
    window.apply_settings(settings)
    assert window.settings() == settings
    assert window.settings().model_name == "FullModel"
    assert window.settings().refinement_mode == "off"
    window.load_video()
    wait_until(app, lambda: window.worker is None)
    assert window.analysis.gray is not None and window.analysis.seek.maximum() == 2
    assert window.analysis.source_view.roi == settings.roi
    assert window.tabs.count() == 3


def test_single_task_updates_controls_and_delivers_result_on_gui_thread(window, app):
    release = threading.Event()
    results = []
    gui_thread = threading.get_ident()

    def action(progress, cancel):
        release.wait(2)
        return threading.get_ident()

    assert window.tasks.start(action, lambda value: results.append((value, threading.get_ident())))
    assert not window.tasks.start(action)
    assert not window.analysis.controls.isEnabled() and window.cancel_button.isEnabled()
    release.set()
    wait_until(app, lambda: window.worker is None)
    assert len(results) == 1 and results[0][0] != gui_thread and results[0][1] == gui_thread
    assert window.analysis.controls.isEnabled() and not window.cancel_button.isEnabled()


def test_closing_active_window_cancels_then_allows_close(window, app):
    def action(progress, cancel):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            check_cancelled(cancel)
            time.sleep(0.005)

    window.tasks.start(action)
    assert window.close() is False
    wait_until(app, lambda: window.worker is None)
    assert window.analysis.controls.isEnabled()
    assert "Cancelling" in window.log.toPlainText() or "Cancelled" in window.log.toPlainText()
    assert window.close() is True


def test_failed_task_restores_controls(window, app):
    def action(progress, cancel):
        raise ValueError("test task failure")

    window.tasks.start(action)
    wait_until(app, lambda: window.worker is None)
    assert "test task failure" in window.log.toPlainText()
    assert window.analysis.controls.isEnabled()


def test_results_playback_uses_acquisition_gaps_and_releases_capture(window, tmp_path):
    video = make_video(tmp_path / "source.mp4", count=3)
    timing = tmp_path / "timing.csv"
    with timing.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["take_id", "host_received_monotonic_ns", "frame_id"])
        for i, stamp in enumerate((0, 1 / 30, 1)):
            writer.writerow([1, 1000000000 + round(stamp * 1e9), i])
    settings = Settings(video=str(video), output=str(tmp_path), timing_csv=str(timing), take_id=1)
    export_records(tmp_path, [dict(epoch=0, start=0, end=10, coverage=0, reason="tracking_gap", probabilities=None)],
                   FakeClassifier(), settings, 10)
    atomic_json(tmp_path / "run.json", {"settings": asdict(settings), "identity": {}})
    window.open_results(tmp_path)
    assert window.tabs.currentIndex() == 2 and window.results.epochs.rowCount() == 1
    window.results.select_epoch(0, 0)
    assert window.results.review_view.image is not None
    window.results.play_tick()
    assert window.results.timer.interval() == 967
    capture = window.results.capture
    window.results.stop_play()
    assert not capture.isOpened() and window.results.capture is None
