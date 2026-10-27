"""Headless desktop smoke test and reviewable screenshots."""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import argparse
from pathlib import Path
import time

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase

from mouse_sleep.gui import Window, style_application
from mouse_sleep.config import ROOT, Settings
from tests.helpers import make_video


def wait_for_task(app, window, failures, timeout=120):
    deadline = time.monotonic() + timeout
    while window.worker is not None:
        app.processEvents()
        if time.monotonic() >= deadline:
            raise TimeoutError("GUI task did not complete")
        time.sleep(0.01)
    app.processEvents()
    if failures:
        raise RuntimeError(failures.pop(0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "verification")
    args = parser.parse_args(argv)
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    app.setFont(QFont("Segoe UI", 10))
    style_application(app)
    window = Window(load_defaults=False)
    failures = []
    for signal in (window.tasks.failed, window.analysis.failed, window.results.failed):
        signal.disconnect(window.on_failure)
        signal.connect(failures.append)
    try:
        video = make_video(directory / "synthetic_gui.mp4", count=90)
        window.apply_settings(Settings(video=str(video), output=str(directory / "gui_run"),
                                       appearance="white", masking_mode="legacy", roi=(5, 5, 80, 54)))
        window.show()
        window.load_video()
        wait_for_task(app, window, failures)
        assert window.analysis.gray is not None
        assert window.settings().model_name == "FullModel"
        assert window.settings().refinement_mode == "off"
        assert window.tabs.count() == 3
        assert window.analysis.seek.maximum() == 2
        assert window.analysis.source_view.roi == (5, 5, 80, 54)
        window.test_segmentation()
        wait_for_task(app, window, failures)
        assert window.analysis.mask_view.image is not None
        window.grab().save(str(directory / "desktop_preview.png"))
        if (directory / "pipeline_white" / "summary.json").is_file():
            window.open_results(directory / "pipeline_white")
            assert not failures, failures
            assert window.results.epochs.rowCount() == 3
            window.results.select_epoch(0, 0)
            assert window.results.review_view.image is not None
            window.grab().save(str(directory / "desktop_results.png"))
        print("GUI smoke test passed", flush=True)
    finally:
        window.tasks.cancel()
        wait_for_task(app, window, [])
        window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
