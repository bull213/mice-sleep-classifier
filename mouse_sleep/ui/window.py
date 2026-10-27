"""Application shell coordinating independent analysis, setup, and review panels."""
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QProgressBar, QPushButton, QPlainTextEdit, QMessageBox, QScrollArea,
)

from ..config import ROOT, Settings
from .analysis import AnalysisPanel
from .results import ResultsPanel
from .setup import ClassifierPanel
from .tasks import TaskController
from .widgets import clock_text


class Window(QMainWindow):
    def __init__(self, *, load_defaults=True):
        super().__init__()
        self.setWindowTitle("Mouse Sleep Analyzer · FLIR video")
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1320, available.width() - 60), min(900, available.height() - 60))
        self.tasks = TaskController(self)
        self.analysis = AnalysisPanel(self.tasks)
        self.setup = ClassifierPanel(self.tasks)
        self.results = ResultsPanel()
        self.defaults = Settings()
        if load_defaults:
            try:
                self.defaults = Settings.load(ROOT / "last_settings.json")
            except (OSError, ValueError, TypeError):
                pass

        outer = QWidget()
        layout = QVBoxLayout(outer)
        title = QLabel("Mouse Sleep Analyzer")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(QLabel("Overhead FLIR video → estimated sleep and awake intervals, every 10 seconds"))
        self.tabs = QTabWidget()
        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        analysis_scroll.setWidget(self.analysis)
        self.tabs.addTab(analysis_scroll, "Analyze video")
        self.tabs.addTab(self.setup, "Classifier setup")
        self.tabs.addTab(self.results, "Results and review")
        layout.addWidget(self.tabs, 1)
        self.status = QLabel("Ready. Select a video, inspect segmentation, then analyze.")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.cancel_button = QPushButton("Cancel task")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_task)
        row = QHBoxLayout()
        row.addWidget(self.progress, 1)
        row.addWidget(self.cancel_button)
        layout.addWidget(self.status)
        layout.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(95)
        self.log.setMaximumBlockCount(250)
        layout.addWidget(self.log)
        self.setCentralWidget(outer)

        self.tasks.progress.connect(self.on_progress)
        self.tasks.failed.connect(self.on_failure)
        self.tasks.busyChanged.connect(self.set_busy)
        self.analysis.statusChanged.connect(self.status.setText)
        self.analysis.failed.connect(self.on_failure)
        self.analysis.analysisFinished.connect(self._analysis_finished)
        self.analysis.output.textChanged.connect(self.results.set_output_folder)
        self.setup.statusChanged.connect(self.status.setText)
        self.setup.classifierReady.connect(self.analysis.classifier.setText)
        self.results.failed.connect(self.on_failure)
        self.results.loaded.connect(lambda: self.tabs.setCurrentIndex(2))
        self.apply_settings(self.defaults)
        self.setup.refresh_classifier_status()
        if self.analysis.video.text() and Path(self.analysis.video.text()).is_file():
            QTimer.singleShot(0, self.load_video)

    @property
    def worker(self):
        return self.tasks.worker

    def settings(self):
        return self.analysis.settings()

    def apply_settings(self, settings):
        self.analysis.apply_settings(settings)

    def load_video(self):
        self.analysis.load_video()

    def test_segmentation(self):
        self.analysis.test_segmentation()

    def open_results(self, folder):
        self.results.open_results(folder)

    @Slot(str, object)
    def _analysis_finished(self, folder, result):
        self.open_results(folder)

    @Slot(bool)
    def set_busy(self, busy):
        self.analysis.set_busy(busy)
        self.setup.train_controls.setEnabled(not busy)
        self.results.set_busy(busy)
        self.cancel_button.setEnabled(busy)
        self.progress.setRange(0, 0 if busy else 1000)
        if not busy:
            self.setup.refresh_classifier_status()

    def cancel_task(self):
        if self.worker is not None:
            self.tasks.cancel()
            self.status.setText("Cancelling after the current frame batch or training step…")
            self.cancel_button.setEnabled(False)

    @Slot(object)
    def on_progress(self, event):
        message = event.get("message", "")
        if event.get("processing_fps"):
            message += f" · {event['processing_fps']:.1f} frames/s"
        if event.get("remaining_seconds") is not None:
            message += f" · approximately {clock_text(event['remaining_seconds'])} remaining"
        self.status.setText(message)
        self.log.appendPlainText(message)
        fraction = event.get("fraction")
        if fraction is None and event.get("duration"):
            fraction = event.get("seconds", 0) / event["duration"]
        if fraction is not None:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(1000 * np.clip(fraction, 0, 1)))

    @Slot(str)
    def on_failure(self, detail):
        self.log.appendPlainText(detail)
        self.status.setText("Task failed. See the error below.")
        QMessageBox.warning(self, "Task failed", detail.splitlines()[-1])

    def closeEvent(self, event):
        if self.worker is not None:
            self.cancel_task()
            self.status.setText("Cancelling the active task. Close again after it finishes.")
            event.ignore()
            return
        self.results.stop_play()
        self.analysis.close_resources()
        event.accept()
