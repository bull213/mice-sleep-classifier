from __future__ import annotations

from pathlib import Path
import csv
import json
import os
import traceback

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
)

from ..config import Settings
from ..video import Timing, inspect_video
from .widgets import ImageView, clock_text


class ResultsPanel(QWidget):
    failed = Signal(str)
    loaded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.output_folder = ""
        self.result_folder = self.capture = None
        self.review_index = 0
        self.review_end = None
        self.busy = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_tick)
        self._build()

    def set_output_folder(self, folder):
        self.output_folder = folder

    def set_busy(self, busy):
        self.busy = busy
        if busy:
            self.stop_play()

    def _build(self):
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        open_results = QPushButton("Open results folder…")
        open_results.clicked.connect(self.choose_results)
        row.addWidget(open_results)
        open_files = QPushButton("Show files in Explorer")
        open_files.clicked.connect(lambda: os.startfile(str(self.result_folder)) if self.result_folder else None)
        row.addWidget(open_files)
        self.play_button = QPushButton("Play / pause selected interval")
        self.play_button.clicked.connect(self.toggle_play)
        row.addWidget(self.play_button)
        row.addStretch()
        layout.addLayout(row)
        self.summary = QLabel("Open a completed or interrupted analysis to review its saved results.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.timeline = QLabel()
        self.timeline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.timeline)
        self.result_tables = QTabWidget()
        self.bouts = QTableWidget(0, 3)
        self.bouts.setHorizontalHeaderLabels(["Sleep start", "Sleep end", "Duration"])
        self.bouts.cellClicked.connect(self.select_bout)
        self.epochs = QTableWidget(0, 6)
        self.epochs.setHorizontalHeaderLabels(["Start", "End", "State", "Sleep probability", "Tracking coverage", "Reason"])
        self.epochs.cellClicked.connect(self.select_epoch)
        for label, table in (("Sleep intervals · click to review", self.bouts), ("All epochs", self.epochs)):
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.result_tables.addTab(table, label)
        layout.addWidget(self.result_tables, 1)
        self.review_view = ImageView()
        self.review_view.setMinimumHeight(200)
        layout.addWidget(self.review_view, 1)
        self.review_label = QLabel("Select an interval to inspect the original video.")
        layout.addWidget(self.review_label)

    def choose_results(self):
        path = QFileDialog.getExistingDirectory(self, "Saved analysis folder", self.output_folder)
        if path:
            self.open_results(path)

    def open_results(self, folder):
        self.stop_play()
        try:
            directory = Path(folder)
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.review_settings = Settings.load(directory / "run.json")
            self.review_timing = Timing(self.review_settings)
            with (directory / "sleep_bouts.csv").open(newline="", encoding="utf-8") as handle:
                self.bout_rows = list(csv.DictReader(handle))
            with (directory / "epochs.csv").open(newline="", encoding="utf-8") as handle:
                self.epoch_rows = list(csv.DictReader(handle))
            self.result_folder = directory
            self.summary.setText(f"Sleep {clock_text(summary['sleep'])} · Awake {clock_text(summary['awake'])} · Unknown {clock_text(summary['unknown'])} · "
                                 f"Unprocessed {clock_text(summary['unprocessed_seconds'])}\n{summary['validation_status']} · {directory}")
            self.timeline.setPixmap(QPixmap(str(directory / "timeline.png")).scaledToWidth(1150, Qt.TransformationMode.SmoothTransformation))
            for table, rows, fields in ((self.bouts, self.bout_rows, ("start_seconds", "end_seconds", "duration_seconds")),
                                       (self.epochs, self.epoch_rows, ("start", "end", "state", "p_sleep", "coverage", "reason"))):
                table.setRowCount(len(rows))
                for i, row in enumerate(rows):
                    for j, field in enumerate(fields):
                        value = row[field]
                        if field in ("start_seconds", "end_seconds", "duration_seconds", "start", "end"):
                            value = clock_text(float(value))
                        elif field in ("p_sleep", "coverage") and value:
                            value = f"{float(value):.1%}"
                        table.setItem(i, j, QTableWidgetItem(value))
            self.loaded.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

    def select_bout(self, row, column):
        item = self.bout_rows[row]
        self.jump_review(float(item["start_seconds"]), float(item["end_seconds"]))

    def select_epoch(self, row, column):
        item = self.epoch_rows[row]
        self.jump_review(float(item["start"]), float(item["end"]))

    def jump_review(self, start, end):
        self.stop_play()
        try:
            info = inspect_video(self.review_settings.video)
            self.review_fps = info.fps
            self.capture = cv2.VideoCapture(self.review_settings.video)
            if not self.capture.isOpened():
                raise ValueError("Cannot open the original recording for review.")
            self.review_index = int(np.searchsorted(self.review_timing.times, start)) if self.review_timing.times is not None else round(start * info.fps)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.review_index)
            self.review_end = end
            self.play_tick()
        except Exception:
            self.failed.emit(traceback.format_exc())

    def play_tick(self):
        if self.capture is None:
            return
        times = self.review_timing.times
        seconds = float(times[self.review_index]) if times is not None and self.review_index < len(times) else self.review_index / self.review_fps
        if self.review_end is not None and seconds >= self.review_end:
            self.timer.stop()
            return
        okay, frame = self.capture.read()
        if not okay:
            self.stop_play()
            return
        self.review_view.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.review_label.setText(f"Original video · {clock_text(seconds)} · relative to first recorded frame")
        self.review_index += 1
        # Playback uses recorded intervals, including acquisition gaps.
        delay = 1000 / self.review_fps
        if times is not None and self.review_index < len(times):
            delay = 1000 * (times[self.review_index] - seconds)
        self.timer.setInterval(max(1, round(delay)))

    def toggle_play(self):
        if self.busy:
            return
        if self.timer.isActive():
            self.timer.stop()
        elif self.capture is not None:
            self.timer.start()

    def stop_play(self):
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

