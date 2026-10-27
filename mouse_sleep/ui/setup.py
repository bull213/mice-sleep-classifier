from __future__ import annotations

from pathlib import Path
import json

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton, QFileDialog

from ..config import ROOT
from .widgets import path_row


class ClassifierPanel(QWidget):
    statusChanged = Signal(str)
    classifierReady = Signal(str)

    def __init__(self, tasks, parent=None):
        super().__init__(parent)
        self.tasks = tasks
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        explanation = QLabel("Train the reusable sleep classifier once from the authors' reference features.\n"
                             "The download is approximately 950 MB and resumes after cancellation.\n"
                             "Validation holds out whole animals; it does not establish accuracy on your FLIR footage.\n"
                             "No EEG labels or retraining are required to analyze your own videos.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.classifier_status = QLabel()
        self.classifier_status.setWordWrap(True)
        layout.addWidget(self.classifier_status)
        self.train_controls = QWidget()
        form = QFormLayout(self.train_controls)
        self.training_csv = QLineEdit()
        self.training_csv.setPlaceholderText("Leave blank to download the official reference CSV automatically")
        self.training_output = QLineEdit(str(ROOT / "classifier"))
        form.addRow("Reference CSV", path_row(self.training_csv, self.choose_training_csv))
        form.addRow("Save classifier to", path_row(self.training_output, self.choose_training_output))
        train = QPushButton("Download / train classifier")
        train.setObjectName("primary")
        train.clicked.connect(self.train_classifier)
        form.addRow(train)
        layout.addWidget(self.train_controls)
        source = QLabel('<a href="https://zenodo.org/records/5180680">Reference dataset on Zenodo</a> · '
                        '<a href="https://zenodo.org/records/5806397">Segmentation models</a>')
        source.setOpenExternalLinks(True)
        layout.addWidget(source)
        layout.addStretch()

    def choose_training_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Official reference features", str(ROOT / "data"), "CSV (*.csv)")
        if path:
            self.training_csv.setText(path)

    def choose_training_output(self):
        path = QFileDialog.getExistingDirectory(self, "Classifier destination", self.training_output.text())
        if path:
            self.training_output.setText(path)

    def train_classifier(self):
        source = self.training_csv.text().strip() or None
        output = self.training_output.text().strip() or str(ROOT / "classifier")
        def done(report):
            self.classifierReady.emit(output)
            self.statusChanged.emit(f"Classifier ready. Held-out reference sleep/awake accuracy: {report['sleep_awake']['accuracy']:.1%}. See validation.json.")
        def action(progress, cancel):
            from ..classifier import train
            return train(source, output, progress, cancel)
        self.tasks.start(action, done)

    def refresh_classifier_status(self):
        directory = Path(self.training_output.text() or ROOT / "classifier")
        try:
            report = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
            self.classifier_status.setText(f"Classifier ready at {directory}\nReference held-out sleep/awake accuracy: {report['sleep_awake']['accuracy']:.1%}")
        except (OSError, ValueError, KeyError):
            self.classifier_status.setText("Classifier setup is required before sleep analysis. Segmentation preview works independently.")

