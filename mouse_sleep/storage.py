"""Analysis persistence; SQLite commits are the authority for resume progress."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sqlite3

import numpy as np

from .config import Settings, atomic_json
from .epochs import EpochRecord


class AnalysisStore:
    def __init__(self, settings: Settings, info, timing, duration: float, identity: dict, *, resume=False):
        self.settings = settings
        self.output = Path(settings.output)
        self.manifest_path = self.output / "run.json"
        self.database = self.output / "analysis.sqlite3"
        self.resume = resume
        self.connection: sqlite3.Connection | None = None
        self.next_epoch = 0
        self.masking_state = None
        self.manifest = {
            "identity": identity, "settings": asdict(settings), "timing": timing.manifest(),
            "source_video": asdict(info), "duration_seconds": duration,
            "status": "running", "completed_epochs": 0,
            "appearance_notice": "White/light sleep classification is experimental"
            if settings.appearance == "white" else "FLIR sleep accuracy requires validation",
        }

    def _validate_resume(self) -> None:
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not self.resume:
                raise ValueError("This folder contains an analysis. Select Resume or choose a new output folder.")
            if old.get("identity") != self.manifest["identity"]:
                raise ValueError("Resume requires the same video, timing, appearance, ROI, settings, and model weights.")
            if not self.database.exists():
                raise ValueError("Resume database is missing.")
        elif self.database.exists():
            raise ValueError("Analysis database exists without its manifest. Choose a new output folder.")

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self._validate_resume()
        self.connection = sqlite3.connect(self.database)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS epochs (epoch INTEGER PRIMARY KEY, start REAL, "
                "end REAL, coverage REAL, reason TEXT, features TEXT, probabilities TEXT)"
            )
            completed = self.connection.execute("SELECT COALESCE(MAX(epoch), -1) FROM epochs").fetchone()[0]
            self.next_epoch = completed + 1
            if self.settings.masking_mode == "enhanced":
                self.connection.execute("CREATE TABLE IF NOT EXISTS masking_epochs "
                                        "(epoch INTEGER PRIMARY KEY, state TEXT NOT NULL, counters TEXT NOT NULL)")
                if self.next_epoch:
                    row = self.connection.execute("SELECT state FROM masking_epochs WHERE epoch = ?", (completed,)).fetchone()
                    count = self.connection.execute("SELECT COUNT(*) FROM masking_epochs").fetchone()[0]
                    if row is None or count != self.next_epoch:
                        raise ValueError("Enhanced resume requires a masking checkpoint for every committed epoch.")
                    self.masking_state = json.loads(row["state"])
            self.manifest["completed_epochs"] = self.next_epoch
            self.save_manifest()
            self.settings.save(self.output / "settings.json")
        except BaseException as exc:
            self._record_error(exc)
            self.close()
            raise
        return self

    def save_manifest(self) -> None:
        atomic_json(self.manifest_path, self.manifest)

    def persist(self, record: EpochRecord, *, masking=None) -> None:
        feature_json = None if record.features is None else json.dumps(
            [None if np.isnan(value) else float(value) for value in record.features], allow_nan=False,
        )
        probabilities = None if record.probabilities is None else json.dumps(record.probabilities)
        if self.settings.masking_mode == "enhanced" and masking is None:
            raise ValueError("Enhanced epochs must commit their masking state atomically.")
        with self.connection:
            self.connection.execute(
                "INSERT INTO epochs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record.epoch, record.start, record.end, record.coverage, record.reason, feature_json, probabilities),
            )
            if masking is not None:
                self.connection.execute("INSERT INTO masking_epochs VALUES (?, ?, ?)",
                                        (record.epoch, json.dumps(masking["state"], allow_nan=False),
                                         json.dumps(masking["counts"], allow_nan=False)))
        self.next_epoch = record.epoch + 1
        self.manifest.update(completed_epochs=self.next_epoch, last_completed_seconds=record.end)
        self.save_manifest()

    def records(self):
        return self.connection.execute(
            "SELECT epoch, start, end, coverage, reason, probabilities FROM epochs ORDER BY epoch"
        ).fetchall()

    def masking_records(self):
        return self.connection.execute(
            "SELECT e.epoch, e.start, e.end, m.counters FROM epochs e "
            "JOIN masking_epochs m ON e.epoch = m.epoch ORDER BY e.epoch"
        ).fetchall()

    def _record_error(self, exc: BaseException) -> None:
        self.manifest.update(status="error", error=str(exc))
        try:
            self.save_manifest()
        except Exception as persistence_error:
            exc.add_note(f"Could not save the error manifest: {persistence_error}")

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is not None:
                self._record_error(exc)
        finally:
            self.close()
