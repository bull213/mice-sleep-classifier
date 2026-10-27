"""Own the single background task and deliver results on the GUI thread."""
from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..config import Cancelled


class Worker(QThread):
    progress = Signal(object)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action
        self.cancel_event = threading.Event()

    def run(self):
        try:
            self.result.emit(self.action(self.progress.emit, self.cancel_event.is_set))
        except Cancelled:
            self.progress.emit({"message": "Cancelled. Saved work can be resumed."})
        except Exception:
            self.failed.emit(traceback.format_exc())


class TaskController(QObject):
    progress = Signal(object)
    failed = Signal(str)
    busyChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._done = None

    def start(self, action, done=None):
        if self.worker is not None:
            return False
        self._done = done
        self.worker = Worker(action, self)
        self.worker.progress.connect(self.progress)
        self.worker.failed.connect(self.failed)
        self.worker.result.connect(self._on_result)
        self.worker.finished.connect(self._on_finished)
        self.busyChanged.emit(True)
        self.worker.start()
        return True

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel_event.set()

    @Slot(object)
    def _on_result(self, result):
        if self._done is not None:
            try:
                self._done(result)
            except Exception:
                self.failed.emit(traceback.format_exc())

    @Slot()
    def _on_finished(self):
        finished = self.worker
        self.worker = self._done = None
        if finished is not None:
            finished.deleteLater()
        self.busyChanged.emit(False)
