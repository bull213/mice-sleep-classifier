"""Application-owned inference resources shared by serialized background jobs."""
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from .config import sha256


class InferenceSession:
    """Keep at most one model, checking checkpoint contents before every lease."""

    def __init__(self, factory=None):
        self.factory = factory
        self._lock = RLock()
        self._model = self._identity = None

    @contextmanager
    def lease(self, checkpoint):
        with self._lock:
            path = str(Path(checkpoint).resolve())
            try:
                identity = (path, sha256(Path(path + ".index")),
                            sha256(Path(path + ".data-00000-of-00001")))
                if identity != self._identity:
                    self.close()
                    factory = self.factory
                    if factory is None:
                        from .segmentation import Segmenter
                        factory = Segmenter
                    self._model = factory(checkpoint)
                    self._identity = identity
                yield self._model
            except BaseException:
                # A failed initialization or inference must not poison the next job.
                self.close()
                raise

    def close(self):
        with self._lock:
            model = self._model
            self._model = self._identity = None
            if model is not None:
                model.close()
