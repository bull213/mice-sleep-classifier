"""Bounded measurement accumulation and calculation of a single epoch."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EPOCH_SECONDS, Settings
from .features import epoch_features, measurements


@dataclass
class EpochRecord:
    epoch: int
    start: float
    end: float
    coverage: float
    reason: str
    features: np.ndarray | None
    probabilities: list[float] | None


@dataclass
class EpochAccumulator:
    settings: Settings
    duration: float
    epoch: int = 0
    times: list[float] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    forced_reason: str = ""

    @property
    def start(self) -> float:
        return self.epoch * EPOCH_SECONDS

    def add(self, stamp: float, mask: np.ndarray, *, device_gap: bool = False) -> None:
        if device_gap:
            self.forced_reason = "substantial_device_frame_gap"
        measurement = measurements(mask, fraction_range=(0.0001, 0.4))
        if measurement is not None:
            self.times.append(stamp)
            self.values.append(measurement)

    def record(self, classifier) -> EpochRecord:
        end = min(self.start + EPOCH_SECONDS, self.duration)
        feature, coverage, reason = None, 0.0, self.forced_reason
        if end - self.start < EPOCH_SECONDS - 1e-6:
            reason = "incomplete_epoch"
        elif not reason:
            feature, coverage, reason = epoch_features(
                self.times, self.values, self.start,
                minimum_coverage=self.settings.minimum_coverage,
                maximum_gap=self.settings.maximum_gap_seconds,
            )
        probabilities = None if feature is None else classifier.predict(feature)[0].tolist()
        return EpochRecord(self.epoch, self.start, end, coverage, reason, feature, probabilities)

    def advance(self) -> None:
        self.epoch += 1
        self.times.clear()
        self.values.clear()
        self.forced_reason = ""
