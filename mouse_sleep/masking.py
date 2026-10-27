"""Selectable masking engines; Legacy retains the original numerical path."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import json
import math

import cv2
import numpy as np

from .config import MASKING_VERSION, atomic_json
from .segmentation import MaskRefiner


@dataclass
class MaskResult:
    mask: np.ndarray
    source: str
    reason: str = ""
    lighting: int | None = None


def render_mask(square, mask):
    view = cv2.cvtColor(cv2.resize(square, (720, 720)), cv2.COLOR_GRAY2RGB)
    selected = cv2.resize(mask, (720, 720), interpolation=cv2.INTER_NEAREST).astype(bool)
    view[selected] = (view[selected] * 0.5 + np.array([30, 230, 100]) * 0.5).astype(np.uint8)
    return view


class LegacyMasker:
    def __init__(self, settings):
        self.settings = settings
        self.refine = MaskRefiner()

    def process(self, mask480, prepared, stamp, *, device_gap=False):
        _, square, valid = prepared
        mask = self.refine(mask480, square, valid, self.settings.refinement_mode, self.settings.threshold)
        return MaskResult(mask, "legacy")

    def checkpoint(self):
        return None

    def advance_epoch(self):
        pass


@dataclass
class _Candidate:
    mask: np.ndarray
    x: float
    y: float
    area: int
    source: str
    rank: int


class EnhancedMasker:
    """Use current-frame evidence only; history selects objects, never manufactures masks."""

    def __init__(self, settings, backgrounds, geometry, state=None):
        self.settings, self.backgrounds, self.geometry = settings, backgrounds, geometry
        self.refine = MaskRefiner()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.delta = np.empty((480, 480), np.float32)
        self.thresholds = np.maximum(settings.background_delta,
                                     settings.background_fraction * backgrounds.references)
        self.empty = np.zeros((1080, 1080), np.uint8)
        floor_area = int(geometry.allowed.sum())
        self.area_limits = (max(6, floor_area * settings.minimum_mouse_area), floor_area * settings.maximum_mouse_area)
        self.last = self.pending = None
        self.pending_count = 0
        self.last_stamp = None
        self.lighting = self.lighting_pending = None
        self.lighting_count = 0
        self.reference_active = True
        self.counts = Counter()
        if state is not None:
            self.restore(state)

    def restore(self, state):
        if state.get("version") != MASKING_VERSION:
            raise ValueError("Saved masking state has an unsupported version.")
        for key in ("last", "pending"):
            value = state[key]
            if value is not None and (set(value) != {"x", "y", "area", "time"}
                    or any(not math.isfinite(v) for v in value.values())
                    or not 0 <= value["x"] < 480 or not 0 <= value["y"] < 480
                    or not 0 < value["area"] <= 480 * 480):
                raise ValueError("Invalid saved mouse tracking state.")
            setattr(self, key, value.copy() if value is not None else None)
        for key in ("pending_count", "lighting_count"):
            value = state[key]
            if type(value) is not int or not 0 <= value <= 3:
                raise ValueError("Invalid saved masking counter.")
            setattr(self, key, value)
        for key in ("lighting", "lighting_pending"):
            value = state[key]
            if value is not None and (type(value) is not int or not 0 <= value < len(self.backgrounds.references)):
                raise ValueError("Invalid saved lighting condition.")
            setattr(self, key, value)
        self.last_stamp = state["last_stamp"]
        if self.last_stamp is not None and not math.isfinite(self.last_stamp):
            raise ValueError("Invalid saved masking timestamp.")
        self.reference_active = state.get("reference_active", True)
        if type(self.reference_active) is not bool:
            raise ValueError("Invalid saved background availability.")

    def checkpoint(self):
        state = {"version": MASKING_VERSION, "last": self.last, "pending": self.pending,
                 "pending_count": self.pending_count, "last_stamp": self.last_stamp,
                 "lighting": self.lighting, "lighting_pending": self.lighting_pending,
                 "lighting_count": self.lighting_count, "reference_active": self.reference_active}
        return {"state": state, "counts": dict(self.counts)}

    def advance_epoch(self):
        self.counts.clear()

    def _reset_position(self):
        self.last = self.pending = None
        self.pending_count = 0

    def _reference(self, image):
        signature = self.geometry.signature(image)
        distances = self.geometry.distances(self.backgrounds.signatures, signature)
        best = int(distances.argmin())
        if distances[best] > self.backgrounds.tolerances[best]:
            if self.reference_active:
                self._reset_position()
            self.reference_active = False
            self.lighting_pending, self.lighting_count = None, 0
            self.counts["background_unavailable"] += 1
            return None
        if not self.reference_active:
            self._reset_position()
        self.reference_active = True
        if self.lighting is None:
            self.lighting = best
        if best != self.lighting:
            if self.lighting_pending != best:
                self._reset_position()
            self.lighting_count = self.lighting_count + 1 if self.lighting_pending == best else 1
            self.lighting_pending = best
            if self.lighting_count < 3:
                self.counts["lighting_transition_frames"] += 1
                return None
            self.lighting = best
            self.counts["lighting_changes"] += 1
        self.lighting_pending, self.lighting_count = None, 0
        return self.lighting

    def _components(self, binary, source, rank):
        count, labels, stats, centers = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        low, high = self.area_limits
        return [_Candidate(labels == index, float(centers[index, 0]), float(centers[index, 1]),
                           int(stats[index, cv2.CC_STAT_AREA]), source, rank)
                for index in range(1, count) if low <= stats[index, cv2.CC_STAT_AREA] <= high]

    @staticmethod
    def _consistent(candidate, previous, stamp):
        ratio = candidate.area / previous["area"]
        elapsed = max(1 / 30, stamp - previous["time"])
        radius = max(8, 2 * math.sqrt(previous["area"])) * min(3, elapsed * 30)
        return 1 / 3 <= ratio <= 3 and math.hypot(candidate.x - previous["x"], candidate.y - previous["y"]) <= radius

    @staticmethod
    def _position(candidate, stamp):
        return {"x": candidate.x, "y": candidate.y, "area": candidate.area, "time": float(stamp)}

    @staticmethod
    def _unique(candidates):
        if not candidates:
            return None
        rank = min(candidate.rank for candidate in candidates)
        best = [candidate for candidate in candidates if candidate.rank == rank]
        if len(best) == 1:
            return best[0]
        # Small changed bedding patches must not veto an otherwise clear mouse.
        # Comparable competing objects remain ambiguous.
        best.sort(key=lambda candidate: candidate.area, reverse=True)
        return best[0] if best[0].area >= 3 * best[1].area else None

    def _select(self, candidates, stamp):
        if self.last is not None and stamp - self.last["time"] > 0.5:
            self._reset_position()
        if self.last is None:
            return self._unique(candidates), "ambiguous_candidates" if candidates else "no_candidate"
        nearby = [candidate for candidate in candidates if self._consistent(candidate, self.last, stamp)]
        chosen = self._unique(nearby)
        if chosen is not None:
            self.pending, self.pending_count = None, 0
            return chosen, ""
        if nearby:
            self.pending, self.pending_count = None, 0
            return None, "ambiguous_candidates"
        candidate = self._unique(candidates)
        if candidate is None:
            self.pending, self.pending_count = None, 0
            return None, "ambiguous_candidates" if candidates else "no_candidate"
        self.pending_count = self.pending_count + 1 if self.pending and self._consistent(candidate, self.pending, stamp) else 1
        self.pending = self._position(candidate, stamp)
        if self.pending_count >= 3:
            self.pending, self.pending_count = None, 0
            self.counts["reacquisitions"] += 1
            return candidate, ""
        return None, "position_or_area_jump"

    def process(self, mask480, prepared, stamp, *, device_gap=False):
        image, square, valid = prepared
        if self.last_stamp is not None and stamp <= self.last_stamp:
            raise ValueError("Enhanced masking requires strictly increasing frame times.")
        if device_gap or (self.last_stamp is not None and stamp - self.last_stamp > self.settings.maximum_gap_seconds):
            self._reset_position()
        self.last_stamp = float(stamp)
        self.counts["frames"] += 1
        reference = self._reference(image)
        neural = self._components(np.asarray(mask480, bool) & self.geometry.allowed, "neural", 2)
        foreground = []
        if reference is not None:
            np.subtract(self.backgrounds.references[reference], image, out=self.delta)
            binary = ((self.delta > self.thresholds[reference]) & self.geometry.allowed
                      & ~self.backgrounds.uncertain[reference]).astype(np.uint8)
            # Remove thin tails/apparatus without the Legacy network's larger opening.
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.opening_kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
            binary &= self.geometry.allowed
            foreground = self._components(binary, "background", 1)
            for candidate in foreground:
                if any(np.count_nonzero(candidate.mask & other.mask) >= 0.25 * min(candidate.area, other.area)
                       for other in neural):
                    candidate.source, candidate.rank = "background_supported", 0
        candidates = foreground
        if not foreground:
            candidates = neural
        elif self.last is not None and reference is not None:
            # A still mouse may be embedded in an automatically estimated reference.
            candidates = foreground + [candidate for candidate in neural
                if np.count_nonzero(candidate.mask & self.backgrounds.uncertain[reference]) > candidate.area * 0.25
                and self._consistent(candidate, self.last, stamp)
                and not any(np.count_nonzero(candidate.mask & other.mask) > 0.25 * min(candidate.area, other.area)
                            for other in foreground)]
        selected, reason = self._select(candidates, stamp)
        if selected is None:
            self.counts["rejected"] += 1
            self.counts[reason] += 1
            return MaskResult(self.empty, "rejected", reason, reference)
        self.last = self._position(selected, stamp)
        self.counts[selected.source] += 1
        result = self.refine(selected.mask, square, valid, "off")
        np.logical_and(result, self.geometry.output_allowed, out=result)
        return MaskResult(result, selected.source, lighting=reference)


def export_masking_diagnostics(store, backgrounds, preparation_seconds):
    fields = ("frames", "background_supported", "background", "neural", "rejected", "no_candidate",
              "ambiguous_candidates", "position_or_area_jump", "background_unavailable",
              "lighting_transition_frames", "lighting_changes", "reacquisitions")
    path = store.output / "masking_diagnostics.csv"
    temporary = path.with_suffix(".csv.tmp")
    totals = Counter()
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "start", "end", *fields])
        writer.writeheader()
        for row in store.masking_records():
            counts = json.loads(row["counters"])
            totals.update(counts)
            writer.writerow({"epoch": row["epoch"], "start": row["start"], "end": row["end"],
                             **{key: counts.get(key, 0) for key in fields}})
    temporary.replace(path)
    atomic_json(store.output / "masking_summary.json", {
        "mode": "enhanced", "version": MASKING_VERSION, "counts": dict(totals),
        "lighting_conditions": len(backgrounds.references), "background_sha256": backgrounds.digest,
        "background_preparation_seconds_this_run": preparation_seconds,
        "notice": "Detection counts describe coverage, not measured masking or sleep accuracy.",
    })
