"""Bounded, immutable background references for Enhanced masking."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import cv2
import numpy as np

from .config import MASKING_VERSION, ROOT, sha256
from .tasks import check_cancelled, ignore_progress
from .video import SquareCrop, frames

SAMPLE_COUNT = 96
CACHE_ERRORS = (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile)


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    def on(p, q, r):
        return _cross(p, q, r) == 0 and all(min(p[i], q[i]) <= r[i] <= max(p[i], q[i]) for i in (0, 1))
    return ((_cross(a, b, c) * _cross(a, b, d) < 0 and _cross(c, d, a) * _cross(c, d, b) < 0)
            or on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b))


class CageGeometry:
    """Boundary masks share the original crop/padding; inference images are unchanged."""

    def __init__(self, settings, shape):
        self.crop = SquareCrop(shape, settings.roi)
        height, width = shape
        x, y, w, h = settings.roi or (0, 0, width, height)
        original = np.zeros(shape, np.uint8)
        if settings.cage_polygon:
            points = tuple(tuple(point) for point in settings.cage_polygon)
            if any(not (x <= px < x + w and y <= py < y + h) for px, py in points):
                raise ValueError("Cage boundary points must lie inside the selected cage region.")
            if len(set(points)) != len(points):
                raise ValueError("Cage boundary points must be distinct; do not repeat the first point.")
            edges = list(zip(points, (*points[1:], points[0])))
            for i, (a, b) in enumerate(edges):
                for j in range(i + 1, len(edges)):
                    if j == i + 1 or (i == 0 and j == len(edges) - 1):
                        continue
                    if _intersects(a, b, *edges[j]):
                        raise ValueError("Cage boundary must not cross or touch itself.")
            polygon = np.asarray(points, np.int32)
            if cv2.contourArea(polygon) < 20:
                raise ValueError("Cage boundary encloses too little area.")
            cv2.fillPoly(original, [polygon], 1)
        else:
            original[y:y+h, x:x+w] = 1
        native = self.crop.apply(original) & self.crop.valid
        self.allowed = cv2.resize(native, (480, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
        if self.allowed.sum() < 100:
            raise ValueError("The cage boundary is too small after fitting to the model.")
        self.output_allowed = cv2.resize(native, (1080, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
        self.grid_weights = cv2.resize(self.allowed.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)
        self.grid_active = self.grid_weights.ravel() > 0.1

    def image(self, gray):
        return cv2.resize(self.crop.apply(gray), (480, 480), interpolation=cv2.INTER_AREA)

    def signature(self, image):
        values = cv2.resize(image.astype(np.float32) * self.allowed, (8, 8), interpolation=cv2.INTER_AREA)
        return np.divide(values, self.grid_weights, out=np.zeros_like(values),
                         where=self.grid_weights > 0.1).ravel()

    def distances(self, signatures, signature):
        return np.sqrt(np.mean((signatures[:, self.grid_active] - signature[self.grid_active]) ** 2, axis=1))


@dataclass
class Backgrounds:
    references: np.ndarray
    signatures: np.ndarray
    tolerances: np.ndarray
    uncertain: np.ndarray
    metadata: dict

    @property
    def digest(self):
        digest = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode())
        for value in (self.references, self.signatures, self.tolerances, self.uncertain):
            digest.update(value.tobytes())
        return digest.hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, references=self.references, signatures=self.signatures,
                                    tolerances=self.tolerances, uncertain=self.uncertain,
                                    metadata=json.dumps(self.metadata, sort_keys=True), digest=self.digest)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, expected_identity):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            if not isinstance(metadata, dict) or metadata.get("identity") != expected_identity:
                raise ValueError("Background references do not match the video, boundary, or reference images.")
            result = cls(*(saved[key].copy() for key in ("references", "signatures", "tolerances", "uncertain")), metadata)
            n = len(result.references)
            if (n not in (1, 2) or result.references.shape != (n, 480, 480)
                    or result.signatures.shape != (n, 64) or result.tolerances.shape != (n,)
                    or result.uncertain.shape != result.references.shape
                    or result.references.dtype != np.float32 or result.signatures.dtype != np.float32
                    or result.tolerances.dtype != np.float32 or result.uncertain.dtype != bool
                    or any(not np.isfinite(a).all() for a in (result.references, result.signatures, result.tolerances))
                    or np.any((result.references < 0) | (result.references > 255))
                    or np.any(result.tolerances <= 0) or str(saved["digest"].item()) != result.digest):
                raise ValueError("Invalid or damaged background references.")
        return result


def background_identity(settings, info, timing):
    def source(path):
        path = Path(path).resolve()
        stat = path.stat()
        return [str(path), stat.st_size, stat.st_mtime_ns]
    return {"version": MASKING_VERSION, "samples": SAMPLE_COUNT, "video": source(settings.video),
            "timing": source(timing.path) if timing.path else None, "take_id": timing.take_id,
            "shape": [info.height, info.width], "roi": list(settings.roi) if settings.roi else None,
            "polygon": [list(p) for p in settings.cage_polygon] if settings.cage_polygon else None,
            "images": [[str(Path(p).resolve()), sha256(Path(p))] for p in settings.background_images]}


def _sample_images(settings, info, timing, geometry, progress, cancel):
    # Sample actual frame positions, avoiding recording pauses and unrecorded tails.
    if timing.times is not None:
        indices = np.unique(np.linspace(0, len(timing.times) - 1, min(SAMPLE_COUNT, len(timing.times))).astype(int))
        targets = timing.times[indices]
    else:
        count = max(1, info.frames or round(info.duration * info.fps))
        indices = np.unique(np.linspace(0, count - 1, min(SAMPLE_COUNT, count)).astype(int))
        targets = indices / info.fps
    full = replace(settings, max_seconds=None)
    samples, positions = [], []
    if info.duration <= 180:
        # One sequential decode is cheaper than repeated seeks on short pilots.
        target = 0
        with closing(frames(full, timing, cancel=cancel)) as source:
            for index, stamp, gray in source:
                if stamp + 1e-7 < targets[target]:
                    continue
                samples.append(geometry.image(gray))
                positions.append([int(index), float(stamp)])
                target += 1
                if target == len(targets):
                    break
                progress({"message": f"Sampling backgrounds: {target}/{len(targets)}"})
    else:
        for number, target in enumerate(targets):
            check_cancelled(cancel)
            with closing(frames(full, timing, float(target), cancel)) as source:
                for index, stamp, gray in source:
                    if stamp + 1e-7 >= target:
                        samples.append(geometry.image(gray))
                        positions.append([int(index), float(stamp)])
                        break
            progress({"message": f"Sampling backgrounds: {number + 1}/{len(targets)}"})
    if not samples:
        raise ValueError("No frames available for background preparation.")
    return np.stack(samples), positions


def build_backgrounds(samples, geometry, identity, *, reference_images=(), positions=()):
    """Deterministic two-condition grouping; no continuously adapting background."""
    signatures = np.stack([geometry.signature(image) for image in samples])
    active = signatures[:, geometry.grid_active]
    order = np.argsort(active.mean(axis=1), kind="stable")
    centers = signatures[[order[0], order[-1]]].copy()
    labels = np.zeros(len(samples), int)
    for _ in range(20):
        distances = np.stack([geometry.distances(centers, value) for value in signatures])
        labels = distances.argmin(axis=1)
        if any(not np.any(labels == i) for i in (0, 1)):
            break
        updated = np.stack([signatures[labels == i].mean(axis=0) for i in (0, 1)])
        if np.allclose(updated, centers, atol=0.01):
            centers = updated
            break
        centers = updated
    separated = geometry.distances(centers, centers[0])[1] >= 12
    if not separated or any(np.count_nonzero(labels == i) < 8 for i in (0, 1)):
        labels = np.zeros(len(samples), int)
    refs, descriptors, tolerances, uncertain = [], [], [], []
    for group in np.unique(labels):
        subset = samples[labels == group]
        reference = np.percentile(subset, 90, axis=0).astype(np.float32)
        signature = signatures[labels == group].mean(axis=0)
        distances = geometry.distances(signatures[labels == group], signature)
        tolerance = max(12.0, min(40.0, float(np.percentile(distances, 95)) + 8))
        # Stable dark structures may include a mouse that never vacated this spot.
        span = np.percentile(subset, 90, axis=0) - np.percentile(subset, 10, axis=0)
        local = cv2.GaussianBlur(reference, (0, 0), 15)
        doubtful = (span < 12) & (local - reference > 12) & geometry.allowed
        refs.append(reference)
        descriptors.append(signature)
        tolerances.append(tolerance)
        uncertain.append(doubtful)
    if len(reference_images) == 2:
        refs, descriptors, tolerances, uncertain = [], [], [], []
    for image in reference_images:
        signature = geometry.signature(image)
        distances = geometry.distances(np.asarray(descriptors), signature) if descriptors else np.empty(0)
        match = int(distances.argmin()) if len(distances) else 0
        if len(reference_images) != 2 and len(distances) and (distances[match] <= 2 * tolerances[match] or len(refs) == 2):
            refs[match], descriptors[match] = image.astype(np.float32), signature
            uncertain[match] = np.zeros_like(geometry.allowed)
        else:
            refs.append(image.astype(np.float32))
            descriptors.append(signature)
            tolerances.append(20.0)
            uncertain.append(np.zeros_like(geometry.allowed))
    return Backgrounds(np.asarray(refs, np.float32), np.asarray(descriptors, np.float32),
                       np.asarray(tolerances, np.float32), np.asarray(uncertain, bool),
                       {"identity": identity, "sample_positions": list(positions),
                        "conditions": len(refs), "provided_references": len(reference_images)})


def prepare_backgrounds(settings, info, timing, *, progress=ignore_progress, cancel=None, saved_path=None):
    geometry = CageGeometry(settings, (info.height, info.width))
    identity = background_identity(settings, info, timing)
    check_cancelled(cancel)
    if saved_path is not None:
        try:
            result = Backgrounds.load(saved_path, identity)
        except CACHE_ERRORS as exc:
            raise ValueError("Saved background references are missing, changed, or damaged. Use a new output folder.") from exc
        if background_identity(settings, info, timing) != identity:
            raise ValueError("Recording or reference images changed while loading backgrounds.")
        return result, geometry
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = ROOT / ".cache" / "masking" / f"{key}.npz"
    if path.exists():
        try:
            result = Backgrounds.load(path, identity)
            if background_identity(settings, info, timing) != identity:
                raise ValueError("Recording or reference images changed while loading backgrounds.")
            progress({"message": "Reusing fixed masking backgrounds"})
            return result, geometry
        except CACHE_ERRORS:
            pass
    progress({"message": "Preparing fixed backgrounds for the recording's lighting conditions"})
    samples, positions = _sample_images(settings, info, timing, geometry, progress, cancel)
    references = []
    for filename in settings.background_images:
        check_cancelled(cancel)
        gray = cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (info.height, info.width):
            raise ValueError("Empty-cage images must have the same dimensions and framing as the original video.")
        references.append(geometry.image(gray))
    check_cancelled(cancel)
    result = build_backgrounds(samples, geometry, identity, reference_images=references, positions=positions)
    if background_identity(settings, info, timing) != identity:
        raise ValueError("Recording or reference images changed during background preparation.")
    check_cancelled(cancel)
    try:
        result.save(path)
    except OSError:
        progress({"message": "Background cache unavailable; using the prepared references for this run"})
    return result, geometry
