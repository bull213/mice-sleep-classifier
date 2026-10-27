"""Extract reusable native-temperature measurements, independent of sleep labels."""
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import tempfile
import zipfile
import zlib

import numpy as np

from .config import ROOT
from .thermal import PATCH_SIZE, ThermalFrame, check_bounds, hottest_patch, thermal_frames


_CACHE_VERSION = 1  # Bump when pixel parsing or native patch measurement semantics change.
_PATCH_DTYPE = np.dtype([("temperature", "<f8"), ("x", "<i8"), ("y", "<i8"), ("edge", "?")])


@dataclass
class ThermalMeasurements:
    shape: tuple
    frame_info: np.ndarray
    patches: np.ndarray
    previews: dict

    def samples(self):
        for number, info in enumerate(self.frame_info):
            frame = ThermalFrame(*(int(v) for v in info), self.previews.get(number))
            patches = []
            for patch in self.patches[number]:
                usable = np.isfinite(patch["temperature"])
                patches.append({"camera_temperature_7x7_c": float(patch["temperature"]) if usable else None,
                                "patch_x": int(patch["x"]) if usable else None,
                                "patch_y": int(patch["y"]) if usable else None,
                                "patch_touches_roi_edge": bool(patch["edge"]),
                                "temperature_reason": "" if usable else "no_finite_7x7_patch"})
            yield number, frame, patches


def _identity(recording_dir, rois, preview_frames):
    sources = []
    for name in ("seek_thermal.csv", "seek_log.csv"):
        path = recording_dir / name
        stat = path.stat()
        sources.append([str(path), stat.st_size, stat.st_mtime_ns])
    return {"version": _CACHE_VERSION, "sources": sources, "rois": [list(roi) for roi in rois],
            "patch_size": PATCH_SIZE, "preview_frames": preview_frames}


def _cache_path(recording_dir, rois, preview_frames):
    # Stable slot: changed sources replace an obsolete cache instead of accumulating copies.
    key = json.dumps([str(recording_dir), rois, preview_frames], sort_keys=True).encode("utf-8")
    return ROOT / ".cache" / "thermal" / (hashlib.sha256(key).hexdigest() + ".npz")


def _load_cache(path, identity):
    try:
        with np.load(path, allow_pickle=False) as archive:
            if json.loads(str(archive["identity"].item())) != identity:
                return None
            shape = tuple(int(v) for v in archive["shape"])
            info, patches = archive["frame_info"], archive["patches"]
            indices, pixels = archive["preview_indices"], archive["preview_pixels"]
        if len(shape) != 2 or min(shape) <= 0:
            return None
        if info.dtype != np.dtype("int64") or info.ndim != 2 or info.shape[1] != 4 or not len(info):
            return None
        if patches.dtype != _PATCH_DTYPE or patches.shape != (len(info), len(identity["rois"])):
            return None
        if (np.any(info[:, 0] < 0) or np.any(info[:, 1:] <= 0)
                or np.any(np.diff(info[:, 0]) <= 0) or np.any(np.diff(info[:, 2]) <= 0)
                or np.any(np.diff(info[:, 1]) < 0) or np.isinf(patches["temperature"]).any()):
            return None
        expected = np.linspace(0, len(info) - 1, min(len(info), identity["preview_frames"]), dtype=int)
        if (indices.dtype != np.dtype("int64") or not np.array_equal(indices, expected)
                or pixels.dtype != np.dtype("float64") or pixels.shape != (len(expected), *shape)):
            return None
        for i, roi in enumerate(identity["rois"]):
            check_bounds(roi, shape)
            selected = patches[:, i]
            finite = np.isfinite(selected["temperature"])
            x, y = selected["x"], selected["y"]
            if (np.any(finite & ((x < roi[0]) | (x > roi[0] + roi[2] - PATCH_SIZE)
                                 | (y < roi[1]) | (y > roi[1] + roi[3] - PATCH_SIZE)))
                    or np.any(~finite & ((x != -1) | (y != -1) | selected["edge"]))):
                return None
        return ThermalMeasurements(shape, info, patches, dict(zip(map(int, indices), pixels)))
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile, zlib.error):
        return None


def _save_cache(path, identity, measurements):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            indices = sorted(measurements.previews)
            np.savez_compressed(handle, identity=json.dumps(identity, sort_keys=True),
                                shape=np.asarray(measurements.shape, dtype=np.int64),
                                frame_info=measurements.frame_info, patches=measurements.patches,
                                preview_indices=np.asarray(indices, dtype=np.int64),
                                preview_pixels=np.stack([measurements.previews[i] for i in indices]))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_measurements(recording_dir, rois, *, preview_only=False, preview_frames=12,
                         use_cache=True, refresh_cache=False, progress=print, frame_source=thermal_frames):
    """Read the raw recording once, or load a validated compact extraction cache.

    Partial previews never read or write the full-recording cache. An injected
    reader also bypasses it so custom input transformations cannot reuse raw results.
    """
    recording_dir = Path(recording_dir).resolve()
    identity = _identity(recording_dir, rois, preview_frames)
    use_cache = use_cache and not preview_only and frame_source is thermal_frames
    path = _cache_path(recording_dir, rois, preview_frames)
    if use_cache and not refresh_cache:
        cached = _load_cache(path, identity)
        if cached is not None:
            if _identity(recording_dir, rois, preview_frames) != identity:
                raise ValueError("Thermal source files changed while loading measurements; retry after recording stops.")
            progress(f"Reusing {len(cached.frame_info):,} cached thermal measurements per cage")
            return cached
    if preview_only:
        preview_indices = set(range(preview_frames))
    else:
        with (recording_dir / "seek_log.csv").open(encoding="utf-8-sig", newline="") as handle:
            total = sum(1 for _ in csv.DictReader(handle))
        preview_indices = set(np.linspace(0, max(0, total - 1), min(total, preview_frames), dtype=int))
    info, patches, previews = [], [], {}
    shape = None
    with closing(frame_source(recording_dir)) as frames:
        for number, frame in enumerate(frames):
            shape = frame.pixels.shape
            info.append((frame.index, frame.take_id, frame.host_ns, frame.utc_ns))
            row = []
            for roi in rois:
                patch = hottest_patch(frame.pixels, roi)
                value = patch["camera_temperature_7x7_c"]
                row.append((value if value is not None else np.nan,
                            patch["patch_x"] if value is not None else -1,
                            patch["patch_y"] if value is not None else -1,
                            patch["patch_touches_roi_edge"]))
            patches.append(row)
            if number in preview_indices:
                previews[number] = frame.pixels
            if number % 250 == 0:
                progress(f"Read thermal frame {number + 1:,}, take {frame.take_id} ({len(rois)} cage regions)")
            if preview_only and number + 1 >= preview_frames:
                break
    if not info:
        raise ValueError("No thermal frames were found.")
    if _identity(recording_dir, rois, preview_frames) != identity:
        raise ValueError("Thermal source files changed while extracting measurements; retry after recording stops.")
    result = ThermalMeasurements(shape, np.asarray(info, dtype=np.int64),
                                 np.asarray(patches, dtype=_PATCH_DTYPE), previews)
    if use_cache:
        try:
            _save_cache(path, identity, result)
        except OSError as exc:
            # Caching is optional; a read-only cache directory must not prevent a report.
            progress(f"Could not save thermal cache; using extracted measurements: {exc}")
    return result
