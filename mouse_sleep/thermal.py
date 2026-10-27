"""Streaming reader for UltramicStudio's native Celsius frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np


PATCH_SIZE = 7


@dataclass
class ThermalFrame:
    index: int
    take_id: int
    host_ns: int
    utc_ns: int
    pixels: np.ndarray | None  # Cached measurements retain pixels only for preview frames.


def rectangle(value, *, minimum=1):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(v) is not int for v in value)
            or min(value[:2]) < 0 or min(value[2:]) < minimum):
        raise ValueError(f"ROI must be integer [x, y, width, height], with dimensions >= {minimum}.")
    return tuple(value)


def rectangles_overlap(a, b):
    return (max(a[0], b[0]) < min(a[0] + a[2], b[0] + b[2])
            and max(a[1], b[1]) < min(a[1] + a[3], b[1] + b[3]))


def check_bounds(roi, shape):
    height, width = shape
    if roi[0] + roi[2] > width or roi[1] + roi[3] > height:
        raise ValueError(f"ROI {roi} extends outside the native {width} x {height} image.")


def _fixed_decimal_pixels(lines, width):
    """Decode the recorder's NN.N, tokens only after validating every byte.

    Other numeric formats go through the general reader, including negative
    temperatures, additional precision and nonfinite values.
    """
    rows = [line.rstrip("\r\n") for line in lines]
    if any(len(row) != width * 5 for row in rows):
        return None
    try:
        data = "".join(rows).encode("ascii")
    except UnicodeEncodeError:
        return None
    tokens = np.frombuffer(data, np.uint8).reshape(len(rows), width, 5)
    if not ((tokens[..., 2] == ord(".")).all() and (tokens[..., 4] == ord(",")).all()):
        return None
    tens, units, decimal = tokens[..., 0], tokens[..., 1], tokens[..., 3]
    if any(not ((digits >= 48) & (digits <= 57)).all() for digits in (tens, units, decimal)):
        return None
    tenths = (tens.astype(np.uint16) - 48) * 100 + (units - 48) * 10 + (decimal - 48)
    return tenths.astype(np.float64) / 10


def _read_pixels(raw, shape, index):
    height, width = shape
    lines = []
    for y in range(height):
        line = raw.readline()
        if not line:
            raise ValueError(f"Truncated thermal frame {index}: missing pixel row {y}.")
        lines.append(line)
    pixels = _fixed_decimal_pixels(lines, width)
    if pixels is not None:
        return pixels
    pixels = np.empty(shape, dtype=np.float64)
    for y, line in enumerate(lines):
        cells = line.strip().removesuffix(",").split(",")
        if len(cells) != width:
            raise ValueError(f"Thermal frame {index}, row {y}: expected {width} pixels.")
        try:
            pixels[y] = cells
        except ValueError as exc:
            raise ValueError(f"Invalid thermal pixels in frame {index}, row {y}.") from exc
    return pixels


def thermal_frames(recording_dir):
    """Read one raw frame at a time; verify its corresponding Seek log row.

    The CSV consists of a metadata row and `height` pixel rows per frame.
    UTC is retained for reference; alignment always uses the integer host clock.
    """
    root = Path(recording_dir)
    previous_ns = previous_index = -1
    previous_take = 0
    shape = None
    with (root / "seek_thermal.csv").open(encoding="utf-8-sig", newline="") as raw, \
            (root / "seek_log.csv").open(encoding="utf-8-sig", newline="") as log:
        rows = csv.DictReader(log)
        required = {"idx", "take_id", "sdk_utc_ns", "host_received_monotonic_ns"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError("seek_log.csv lacks frame IDs, take IDs, or acquisition timestamps.")
        for number, header in enumerate(raw):
            # Each loop consumes all pixel rows before the next metadata row.
            metadata = {}
            for item in next(csv.reader([header])):
                if "=" in item:
                    key, value = item.split("=", 1)
                    if key in metadata and metadata[key] != value:
                        raise ValueError(f"Conflicting thermal metadata: {key}.")
                    metadata[key] = value
            try:
                take = int(metadata["take_id"])
                host_ns = int(metadata["host_received_monotonic_ns"])
                utc_ns = int(metadata["timestamp_utc_ns"])
                width, height = int(metadata["width"]), int(metadata["height"])
                fmt = tuple(int(metadata[k]) for k in ("type", "channels", "pixel_depth"))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid thermal metadata at frame {number}.") from exc
            if fmt != (16, 1, 32):
                raise ValueError("Expected native float Celsius thermography (type 16, one channel, 32 bits).")
            if min(width, height, take, host_ns, utc_ns) <= 0:
                raise ValueError("Thermal dimensions, take IDs, and timestamps must be positive.")
            if shape is not None and shape != (height, width):
                raise ValueError("Thermal resolution changed during the recording; redefine the cage regions.")
            shape = (height, width)
            row = next(rows, None)
            if row is None:
                raise ValueError("Raw thermal data has more frames than seek_log.csv.")
            try:
                index = int(row["idx"])
                logged = tuple(int(row[k]) for k in
                               ("take_id", "host_received_monotonic_ns", "sdk_utc_ns"))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid Seek log row at frame {number}.") from exc
            if logged != (take, host_ns, utc_ns):
                raise ValueError(f"Raw thermal frame {number} does not match seek_log.csv.")
            if host_ns <= previous_ns or index <= previous_index or take < previous_take:
                raise ValueError("Seek frame IDs/times must increase, with takes in recording order.")
            pixels = _read_pixels(raw, shape, index)
            previous_ns, previous_index, previous_take = host_ns, index, take
            yield ThermalFrame(index, take, host_ns, utc_ns, pixels)
        if next(rows, None) is not None:
            raise ValueError("seek_log.csv has more frames than the raw thermal data.")


def hottest_patch(pixels, roi):
    """Maximum-mean complete 7x7 patch, wholly inside one native cage ROI.

    A patch containing any nonfinite pixel is ineligible. Equal means select
    the first patch in row-major order. Returned coordinates are native pixels.
    """
    roi = rectangle(roi, minimum=PATCH_SIZE)
    check_bounds(roi, pixels.shape)
    x, y, width, height = roi
    crop = pixels[y:y + height, x:x + width]
    finite = np.isfinite(crop)

    def sums(array):
        integral = np.pad(array.cumsum(0, dtype=np.float64).cumsum(1), ((1, 0), (1, 0)))
        k = PATCH_SIZE
        return integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]

    means = sums(np.where(finite, crop, 0)) / PATCH_SIZE ** 2
    valid = (sums(finite) == PATCH_SIZE ** 2) & np.isfinite(means)
    if not valid.any():
        return {"camera_temperature_7x7_c": None, "patch_x": None, "patch_y": None,
                "patch_touches_roi_edge": False, "temperature_reason": "no_finite_7x7_patch"}
    py, px = np.unravel_index(np.where(valid, means, -np.inf).argmax(), means.shape)
    return {"camera_temperature_7x7_c": float(means[py, px]),
            "patch_x": int(x + px), "patch_y": int(y + py),
            "patch_touches_roi_edge": bool(px == 0 or py == 0 or
                                           px == width - PATCH_SIZE or py == height - PATCH_SIZE),
            "temperature_reason": ""}
