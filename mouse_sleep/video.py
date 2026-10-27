from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
from pathlib import Path
import csv
import json
import re

import av
from av.video.reformatter import VideoReformatter
import cv2
import numpy as np

from .config import Cancelled, Settings, SAMPLE_RATE, atomic_json
from .tasks import CancelCallback, ProgressCallback, ignore_progress


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    frames: int | None


def inspect_video(path) -> VideoInfo:
    if Path(path).name.startswith("prepared_480") and (Path(path).parent / "prepared_video.json").exists():
        raise ValueError("This is a fitted preview derivative. Select the original recording for analysis so acquisition timing and pixel detail are preserved.")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("This file does not contain a video stream.")
        stream = container.streams.video[0]
        rate = float(stream.average_rate or stream.guessed_rate or 0)
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else float((container.duration or 0) / av.time_base)
        return VideoInfo(stream.width, stream.height, rate, duration, stream.frames or None)


def validate_rate(info):
    if info.fps < 29.0:
        raise ValueError(f"Video is {info.fps:.2f} fps. The published classifier needs approximately 30 fps or higher. Record at 30 fps; resizing or duplicating frames cannot restore missing motion.")


class Timing:
    """Map each decoded frame to its original acquisition time using compact arrays."""

    def __init__(self, settings: Settings):
        self.source = "video_presentation_timestamps"
        self.path = None
        self.take_id = settings.take_id
        self.times = None
        self.frame_ids = None
        self.origin_ns = None
        self.wall_start = None
        self.stop_seconds = None
        video = Path(settings.video)
        match = re.fullmatch(r"flir_cam(\d+)(?:_take(\d+))?", video.stem, re.IGNORECASE)
        path = Path(settings.timing_csv) if settings.timing_csv else (video.parent / f"flir_log_cam{match[1]}.csv" if match else None)
        if path is None or not path.is_file():
            if settings.timing_csv:
                raise ValueError("The selected timing CSV does not exist.")
            return
        self.take_id = self.take_id or (int(match[2] or 1) if match else None)
        # Only selected take rows are retained; no 24-hour pandas frame table.
        ns, ids = [], []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            required = {"take_id", "host_received_monotonic_ns", "frame_id"}
            if not required.issubset(columns):
                raise ValueError("Timing CSV needs take_id, host_received_monotonic_ns and frame_id columns.")
            takes = set()
            first_wall = None
            for row in reader:
                take = int(row["take_id"])
                takes.add(take)
                if self.take_id is not None and take != self.take_id:
                    continue
                ns.append(int(row["host_received_monotonic_ns"]))
                ids.append(int(row["frame_id"]))
                if first_wall is None and row.get("pc_time"):
                    first_wall = float(row["pc_time"])
            if self.take_id is None and len(takes) != 1:
                raise ValueError("Select a take ID when a timing CSV contains multiple takes.")
            if self.take_id is None:
                self.take_id = next(iter(takes), None)
        if not ns:
            raise ValueError("No timing rows match the selected take.")
        times_ns = np.asarray(ns, dtype=np.int64)
        self.frame_ids = np.asarray(ids, dtype=np.int64)
        if np.any(np.diff(times_ns) <= 0) or np.any(np.diff(self.frame_ids) <= 0):
            raise ValueError("Frame IDs and acquisition times must increase within a take.")
        self.origin_ns = int(times_ns[0])
        self.wall_start = first_wall
        self.times = (times_ns - self.origin_ns) / 1e9
        self.source = "recorder_host_frame_receipt"
        self.path = str(path.resolve())
        sync_path = video.parent / f"take{self.take_id:02d}_sync.json"
        if sync_path.is_file():
            report = json.loads(sync_path.read_text(encoding="utf-8-sig"))
            stop = report.get("shared_stop_monotonic_ns", report.get("stop_monotonic_ns"))
            if stop is not None:
                self.stop_seconds = max(0, (int(stop) - self.origin_ns) / 1e9)

    def time(self, index, pts_time):
        if self.times is None:
            return pts_time
        if index >= len(self.times):
            raise ValueError("Video has more frames than its timing CSV; cannot align this recording safely.")
        return float(self.times[index])

    def duration(self, info):
        if self.times is not None:
            return max(float(self.times[-1]) + 1 / SAMPLE_RATE, self.stop_seconds or 0)
        return info.duration

    def manifest(self):
        return {"source": self.source, "csv": self.path, "take_id": self.take_id,
                "origin_host_monotonic_ns": self.origin_ns, "first_frame_unix_seconds": self.wall_start,
                "time_zero": "first recorded FLIR frame", "accuracy": "host receipt times are not exposure times" if self.times is not None else "container presentation timestamps"}


def frames(settings, timing: Timing, start_seconds=0.0, cancel=None):
    with av.open(str(settings.video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # VideoFrame.to_ndarray(format=...) creates a converter for each frame.
        # Keep FFmpeg's conversion context for the lifetime of this decoder.
        converter = VideoReformatter()
        fps = float(stream.average_rate or stream.guessed_rate or SAMPLE_RATE)
        first_pts = float((stream.start_time or 0) * stream.time_base)
        initial_index = 0
        if start_seconds > 0:
            # Acquisition timestamps may differ markedly from fixed-30-fps MP4 time.
            target_index = int(np.searchsorted(timing.times, start_seconds)) if timing.times is not None else int(start_seconds * fps)
            seek_time = max(0, (target_index - int(fps * 2)) / fps)
            container.seek(int((seek_time + first_pts) / float(stream.time_base)), stream=stream, backward=True)
            initial_index = None
        previous_pts, previous_time = None, None
        index = initial_index
        decoded = 0
        for frame in container.decode(stream):
            if cancel is not None and cancel():
                raise Cancelled()
            if frame.pts is None:
                if timing.times is None:
                    raise ValueError("Video frame lacks presentation time; provide a recorder timing CSV.")
                if index is None:
                    raise ValueError("Cannot resume a video with missing frame timestamps.")
                pts = index / fps
            else:
                pts = float(frame.pts * frame.time_base) - first_pts
            if index is None:
                # The recorder's MP4 stores one frame per encoded 1/30-second slot.
                index = round(pts * fps) if timing.times is not None else 0
            if timing.times is not None and abs(pts - index / fps) > 0.5 / fps:
                raise ValueError("Recorder CSV alignment requires a fixed-rate original MP4 with one encoded frame per frame slot.")
            if previous_pts is not None and pts <= previous_pts:
                raise ValueError("Video presentation timestamps are not strictly increasing.")
            time = timing.time(index, pts)
            if previous_time is not None and time <= previous_time:
                raise ValueError("Mapped frame times are not strictly increasing.")
            previous_pts, previous_time = pts, time
            if settings.max_seconds is not None and time >= settings.max_seconds:
                break
            if time >= start_seconds - 1 / SAMPLE_RATE:
                yield index, time, converter.reformat(frame, format="gray").to_ndarray()
            index += 1
            decoded += 1
        if timing.times is not None and settings.max_seconds is None and index != len(timing.times):
            raise ValueError(f"Video ends at frame {index}, but timing CSV has {len(timing.times)} rows.")


def sampled_frames(settings, timing: Timing, start_seconds=0.0, cancel: CancelCallback = None):
    """Select at most one original frame per 30 Hz slot without inventing frames."""
    next_slot = round(start_seconds * SAMPLE_RATE)
    with closing(frames(settings, timing, start_seconds, cancel)) as source:
        for index, stamp, gray in source:
            slot = round(stamp * SAMPLE_RATE)
            if slot < next_slot:
                continue
            next_slot = slot + 1
            yield index, stamp, gray


class SquareCrop:
    """Geometry shared by frames of the same size; image pixels remain owned per frame."""

    def __init__(self, shape, roi=None):
        self.shape = tuple(shape[:2])
        height, width = self.shape
        x, y, w, h = roi or (0, 0, width, height)
        if x + w > width or y + h > height:
            raise ValueError("Selected cage region extends outside the video.")
        self.region = (slice(y, y + h), slice(x, x + w))
        side = max(w, h)
        left, top = (side - w) // 2, (side - h) // 2
        self.border = (top, side - h - top, left, side - w - left)
        self.valid = np.zeros((side, side), np.uint8)
        self.valid[top:top + h, left:left + w] = 1

    def apply(self, gray):
        if gray.shape[:2] != self.shape:
            raise ValueError("Frame dimensions do not match the selected crop geometry.")
        return cv2.copyMakeBorder(gray[self.region], *self.border, cv2.BORDER_REPLICATE)


def square_image(gray, roi=None):
    crop = SquareCrop(gray.shape, roi)
    return crop.apply(gray), crop.valid


def prepare(settings: Settings, progress: ProgressCallback = ignore_progress, cancel: CancelCallback = None):
    settings.validate()
    info = inspect_video(settings.video)
    validate_rate(info)
    timing = Timing(settings)
    output = Path(settings.output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "prepared_480.mp4"
    mapping_path = output / "prepared_frames.csv"
    if destination.exists() or mapping_path.exists():
        raise ValueError("Prepared output already exists. Choose a new output folder.")
    temporary = output / "prepared_480.partial.mp4"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), SAMPLE_RATE, (480, 480), True)
    if not writer.isOpened():
        raise OSError("Cannot open the MP4 preparation writer.")
    count = 0
    crop = None
    duration = min(timing.duration(info), settings.max_seconds or float("inf"))
    status = "error"
    try:
        with mapping_path.open("w", newline="", encoding="utf-8") as handle, closing(sampled_frames(settings, timing, cancel=cancel)) as source:
            mapping = csv.writer(handle)
            mapping.writerow(["prepared_frame", "source_frame", "source_seconds", "prepared_seconds"])
            for index, time, gray in source:
                if crop is None or crop.shape != gray.shape[:2]:
                    crop = SquareCrop(gray.shape, settings.roi)
                square = crop.apply(gray)
                prepared = cv2.resize(square, (480, 480), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR))
                mapping.writerow([count, index, f"{time:.9f}", f"{count / SAMPLE_RATE:.9f}"])
                count += 1
                if count % 300 == 0:
                    progress({"message": "Preparing video", "seconds": time, "duration": duration})
        status = "complete"
    except Cancelled:
        status = "cancelled"
        raise
    finally:
        writer.release()
        atomic_json(output / "prepared_video.json", {"status": status, "settings": settings.identity(), "timing": timing.manifest(), "frames": count,
                    "notice": "Segmentation preview derivative. Analyze the ORIGINAL video for sleep features; gaps are recorded in prepared_frames.csv and are not filled."})
        if status == "complete":
            temporary.replace(destination)
    progress({"message": f"Prepared {count} frames", "seconds": duration, "duration": duration})
    return destination
