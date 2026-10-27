import csv
import numpy as np
import pytest
from mouse_sleep.config import Settings
from mouse_sleep.video import Timing, frames, inspect_video, validate_rate, prepare
from mouse_sleep.video import sampled_frames
from tests.helpers import make_video


@pytest.mark.parametrize("times,start,expected", [
    ([0, 1 / 30, 2 / 30], 0, [0, 1, 2]),
    ([i / 60 for i in range(6)], 0, [0, 2, 3]),
    ([0, 1 / 30, 2, 2 + 1 / 30], 0, [0, 1, 2, 3]),
    ([9.95, 10 - 1 / 30, 10, 10 + 1 / 30], 10, [2, 3]),
])
def test_sampling_retains_original_frames_and_timestamps(monkeypatch, times, start, expected):
    closed = []
    image = np.zeros((2, 2), np.uint8)

    def source(*args):
        try:
            yield from ((i, stamp, image) for i, stamp in enumerate(times))
        finally:
            closed.append(True)

    monkeypatch.setattr("mouse_sleep.video.frames", source)
    selected = list(sampled_frames(Settings(), None, start))
    assert [row[0] for row in selected] == expected
    assert [row[1] for row in selected] == [times[i] for i in expected]
    assert all(row[2] is image for row in selected)
    assert closed == [True]


def test_closing_sampler_closes_decoder(monkeypatch):
    closed = []

    def source(*args):
        try:
            yield 0, 0, None
            yield 1, 1 / 30, None
        finally:
            closed.append(True)

    monkeypatch.setattr("mouse_sleep.video.frames", source)
    sampled = sampled_frames(Settings(), None)
    next(sampled)
    sampled.close()
    assert closed == [True]


def test_recorder_take_and_acquisition_gaps(tmp_path):
    video = make_video(tmp_path / "flir_cam0_take02.mp4", count=75)
    path = tmp_path / "flir_log_cam0.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["take_id", "host_received_monotonic_ns", "frame_id", "pc_time"])
        writer.writerow([1, 100000, 1, 1700000000])
        for i in range(75):
            writer.writerow([2, 1000000000 + round((i / 30 + (1 if i >= 30 else 0)) * 1e9), 100 + i + (30 if i >= 30 else 0), 1700000001 + i / 30])
    settings = Settings(video=str(video), output=str(tmp_path / "out"))
    timing = Timing(settings)
    assert timing.take_id == 2
    assert timing.time(30, 1) == 2
    assert timing.duration(inspect_video(video)) == pytest.approx(3.5)
    observed = list(frames(settings, timing))
    assert len(observed) == 75 and observed[30][1] == 2
    resumed = list(frames(settings, timing, 2))
    assert resumed[0][0] == 30
    assert resumed[-1][0] == 74


def test_timestamp_csv_count_mismatch_is_rejected(tmp_path):
    video = make_video(tmp_path / "flir_cam0.mp4", count=30)
    with (tmp_path / "flir_log_cam0.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["take_id", "host_received_monotonic_ns", "frame_id"])
        for i in range(31):
            writer.writerow([1, 1000000000 + round(i / 30 * 1e9), i])
    settings = Settings(video=str(video), output=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="timing CSV"):
        list(frames(settings, Timing(settings)))
    with pytest.raises(ValueError, match="timing CSV"):
        list(frames(settings, Timing(settings), 0.5))


def test_prepare_keeps_original_and_exports_timing_map(tmp_path):
    video = make_video(tmp_path / "source.mp4", count=31)
    before = video.read_bytes()
    output = tmp_path / "prepared"
    prepared = prepare(Settings(video=str(video), output=str(output)))
    assert video.read_bytes() == before and prepared.is_file()
    assert len(list(csv.DictReader((output / "prepared_frames.csv").open()))) == 31
    with pytest.raises(ValueError, match="preview derivative"):
        inspect_video(prepared)
    low_rate = make_video(tmp_path / "slow.mp4", count=30, fps=15)
    with pytest.raises(ValueError, match="30 fps"):
        validate_rate(inspect_video(low_rate))
