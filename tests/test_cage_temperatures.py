import csv
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from mouse_sleep.cage_temperatures import analyze_temperatures, load_config, load_runs
from mouse_sleep.segmentation import network_image
from mouse_sleep.thermal import hottest_patch, thermal_frames


# Above 2**53: conversion of the host clock to float would lose boundary ns.
ORIGIN = 9_007_199_254_741_111
ROIS = [[1, 1, 12, 14], [17, 1, 12, 14]]


def write_case(tmp_path, offsets=None):
    recording = tmp_path / "recording"
    recording.mkdir()
    takes = {1: (ORIGIN, 30), 2: (ORIGIN + 46_000_000_000, 20)}
    optical = recording / "flir_log_cam0.csv"
    with optical.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["take_id", "host_received_monotonic_ns", "frame_id"])
        for take, (origin, duration) in takes.items():
            for frame in range(duration * 30):
                writer.writerow([take, origin + round(frame * 1e9 / 30), frame + 100])
            (recording / f"take{take:02d}_sync.json").write_text(json.dumps({
                "shared_start_monotonic_ns": origin, "shared_stop_monotonic_ns": origin + duration * 1_000_000_000,
            }))
    offsets = offsets or {1: range(30), 2: range(20)}
    with (recording / "seek_thermal.csv").open("w", newline="") as raw, \
            (recording / "seek_log.csv").open("w", newline="") as log:
        raw_writer, log_writer = csv.writer(raw), csv.writer(log)
        log_writer.writerow(["idx", "take_id", "sdk_utc_ns", "host_received_monotonic_ns"])
        index = 0
        for take, seconds in offsets.items():
            for second in seconds:
                host = takes[take][0] + round(second * 1e9)
                utc = 1_790_000_000_000_000_000 + host - ORIGIN
                pixels = np.full((18, 32), 20.0)
                if take == 1:
                    left = 30 if second < 10 else 28 if second < 20 else 90
                    right = 35 if second < 10 else 40 if second < 20 else 37
                else:
                    left, right = (26, 39) if second < 10 else (32, 36)
                pixels[4:11, 3:10] = left
                pixels[4:11, 19:26] = right
                pixels[:, 14:16] = 100  # Hotter object outside both cages.
                raw_writer.writerow([f"take_id={take}", f"host_received_monotonic_ns={host}",
                                     f"timestamp_utc_ns={utc}", "width=32", "height=18", "type=16",
                                     "channels=1", "pixel_depth=32", "blank", ""])
                raw_writer.writerows([*row, ""] for row in pixels)
                log_writer.writerow([index, take, utc, host])
                index += 1
    config = {"schema_version": 1, "calibration": {"slope": 1, "intercept_c": 0}, "cages": []}
    labels = {"A": {1: ["awake", "sleep", "unknown"], 2: ["sleep", "awake"]},
              "B": {1: ["sleep", "awake", "sleep"], 2: ["awake", "sleep"]}}
    for cage_index, mouse in enumerate(("A", "B")):
        cage = {"mouse_id": mouse, "thermal_roi": ROIS[cage_index], "analysis_dirs": [], "exclude_intervals": []}
        for take, (origin, duration) in takes.items():
            directory = tmp_path / mouse / f"take{take:02d}"
            directory.mkdir(parents=True)
            video = recording / ("flir_cam0.mp4" if take == 1 else f"flir_cam0_take{take:02d}.mp4")
            manifest = {"status": "complete", "completed_epochs": duration // 10, "duration_seconds": duration,
                        "timing": {"source": "recorder_host_frame_receipt", "take_id": take,
                                   "origin_host_monotonic_ns": origin, "csv": str(optical)},
                        "settings": {"video": str(video), "roi": [544 * cage_index, 0, 500, 588]},
                        "source_video": {"width": 1044, "height": 588, "fps": 30, "duration": duration, "frames": duration * 30}}
            (directory / "run.json").write_text(json.dumps(manifest))
            with (directory / "epochs.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["epoch", "start", "end", "state", "reason"])
                for epoch, state in enumerate(labels[mouse][take]):
                    writer.writerow([epoch, epoch * 10, (epoch + 1) * 10, state,
                                     "bad_tracking" if state == "unknown" else ""])
            cage["analysis_dirs"].append(str(directory.relative_to(tmp_path)))
        config["cages"].append(cage)
    config_path = tmp_path / "cages.json"
    config_path.write_text(json.dumps(config))
    return recording, config_path, config


@pytest.fixture
def capture_reports(monkeypatch):
    """Exercise raw IO, alignment, and calculations without repeatedly rendering figures."""
    from mouse_sleep import temperature_reports
    captured = {}

    def capture(data, summary, path):
        captured[summary["mouse_id"]] = (data, summary)

    monkeypatch.setattr(temperature_reports, "write_cage_report", capture)
    monkeypatch.setattr(temperature_reports, "write_preview", lambda *args: None)
    return captured


def test_two_cages_full_outputs_preserve_identity_states_and_pause(tmp_path):
    recording, config_path, _ = write_case(tmp_path)
    output = tmp_path / "result"
    result = analyze_temperatures(recording, config_path, output, preview_frames=2, progress=lambda _: None)
    a = json.loads((output / "A" / "summary.json").read_text())
    b = json.loads((output / "B" / "summary.json").read_text())
    assert a["comparisons"]["all"]["sleep_minus_awake_c"] == pytest.approx(-4)
    assert b["comparisons"]["all"]["sleep_minus_awake_c"] == pytest.approx(-3.5)
    assert a["comparisons"]["all"]["sleep"]["samples"] == 20
    assert b["comparisons"]["all"]["sleep"]["samples"] == 30
    assert a["unknown_state_samples"] == 10 and b["unknown_state_samples"] == 0
    assert result["gaps"][0]["duration_seconds"] == 16
    samples = pd.read_csv(output / "A" / "temperature_samples.csv")
    assert len(samples) == 50 and not samples.elapsed_seconds.between(30, 46, inclusive="left").any()
    assert samples.loc[samples.elapsed_seconds == 46, "take_seconds"].item() == 0
    assert samples.host_received_monotonic_ns.iloc[0] == ORIGIN
    assert (samples.patch_x == 3).all() and (samples.patch_y == 4).all()
    assert a["sleep_changes"][0]["sleep_minus_preceding_awake_c"] == -2
    assert a["sleep_changes"][1]["sleep_minus_preceding_awake_c"] is None  # New take starts asleep.
    for mouse in ("A", "B"):
        folder = output / mouse
        assert (folder / "temperature_awake_vs_sleep.png").stat().st_size > 5000
        assert (folder / "temperature_awake_vs_sleep.pdf").read_bytes().startswith(b"%PDF")
        assert (folder / "temperature_analysis.md").is_file()
        assert (folder / "state_intervals.csv").is_file()
        assert (folder / "sleep_temperature_changes.csv").is_file()
    assert (output / "thermal_preview.png").is_file()
    assert a["geometry"][0]["network_active_width"] == round(500 * 480 / 588)


def test_native_patches_ignore_other_cage_and_nonfinite_pixels():
    pixels = np.full((18, 32), 20.0)
    pixels[4:11, 3:10] = 30
    pixels[4:11, 19:26] = 70
    pixels[1, 1] = np.nan
    left = hottest_patch(pixels, ROIS[0])
    assert left["camera_temperature_7x7_c"] == 30
    assert (left["patch_x"], left["patch_y"]) == (3, 4)
    assert not left["patch_touches_roi_edge"]
    pixels[1:15, 1:13] = np.nan
    assert hottest_patch(pixels, ROIS[0])["temperature_reason"] == "no_finite_7x7_patch"
    assert hottest_patch(pixels, ROIS[1])["camera_temperature_7x7_c"] == 70
    with pytest.raises(ValueError, match="outside"):
        hottest_patch(pixels, (30, 0, 7, 7))


def test_exact_epoch_boundaries_and_camera_start_offset(tmp_path, capture_reports):
    recording, config_path, _ = write_case(tmp_path, {
        1: [-.25, 0, 9.999999999, 10, 19.999999999, 20, 30], 2: [0, 10],
    })
    # Last thermal sample can coincide with the stop; take 2 still starts at 46 s.
    analyze_temperatures(recording, config_path, tmp_path / "result", progress=lambda _: None)
    data, _ = capture_reports["A"]
    assert data.loc[data.take_id == 1, "state"].tolist() == ["unknown", "awake", "awake", "sleep", "sleep", "unknown", "unknown"]
    assert data.elapsed_seconds.iloc[0] == 0
    assert data.take_seconds.iloc[0] == -.25
    assert data.loc[data.take_seconds == 10, "state"].tolist() == ["sleep", "awake"]


@pytest.mark.parametrize("kind", ["exclusion", "thermal_gap", "unknown_epoch"])
def test_temperature_baselines_never_bridge_missing_data(tmp_path, capture_reports, kind):
    offsets = {1: [s for s in range(30) if kind != "thermal_gap" or s not in (8, 9, 10)], 2: range(20)}
    recording, config_path, config = write_case(tmp_path, offsets)
    if kind == "exclusion":
        config["cages"][0]["exclude_intervals"] = [{"take_id": 1, "start_seconds": 9, "end_seconds": 11, "reason": "spout"}]
        config_path.write_text(json.dumps(config))
    elif kind == "unknown_epoch":
        path = tmp_path / "A" / "take01" / "epochs.csv"
        path.write_text(path.read_text().replace("0,0,10,awake,", "0,0,10,unknown,bad_tracking"))
    analyze_temperatures(recording, config_path, tmp_path / "result", progress=lambda _: None)
    data, summary = capture_reports["A"]
    assert summary["sleep_changes"][0]["sleep_minus_preceding_awake_c"] is None
    if kind == "exclusion":
        rejected = data.loc[~data.temperature_usable]
        assert rejected.take_seconds.tolist() == [9, 10]
        assert rejected.temperature_reason.tolist() == ["spout", "spout"]
        assert capture_reports["B"][1]["excluded_temperature_samples"] == 0


@pytest.mark.parametrize("error", ["thermal_overlap", "optical_overlap", "reused_run", "wrong_origin", "incomplete", "missing_take"])
def test_mixed_or_unfinished_inputs_are_rejected(tmp_path, error):
    recording, config_path, config = write_case(tmp_path)
    if error == "thermal_overlap":
        config["cages"][1]["thermal_roi"] = [3, 1, 12, 14]
    elif error == "reused_run":
        config["cages"][1]["analysis_dirs"][0] = config["cages"][0]["analysis_dirs"][0]
    elif error == "missing_take":
        config["cages"][1]["analysis_dirs"].pop()
    else:
        path = tmp_path / "B" / "take01" / "run.json"
        manifest = json.loads(path.read_text())
        if error == "optical_overlap":
            manifest["settings"]["roi"] = [0, 0, 500, 588]
        elif error == "wrong_origin":
            manifest["timing"]["origin_host_monotonic_ns"] += 1
        elif error == "incomplete":
            manifest["status"] = "cancelled"
        path.write_text(json.dumps(manifest))
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        cages, _, _ = load_config(config_path)
        load_runs(cages, recording)


def test_raw_csv_truncation_and_log_disagreement_preserve_previous_report(tmp_path, capture_reports):
    recording, config_path, _ = write_case(tmp_path)
    raw = recording / "seek_thermal.csv"
    original = raw.read_text()
    raw.write_text("\n".join(original.splitlines()[:-1]) + "\n")
    output = tmp_path / "result"
    output.mkdir()
    previous = output / "run_summary.json"
    previous.write_text('{"previous": true}')
    with pytest.raises(ValueError, match="Truncated"):
        analyze_temperatures(recording, config_path, output, progress=lambda _: None)
    assert previous.read_text() == '{"previous": true}'
    raw.write_text(original.replace(f"host_received_monotonic_ns={ORIGIN}", f"host_received_monotonic_ns={ORIGIN + 1}", 1))
    with pytest.raises(ValueError, match="does not match"):
        next(thermal_frames(recording))


def test_preview_works_before_analyses_exist_and_default_calibration(tmp_path):
    recording, config_path, config = write_case(tmp_path)
    config.pop("calibration")
    for cage in config["cages"]:
        cage["analysis_dirs"] = ["not_analyzed_yet"]
    config_path.write_text(json.dumps(config))
    output = tmp_path / "preview"
    result = analyze_temperatures(recording, config_path, output, preview_only=True, preview_frames=1, progress=lambda _: None)
    assert result["calibration"] == {"slope": .9674, "intercept_c": 2.2661}
    assert result["mode"] == "preview"
    assert (output / "thermal_preview.png").is_file()
    assert not (output / "A").exists()


def test_empty_sleep_state_still_produces_two_panel_figure(tmp_path, capture_reports):
    recording, config_path, _ = write_case(tmp_path)
    for path in (tmp_path / "A").rglob("epochs.csv"):
        path.write_text(path.read_text().replace("sleep", "awake"))
    analyze_temperatures(recording, config_path, tmp_path / "result", progress=lambda _: None)
    data, summary = capture_reports["A"]
    assert summary["comparisons"]["all"]["sleep"]["samples"] == 0
    assert summary["comparisons"]["all"]["sleep_minus_awake_c"] is None
    from mouse_sleep.temperature_reports import create_figure
    figure = create_figure(data, summary)
    assert len(figure.axes) == 2
    assert [a.get_title()[:1] for a in figure.axes] == ["A", "B"]
    figure.savefig(tmp_path / "no_sleep.png")
    figure.clear()


def test_unusable_cage_does_not_affect_neighbor_and_calibration_is_applied(tmp_path, capture_reports, monkeypatch):
    recording, config_path, config = write_case(tmp_path)
    config.pop("calibration")
    config_path.write_text(json.dumps(config))

    def bad_left_cage(root):
        for frame in thermal_frames(root):
            frame.pixels[1:15, 1:13] = np.nan
            yield frame

    monkeypatch.setattr("mouse_sleep.cage_temperatures.thermal_frames", bad_left_cage)
    analyze_temperatures(recording, config_path, tmp_path / "result", progress=lambda _: None)
    data, a = capture_reports["A"]
    b = capture_reports["B"][1]
    assert a["excluded_temperature_samples"] == 50
    assert a["comparisons"]["all"]["awake"]["mean_c"] is None
    assert a["comparisons"]["all"]["sleep_minus_awake_c"] is None
    assert b["excluded_temperature_samples"] == 0
    assert b["comparisons"]["all"]["sleep"]["mean_c"] == pytest.approx(36 * .9674 + 2.2661)
    from mouse_sleep.temperature_reports import create_figure
    figure = create_figure(data, a)
    figure.savefig(tmp_path / "unusable.png")
    assert len(figure.axes) == 2
    figure.clear()


@pytest.mark.parametrize("right_roi", [(544, 0, 500, 588), (600, 90, 444, 400)])
def test_two_optical_crops_preserve_body_proportions_and_exclude_neighbor(right_roi):
    source = np.zeros((588, 1044), np.uint8)
    cv2.circle(source, (250, 294), 40, 80, -1)
    cv2.circle(source, (794, 294), 40, 200, -1)
    for roi, intensity in [((0, 0, 500, 588), 80), (right_roi, 200)]:
        fitted, _, _ = network_image(source, roi)
        assert fitted.shape == (480, 480) and fitted.max() == intensity
        y, x = np.where(fitted > intensity / 2)
        assert abs((x.max() - x.min()) - (y.max() - y.min())) <= 1
