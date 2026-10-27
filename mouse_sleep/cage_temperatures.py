"""Join independent cage analyses to native thermal measurements on the host clock."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import csv
import json
import math
import re
import tempfile

import numpy as np
import pandas as pd

from .config import atomic_json
from .thermal import PATCH_SIZE, check_bounds, hottest_patch, rectangle, rectangles_overlap, thermal_frames
from .thermal_measurements import extract_measurements


@dataclass
class Cage:
    mouse_id: str
    thermal_roi: tuple
    analysis_dirs: list[Path]
    exclusions: list[dict]


@dataclass
class Run:
    path: Path
    manifest: dict
    take_id: int
    origin_ns: int
    epochs: list[dict]
    phases: list[dict]
    starts_ns: np.ndarray
    ends_ns: np.ndarray

    def label(self, host_ns):
        index = int(np.searchsorted(self.starts_ns, host_ns, side="right")) - 1
        if index < 0 or host_ns >= self.ends_ns[index]:
            return {"epoch": None, "phase_id": None, "state": "unknown", "state_reason": "outside_analyzed_epochs"}
        row = self.epochs[index]
        return {"epoch": row["epoch"], "phase_id": row["phase_id"], "state": row["state"],
                "state_reason": row["reason"]}


def load_config(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1:
        raise ValueError("Cage temperature configuration needs schema_version 1.")
    calibration = data.get("calibration", {"slope": 0.9674, "intercept_c": 2.2661})
    slope, intercept = float(calibration["slope"]), float(calibration["intercept_c"])
    gap = float(data.get("max_thermal_gap_seconds", 2.5))
    if not all(math.isfinite(v) for v in (slope, intercept, gap)) or min(slope, gap) <= 0:
        raise ValueError("Calibration must be finite, with positive slope and thermal gap threshold.")
    cages = []
    for item in data.get("cages", []):
        name = item["mouse_id"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name)
                or PureWindowsPath(name).is_reserved()):
            raise ValueError("Mouse IDs must be safe, unique folder names using letters, digits, '_' or '-'.")
        roi = rectangle(item["thermal_roi"], minimum=PATCH_SIZE)
        directories = [(path.parent / p).resolve() for p in item.get("analysis_dirs", [])]
        exclusions = item.get("exclude_intervals", [])
        for exclusion in exclusions:
            start, end = float(exclusion["start_seconds"]), float(exclusion["end_seconds"])
            if (type(exclusion["take_id"]) is not int or exclusion["take_id"] < 1
                    or not math.isfinite(start) or not math.isfinite(end) or end <= start
                    or not isinstance(exclusion.get("reason"), str) or not exclusion["reason"].strip()):
                raise ValueError("Exclusions need a positive take ID, finite start < end, and a reason.")
        for other in cages:
            if name.casefold() == other.mouse_id.casefold():
                raise ValueError("Mouse IDs must be unique (including letter case).")
            if rectangles_overlap(roi, other.thermal_roi):
                raise ValueError("Thermal cage regions overlap; select a separate region for each mouse.")
        cages.append(Cage(name, roi, directories, exclusions))
    if not cages:
        raise ValueError("Configure at least one cage.")
    return cages, {"slope": slope, "intercept_c": intercept}, gap


def _flir_bounds(path):
    bounds = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            take, stamp, frame = (int(row[k]) for k in ("take_id", "host_received_monotonic_ns", "frame_id"))
            if take not in bounds:
                bounds[take] = {"first": stamp, "last": stamp, "frame": frame, "count": 1}
            else:
                item = bounds[take]
                if stamp <= item["last"] or frame <= item["frame"]:
                    raise ValueError(f"Non-increasing FLIR timing in {path.name}, take {take}.")
                item.update(last=stamp, frame=frame, count=item["count"] + 1)
    return bounds


def _load_run(path, recording_dir, log_cache):
    manifest = json.loads((path / "run.json").read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Finish or resume the sleep analysis before temperature analysis: {path}")
    timing = manifest["timing"]
    if timing.get("source") != "recorder_host_frame_receipt" or timing.get("origin_host_monotonic_ns") is None:
        raise ValueError(f"{path}: shared recorder timestamps are required; video playback times cannot align Seek.")
    take, origin = int(timing["take_id"]), int(timing["origin_host_monotonic_ns"])
    log_path = recording_dir / Path(timing["csv"]).name
    if log_path not in log_cache:
        log_cache[log_path] = _flir_bounds(log_path)
    bounds = log_cache[log_path].get(take)
    if bounds is None or bounds["first"] != origin:
        raise ValueError(f"{path}: analysis origin/take does not match this recording's FLIR log.")
    info = manifest["source_video"]
    if info.get("frames") and info["frames"] != bounds["count"]:
        raise ValueError(f"{path}: FLIR frame count differs from the analyzed video.")
    if manifest["settings"].get("roi") is None:
        raise ValueError(f"{path}: no optical cage region was selected. Analyze this mouse with its own cage ROI first.")
    roi = rectangle(manifest["settings"]["roi"])
    check_bounds(roi, (info["height"], info["width"]))
    epochs, phases, starts, ends = [], [], [], []
    duration = float(manifest["duration_seconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{path}: invalid analysis duration.")
    with (path / "epochs.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            start, end = float(row["start"]), float(row["end"])
            state, reason = row["state"], row.get("reason", "")
            if (not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration + 1e-6
                    or state not in ("awake", "sleep", "unknown") or (state != "unknown" and reason)):
                raise ValueError(f"{path}: invalid epoch bounds or state.")
            left, right = origin + round(start * 1e9), origin + round(end * 1e9)
            if starts and (left < ends[-1] or int(row["epoch"]) <= epochs[-1]["epoch"]):
                raise ValueError(f"{path}: epochs must increase without overlapping.")
            if phases and starts and left == ends[-1] and phases[-1]["state"] == state:
                phases[-1]["end_seconds"] = end
            else:
                phases.append({"take_id": take, "phase_id": f"take{take}_phase{len(phases) + 1}",
                               "state": state, "start_seconds": start, "end_seconds": end})
            epochs.append({"epoch": int(row["epoch"]), "state": state, "reason": reason,
                           "phase_id": phases[-1]["phase_id"]})
            starts.append(left)
            ends.append(right)
    if not epochs or len(epochs) != manifest["completed_epochs"]:
        raise ValueError(f"{path}: epochs.csv does not match the completed analysis.")
    return Run(path, manifest, take, origin, epochs, phases,
               np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64))


def load_runs(cages, recording_dir):
    runs, log_cache, used_paths = {}, {}, set()
    for cage in cages:
        if not cage.analysis_dirs:
            raise ValueError(f"{cage.mouse_id}: add completed analysis_dirs, or use --preview-only first.")
        by_take = {}
        for path in cage.analysis_dirs:
            if path in used_paths:
                raise ValueError("Each cage/take needs a separate analysis folder; a folder was reused.")
            used_paths.add(path)
            run = _load_run(path, recording_dir, log_cache)
            if run.take_id in by_take:
                raise ValueError(f"{cage.mouse_id}: two analysis folders identify the same take.")
            by_take[run.take_id] = run
        if any(e["take_id"] not in by_take for e in cage.exclusions):
            raise ValueError(f"{cage.mouse_id}: exclusion refers to an unconfigured take.")
        runs[cage.mouse_id] = by_take
    reference = runs[cages[0].mouse_id]
    for cage in cages[1:]:
        if set(runs[cage.mouse_id]) != set(reference):
            raise ValueError("All cages must have analyses for the same takes.")
    for take in reference:
        selected = [runs[c.mouse_id][take] for c in cages]
        first = selected[0]
        for index, run in enumerate(selected):
            if (run.origin_ns != first.origin_ns or
                    Path(run.manifest["settings"]["video"]).resolve() != Path(first.manifest["settings"]["video"]).resolve()
                    or run.manifest["source_video"] != first.manifest["source_video"]):
                raise ValueError("Cage analyses for a take must use the same original video and timestamp origin.")
            roi = run.manifest["settings"]["roi"]
            if any(rectangles_overlap(roi, other.manifest["settings"]["roi"]) for other in selected[:index]):
                raise ValueError("Optical cage regions overlap; analyze each mouse in a separate region.")
    return runs


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {"samples": int(values.size), "mean_c": float(values.mean()) if values.size else None,
            "sd_c": float(values.std(ddof=1)) if values.size > 1 else None}


def difference(a, b):
    return a - b if a is not None and b is not None else None


def summarize(data, runs, origin_ns):
    usable = data.loc[data.temperature_usable]
    # Build groups once; repeated full-table scans per bout become expensive on
    # day-long recordings with thousands of state transitions.
    by_phase = {key: group for key, group in usable.groupby("phase_id", sort=False)}
    empty = usable.iloc[:0]
    phases = []
    for run in sorted(runs.values(), key=lambda r: r.origin_ns):
        for phase in run.phases:
            offset = (run.origin_ns - origin_ns) / 1e9
            phases.append({**phase, "elapsed_start_seconds": offset + phase["start_seconds"],
                           "elapsed_end_seconds": offset + phase["end_seconds"],
                           **statistics(by_phase.get(phase["phase_id"], empty).calibrated_temperature_c)})
    comparisons = {}
    for key, take in [(f"take{t}", t) for t in sorted(runs)] + [("all", None)]:
        selected = usable if take is None else usable.loc[usable.take_id == take]
        by_state = {state: statistics(selected.loc[selected.state == state, "calibrated_temperature_c"])
                    for state in ("awake", "sleep")}
        comparisons[key] = {**by_state, "sleep_minus_awake_c": difference(by_state["sleep"]["mean_c"], by_state["awake"]["mean_c"])}
    changes = []
    for index, phase in enumerate(phases):
        if phase["state"] != "sleep":
            continue
        selected = by_phase.get(phase["phase_id"], empty)
        first = selected.loc[selected.take_seconds < phase["start_seconds"] + 60]
        last = selected.loc[selected.take_seconds >= phase["end_seconds"] - 60]
        first_stats, last_stats = (statistics(v.calibrated_temperature_c) for v in (first, last))
        continuous = not selected.empty and selected.measurement_segment.nunique() == 1
        change = {"take_id": phase["take_id"], "phase_id": phase["phase_id"],
                  "start_seconds": phase["start_seconds"], "end_seconds": phase["end_seconds"],
                  "sleep_mean_c": phase["mean_c"], "sleep_samples": phase["samples"],
                  "first_sleep_minute_mean_c": first_stats["mean_c"], "first_sleep_minute_samples": first_stats["samples"],
                  "last_sleep_minute_mean_c": last_stats["mean_c"], "last_sleep_minute_samples": last_stats["samples"],
                  "last_minus_first_sleep_minute_c": difference(last_stats["mean_c"], first_stats["mean_c"]) if continuous else None,
                  "preceding_awake_mean_c": None, "last_awake_minute_mean_c": None,
                  "last_awake_minute_samples": 0, "sleep_minus_preceding_awake_c": None,
                  "sleep_minus_last_awake_minute_c": None, "first_sleep_minus_last_awake_minute_c": None,
                  "baseline_reason": "no_contiguous_awake_baseline"}
        previous = phases[index - 1] if index else None
        if (previous and previous["take_id"] == phase["take_id"] and previous["state"] == "awake"
                and abs(previous["end_seconds"] - phase["start_seconds"]) < 1e-8):
            awake = by_phase.get(previous["phase_id"], empty)
            before = awake.loc[awake.take_seconds >= phase["start_seconds"] - 60]
            # Neither missing thermal data nor manually rejected samples may bridge a baseline.
            if (continuous and not before.empty and awake.measurement_segment.nunique() == 1
                    and int(awake.measurement_segment.iloc[-1]) == int(selected.measurement_segment.iloc[0])):
                baseline = statistics(before.calibrated_temperature_c)
                change.update(preceding_awake_mean_c=previous["mean_c"], last_awake_minute_mean_c=baseline["mean_c"],
                              last_awake_minute_samples=baseline["samples"], baseline_reason="",
                              sleep_minus_preceding_awake_c=difference(phase["mean_c"], previous["mean_c"]),
                              sleep_minus_last_awake_minute_c=difference(phase["mean_c"], baseline["mean_c"]),
                              first_sleep_minus_last_awake_minute_c=difference(first_stats["mean_c"], baseline["mean_c"]))
            else:
                change["baseline_reason"] = "missing_or_excluded_temperature"
        changes.append(change)
    return {"samples": len(data), "usable_temperature_samples": len(usable),
            "excluded_temperature_samples": int((~data.temperature_usable).sum()),
            "unknown_state_samples": int((data.state == "unknown").sum()),
            "comparisons": comparisons, "phases": phases, "sleep_changes": changes}


def _sample(frame, cage, run, origin_ns, calibration, previous, segment, max_gap, *, patch=None):
    patch = hottest_patch(frame.pixels, cage.thermal_roi) if patch is None else dict(patch)
    seconds = (frame.host_ns - run.origin_ns) / 1e9
    rejected = [e["reason"] for e in cage.exclusions if e["take_id"] == frame.take_id
                and float(e["start_seconds"]) <= seconds < float(e["end_seconds"])]
    reason = "; ".join(filter(None, [patch.pop("temperature_reason"), *rejected]))
    raw = patch["camera_temperature_7x7_c"]
    temperature = raw * calibration["slope"] + calibration["intercept_c"] if raw is not None else None
    if temperature is not None and not math.isfinite(temperature):
        reason = "; ".join(filter(None, [reason, "nonfinite_calibrated_temperature"]))
        temperature = None
    good = not reason
    if (previous is None or frame.take_id != previous["take_id"] or not good or not previous["temperature_usable"]
            or frame.host_ns - previous["host_received_monotonic_ns"] > round(max_gap * 1e9)):
        segment += 1
    return {"mouse_id": cage.mouse_id, "frame_index": frame.index, "take_id": frame.take_id,
            "host_received_monotonic_ns": frame.host_ns, "timestamp_utc_ns": frame.utc_ns,
            "elapsed_seconds": (frame.host_ns - origin_ns) / 1e9, "take_seconds": seconds,
            **run.label(frame.host_ns), **patch, "patch_size": PATCH_SIZE,
            "calibrated_temperature_c": temperature, "temperature_usable": good,
            "temperature_reason": reason, "measurement_segment": segment}


def _geometry(runs):
    result = []
    for take, run in sorted(runs.items()):
        x, y, width, height = run.manifest["settings"]["roi"]
        scale = 480 / max(width, height)
        result.append({"take_id": take, "optical_roi": [x, y, width, height],
                       "source_width": run.manifest["source_video"]["width"],
                       "source_height": run.manifest["source_video"]["height"],
                       "network_active_width": round(width * scale), "network_active_height": round(height * scale),
                       "network_scale": scale, "upsampling": scale > 1,
                       "aspect_ratio_preserved": True, "analysis_directory": str(run.path)})
    return result


def _recording_gaps(recording_dir, observed, runs, origin_ns):
    gaps = []
    ordered = sorted(observed, key=lambda t: observed[t][0])
    for previous, current in zip(ordered, ordered[1:]):
        end = max(observed[previous][1], int(runs[previous].ends_ns[-1]))
        start = min(observed[current][0], runs[current].origin_ns)
        method = "between_observed_take_bounds"
        files = [recording_dir / f"take{t:02d}_sync.json" for t in (previous, current)]
        if all(p.is_file() for p in files):
            before, after = [json.loads(p.read_text(encoding="utf-8-sig")) for p in files]
            stop = before.get("shared_stop_monotonic_ns")
            restart = after.get("shared_start_monotonic_ns")
            if stop is not None and restart is not None:
                stop, restart = int(stop), int(restart)
                if not observed[previous][1] <= stop < restart <= observed[current][0]:
                    raise ValueError("Take stop/start metadata conflicts with recorded thermal timestamps.")
                end, start, method = stop, restart, "recorder_shared_stop_to_start"
        if start > end:
            gaps.append({"after_take": previous, "before_take": current,
                         "start_seconds": (end - origin_ns) / 1e9, "end_seconds": (start - origin_ns) / 1e9,
                         "duration_seconds": (start - end) / 1e9, "source": method})
    return gaps


def analyze_temperatures(recording_dir, config_path, output, *, preview_only=False, preview_frames=12,
                         use_cache=True, refresh_cache=False, progress=print):
    """One raw-CSV pass for all cages; publish reports only after successful validation."""
    from .temperature_reports import write_preview, write_cage_report

    recording_dir, output = Path(recording_dir).resolve(), Path(output).resolve()
    cages, calibration, max_gap = load_config(config_path)
    if not 1 <= preview_frames <= 30:
        raise ValueError("Preview frames must be between 1 and 30.")
    runs = {} if preview_only else load_runs(cages, recording_dir)
    for cage in cages:
        for geometry in _geometry(runs.get(cage.mouse_id, {})):
            width, height = geometry["optical_roi"][2:]
            note = " (upsampling: inspect native detail in the pilot)" if geometry["upsampling"] else ""
            progress(f"{cage.mouse_id}, take {geometry['take_id']}: {width} x {height} native optical crop -> "
                     f"{geometry['network_active_width']} x {geometry['network_active_height']} active network pixels; "
                     f"proportions preserved{note}")
    output.mkdir(parents=True, exist_ok=True)
    previews, rows = [], {c.mouse_id: [] for c in cages}
    measurements = extract_measurements(
        recording_dir, [c.thermal_roi for c in cages], preview_only=preview_only,
        preview_frames=preview_frames, use_cache=use_cache, refresh_cache=refresh_cache,
        progress=progress, frame_source=thermal_frames,
    )
    origin_ns = None
    observed = {}
    dimensions = list(reversed(measurements.shape))
    for number, frame, patches in measurements.samples():
        if origin_ns is None:
            origin_ns = min([frame.host_ns] + [r.origin_ns for rs in runs.values() for r in rs.values()])
        observed.setdefault(frame.take_id, [frame.host_ns, frame.host_ns])[1] = frame.host_ns
        samples = []
        for cage, patch in zip(cages, patches, strict=True):
            if preview_only:
                samples.append({"mouse_id": cage.mouse_id, **patch})
                continue
            if frame.take_id not in runs[cage.mouse_id]:
                raise ValueError(f"{cage.mouse_id}: no completed analysis configured for recorded take {frame.take_id}.")
            previous = rows[cage.mouse_id][-1] if rows[cage.mouse_id] else None
            sample = _sample(frame, cage, runs[cage.mouse_id][frame.take_id], origin_ns, calibration,
                             previous, previous["measurement_segment"] if previous else 0, max_gap, patch=patch)
            rows[cage.mouse_id].append(sample)
            samples.append(sample)
        if number in measurements.previews:
            previews.append((frame, samples))
    del measurements
    reference = runs.get(cages[0].mouse_id, {})
    if not preview_only and set(reference) != set(observed):
        raise ValueError("Configured analyses and recorded thermal takes do not match.")
    gaps = [] if preview_only else _recording_gaps(recording_dir, observed, reference, origin_ns)
    common = {"schema_version": 1, "mode": "preview" if preview_only else "analysis",
              "recording_directory": str(recording_dir), "config_file": str(Path(config_path).resolve()),
              "thermal_dimensions": dimensions, "origin_host_monotonic_ns": origin_ns,
              "time_zero": "earliest first FLIR frame or thermal sample in this session",
              "timing": "shared host receipt clock; not hardware exposure synchronization",
              "calibration": calibration, "patch_size": PATCH_SIZE, "max_thermal_gap_seconds": max_gap,
              "gaps": gaps, "mouse_ids": [c.mouse_id for c in cages]}
    # Failed reads never replace an earlier successful report. Only compact rows
    # and a bounded number of preview frames are retained, not the thermal movie.
    with tempfile.TemporaryDirectory(prefix=".temperature-", dir=output) as staging:
        staging = Path(staging)
        write_preview(previews, cages, staging / "thermal_preview.png")
        if preview_only:
            atomic_json(staging / "thermal_preview.json", {**common, "previewed_frames": len(previews),
                        "note": "First frames only; full recording and sleep alignment have not been checked.",
                        "cages": [{"mouse_id": c.mouse_id, "thermal_roi": list(c.thermal_roi)} for c in cages]})
        else:
            atomic_json(staging / "run_summary.json", common)
            for cage in cages:
                data = pd.DataFrame(rows.pop(cage.mouse_id))
                summary = {**common, "mouse_id": cage.mouse_id, "thermal_roi": list(cage.thermal_roi),
                           "exclude_intervals": cage.exclusions, "geometry": _geometry(runs[cage.mouse_id]),
                           "label_source": "independent video-based sleep predictions",
                           "temperature_method": "hottest complete native 7x7 mean within cage; patch identity requires visual review",
                           **summarize(data, runs[cage.mouse_id], origin_ns)}
                write_cage_report(data, summary, staging / cage.mouse_id)
        for source in staging.rglob("*"):
            if source.is_file():
                target = output / source.relative_to(staging)
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
    progress(f"{'Preview' if preview_only else 'Temperature analysis'} saved to {output}")
    return common
