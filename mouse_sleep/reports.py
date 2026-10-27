from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from .classifier import viterbi
from .config import atomic_json


def scored_rows(records, classifier, minimum_confidence):
    rows = []
    for record in records:
        row = dict(record)
        row["state"] = "unknown"
        row["p_awake"] = row["p_sleep"] = row["p_nrem"] = row["p_rem"] = None
        if row["probabilities"]:
            p = np.asarray(json.loads(row["probabilities"]))
            row.update(p_awake=float(p[0]), p_sleep=float(p[1] + p[2]), p_nrem=float(p[1]), p_rem=float(p[2]))
            if max(row["p_awake"], row["p_sleep"]) < minimum_confidence:
                row["reason"] = "low_confidence"
        rows.append(row)
    # Split before smoothing at unknown data and every timing discontinuity.
    pending = []
    start = None
    for i in range(len(rows) + 1):
        valid = i < len(rows) and rows[i]["probabilities"] and not rows[i]["reason"]
        discontinuity = i > 0 and i < len(rows) and abs(rows[i]["start"] - rows[i - 1]["end"]) > 1e-5
        if start is not None and (not valid or discontinuity):
            pending.append((start, i))
            start = None
        if valid and start is None:
            start = i
    while pending:
        start, end = pending.pop()
        p = np.asarray([json.loads(rows[i]["probabilities"]) for i in range(start, end)])
        predictions = viterbi(p, classifier.transition, classifier.initial)
        conflicts = []
        for offset, stage in enumerate(predictions):
            row = rows[start + offset]
            confidence = row["p_awake"] if stage == 0 else row["p_sleep"]
            if confidence < minimum_confidence:
                row["state"], row["reason"] = "unknown", "low_confidence_after_smoothing"
                conflicts.append(start + offset)
            else:
                row["state"] = "awake" if stage == 0 else "sleep"
        # Re-decode each remaining part so a newly unknown epoch cannot carry state across a gap.
        if conflicts:
            boundaries = [start - 1, *conflicts, end]
            pending.extend((left + 1, right) for left, right in zip(boundaries[:-1], boundaries[1:]) if right > left + 1)
    return rows


def summaries(rows):
    bouts = []
    totals = {"sleep": 0.0, "awake": 0.0, "unknown": 0.0}
    hours = {}
    for row in rows:
        start, end, state = row["start"], row["end"], row["state"]
        totals[state] += end - start
        if state == "sleep":
            if bouts and abs(bouts[-1]["end_seconds"] - start) < 1e-5:
                bouts[-1]["end_seconds"] = end
                bouts[-1]["duration_seconds"] = end - bouts[-1]["start_seconds"]
            else:
                bouts.append({"start_seconds": start, "end_seconds": end, "duration_seconds": end - start})
        cursor = start
        while cursor < end - 1e-7:
            hour = int(cursor // 3600)
            stop = min(end, (hour + 1) * 3600)
            hours.setdefault(hour, {"hour": hour, "sleep_seconds": 0.0, "awake_seconds": 0.0, "unknown_seconds": 0.0})[state + "_seconds"] += stop - cursor
            cursor = stop
    scored = totals["sleep"] + totals["awake"]
    totals["sleep_fraction_of_scorable_time"] = totals["sleep"] / scored if scored else None
    totals["analyzed_seconds"] = sum(totals[state] for state in ("sleep", "awake", "unknown"))
    return bouts, list(hours.values()), totals


def write_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export_reports(output, connection, classifier, settings, total_duration):
    """Compatibility entry point for callers with an existing SQLite connection."""
    records = connection.execute("SELECT epoch, start, end, coverage, reason, probabilities FROM epochs ORDER BY epoch").fetchall()
    return export_records(output, records, classifier, settings, total_duration)


def export_records(output, records, classifier, settings, total_duration):
    """Export ordered committed epoch records without depending on SQLite."""
    output = Path(output)
    rows = scored_rows(records, classifier, settings.minimum_confidence)
    bouts, hourly, totals = summaries(rows)
    totals.update(expected_video_seconds=total_duration, unprocessed_seconds=max(0, total_duration - totals["analyzed_seconds"]),
                  prediction_type="estimated video-based sleep", appearance=settings.appearance,
                  validation_status="FLIR setup unvalidated" if settings.appearance == "dark" else "Experimental transfer to white/light mice; FLIR setup unvalidated")
    write_csv(output / "epochs.csv", ["epoch", "start", "end", "state", "p_sleep", "p_awake", "p_nrem", "p_rem", "coverage", "reason"], rows)
    write_csv(output / "sleep_bouts.csv", ["start_seconds", "end_seconds", "duration_seconds"], bouts)
    write_csv(output / "hourly_sleep.csv", ["hour", "sleep_seconds", "awake_seconds", "unknown_seconds"], hourly)
    atomic_json(output / "summary.json", totals)
    plot_timeline(output / "timeline.png", rows, total_duration, totals["validation_status"])
    return totals


def plot_timeline(path, rows, total_duration, validation_status):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    figure = Figure(figsize=(12, 3), constrained_layout=True)
    axis = figure.subplots()
    colors = {"sleep": "#259685", "awake": "#d99531", "unknown": "#b4bbc7"}
    # Merge display runs so a 24-hour plot stays small.
    runs = []
    for row in rows:
        if runs and runs[-1][2] == row["state"] and abs(runs[-1][1] - row["start"]) < 1e-5:
            runs[-1] = (runs[-1][0], row["end"], row["state"])
        else:
            runs.append((row["start"], row["end"], row["state"]))
    for state, color in colors.items():
        axis.broken_barh([(start / 3600, (end - start) / 3600) for start, end, value in runs if value == state], (0, 1), facecolors=color, label=state)
    axis.set(xlabel="Hours since first recorded FLIR frame", yticks=[], ylim=(0, 1), xlim=(0, max(total_duration / 3600, 0.001)), title="Estimated sleep — " + validation_status)
    axis.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.0))
    figure.savefig(path, dpi=130)
