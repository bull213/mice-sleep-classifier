"""Per-mouse temperature tables and two-panel figures (timeline and comparison)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from .config import atomic_json
from .thermal import PATCH_SIZE


COLORS = {"awake": "#C57B21", "sleep": "#4965B2", "unknown": "#89939F"}
INK = "#243449"


def write_preview(previews, cages, path):
    """Bounded contact sheet; native pixel axes make cage/patch placement reviewable."""
    columns = min(3, len(previews))
    rows = (len(previews) + columns - 1) // columns
    figure = Figure(figsize=(5 * columns, 3.9 * rows), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    limits = [np.percentile(frame.pixels[np.isfinite(frame.pixels)], [1, 99.9])
              for frame, _ in previews if np.isfinite(frame.pixels).any()]
    low = min(v[0] for v in limits) if limits else 0
    high = max(v[1] for v in limits) if limits else 1
    if high <= low:
        high = low + 1
    cage_colors = ["#00FFFF", "#FF88FF", "#A6FF00", "#FFFFFF"]
    for axis, (frame, samples) in zip(axes, previews):
        axis.imshow(frame.pixels, cmap="inferno", interpolation="nearest", vmin=low, vmax=high)
        for index, (cage, sample) in enumerate(zip(cages, samples)):
            color = cage_colors[index % len(cage_colors)]
            x, y, width, height = cage.thermal_roi
            axis.add_patch(Rectangle((x - .5, y - .5), width, height, fill=False, edgecolor=color, linewidth=1.5))
            axis.text(x + 3, y + 3, cage.mouse_id, color=color, fontsize=8, va="top",
                      bbox={"facecolor": "black", "alpha": .65, "edgecolor": "none", "pad": 2})
            if sample["patch_x"] is not None:
                px, py = sample["patch_x"], sample["patch_y"]
                axis.add_patch(Rectangle((px - .5, py - .5), PATCH_SIZE, PATCH_SIZE,
                                         fill=False, edgecolor=color, linewidth=1.3))
        axis.set(title=f"Take {frame.take_id} | thermal frame {frame.index}",
                 xlabel="Native x pixel", ylabel="Native y pixel")
    for axis in axes[len(previews):]:
        axis.set_visible(False)
    figure.suptitle("Thermal cage regions and candidate 7 x 7 patches\n"
                    "Inspect that each patch lies on its mouse; color is display only.", fontsize=13)
    figure.savefig(path, dpi=140)
    figure.clear()


def _number(value, *, signed=False):
    return "—" if value is None else format(value, "+.3f" if signed else ".3f")


def create_figure(data, summary):
    figure = Figure(figsize=(12.8, 8), layout="constrained")
    timeline, comparison = figure.subplots(2, 1, gridspec_kw={"height_ratios": [1.45, 1]})
    figure.suptitle(f"{summary['mouse_id']} — temperature during predicted awake and sleep\n"
                    "Calibrated native 7 × 7 patch mean • Independent cage analysis", fontsize=15, color=INK)
    end = max(float(data.elapsed_seconds.max()),
              max((p["elapsed_end_seconds"] for p in summary["phases"]), default=0)) / 60
    timeline.set(xlim=(0, max(end, 1 / 60)), xlabel="Elapsed minutes on the shared recording clock",
                 ylabel="Calibrated temperature (°C)", title="A   Temperature timeline")
    # Unknown/unscored space is gray; known intervals overlay it. Runs and pauses
    # remain distinct even when adjacent samples have identical temperatures.
    timeline.axvspan(0, max(end, 1 / 60), color=COLORS["unknown"], alpha=.08, linewidth=0)
    for state in ("awake", "sleep"):
        spans = [(p["elapsed_start_seconds"] / 60, (p["elapsed_end_seconds"] - p["elapsed_start_seconds"]) / 60)
                 for p in summary["phases"] if p["state"] == state]
        timeline.broken_barh(spans, (0, 1), transform=timeline.get_xaxis_transform(),
                             facecolors=COLORS[state], alpha=.12, linewidth=0)
    for gap in summary["gaps"]:
        timeline.axvspan(gap["start_seconds"] / 60, gap["end_seconds"] / 60,
                         facecolor="white", edgecolor="#89939F", hatch="////", linewidth=.6, zorder=3)
    good = data.temperature_usable.to_numpy(dtype=bool)
    x = data.elapsed_seconds.to_numpy(dtype=float) / 60
    y = data.calibrated_temperature_c.to_numpy(dtype=float)
    states = data.state.to_numpy()
    colors = np.asarray([COLORS[s] for s in states])
    phases = data.phase_id.fillna("").to_numpy()
    segment = data.measurement_segment.to_numpy()
    adjacent = good[:-1] & good[1:] & (segment[:-1] == segment[1:]) & (phases[:-1] == phases[1:])
    points = np.column_stack((x, y))
    lines = np.stack((points[:-1], points[1:]), axis=1)[adjacent]
    timeline.add_collection(LineCollection(lines, colors=colors[:-1][adjacent], linewidths=.85, rasterized=True))
    timeline.scatter(x[good], y[good], c=colors[good], s=3, linewidths=0, rasterized=True, zorder=4)
    timeline.autoscale_view(scalex=False, scaley=True)
    if not good.any():
        timeline.set_yticks([])
        timeline.text(.5, .5, "No usable temperature samples", transform=timeline.transAxes, ha="center")
    handles = [Patch(facecolor=color, label=state.capitalize()) for state, color in COLORS.items()]
    if summary["gaps"]:
        handles.append(Patch(facecolor="white", edgecolor="#89939F", hatch="////", label="No recording"))
    timeline.legend(handles=handles, loc="best", ncols=len(handles), fontsize=9)

    groups = list(summary["comparisons"].items())
    # Many takes remain in the tables; keep the figure legible.
    if len(groups) > 7:
        groups = [("all", summary["comparisons"]["all"])]
    bounds = []
    for index, (_, stats) in enumerate(groups):
        for state, offset in (("awake", -.14), ("sleep", .14)):
            item = stats[state]
            if item["mean_c"] is None:
                continue
            mean, sd = item["mean_c"], item["sd_c"]
            bounds.extend([mean - (sd or 0), mean + (sd or 0)])
            comparison.errorbar(index + offset, mean, yerr=sd, fmt="o", color=COLORS[state],
                                markersize=6, capsize=4, linewidth=1.5)
            comparison.annotate(f"n={item['samples']:,}", (index + offset, mean - (sd or 0)),
                                xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8)
        delta = _number(stats["sleep_minus_awake_c"], signed=True)
        comparison.text(index, .97, f"Sleep − awake\n{delta} °C", transform=comparison.get_xaxis_transform(),
                        ha="center", va="top", fontsize=10)
    if bounds:
        span = max(max(bounds) - min(bounds), .5)
        comparison.set_ylim(min(bounds) - .35 * span, max(bounds) + .65 * span)
    else:
        comparison.text(.5, .4, "No usable temperatures with scored sleep/awake labels",
                        transform=comparison.transAxes, ha="center")
    comparison.set(xticks=range(len(groups)), xticklabels=["All takes" if k == "all" else k.replace("take", "Take ") for k, _ in groups],
                   xlim=(-.6, len(groups) - .4), ylabel="Calibrated temperature (°C)",
                   title="B   Awake versus sleep temperature",
                   xlabel="Means ± sample SD; n = thermal samples, not independent animals.\n"
                          "Unknown states and unusable temperatures are excluded from these comparisons.")
    for axis in (timeline, comparison):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    return figure


def write_cage_report(data, summary, output):
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "temperature_samples.csv", index=False)
    pd.DataFrame(summary["phases"]).to_csv(output / "state_intervals.csv", index=False)
    changes = pd.DataFrame(summary["sleep_changes"])
    if changes.empty:
        changes = pd.DataFrame(columns=["take_id", "phase_id", "start_seconds", "end_seconds", "sleep_mean_c", "baseline_reason"])
    changes.to_csv(output / "sleep_temperature_changes.csv", index=False)
    atomic_json(output / "summary.json", summary)
    figure = create_figure(data, summary)
    figure.savefig(output / "temperature_awake_vs_sleep.png", dpi=180)
    figure.savefig(output / "temperature_awake_vs_sleep.pdf")
    figure.clear()
    lines = [f"# {summary['mouse_id']}: temperature and predicted sleep", "",
             f"Processed {summary['samples']:,} recorded thermal samples. "
             f"{summary['excluded_temperature_samples']:,} have unusable/excluded temperatures; "
             f"{summary['unknown_state_samples']:,} have unknown sleep state (these counts can overlap).", "",
             "| Scope | Awake mean ± SD (°C) | Awake n | Sleep mean ± SD (°C) | Sleep n | Sleep − awake (°C) |",
             "| --- | --- | ---: | --- | ---: | ---: |"]
    for name, stats in summary["comparisons"].items():
        awake, sleep = stats["awake"], stats["sleep"]
        lines.append(f"| {name} | {_number(awake['mean_c'])} ± {_number(awake['sd_c'])} | {awake['samples']} | "
                     f"{_number(sleep['mean_c'])} ± {_number(sleep['sd_c'])} | {sleep['samples']} | "
                     f"{_number(stats['sleep_minus_awake_c'], signed=True)} |")
    lines += ["", "## Temperature change within sleep intervals", "",
              "| Take | Sleep start–end (s from first FLIR frame) | Sleep − preceding awake (°C) | Sleep − last awake minute (°C) | First sleep − last awake minute (°C) | Last − first sleep minute (°C) | Baseline status |",
              "| ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in summary["sleep_changes"]:
        values = [_number(row[k], signed=True) for k in ("sleep_minus_preceding_awake_c", "sleep_minus_last_awake_minute_c",
                                                         "first_sleep_minus_last_awake_minute_c", "last_minus_first_sleep_minute_c")]
        lines.append(f"| {row['take_id']} | {row['start_seconds']:.3f}–{row['end_seconds']:.3f} | "
                     + " | ".join(values) + f" | {row['baseline_reason'] or 'available'} |")
    lines += ["", "## Timing and measurement", "",
              "- State labels come from this mouse's completed analysis. NREM and REM are already combined as sleep by the analyzer.",
              "- Alignment uses integer host receipt timestamps and half-open epoch intervals [start, end). Camera UTC is retained for reference only.",
              "- All raw thermal samples are saved in temperature_samples.csv, including unknown states and exclusions. No interpolation or temperature smoothing is used for statistics.",
              "- Means weight usable original samples equally. SD uses ddof=1 and is absent with fewer than two samples. Absent states/baselines produce empty values, not zero.",
              "- Sleep baselines require immediately preceding awake time in the same take, without unknown intervals, thermal gaps, or rejected temperature samples between compared periods.",
              "- First/last-minute windows are clipped to the sleep interval and can overlap for short bouts; their sample counts are in sleep_temperature_changes.csv.",
              "- The timeline preserves pauses and breaks lines at state boundaries, thermal gaps, and excluded temperatures. Unknown-state temperatures may appear in gray but do not enter state comparisons.",
              f"- Thermal gaps exceed {summary['max_thermal_gap_seconds']:g} s between samples. Choose this threshold for the actual thermal acquisition cadence.",
              f"- Calibration: temperature = {summary['calibration']['slope']:g} × camera 7×7 mean + {summary['calibration']['intercept_c']:g} °C.",
              "- Each 7×7 patch is entirely inside the native cage region and has finite pixels. The hottest patch is a candidate measurement, not automatic identification of mouse tissue; inspect thermal_preview.png and exclude wrong-object samples.",
              "- A patch touching a cage-region edge is flagged for review. Camera geometry and crop scale are recorded in summary.json.",
              "- These are surface-temperature measurements and estimated video-based sleep. Repeated thermal samples are not independent animals; good segmentation does not establish sleep accuracy.", ""]
    if summary["gaps"]:
        lines += ["## Recording pauses", ""]
        lines += [f"- Take {g['after_take']} → {g['before_take']}: {g['duration_seconds']:.6f} s ({g['source']})."
                  for g in summary["gaps"]]
    (output / "temperature_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
