"""Time a real analysis pilot and optionally compare it with a saved baseline."""
from dataclasses import replace
from pathlib import Path
import argparse
import ctypes
import json
import os
import platform
import sqlite3
import sys
import time

from mouse_sleep.config import Settings, atomic_json


def peak_memory_bytes():
    """OS process peak, including native TensorFlow/OpenCV allocations."""
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in
                          ("peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                           "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile")]]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        if not query(process(), ctypes.byref(counters), counters.cb):
            return None
        return counters.peak_working_set
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024
    except ImportError:
        return None


def compare_results(baseline, output):
    names = ("epochs.csv", "sleep_bouts.csv", "hourly_sleep.csv", "summary.json", "timeline.png")
    matches = {name: (baseline / name).read_bytes() == (output / name).read_bytes() for name in names}
    records = []
    for folder in (baseline, output):
        with sqlite3.connect((folder / "analysis.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
            records.append(db.execute("SELECT * FROM epochs ORDER BY epoch").fetchall())
    matches["all_epoch_features_and_probabilities"] = records[0] == records[1]
    return matches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New pilot output folder")
    parser.add_argument("--seconds", type=float, default=60.5)
    parser.add_argument("--compare", type=Path, help="Saved baseline with the same source and analysis settings")
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    settings = replace(Settings.load(args.config), output=str(args.output), max_seconds=args.seconds)
    began = time.perf_counter()
    from mouse_sleep.pipeline import analyze
    manifest = analyze(settings, progress=lambda event: print(event.get("message", ""), flush=True))
    elapsed = time.perf_counter() - began
    report = {"python": platform.python_version(), "logical_cpus": os.cpu_count(),
              "video_seconds": manifest["duration_seconds"], "model": settings.model_name,
              "batch_size": settings.batch_size, "wall_seconds": elapsed,
              "processing_fps": manifest["duration_seconds"] * 30 / elapsed,
              "peak_process_memory_bytes": peak_memory_bytes(),
              "includes": "pipeline imports, input validation, model loading, analysis and reports"}
    if args.compare:
        reference = Settings.load(args.compare / "run.json")
        if reference.identity() != settings.identity():
            raise ValueError("Baseline and pilot must use identical source metadata and settings.")
        report["exact_matches"] = compare_results(args.compare, args.output)
    atomic_json(args.output / "performance.json", report)
    print(json.dumps(report, indent=2))
    return 0 if all(report.get("exact_matches", {}).values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
