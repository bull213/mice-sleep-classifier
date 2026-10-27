"""Exercise the actual checkpoints and trained classifier on a synthetic MP4.

The generated shape is a software fixture, not biological validation data.
"""
from pathlib import Path
import json
import time
from mouse_sleep.config import ROOT, Settings, atomic_json
from mouse_sleep.pipeline import analyze
from mouse_sleep.video import inspect_video
from tests.helpers import make_video

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "verification")
    args = parser.parse_args(argv)
    directory = args.output
    directory.mkdir(exist_ok=True)
    source = directory / "synthetic_pipeline.mp4"
    if not source.exists():
        make_video(source, count=615)
    results = []
    for appearance in ("dark", "white"):
        destination = directory / ("pipeline_" + appearance)
        settings = Settings(video=str(source), output=str(destination), appearance=appearance, masking_mode="legacy", batch_size=4, roi=(8, 0, 64, 64))
        result = analyze(settings, resume=(destination / "run.json").exists(), progress=lambda event: print(event.get("message", ""), flush=True))
        assert result["status"] == "complete"
        assert result["completed_epochs"] == 3
        assert result["summary"]["analyzed_seconds"] == 20.5
        for filename in ("epochs.csv", "sleep_bouts.csv", "hourly_sleep.csv", "timeline.png", "summary.json", "run.json", "settings.json", "analysis.sqlite3"):
            assert (destination / filename).is_file()
        results.append({"appearance": appearance, "model": settings.model_name, "status": result["status"], "summary": result["summary"], "elapsed_seconds": result["elapsed_seconds_this_run"]})
    atomic_json(directory / "end_to_end.json", {"synthetic_fixture_only": True, "checks": results})
    print("Actual-model MP4-to-report smoke tests passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
