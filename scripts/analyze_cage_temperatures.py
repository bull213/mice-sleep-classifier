"""Postprocess separate cage analyses without rerunning segmentation."""
from pathlib import Path
import argparse
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mouse_sleep.cage_temperatures import analyze_temperatures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", required=True, type=Path, help="Original recording with raw Seek data and timing logs")
    parser.add_argument("--config", required=True, type=Path, help="Per-mouse thermal ROIs and completed sleep-analysis directories")
    parser.add_argument("--output", required=True, type=Path, help="Destination for separate mouse reports and thermal preview")
    parser.add_argument("--preview-only", action="store_true", help="Inspect the first thermal frames without requiring sleep analyses")
    parser.add_argument("--preview-frames", default=12, type=int, help="1–30 contact-sheet frames; spread over the recording in full analysis (default: 12)")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--no-cache", action="store_true", help="Read original thermal data without reading or writing a cache")
    cache.add_argument("--refresh-cache", action="store_true", help="Rebuild cached thermal measurements from the original recording")
    args = parser.parse_args(argv)
    try:
        analyze_temperatures(args.recording_dir, args.config, args.output,
                             preview_only=args.preview_only, preview_frames=args.preview_frames,
                             use_cache=not args.no_cache, refresh_cache=args.refresh_cache)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Temperature analysis failed: {exc}\n")


if __name__ == "__main__":
    main()
