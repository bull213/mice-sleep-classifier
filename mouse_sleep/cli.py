from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

from .config import ROOT, Cancelled, Settings


def cage_boundary(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return tuple(tuple(point) for point in value)
    except (OSError, ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Boundary file must contain a JSON array of [x, y] points.") from exc


def parser():
    result = argparse.ArgumentParser(description="Estimate single-mouse sleep from FLIR video. Outputs are video-based estimates, not EEG measurements.")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("gui", help="Open the desktop application")
    trainer = commands.add_parser("train", help="One-time classifier setup from published reference data")
    trainer.add_argument("--dataset", help="Local published CSV; omit to download it")
    trainer.add_argument("--output", default=str(ROOT / "classifier"))
    for name in ("prepare", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--config", help="Saved GUI/CLI settings JSON")
        command.add_argument("--video")
        command.add_argument("--output")
        command.add_argument("--appearance", choices=["dark", "white"])
        command.add_argument("--model", choices=["auto", "24Hr_OFA", "FullModel", "AllCoatColor_RegularOFA"])
        command.add_argument("--refinement", choices=["auto", "off", "dark", "bright"])
        command.add_argument("--threshold", type=int)
        command.add_argument("--masking-mode", choices=["enhanced", "legacy"], help="Fresh settings use Enhanced; old configs retain Legacy")
        command.add_argument("--cage-boundary", dest="cage_polygon", type=cage_boundary, metavar="JSON_FILE",
                             help="Mouse-accessible boundary in original video coordinates")
        command.add_argument("--background-image", dest="background_images", action="append", metavar="IMAGE",
                             help="Optional empty-cage reference; repeat for the second lighting condition")
        command.add_argument("--background-delta", type=float)
        command.add_argument("--background-fraction", type=float)
        command.add_argument("--minimum-mouse-area", type=float, help="Minimum fraction of the allowed cage area")
        command.add_argument("--maximum-mouse-area", type=float, help="Maximum fraction of the allowed cage area")
        command.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
        command.add_argument("--timing-csv")
        command.add_argument("--take-id", type=int)
        command.add_argument("--max-seconds", type=float, help="Analyze/prepare an initial subset for a pilot run")
        command.add_argument("--batch-size", type=int)
        command.add_argument("--minimum-confidence", type=float)
        command.add_argument("--minimum-coverage", type=float)
        command.add_argument("--maximum-gap-seconds", type=float)
        command.add_argument("--classifier", help="Folder containing a trained classifier")
        command.add_argument("--save-config")
        if name == "analyze":
            command.add_argument("--resume", action="store_true")
    return result


def progress(event):
    message = event.get("message", "")
    if event.get("processing_fps"):
        message += f" | {event['processing_fps']:.1f} fps"
    if event.get("remaining_seconds") is not None:
        message += f" | estimated remaining {event['remaining_seconds'] / 3600:.1f} h"
    print(message, flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import main as gui
            gui()
        elif args.command == "train":
            from .classifier import train
            train(args.dataset, args.output, progress)
        else:
            settings = Settings.load(args.config) if args.config else Settings()
            for field in fields(Settings):
                value = getattr(args, field.name, None)
                if value is not None:
                    setattr(settings, field.name, tuple(value) if field.name in ("roi", "background_images") else value)
            settings.validate()
            if args.save_config:
                settings.save(args.save_config)
            if args.command == "prepare":
                from .video import prepare
                print(prepare(settings, progress))
            else:
                from .pipeline import analyze
                manifest = analyze(settings, resume=args.resume, progress=progress)
                print(json.dumps(manifest["summary"], indent=2))
                if manifest["status"] != "complete":
                    return 130
        return 0
    except (Cancelled, KeyboardInterrupt):
        print("Cancelled. Completed analysis epochs can be resumed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
