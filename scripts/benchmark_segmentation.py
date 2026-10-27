"""Compare optimized inference with the original TensorFlow mask cleanup.

Use --config last_settings.json to sample the selected recording and ROI.
Without a config, use synthetic frames (a software benchmark only).
"""
from contextlib import closing
from itertools import islice
from pathlib import Path
import argparse
import os
import platform
import time

import cv2
import numpy as np

from mouse_sleep.config import MODEL_STEPS, ROOT, Settings, atomic_json
from mouse_sleep.segmentation import Segmenter, network_image
from mouse_sleep.video import Timing, sampled_frames
from tests.reference_morphology import reference_masks


def sample_images(settings, count):
    if settings.video:
        source = sampled_frames(settings, Timing(settings))
        with closing(source):
            images = [network_image(gray, settings.roi)[0] for _, _, gray in islice(source, count)]
        if not images:
            raise ValueError("The selected recording contains no usable frames.")
        return np.asarray(images, dtype=np.float32)[..., None]
    rng = np.random.default_rng(52)
    images = rng.normal(130, 12, (count, 480, 480, 1)).astype(np.float32)
    images[::2, 150:260, 200:270] = 15
    images[1::2, 150:260, 200:270] = 240
    return images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", choices=MODEL_STEPS)
    parser.add_argument("--batch-size", type=int, choices=range(1, 33))
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "verification" / "performance" / "segmentation.json")
    args = parser.parse_args(argv)
    if args.batches < 1 or args.repeats < 1:
        parser.error("--batches and --repeats must be positive")
    settings = Settings.load(args.config) if args.config else Settings(model="FullModel")
    if args.model:
        settings.model = args.model
    if args.batch_size:
        settings.batch_size = args.batch_size
    if not 1 <= settings.batch_size <= 32:
        parser.error("The settings batch size must be between 1 and 32")
    images = sample_images(settings, settings.batch_size * args.batches)
    batches = [images[start:start + settings.batch_size] for start in range(0, len(images), settings.batch_size)]
    timings = {"original": [], "optimized": []}
    with Segmenter(settings.checkpoint) as model:
        with model.graph.as_default():
            original_masks = reference_masks(model.tf, model.probabilities)

        def original():
            return [model.session.run(original_masks, {model.inputs: batch}) for batch in batches]

        def optimized():
            return [model.predict(batch) for batch in batches]

        # Warm both execution paths and require exact mask equality before timing.
        for expected, actual in zip(original(), optimized(), strict=True):
            np.testing.assert_array_equal(actual, expected)
        methods = [("original", original), ("optimized", optimized)]
        for repeat in range(args.repeats):
            for name, predict in methods[::1 if repeat % 2 == 0 else -1]:
                began = time.perf_counter()
                predict()
                elapsed = time.perf_counter() - began
                timings[name].append(elapsed)
                print(f"{name}: {len(images) / elapsed:.1f} frames/s", flush=True)
        speedup = float(np.median(timings["original"]) / np.median(timings["optimized"]))
        report = {
            "python": platform.python_version(), "tensorflow": model.tf.__version__,
            "opencv": cv2.__version__, "logical_cpus": os.cpu_count(),
            "model": settings.model_name, "batch_size": settings.batch_size,
            "frames_per_repeat": len(images), "repeats": args.repeats,
            "input": "recording" if settings.video else "synthetic",
            "mask_parity": "exact", "startup_excluded": True,
            "seconds": timings,
            "median_fps": {name: len(images) / float(np.median(values)) for name, values in timings.items()},
            "speedup": speedup,
        }
    atomic_json(args.output, report)
    print(f"Segmentation speedup: {speedup:.2f}x; identical masks. Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
