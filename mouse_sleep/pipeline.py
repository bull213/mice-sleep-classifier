"""Coordinate streaming analysis, committed epochs, and final reports."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import json
import time

from .classifier import SleepClassifier
from .config import Cancelled, Settings, EPOCH_SECONDS, SAMPLE_RATE, sha256
from .epochs import EpochAccumulator
from .locking import OutputLock
from .reports import export_records
from .segmentation import FramePreparer, Segmenter
from .masking import LegacyMasker, EnhancedMasker, export_masking_diagnostics
from .backgrounds import prepare_backgrounds
from .storage import AnalysisStore
from .tasks import CancelCallback, ProgressCallback, check_cancelled, ignore_progress
from .video import Timing, inspect_video, sampled_frames, validate_rate


class _FrameProcessor:
    def __init__(self, settings, timing, duration, classifier, store, progress, began, masker):
        self.settings, self.timing = settings, timing
        self.duration, self.classifier = duration, classifier
        self.store, self.progress, self.began = store, progress, began
        self.initial_epoch = store.next_epoch
        self.epoch = EpochAccumulator(settings, duration, store.next_epoch)
        self.prepare_frame = FramePreparer(settings.roi)
        self.masker = masker

    def persist(self):
        record = self.epoch.record(self.classifier)
        self.store.persist(record, masking=self.masker.checkpoint())
        self.masker.advance_epoch()
        self.epoch.advance()
        elapsed = time.monotonic() - self.began
        processed = (self.epoch.epoch - self.initial_epoch) * EPOCH_SECONDS
        speed = processed / elapsed if elapsed else 0
        self.progress({
            "message": f"Analyzed {record.end / 3600:.2f} / {self.duration / 3600:.2f} hours",
            "seconds": record.end, "duration": self.duration, "processing_fps": speed * SAMPLE_RATE,
            "remaining_seconds": (self.duration - record.end) / speed if speed else None,
        })

    def process_batch(self, batch, segmenter):
        masks = segmenter.predict([item[2] for item in batch])
        for (index, stamp, image, square, valid), mask480 in zip(batch, masks, strict=True):
            epoch = int((stamp + 1e-8) // EPOCH_SECONDS)
            if epoch < self.epoch.epoch:
                continue
            while self.epoch.epoch < epoch and self.epoch.start < self.duration:
                self.persist()
            if self.epoch.start >= self.duration:
                return
            ids = self.timing.frame_ids
            device_gap = ids is not None and index > 0 and ids[index] - ids[index - 1] > 4
            result = self.masker.process(mask480, (image, square, valid), stamp, device_gap=device_gap)
            self.epoch.add(stamp, result.mask, device_gap=device_gap)

    def run(self, segmenter_factory, cancel):
        if self.epoch.start >= self.duration:
            return
        self.progress({"message": "Loading pretrained segmentation weights"})
        with segmenter_factory(self.settings.checkpoint) as segmenter:
            batch = []
            with closing(sampled_frames(self.settings, self.timing, self.epoch.start, cancel)) as source:
                for index, stamp, gray in source:
                    check_cancelled(cancel)
                    image, square, valid = self.prepare_frame(gray)
                    batch.append((index, stamp, image, square, valid))
                    if len(batch) >= self.settings.batch_size:
                        self.process_batch(batch, segmenter)
                        batch.clear()
            if batch:
                self.process_batch(batch, segmenter)
            while self.epoch.start < self.duration - 1e-7:
                self.persist()


def analyze(settings: Settings, *, resume=False, progress: ProgressCallback = ignore_progress,
            cancel: CancelCallback = None, segmenter_factory=Segmenter, classifier_factory=SleepClassifier):
    settings.validate()
    with OutputLock(settings.output):
        return _analyze(settings, resume=resume, progress=progress, cancel=cancel,
                        segmenter_factory=segmenter_factory, classifier_factory=classifier_factory)


def _analyze(settings: Settings, *, resume=False, progress: ProgressCallback = ignore_progress,
             cancel: CancelCallback = None, segmenter_factory=Segmenter, classifier_factory=SleepClassifier):
    info = inspect_video(settings.video)
    validate_rate(info)
    timing = Timing(settings)
    if timing.times is not None and info.frames is not None and len(timing.times) != info.frames:
        raise ValueError(f"Video declares {info.frames} frames, but the selected timing CSV contains {len(timing.times)} rows.")
    if timing.path:
        settings.timing_csv, settings.take_id = timing.path, timing.take_id
    duration = min(timing.duration(info), settings.max_seconds or float("inf"))
    if duration <= 0:
        raise ValueError("Cannot determine video duration.")
    classifier = classifier_factory(settings.classifier_dir)
    checkpoint_data = Path(str(settings.checkpoint) + ".data-00000-of-00001")
    identity = {"settings": settings.identity(), "segmentation_sha256": sha256(checkpoint_data),
                "classifier_sha256": classifier.schema["model_sha256"]}
    backgrounds = geometry = None
    preparation_seconds = 0.0
    if settings.masking_mode == "enhanced":
        identity["segmentation_index_sha256"] = sha256(Path(str(settings.checkpoint) + ".index"))
        saved_manifest = Path(settings.output) / "run.json"
        existing = json.loads(saved_manifest.read_text(encoding="utf-8")) if saved_manifest.exists() else None
        if existing is not None:
            if not resume:
                raise ValueError("This folder contains an analysis. Select Resume or choose a new output folder.")
            if any(existing.get("identity", {}).get(key) != value for key, value in identity.items()):
                raise ValueError("Resume requires the same video, masking mode, boundary, settings, and model weights.")
        preparing = time.monotonic()
        backgrounds, geometry = prepare_backgrounds(
            settings, info, timing, progress=progress, cancel=cancel,
            saved_path=Path(settings.output) / "masking_background.npz" if existing else None)
        preparation_seconds = time.monotonic() - preparing
        identity["masking_background_sha256"] = backgrounds.digest
    with AnalysisStore(settings, info, timing, duration, identity, resume=resume) as store:
        began = time.monotonic()
        if backgrounds is not None:
            if not resume or not (store.output / "masking_background.npz").exists():
                backgrounds.save(store.output / "masking_background.npz")
            store.manifest["masking"] = {"mode": "enhanced", "background": backgrounds.metadata,
                                         "background_preparation_seconds": preparation_seconds}
            masker = EnhancedMasker(settings, backgrounds, geometry, store.masking_state)
        else:
            masker = LegacyMasker(settings)
        processor = _FrameProcessor(settings, timing, duration, classifier, store, progress, began, masker)
        try:
            processor.run(segmenter_factory, cancel)
            store.manifest["status"] = "complete"
        except (Cancelled, KeyboardInterrupt):
            store.manifest["status"] = "cancelled"
        progress({"message": "Writing sleep intervals and timeline"})
        totals = export_records(store.output, store.records(), classifier, settings, duration)
        if backgrounds is not None:
            export_masking_diagnostics(store, backgrounds, preparation_seconds)
        store.manifest.update(summary=totals, elapsed_seconds_this_run=time.monotonic() - began)
        store.save_manifest()
        manifest = store.manifest
    complete = manifest["status"] == "complete"
    progress({"message": "Analysis complete" if complete else "Cancelled; completed epochs saved for Resume",
              "fraction": 1.0 if complete else None})
    return manifest
