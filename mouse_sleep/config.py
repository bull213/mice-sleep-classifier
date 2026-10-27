from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
MODEL_STEPS = {"24Hr_OFA": 255000, "FullModel": 415000, "AllCoatColor_RegularOFA": 234000}
SEED = 1438939568
SAMPLE_RATE = 30
EPOCH_SECONDS = 10
MEASUREMENT_SIZE = 1080
MASKING_VERSION = 2
MASKING_FIELDS = ("masking_mode", "cage_polygon", "background_images", "background_delta",
                  "background_fraction", "minimum_mouse_area", "maximum_mouse_area")


class Cancelled(Exception):
    """A resumable task was cancelled by its caller."""


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Settings:
    schema_version: int = 1
    video: str = ""
    output: str = ""
    appearance: str = "dark"
    model: str = "auto"
    refinement: str = "auto"
    threshold: int = 30
    roi: tuple[int, int, int, int] | None = None  # x, y, width, height in original pixels
    batch_size: int = 4
    minimum_coverage: float = 0.95
    maximum_gap_seconds: float = 0.12
    minimum_confidence: float = 0.6
    timing_csv: str = ""
    take_id: int | None = None
    classifier: str = ""
    max_seconds: float | None = None
    masking_mode: str = "enhanced"
    cage_polygon: tuple[tuple[int, int], ...] | None = None  # original video coordinates
    background_images: tuple[str, ...] = ()
    background_delta: float = 12.0
    background_fraction: float = 0.45
    minimum_mouse_area: float = 0.001
    maximum_mouse_area: float = 0.25

    @property
    def model_name(self):
        if self.model != "auto":
            return self.model
        if self.appearance == "white":
            return "FullModel"
        return "AllCoatColor_RegularOFA" if self.masking_mode == "enhanced" else "24Hr_OFA"

    @property
    def refinement_mode(self):
        return ("dark" if self.appearance == "dark" else "off") if self.refinement == "auto" else self.refinement

    @property
    def checkpoint(self):
        return ROOT / "pretrained_models" / self.model_name / f"model.ckpt-{MODEL_STEPS[self.model_name]}"

    @property
    def classifier_dir(self):
        return Path(self.classifier).expanduser().resolve() if self.classifier else ROOT / "classifier"

    def validate(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported settings version. Expected schema_version 1.")
        if self.appearance not in ("dark", "white") or self.model not in ("auto", *MODEL_STEPS):
            raise ValueError("Choose dark/white appearance and an available segmentation model.")
        if self.refinement not in ("auto", "off", "dark", "bright"):
            raise ValueError("Refinement must be auto, off, dark, or bright.")
        if self.masking_mode not in ("legacy", "enhanced"):
            raise ValueError("Masking mode must be Enhanced or Legacy.")
        if self.masking_mode == "enhanced":
            if self.appearance != "dark":
                raise ValueError("Enhanced masking currently supports dark mice. Select Legacy for white/light mice.")
            if (not math.isfinite(self.background_delta) or not 0 < self.background_delta <= 255
                    or not math.isfinite(self.background_fraction) or not 0 < self.background_fraction < 1):
                raise ValueError("Background contrast must be 0–255 levels and a fraction between 0 and 1.")
            if not 0 < self.minimum_mouse_area < self.maximum_mouse_area <= 0.4:
                raise ValueError("Mouse area limits must be increasing fractions between 0 and 0.4.")
            if len(self.background_images) > 2 or any(not Path(p).is_file() for p in self.background_images):
                raise ValueError("Select up to two existing empty-cage images, one per lighting condition.")
            if self.cage_polygon is not None and (len(self.cage_polygon) < 3 or any(
                    len(point) != 2 or any(type(v) is not int or v < 0 for v in point)
                    for point in self.cage_polygon)):
                raise ValueError("Cage boundary needs at least three points in original pixel coordinates.")
        if not Path(self.video).is_file():
            raise ValueError("Select an existing video file.")
        if not self.output:
            raise ValueError("Select an output folder.")
        if not 1 <= self.batch_size <= 32 or not 0 <= self.threshold <= 255:
            raise ValueError("Batch size must be 1–32 and brightness threshold 0–255.")
        if not 0.5 <= self.minimum_coverage <= 1 or not 0.5 <= self.minimum_confidence <= 1:
            raise ValueError("Coverage and confidence thresholds must be between 0.5 and 1.")
        if not math.isfinite(self.maximum_gap_seconds) or not 0 < self.maximum_gap_seconds <= 1:
            raise ValueError("Maximum gap must be between 0 and 1 second.")
        if self.max_seconds is not None and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("Maximum duration must be a positive finite number.")
        if self.roi is not None and (len(self.roi) != 4 or any(type(v) is not int for v in self.roi) or min(self.roi[:2]) < 0 or min(self.roi[2:]) <= 0):
            raise ValueError("ROI must contain integer x, y, width, height with positive dimensions.")
        if self.take_id is not None and self.take_id <= 0:
            raise ValueError("Take ID must be positive.")
        return self

    def save(self, path):
        atomic_json(Path(path), asdict(self))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(data.get("settings"), dict) and "identity" in data:
            data = data["settings"]
        if data.get("roi") is not None:
            data["roi"] = tuple(data["roi"])
        # A missing field always means the original detector, including old manifests.
        data.setdefault("masking_mode", "legacy")
        if data.get("cage_polygon") is not None:
            data["cage_polygon"] = tuple(tuple(point) for point in data["cage_polygon"])
        if "background_images" in data:
            data["background_images"] = tuple(data["background_images"])
        return cls(**data)

    def identity(self):
        data = asdict(self)
        # JSON stores tuples as arrays. Compare the same representation on resume.
        data["roi"] = list(self.roi) if self.roi is not None else None
        if self.masking_mode == "legacy":
            # Retain the exact identity of pre-mode runs so they remain resumable.
            for key in MASKING_FIELDS:
                data.pop(key)
        else:
            data["masking_version"] = MASKING_VERSION
            data["cage_polygon"] = [list(point) for point in self.cage_polygon] if self.cage_polygon else None
            data["background_images"] = [
                {"path": str(Path(path).resolve()), "sha256": sha256(Path(path))}
                for path in self.background_images
            ]
        data["video"] = str(Path(self.video).resolve())
        data.pop("output")
        video = Path(self.video).stat()
        data["video_size"] = video.st_size
        data["video_mtime_ns"] = video.st_mtime_ns
        for key in ("timing_csv",):
            if data[key]:
                stat = Path(data[key]).stat()
                data[key] = [str(Path(data[key]).resolve()), stat.st_size, stat.st_mtime_ns]
        return data
