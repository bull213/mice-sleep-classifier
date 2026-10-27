from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import time

import requests

from .config import ROOT, Cancelled
from .tasks import CancelCallback, ProgressCallback, ignore_progress

DATASET_NAME = "df_features_10s_17_full_hu.csv"
DATASET_SIZE = 949591005
DATASET_MD5 = "5257267cd0af75b9d5362d971e511c22"
DATASET_RECORD = "https://zenodo.org/records/5180680"
DATASET_URLS = [
    f"https://zenodo.org/api/records/5180680/files/{DATASET_NAME}/content",
    f"https://zenodo.org/records/5180680/files/{DATASET_NAME}?download=1",
]


def verify_dataset(path, cancel=None):
    path = Path(path)
    if path.stat().st_size != DATASET_SIZE:
        raise ValueError(f"Published training CSV must be {DATASET_SIZE:,} bytes; this copy is incomplete or different.")
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if cancel is not None and cancel():
                raise Cancelled()
            digest.update(block)
    if digest.hexdigest() != DATASET_MD5:
        raise ValueError("The published training data checksum does not match. Download a fresh copy from Zenodo.")
    return path


def download_dataset(directory=None, progress: ProgressCallback = ignore_progress, cancel: CancelCallback = None):
    directory = Path(directory or ROOT / "data")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / DATASET_NAME
    if target.exists():
        progress({"message": "Verifying cached training data"})
        return verify_dataset(target, cancel)
    partial = target.with_suffix(".csv.part")
    if partial.exists() and partial.stat().st_size == DATASET_SIZE:
        verify_dataset(partial, cancel)
        partial.replace(target)
        return target
    failures = []
    # Bounded byte-range requests avoid keeping a slow 950 MB connection open.
    # Only verified-length, contiguous chunks are appended to the resumable file.
    try:
        _range_download(partial, progress, cancel)
        progress({"message": "Verifying training data checksum"})
        verify_dataset(partial, cancel)
        partial.replace(target)
        return target
    except Cancelled:
        raise
    except (requests.RequestException, OSError, ValueError) as exc:
        failures.append(str(exc))
        progress({"message": f"Parallel download interrupted ({exc}); continuing from the saved position."})
    with requests.Session() as session:
        session.headers["User-Agent"] = "MouseSleepAnalyzer/1.0 (academic video-analysis tool)"
        for url in DATASET_URLS:
            try:
                _stream_download(session, url, partial, progress, cancel)
                progress({"message": "Verifying training data checksum"})
                verify_dataset(partial, cancel)
                partial.replace(target)
                return target
            except Cancelled:
                raise
            except (requests.RequestException, OSError, ValueError) as exc:
                failures.append(str(exc))
    raise RuntimeError("Automatic download failed. Open " + DATASET_RECORD + ", download " + DATASET_NAME + ", then select that CSV in Classifier setup. " + " | ".join(failures))


def _stream_download(session, url, partial, progress, cancel):
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(url, stream=True, timeout=(20, 90), headers=headers) as response:
        response.raise_for_status()
        if response.status_code == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Download server returned an unexpected byte range.")
            mode = "ab"
        else:
            offset, mode = 0, "wb"
        done = offset
        update = 0.0
        with partial.open(mode) as handle:
            for block in response.iter_content(1024 * 1024):
                if cancel is not None and cancel():
                    raise Cancelled()
                if block:
                    handle.write(block)
                    done += len(block)
                    if done > DATASET_SIZE:
                        raise ValueError("Downloaded file exceeds the published dataset size.")
                if time.monotonic() - update >= 1:
                    progress({"message": f"Downloading training data: {done / 1e6:.0f} / {DATASET_SIZE / 1e6:.0f} MB", "fraction": done / DATASET_SIZE})
                    update = time.monotonic()


class _RangeDownload:
    """State shared by parallel range workers for one resumable download."""

    def __init__(self, partial, progress, cancel):
        self.partial, self.progress, self.cancel = partial, progress, cancel
        self.chunk_size = 4 * 1024 * 1024
        self.offset = partial.stat().st_size if partial.exists() else 0
        if self.offset > DATASET_SIZE:
            raise ValueError("Cached partial download exceeds the published size.")
        self.chunks = partial.parent / (partial.name + ".chunks")
        self.chunks.mkdir(exist_ok=True)
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.downloaded = self.offset
        self.last_update = 0.0

    def check_cancel(self):
        if self.stopped.is_set() or (self.cancel is not None and self.cancel()):
            raise Cancelled()

    def fetch(self, bounds):
        start, end = bounds
        cached = self.chunks / f"{start}-{end}.bin"
        if cached.exists() and cached.stat().st_size == end - start:
            return cached
        last_error = None
        for attempt in range(6):
            self.check_cancel()
            try:
                with requests.get(DATASET_URLS[attempt % len(DATASET_URLS)], headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": "MouseSleepAnalyzer/1.0"}, stream=True, timeout=(20, 30)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{DATASET_SIZE}":
                        raise ValueError("Server does not support the requested byte range.")
                    pending = cached.with_suffix(".pending")
                    size = 0
                    with pending.open("wb") as handle:
                        for block in response.iter_content(128 * 1024):
                            self.check_cancel()
                            size += len(block)
                            if size > end - start:
                                raise ValueError("Range download exceeded its expected size.")
                            handle.write(block)
                            with self.lock:
                                self.downloaded += len(block)
                                if time.monotonic() - self.last_update >= 2:
                                    self.progress({"message": f"Downloading training data: {min(self.downloaded, DATASET_SIZE) / 1e6:.0f} / {DATASET_SIZE / 1e6:.0f} MB", "fraction": min(self.downloaded / DATASET_SIZE, 0.99)})
                                    self.last_update = time.monotonic()
                    if size != end - start:
                        raise ValueError("Incomplete byte range.")
                    pending.replace(cached)
                    return cached
            except (requests.RequestException, OSError, ValueError) as exc:
                last_error = exc
                self.progress({"message": f"Retrying training-data range at {start / 1e6:.0f} MB: {exc}"})
                if self.stopped.wait(min(2 ** attempt, 30)):
                    raise Cancelled()
        raise last_error

    def run(self):
        bounds = []
        cursor = self.offset
        while cursor < DATASET_SIZE:
            end = min(((cursor // self.chunk_size) + 1) * self.chunk_size, DATASET_SIZE)
            bounds.append((cursor, end))
            cursor = end
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.fetch, item) for item in bounds]
            try:
                with self.partial.open("ab") as handle:
                    for bounds_item, future in zip(bounds, futures, strict=True):
                        self.check_cancel()
                        cached = future.result()
                        with cached.open("rb") as source:
                            for block in iter(lambda: source.read(1024 * 1024), b""):
                                handle.write(block)
                        handle.flush()
                        # Keep downloaded chunks as a recovery cache until verification.
            finally:
                self.stopped.set()
                for future in futures:
                    future.cancel()


def _range_download(partial, progress, cancel):
    _RangeDownload(partial, progress, cancel).run()
