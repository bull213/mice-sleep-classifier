"""Shared, UI-independent progress and cancellation contracts."""
from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from .config import Cancelled


class ProgressEvent(TypedDict, total=False):
    message: str
    seconds: float
    duration: float
    processing_fps: float
    remaining_seconds: float | None
    fraction: float | None


ProgressCallback = Callable[[ProgressEvent], None]
CancelCallback = Callable[[], bool] | None


def ignore_progress(event: ProgressEvent) -> None:
    pass


def check_cancelled(cancel: CancelCallback) -> None:
    if cancel is not None and cancel():
        raise Cancelled()
