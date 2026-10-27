"""Label-free port of the MIT MouseSleep feature calculations."""
from __future__ import annotations

import warnings
import cv2
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew

from .config import SAMPLE_RATE

SIGNALS = ["m00", "perimeter", "w", "l", "wl_ratio", "dx", "dy", "dx2_plus_dy2", *[f"hu{i}" for i in range(7)]]
SUFFIXES = ["k", "k_psd", "s_psd", "MPL_1", "MPL_3", "MPL_5", "MPL_8", "MPL_15", "Tot_PSD", "Max_PSD", "Ave_PSD", "Std_PSD", "Ave_Signal", "Std_Signal", "Top_Signal"]
FEATURE_NAMES = [f"{name}__{suffix}" for name in SIGNALS for suffix in SUFFIXES]
_FILTER = signal.butter(5, [1 / SAMPLE_RATE, 29 / SAMPLE_RATE], "bandpass")
_EPOCH_GRID = np.arange(300) / SAMPLE_RATE


def measurements(mask, *, fraction_range=None):
    moments = cv2.moments(mask)
    area = moments["m00"]
    if area < 20:
        return None
    if fraction_range is not None and not fraction_range[0] <= area / mask.size <= fraction_range[1]:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
    x, y = moments["m10"] / area, moments["m01"] / area
    a = moments["m20"] / area - x * x
    b = 2 * (moments["m11"] / area - x * y)
    c = moments["m02"] / area - y * y
    term = np.sqrt(b * b + (a - c) ** 2)
    w = np.sqrt(max(0, 8 * (a + c - term))) / 2
    length = np.sqrt(max(0, 8 * (a + c + term))) / 2
    if w <= 0 or length <= 0:
        return None
    return np.array([area, perimeter, w, length, w / length, x, y, *cv2.HuMoments(moments).ravel()], dtype=np.float64)


def spectral_features(wave):
    # Preserve upstream normalized filter cutoffs and Welch PSD scaling exactly.
    a, b = _FILTER
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filtered = signal.filtfilt(a, b, wave)
        frequencies, psd = signal.welch(filtered)
        frequencies *= SAMPLE_RATE
        result = [kurtosis(filtered), kurtosis(psd), skew(psd)]
        result += [np.mean(psd[(frequencies >= lower) & (frequencies < upper)]) for lower, upper in [(0.1, 1), (1, 3), (3, 5), (5, 8), (8, 15)]]
        maximum = np.max(psd)
        result += [np.sum(psd), maximum, np.mean(psd), np.std(psd), np.mean(filtered), np.std(filtered), frequencies[np.flatnonzero(psd == maximum)[0]]]
    return np.asarray(result)


def epoch_features(times, values, epoch_start, *, minimum_coverage=0.95, maximum_gap=0.12):
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    grid = epoch_start + _EPOCH_GRID
    if len(times) < 2:
        return None, 0.0, "mouse_not_tracked"
    # Assess actual time support, not the number of interpolated samples.
    positions = np.searchsorted(times, grid)
    previous = times[np.clip(positions - 1, 0, len(times) - 1)]
    following = times[np.clip(positions, 0, len(times) - 1)]
    distance = np.minimum(np.abs(grid - previous), np.abs(grid - following))
    coverage = float(np.mean(distance <= 0.75 / SAMPLE_RATE))
    gaps = np.diff(times)
    if coverage < minimum_coverage or np.any(gaps > maximum_gap):
        return None, coverage, "tracking_or_timing_gap"
    if times[0] > grid[0] + maximum_gap or times[-1] < grid[-1] - maximum_gap:
        return None, coverage, "tracking_or_timing_gap"
    sampled = np.column_stack([np.interp(grid, times, values[:, i]) for i in range(values.shape[1])])
    dx = np.r_[0, np.diff(sampled[:, 5])]
    dy = np.r_[0, np.diff(sampled[:, 6])]
    waves = np.column_stack([sampled[:, :5], dx, dy, dx * dx + dy * dy, sampled[:, 7:]])
    features = np.concatenate([spectral_features(waves[:, i]) for i in range(len(SIGNALS))])
    # Undefined kurtosis for truly constant measurements is missing, not artificial zero.
    if np.any(np.isinf(features)) or np.mean(np.isnan(features)) > 0.2:
        return None, coverage, "invalid_features"
    return features, coverage, ""
