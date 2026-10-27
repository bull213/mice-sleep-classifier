import numpy as np
from scipy import signal
from mouse_sleep.features import FEATURE_NAMES, SUFFIXES, spectral_features, epoch_features
from tests.reference_spectral import get_freq_feature


def test_features_match_author_reference():
    rng = np.random.default_rng(152)
    t = np.arange(300) / 30
    wave = 2100 + 25 * np.sin(2 * np.pi * 3.6 * t) + rng.normal(0, 3, 300)
    a, b = signal.butter(5, [1 / 30, 29 / 30], "bandpass")
    reference = get_freq_feature(wave, a, b, 30)
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17]
    np.testing.assert_allclose(spectral_features(wave), np.asarray(reference)[indices], rtol=1e-12, atol=1e-12)
    assert len(FEATURE_NAMES) == 225
    assert FEATURE_NAMES[:15] == ["m00__" + suffix for suffix in SUFFIXES]


def test_missing_tracking_is_unknown_not_stationary():
    times = np.arange(300) / 30
    values = np.column_stack([np.sin(times * (i + 1)) + i for i in range(14)])
    feature, coverage, reason = epoch_features(times, values, 0)
    assert len(feature) == 225 and coverage == 1 and reason == ""
    keep = (times < 4) | (times > 4.3)
    feature, _, reason = epoch_features(times[keep], values[keep], 0)
    assert feature is None and reason == "tracking_or_timing_gap"
    assert epoch_features([], [], 0)[2] == "mouse_not_tracked"
