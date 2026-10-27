"""Verbatim reference function from MouseSleep (MIT, KumarLabJax 2021).

Source: dataset-creation/Sleep_feature_generation.py, main branch.
See licenses/MouseSleep_LICENSE.txt for the complete copyright notice.
"""
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew


def get_freq_feature(wave, a, b, samplerate):
    wave = signal.filtfilt(a, b, wave)
    freqs, psd = signal.welch(wave)
    freqs = freqs * samplerate
    k = kurtosis(wave)
    k_psd = kurtosis(psd)
    s_psd = skew(psd)
    idx_1 = np.logical_and(freqs >= 0.1, freqs < 1)
    idx_3 = np.logical_and(freqs >= 1, freqs < 3)
    idx_5 = np.logical_and(freqs >= 3, freqs < 5)
    idx_8 = np.logical_and(freqs >= 5, freqs < 8)
    idx_15 = np.logical_and(freqs >= 8, freqs < 15)
    MPL_1 = np.mean(psd[idx_1])
    MPL_3 = np.mean(psd[idx_3])
    MPL_5 = np.mean(psd[idx_5])
    MPL_8 = np.mean(psd[idx_8])
    MPL_15 = np.mean(psd[idx_15])
    Tot_PSD = np.sum(psd)
    Max_PSD = max(psd)
    Min_PSD = min(psd)
    Ave_PSD = np.average(psd)
    Std_PSD = np.std(psd)
    Ave_Signal = np.average(wave)
    Std_Signal = np.std(wave)
    Max_Signal = max(wave)
    Min_Signal = min(wave)
    Top_Signal = freqs[psd == Max_PSD][0]
    Med_Signal = np.median(wave)
    Med_PSD = np.median(psd)
    return [k, k_psd, s_psd, MPL_1, MPL_3, MPL_5, MPL_8, MPL_15, Tot_PSD, Max_PSD, Min_PSD, Ave_PSD, Std_PSD,
            Ave_Signal, Std_Signal, Max_Signal, Min_Signal, Top_Signal, Med_Signal, Med_PSD]
