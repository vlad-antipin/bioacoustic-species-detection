import numpy as np
import librosa
from numpy.typing import NDArray
from typing import Callable

# A bunch of heuristics to check the quality of an audio
# proposed by Claude


def spectral_flatness_score(audio: NDArray, sr: int) -> float:
    """
    Compute mean spectral flatness across time frames.
    Low value = tonal/structured (biological vocalization).
    High value = flat/noise-like (wind, rain).
    Returns (1 - flatness) so that higher score = better chunk.
    """
    flatness = librosa.feature.spectral_flatness(y=audio)  # shape: (1, T)
    return 1.0 - float(np.mean(flatness))


def band_snr_score(
    audio: NDArray, sr: int, f_low: int = 500, f_high: int = 10_000
) -> float:
    """
    SNR of the target band (500 Hz-10 kHz) relative to the noise floor
    estimated from the lowest-energy bins outside the band.
    Broadband default covers birds, frogs, and most insects.
    """
    S = np.abs(librosa.stft(audio))  # (freq_bins, time_frames)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    in_band = (freqs >= f_low) & (freqs <= f_high)
    out_band = ~in_band

    signal_power = np.mean(S[in_band] ** 2)
    noise_power = np.mean(S[out_band] ** 2) + 1e-10  # avoid division by zero

    return float(np.log1p(signal_power / noise_power))  # log for numerical stability


def activity_index(audio: NDArray, sr: int, threshold_db: float = -40.0) -> float:
    """
    Fraction of spectrogram cells exceeding threshold above local noise floor.
    High value = dense acoustic events (insects, chorus, active bird).
    """
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    noise_floor = np.percentile(S_db, 10)  # robust noise floor estimate
    active_cells = np.mean(S_db > (noise_floor + abs(threshold_db)))
    return float(active_cells)


def chunk_quality_score(
    audio: NDArray, sr: int, w1: float = 0.5, w2: float = 0.3, w3: float = 0.2
) -> float:
    """
    Weighted combination of three complementary quality signals.

    w1: tonality (spectral flatness) — most reliable cross-taxon signal
    w2: band SNR                     — penalises broadband noise
    w3: activity index               — rewards dense vocalisation events
    """
    sf = spectral_flatness_score(audio, sr)
    snr = band_snr_score(audio, sr)
    ai = activity_index(audio, sr)

    # Each sub-score has different scale; normalise band_snr (log scale, can be > 1)
    # Flatness and activity are already in [0, 1]
    snr_normalised = np.tanh(snr)  # smooth squash to [0, 1]

    return w1 * sf + w2 * snr_normalised + w3 * ai
