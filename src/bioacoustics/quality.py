import numpy as np
import librosa
from numpy.typing import NDArray

# A bunch of heuristics to check the quality of an audio
# proposed by Claude


def spectral_flatness_score(audio: NDArray, sr: int) -> float:
    """Returns 1 - mean spectral flatness; higher = more tonal/structured."""
    flatness = librosa.feature.spectral_flatness(y=audio)
    return 1.0 - float(np.mean(flatness))


def band_snr_score(
    audio: NDArray, sr: int, f_low: int = 500, f_high: int = 10_000
) -> float:
    """Log SNR of the 500 Hz–10 kHz band relative to out-of-band noise floor."""
    S = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    in_band = (freqs >= f_low) & (freqs <= f_high)
    out_band = ~in_band

    signal_power = np.mean(S[in_band] ** 2)
    noise_power = np.mean(S[out_band] ** 2) + 1e-10  # avoid division by zero

    return float(np.log1p(signal_power / noise_power))


def activity_index(audio: NDArray, sr: int, threshold_db: float = -40.0) -> float:
    """Fraction of spectrogram cells above local noise floor + threshold."""
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    noise_floor = np.percentile(S_db, 10)
    active_cells = np.mean(S_db > (noise_floor + abs(threshold_db)))
    return float(active_cells)


def chunk_quality_score(
    audio: NDArray, sr: int, w1: float = 0.5, w2: float = 0.3, w3: float = 0.2
) -> float:
    """Weighted combination: w1=tonality, w2=band SNR, w3=activity index."""
    sf = spectral_flatness_score(audio, sr)
    snr = band_snr_score(audio, sr)
    ai = activity_index(audio, sr)

    # band_snr is log-scale and can exceed 1; squash to [0, 1] to match the others
    snr_normalised = np.tanh(snr)

    return w1 * sf + w2 * snr_normalised + w3 * ai
