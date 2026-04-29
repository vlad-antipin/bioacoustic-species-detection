import numpy as np
from numpy.typing import NDArray
import pandas as pd
import librosa

from .data import SR
from .data import load_audio
from .preprocessing import get_labels
from tqdm.auto import tqdm


def get_spectrogram(audio, ref=np.max, n_fft=2048):
    # TODO: automatically set hop_length
    S = librosa.stft(audio, n_fft=n_fft, hop_length=512)
    frequencies = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(S.shape[1]), sr=SR, n_fft=n_fft, hop_length=512
    )
    S_db = librosa.amplitude_to_db(np.abs(S), ref=ref)
    return S_db, frequencies, times

def get_mel_spectrogram(audio, ref=np.max, n_fft=2048, n_mels=128):
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=n_fft, hop_length=512, n_mels=n_mels
    )
    frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=SR / 2)
    times = librosa.frames_to_time(
        np.arange(S.shape[1]), sr=SR, n_fft=n_fft, hop_length=512
    )
    S_db = librosa.power_to_db(S, ref=ref)
    return S_db, frequencies, times


def get_mfcc(audio, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=n_mfcc)
    return mfcc


def get_chroma_stft(audio):
    chroma = librosa.feature.chroma_stft(y=audio, sr=SR)
    return chroma


def add_basic_signal_stats(audio, features):
    features["mean"] = np.mean(audio)
    features["std"] = np.std(audio)
    features["max"] = np.max(audio)
    features["min"] = np.min(audio)
    features["rms"] = np.sqrt(np.mean(audio**2))  # signal energy
    
def add_percentiles(name, values, features):
    p = np.percentile(values, [10, 25, 50, 75, 90])
    for i, perc in zip([10, 25, 50, 75, 90], p):
        features[f"{name}_p{i}"] = perc

def add_zero_crossing_rate(audio, features, include_percentiles=True):
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features["zcr_mean"] = zcr.mean()
    features["zcr_std"] = zcr.std()
    
    if include_percentiles:
        add_percentiles("zcr", zcr, features)
    
def add_spectrogram(audio, features):
    S = np.abs(librosa.stft(audio))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    features["spec_mean"] = S_db.mean()
    features["spec_std"] = S_db.std()
    features["spec_min"] = S_db.min()
    features["spec_max"] = S_db.max()

def add_spectral_features(audio, features, include_percentiles=True):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SR)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SR)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR)[0]

    features["centroid_mean"] = spectral_centroid.mean()
    features["centroid_std"] = spectral_centroid.std()

    features["bandwidth_mean"] = spectral_bandwidth.mean()
    features["bandwidth_std"] = spectral_bandwidth.std()

    features["rolloff_mean"] = spectral_rolloff.mean()
    features["rolloff_std"] = spectral_rolloff.std()
    
    if include_percentiles:
        add_percentiles("centroid", spectral_centroid, features)
        add_percentiles("bandwidth", spectral_bandwidth, features)
        add_percentiles("rolloff", spectral_rolloff, features)

def add_mfcc(audio, features, include_delta=True):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20)

    for i in range(mfcc.shape[0]):
        features[f"mfcc_{i}_mean"] = mfcc[i].mean()
        features[f"mfcc_{i}_std"] = mfcc[i].std()
    
    if include_delta:
        delta_mfcc = librosa.feature.delta(mfcc)

        for i in range(delta_mfcc.shape[0]):
            features[f"delta_mfcc_{i}_mean"] = delta_mfcc[i].mean()
            features[f"delta_mfcc_{i}_std"] = delta_mfcc[i].std()

def add_chroma(audio, features):
    chroma = librosa.feature.chroma_stft(y=audio, sr=SR)

    for i in range(chroma.shape[0]):
        features[f"chroma_{i}_mean"] = chroma[i].mean()
        features[f"chroma_{i}_std"] = chroma[i].std()

def add_log_mel(audio, features):
    mel = librosa.feature.melspectrogram(y=audio, sr=SR)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    features["mel_mean"] = mel_db.mean()
    features["mel_std"] = mel_db.std()

    # also per-band statistics (compressed version)
    for i in range(min(20, mel_db.shape[0])):  # limit dimensionality
        features[f"mel_band_{i}_mean"] = mel_db[i].mean()
        features[f"mel_band_{i}_std"] = mel_db[i].std()

def add_autocorrelation(audio, features):
    # captures rhythmic patterns (e.g., repeating chirps)
    autocorr = np.correlate(audio, audio, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    features["autocorr_mean"] = autocorr.mean()
    features["autocorr_std"] = autocorr.std()
    features["autocorr_max"] = autocorr.max()

def add_rms_energy_stats(audio, features):
    rms_frame = librosa.feature.rms(y=audio)[0]

    features["rms_frame_mean"] = rms_frame.mean()
    features["rms_frame_std"] = rms_frame.std()

    # how "bursty" the signal is
    features["rms_frame_max"] = rms_frame.max()
    features["rms_frame_min"] = rms_frame.min()
    
def get_features(audio: NDArray) -> pd.Series:
    """Extract a rich set of audio features from a waveform."""

    if len(audio) < SR * 0.5:  # shorter than 0.5 sec
        return pd.Series({})

    features = {}

    add_basic_signal_stats(audio, features)
    
    add_zero_crossing_rate(audio, features, include_percentiles=True)
    
    add_spectrogram(audio, features)
    
    add_spectral_features(audio, features, include_percentiles=True)

    add_mfcc(audio, features, include_delta=True)
    
    add_chroma(audio, features)
    
    add_log_mel(audio, features)
    
    # NOTE: autocorrelation takes forever ...
    # add_autocorrelation(audio, features)
    
    add_rms_energy_stats(audio, features)

    return pd.Series(features)


def prepare_data(df, df_taxonomy, sample_idx):

    df = df.iloc[sample_idx]
    y_class, y_primary = get_labels(df, df_taxonomy)

    features = [
        get_features(load_audio(sample))
        for _, sample in tqdm(df.iterrows(), total=len(df), desc="Extracting features")
    ]
    X = pd.DataFrame(features, index=sample_idx)

    mask = ~X.isna().any(axis=1)

    X = X[mask]
    y_primary = y_primary[mask]
    y_class = y_class[mask]

    return {
        "X": X,
        "y_primary": y_primary,
        "y_class": y_class,
    }
