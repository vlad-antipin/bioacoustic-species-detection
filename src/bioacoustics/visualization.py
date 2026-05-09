import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display

from .config import SR, HOP_LENGTH

import matplotlib.pyplot as plt

def set_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 100,

        "axes.titlesize": 12,
        "axes.labelsize": 10,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.alpha": 0.2,

        "lines.linewidth": 1.8,

        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "legend.fontsize": 9,

        "font.family": "DejaVu Sans",
    })


def plot_label_frequency(df_label, log=True, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    label_freq = df_label.mean()
    label_freq.plot(kind="bar", ax=ax)
    if log:
        ax.set_yscale("log")
    ax.set_title("Label frequency" + " (log scale)" if log else "")
    ax.set_ylabel("P(label = 1)")
    ax.tick_params(axis="x", labelrotation=45)


def plot_active_labels(df_label, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    row_counts = df_label.sum(axis=1)
    ax.hist(row_counts, bins=range(int(row_counts.max()) + 2), align="left")
    ax.set_title("Number of active labels per sample")
    ax.set_xlabel("labels per sample")
    ax.set_ylabel("count")


def plot_label_concurrence(df_label, normalize=True, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    cooc = df_label.T.dot(df_label)
    if normalize:
        diag = np.diag(cooc)
        union = diag[:, None] + diag[None, :] - cooc
        cooc /= union

    sns.heatmap(cooc, cmap="viridis", ax=ax)
    ax.set_title("Label co-occurrence matrix" + " (normalized)" if normalize else "")


def plot_waveform(audio, ax=None, title=None,sr=SR):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    librosa.display.waveshow(audio, sr=sr, alpha=0.5, ax=ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title if title is not None else "Waveform")

    return ax


def plot_autocorrelation(audio, ax=None, title=None, sr=SR):
    if ax is None:
        _, ax = plt.subplots()

    autocorr = np.correlate(audio, audio, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    librosa.display.waveshow(autocorr, sr=sr, alpha=0.5, ax=ax)
    ax.set_title(title if title is not None else "Autocorrelation")
    ax.set_ylabel("autocorrelation")
    ax.set_xlabel("lag (s)")


def plot_cepstrum_pipeline(audio, sr=SR):

    X = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), d=1 / sr)

    pos_idx = freqs >= 0
    freqs = freqs[pos_idx]
    X = X[pos_idx]

    power_spectrum = np.abs(X) ** 2

    log_spectrum = np.log1p(power_spectrum)

    log_mag = np.log1p(np.abs(X))
    cepstrum = np.fft.ifft(log_mag).real

    quefrency = np.arange(len(cepstrum)) / sr

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(freqs, power_spectrum)
    plt.title("Power Spectrum (|FFT|^2)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")

    plt.subplot(3, 1, 2)
    plt.plot(freqs, log_spectrum)
    plt.title("Log Power Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Log Power")

    plt.subplot(3, 1, 3)
    plt.plot(quefrency, cepstrum)
    plt.title("Cepstrum (IFFT of log spectrum)")
    plt.xlabel("Quefrency (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def plot_spectrogram(S_db, ax=None, title=None,sr=SR, hop_length=HOP_LENGTH, y_axis="linear"):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis=y_axis,
        cmap="viridis",
        ax=ax,
    )

    ax.set_title(title if title is not None else "Spectrogram")

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", shrink=0.8)
    cbar.set_label("Intensity (dB)")

    return ax


def plot_mfcc(mfccs, ax=None, title=None, sr=SR, hop_length=HOP_LENGTH):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    img = librosa.display.specshow(
        mfccs,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="frames",
        cmap="viridis",
        ax=ax,
    )

    ax.set_ylabel("MFCC Coefficients")
    ax.set_title(title if title is not None else "MFCC")

    cbar = fig.colorbar(img, ax=ax, shrink=0.8)
    cbar.set_label("Coefficient value")

    return ax


def plot_chroma_stft(chroma, ax=None, title=None, sr=SR, hop_length=HOP_LENGTH, show=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    img = librosa.display.specshow(
        chroma,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="chroma",
        cmap="viridis",
        ax=ax,
    )

    ax.set_title(title if title is not None else "Chroma STFT")

    cbar = fig.colorbar(img, ax=ax, shrink=0.8)
    cbar.set_label("Intensity")

    fig.tight_layout()

    if show:
        plt.show()

    return ax


def plot_onsets(audio, sr=SR):

    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    # Detect onset frames
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    times = librosa.times_like(onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    fig, ax = plt.subplots()

    librosa.display.waveshow(audio, sr=sr, alpha=0.5, ax=ax)

    for onset_time in onset_times:
        ax.axvline(onset_time, linestyle="--", alpha=0.8)

    ax.set_title("Waveform with Detected Onsets")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    plt.show()

    fig, ax = plt.subplots()

    ax.plot(times, onset_env)

    ax.vlines(onset_times, ymin=0, ymax=onset_env.max(), linestyles="--")

    ax.set_title("Onset Strength Envelope")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Onset Strength")

    plt.show()
