import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display

from .data import SR


def plot_label_frequency(df_label, log=True, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    label_freq = df_label.mean()
    label_freq.plot(kind="bar", ax=ax)
    if log:
        ax.set_yscale("log")
    ax.set_title("Label frequency" + " (log scale)" if log else "")
    ax.set_ylabel("P(label = 1)")
    ax.tick_params(axis="x", labelrotation=45)


def plot_active_labels(df_label, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    row_counts = df_label.sum(axis=1)
    ax.hist(row_counts, bins=range(int(row_counts.max()) + 2), align="left")
    ax.set_title("Number of active labels per sample")
    ax.set_xlabel("labels per sample")
    ax.set_ylabel("count")


def plot_label_concurrence(df_label, normalize=True, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    cooc = df_label.T.dot(df_label)
    if normalize:
        diag = np.diag(cooc)
        union = diag[:, None] + diag[None, :] - cooc
        cooc /= union

    sns.heatmap(cooc, cmap="viridis", ax=ax)
    ax.set_title("Label co-occurrence matrix" + " (normalized)" if normalize else "")


def plot_waveform(audio, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    librosa.display.waveshow(audio, sr=SR, ax=ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title if title is not None else "Waveform")

    return ax


def plot_cepstrum_pipeline(audio):

    X = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), d=1 / SR)

    pos_idx = freqs >= 0
    freqs = freqs[pos_idx]
    X = X[pos_idx]

    power_spectrum = np.abs(X) ** 2

    log_spectrum = np.log1p(power_spectrum)

    log_mag = np.log1p(np.abs(X))
    cepstrum = np.fft.ifft(log_mag).real

    quefrency = np.arange(len(cepstrum)) / SR

    plt.figure(figsize=(15, 10))

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


def plot_spectrogram(S_db, ax=None, title=None, hop_length=512, y_axis="linear"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    img = librosa.display.specshow(
        S_db,
        sr=SR,
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


def plot_mfcc(mfccs, ax=None, title=None, hop_length=512):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    img = librosa.display.specshow(
        mfccs,
        sr=SR,
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


def plot_chroma_stft(chroma, ax=None, title=None, hop_length=512, show=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    img = librosa.display.specshow(
        chroma,
        sr=SR,
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
