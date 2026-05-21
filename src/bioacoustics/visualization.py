import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import seaborn as sns
import librosa.display

from .config import SR, HOP_LENGTH

CLASS_COLORS = {
    "Amphibia": "tab:green",
    "Aves": "tab:blue",
    "Mammalia": "tab:red",
    "Reptilia": "tab:orange",
    "Insecta": "tab:purple",
}


def set_style():
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 100,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # "axes.grid": True,
            # "grid.alpha": 0.2,
            "lines.linewidth": 1.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def plot_label_frequency(df_label, log=True, ax=None):
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

    label_freq = df_label.mean()
    label_freq.plot(kind="bar", ax=ax)
    if log:
        ax.set_yscale("log")
    ax.set_title("Label frequency" + " (log scale)" if log else "")
    ax.set_ylabel("P(label = 1)")
    ax.tick_params(axis="x", labelrotation=45)


def plot_active_labels(df_label, ax=None):
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

    row_counts = df_label.sum(axis=1)
    ax.hist(row_counts, bins=range(int(row_counts.max()) + 2), align="left")
    ax.set_title("Number of active labels per sample")
    ax.set_xlabel("labels per sample")
    ax.set_ylabel("count")


def plot_label_concurrence(df_label, normalize=True, ax=None):
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

    cooc = df_label.T.dot(df_label)
    if normalize:
        diag = np.diag(cooc)
        union = diag[:, None] + diag[None, :] - cooc
        cooc /= union

    sns.heatmap(cooc, cmap="viridis", ax=ax)
    ax.set_title("Label co-occurrence matrix" + " (normalized)" if normalize else "")


def plot_waveform(audio, ax=None, title=None, sr=SR):
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

    librosa.display.waveshow(audio, sr=sr, alpha=0.5, ax=ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title if title is not None else "Waveform")

    return ax


def plot_autocorrelation(audio, ax=None, title=None, sr=SR):
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

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


def plot_spectrogram(
    S_db, ax=None, title=None, sr=SR, hop_length=HOP_LENGTH, y_axis="linear"
):
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

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
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

    img = librosa.display.specshow(
        mfccs,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="frames",
        cmap="RdBu_r",
        vmin=np.percentile(mfccs[1:], 2),
        vmax=np.percentile(mfccs[1:], 98),
        ax=ax,
    )

    ax.set_ylabel("MFCC Coefficient Index")
    ax.set_xlabel("Time (s)")
    ax.set_title(title if title is not None else "MFCC")

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Coefficient value")

    fig.tight_layout()
    return ax


def plot_chroma_stft(
    chroma, ax=None, title=None, sr=SR, hop_length=HOP_LENGTH, show=True
):
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

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


def plot_importance_heatmap(df, top_n=20):

    # Keep only top-N features by mean importance across classes

    top_features = df.mean().nlargest(top_n).index
    df_top = df[top_features]

    # Plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(df_top) * 0.12)))
    sns.heatmap(
        df_top,
        ax=ax,
        cmap="viridis",
        xticklabels=True,
        yticklabels=(len(df) < 10),
        cbar_kws={"label": "Feature Importance"},
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Class (estimator index)")
    ax.set_title(f"Top {top_n} Features by Mean Importance Across All Classes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_importance_mean(df, top_n=20):
    top_features = df.mean().nlargest(top_n).index
    fig, ax = plt.subplots(figsize=(10, 5))
    df[top_features].mean().sort_values(ascending=True).plot.barh(ax=ax)
    ax.set_title("Mean Feature Importance Across All Classes")
    ax.set_xlabel("Mean Importance")
    plt.tight_layout()
    plt.show()


def plot_class_distribution(data_train, data_train_soundscapes):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    data_train["y_class"].sum().plot.bar(ax=axes[0])
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Class Distribution of Train Audio")
    axes[0].tick_params(axis="x", rotation=0)

    data_train_soundscapes["y_class"].sum().plot.bar(ax=axes[1])
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Class Distribution of Train Soundscapes")
    axes[1].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    plt.show()


def plot_species_distribution(
    data_train, data_train_soundscapes, primary_to_class, class_colors=CLASS_COLORS
):

    #
    # Counts
    #
    counts_train = data_train["y_primary"].sum()
    counts_sound = data_train_soundscapes["y_primary"].sum()

    #
    # Build dataframe using SOUND frequencies
    # to define the global ordering
    #
    df_order = pd.DataFrame(
        {
            "sound_count": counts_sound,
        }
    )

    df_order["class"] = df_order.index.map(primary_to_class)

    # same class order everywhere
    class_order = [
        "Amphibia",
        "Aves",
        "Mammalia",
        "Reptilia",
        "Insecta",
    ]

    df_order["class"] = pd.Categorical(
        df_order["class"],
        categories=class_order,
        ordered=True,
    )

    #
    # Sort:
    #   1. by class
    #   2. by soundscape frequency
    #
    df_order = df_order.sort_values(
        by=["class", "sound_count"],
        ascending=[True, False],
    )

    # final common species ordering
    species_order = df_order.index

    # Reindex both datasets with same ordering
    counts_train = counts_train.reindex(species_order)
    counts_sound = counts_sound.reindex(species_order)

    # Colors
    classes = [primary_to_class[label] for label in species_order]
    colors = [class_colors[c] for c in classes]

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(32, 8))

    counts_train.plot.bar(
        ax=axes[0],
        color=colors,
    )

    axes[0].set_xlabel("Species")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Class Distribution of Train Audio")

    counts_sound.plot.bar(
        ax=axes[1],
        color=colors,
    )

    axes[1].set_xlabel("Species")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Class Distribution of Train Soundscapes")

    # Legend
    legend_elements = [
        Patch(facecolor=color, label=cls) for cls, color in class_colors.items()
    ]

    fig.legend(handles=legend_elements)

    fig.tight_layout()
    plt.show()


def plot_corr_matrix(corr_matrix):
    plt.figure(figsize=(20, 16))

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        square=True,
    )

    plt.title("Correlation Matrix")
    plt.show()


def plot_feature_distribution(
    data,
    feature,
    figsize=(10, 5),
    jitter=True,
    alpha=0.5,
    point_size=3,
):
    x = data["X"][[feature]]
    y = data["y_class"]

    # Convert one-hot labels to long format
    df_plot = (
        x.join(y)
        .melt(
            id_vars=feature,
            var_name="category",
            value_name="active",
        )
        .query("active == 1")
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Boxplot
    sns.boxplot(
        data=df_plot,
        x="category",
        y=feature,
        ax=ax,
    )

    # Individual points
    sns.stripplot(
        data=df_plot,
        x="category",
        y=feature,
        ax=ax,
        jitter=jitter,
        alpha=alpha,
        size=point_size,
    )

    ax.set_title(f"Distribution of {feature} by category")
    ax.set_xlabel("Category")
    ax.set_ylabel(feature)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_dim_reduction(
    X_pca, y, all_class_combinations=False, class_colors=CLASS_COLORS
):
    # TODO: Investigate whether it overrides some of the class
    if all_class_combinations:
        y_labels = y.apply(
            lambda row: ", ".join(row.index[row == 1]),
            axis=1,
        )
        df_pca = pd.DataFrame(
            {
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1],
                "y_class": y_labels.values,
            }
        )
    else:
        df_pca = (
            pd.DataFrame(
                {
                    "PC1": X_pca[:, 0],
                    "PC2": X_pca[:, 1],
                },
                index=y.index,
            )
            .join(y)
            .melt(
                id_vars=["PC1", "PC2"],
                var_name="y_class",
                value_name="active",
            )
            .query("active == 1")
        )

    plt.figure(figsize=(10, 8))

    for cls in df_pca["y_class"].unique():
        subset = df_pca[df_pca["y_class"] == cls]

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            color=class_colors[cls] if not all_class_combinations else None,
            label=cls,
            alpha=0.5,
        )

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("Feature Space Dimension Reduction")
    plt.legend()
    plt.show()


def plot_corr_cirle(X, pca):
    # Feature names
    features = X.columns

    # Loadings
    loadings = pca.components_.T

    # Scale by explained variance
    loadings = loadings * np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, color="gray", fill=False)
    ax.add_artist(circle)

    # Draw arrows
    for i, feature in enumerate(features):
        x_vector = loadings[i, 0]
        y_vector = loadings[i, 1]

        ax.arrow(
            0,
            0,
            x_vector,
            y_vector,
            color="tab:blue",
            alpha=0.5,
            head_width=0.02,
        )

        ax.text(
            x_vector * 1.05,
            y_vector * 1.05,
            feature,
            fontsize=8,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Correlation Circle")

    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()
