import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch, Rectangle
import seaborn as sns
import librosa.display
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .config import RESULTS_DIR

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

def savefig(fname):
    plt.savefig(RESULTS_DIR / "figures"/ f"{fname}.pdf")
    

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

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
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

    top_features = df.mean().nlargest(top_n).index
    df_top = df[top_features]

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
    # ax.set_title(f"Top {top_n} Features by Mean Importance Across All Classes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


def plot_importance_mean(df, top_n=20):
    top_features = df.mean().nlargest(top_n).index
    fig, ax = plt.subplots(figsize=(10, 5))
    df[top_features].mean().sort_values(ascending=True).plot.barh(ax=ax)
    # ax.set_title("Mean Feature Importance Across All Classes")
    ax.set_xlabel("Mean Importance")
    plt.tight_layout()


def plot_class_distribution(data_train, data_train_soundscapes, save_file=None):
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
    if save_file:
        fig.savefig(save_file)


def plot_species_distribution(
    counts_train, counts_sound, primary_to_class, class_colors=CLASS_COLORS, save_file=None
):

    df_order = pd.DataFrame(
        {
            "sound_count": counts_sound,
        }
    )

    df_order["class"] = df_order.index.map(primary_to_class)

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

    df_order = df_order.sort_values(
        by=["class", "sound_count"],
        ascending=[True, False],
    )

    species_order = df_order.index

    counts_train = counts_train.reindex(species_order)
    counts_sound = counts_sound.reindex(species_order)

    classes = [primary_to_class[label] for label in species_order]
    colors = [class_colors[c] for c in classes]

    fig, axes = plt.subplots(2, 1, figsize=(6, 4))

    counts_train.plot.bar(ax=axes[0], color=colors, width=0.6)

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

    axes[0].set_xticks([])
    axes[1].set_xticks([])
    legend_elements = [
        Patch(facecolor=color, label=cls) for cls, color in class_colors.items()
    ]

    fig.legend(handles=legend_elements)

    fig.tight_layout()
    if save_file:
        fig.savefig(save_file)
    plt.show()


def plot_corr_matrix(corr_matrix, save_file=None):
    plt.figure(figsize=(20, 16))

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        square=True,
    )

    plt.title("Correlation Matrix")
    if save_file:
        plt.savefig(save_file)
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
    X_pca, y, title=None, all_class_combinations=False, class_colors=CLASS_COLORS, save_file=None
):
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
                {"PC1": X_pca[:, 0], "PC2": X_pca[:, 1]},
                index=y.index,
            )
            .join(y)
            .melt(id_vars=["PC1", "PC2"], var_name="y_class", value_name="active")
            .query("active == 1")
            .drop(columns="active")
        )

    plt.figure(figsize=(10, 8))

    for cls in df_pca["y_class"].unique():
        subset = df_pca[df_pca["y_class"] == cls]

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            color=class_colors.get(cls, "gray") if not all_class_combinations else None,
            label=cls,
            alpha=0.5,
        )

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(title if title else "Feature Space Dimension Reduction")
    plt.legend()
    if save_file:
        plt.savefig(save_file)
    plt.show()


def _multilabel_inputs(y_true, y_proba):
    """Normalize y_true / y_proba to arrays and compute per-label AUC/AP.

    Returns (labels, y_true_arr, y_proba_arr, supported, aucs, aps).
    `supported` lists label indices that have both positive and negative examples.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    if hasattr(y_true, "columns"):
        labels = list(y_true.columns)
        y_true_arr = y_true.to_numpy().astype(int)
    else:
        labels = [str(i) for i in range(y_true.shape[1])]
        y_true_arr = np.asarray(y_true, dtype=int)

    y_proba_arr = np.asarray(y_proba)
    n = len(labels)

    supported = [j for j in range(n) if 0 < y_true_arr[:, j].sum() < len(y_true_arr)]
    aucs = {j: roc_auc_score(y_true_arr[:, j], y_proba_arr[:, j]) for j in supported}
    aps = {
        j: average_precision_score(y_true_arr[:, j], y_proba_arr[:, j])
        for j in supported
    }

    return labels, y_true_arr, y_proba_arr, supported, aucs, aps


def _small_multiples_grid(n, max_cols=3):
    """Return (nrows, ncols) for a small-multiples layout."""
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def plot_multilabel_roc_pr(y_true, y_proba):
    """ROC and Precision-Recall curves overlaid for every label."""
    from sklearn.metrics import roc_curve, precision_recall_curve

    labels, y_true_arr, y_proba_arr, supported, aucs, aps = _multilabel_inputs(
        y_true, y_proba
    )
    n = len(labels)
    cmap = plt.cm.get_cmap("tab10", n)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Per-label Curves", fontsize=13, fontweight="bold")

    for j in supported:
        fpr, tpr, _ = roc_curve(y_true_arr[:, j], y_proba_arr[:, j])
        ax_roc.plot(
            fpr, tpr, color=cmap(j), lw=1.8, label=f"{labels[j]}  AUC={aucs[j]:.2f}"
        )
        prec, rec, _ = precision_recall_curve(y_true_arr[:, j], y_proba_arr[:, j])
        ax_pr.plot(
            rec, prec, color=cmap(j), lw=1.8, label=f"{labels[j]}  AP={aps[j]:.2f}"
        )

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax_roc.set(
        xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves"
    )
    ax_roc.legend(fontsize=8)

    for j in supported:
        ax_pr.axhline(y_true_arr[:, j].mean(), color=cmap(j), lw=0.7, ls=":", alpha=0.5)
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax_pr.legend(fontsize=8)

    fig.tight_layout()
    plt.show()


def plot_multilabel_score_distributions(y_true, y_proba, threshold=0.5, max_labels=12):
    """Per-label histograms of predicted scores split by true label."""
    labels, y_true_arr, y_proba_arr, *_ = _multilabel_inputs(y_true, y_proba)
    show_n = min(len(labels), max_labels)
    nrows, ncols = _small_multiples_grid(show_n)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False
    )
    fig.suptitle(
        "Score Distributions (positive vs. negative)", fontsize=13, fontweight="bold"
    )

    bins = np.linspace(0, 1, 30)
    for idx in range(show_n):
        ax = axes[idx // ncols][idx % ncols]
        neg = y_proba_arr[y_true_arr[:, idx] == 0, idx]
        pos = y_proba_arr[y_true_arr[:, idx] == 1, idx]
        ax.hist(
            neg,
            bins=bins,
            density=True,
            alpha=0.65,
            color="tab:blue",
            label=f"y=0  (n={len(neg)})",
        )
        ax.hist(
            pos,
            bins=bins,
            density=True,
            alpha=0.65,
            color="tab:orange",
            label=f"y=1  (n={len(pos)})",
        )
        ax.axvline(threshold, color="crimson", lw=1.4, ls="--")
        ax.set_title(labels[idx], fontsize=10)
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Density")
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(show_n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_multilabel_confusion_breakdown(y_true, y_proba, threshold=0.5):
    """Stacked TP/FP/FN/TN bar chart per label with precision & recall annotations."""
    labels, y_true_arr, y_proba_arr, *_ = _multilabel_inputs(y_true, y_proba)
    n = len(labels)
    y_pred_arr = (y_proba_arr >= threshold).astype(int)

    tp = (y_pred_arr & y_true_arr).sum(axis=0)
    fp = (y_pred_arr & (1 - y_true_arr)).sum(axis=0)
    fn = ((1 - y_pred_arr) & y_true_arr).sum(axis=0)
    tn = ((1 - y_pred_arr) & (1 - y_true_arr)).sum(axis=0)

    x, w = np.arange(n), 0.55
    fig, ax = plt.subplots(figsize=(max(8, n * 1.3), 4.5))
    ax.bar(x, tp, w, label="TP", color="tab:green")
    ax.bar(x, fp, w, bottom=tp, label="FP", color="tab:red")
    ax.bar(x, fn, w, bottom=tp + fp, label="FN", color="tab:orange")
    ax.bar(x, tn, w, bottom=tp + fp + fn, label="TN", color="tab:blue", alpha=0.35)

    for xi, (t, f, fn_) in enumerate(zip(tp, fp, fn)):
        prec_ = t / (t + f) if (t + f) > 0 else float("nan")
        rec_ = t / (t + fn_) if (t + fn_) > 0 else float("nan")
        # ax.text(
        #     xi,
        #     tp[xi] + fp[xi] + fn[xi] + tn[xi] + 1,
        #     f"P={prec_:.2f}\nR={rec_:.2f}",
        #     ha="center",
        #     va="bottom",
        #     fontsize=7.5,
        # )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    # ax.set_title(
    #     f"Prediction Breakdown per Label  (threshold={threshold})",
    #     fontsize=12,
    #     fontweight="bold",
    # )
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    # plt.show()


def plot_multilabel_calibration(y_true, y_proba, max_labels=12):
    """Reliability diagrams (calibration curves) per label."""
    from sklearn.calibration import calibration_curve

    labels, y_true_arr, y_proba_arr, supported, *_ = _multilabel_inputs(y_true, y_proba)
    show_n = min(len(labels), max_labels)
    nrows, ncols = _small_multiples_grid(show_n)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False
    )
    fig.suptitle("Calibration (Reliability Diagrams)", fontsize=13, fontweight="bold")

    for idx in range(show_n):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        if idx in supported and y_true_arr[:, idx].sum() >= 5:
            n_bins = min(10, max(3, int(y_true_arr[:, idx].sum() // 3)))
            try:
                frac, mean_pred = calibration_curve(
                    y_true_arr[:, idx],
                    y_proba_arr[:, idx],
                    n_bins=n_bins,
                    strategy="quantile",
                )
                ax.plot(mean_pred, frac, "o-", color="tab:blue", ms=5, lw=1.5)
                ax.fill_between(
                    mean_pred, frac, mean_pred, alpha=0.15, color="tab:blue"
                )
            except ValueError:
                ax.text(
                    0.5,
                    0.5,
                    "too few samples",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="gray",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "no positive samples",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="gray",
            )
        ax.set_title(labels[idx], fontsize=10)
        ax.set(
            xlabel="Mean predicted score",
            ylabel="Fraction of positives",
            xlim=(0, 1),
            ylim=(0, 1),
        )

    for idx in range(show_n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_multilabel_errors(y_true, y_proba, threshold=0.5, max_labels_grid=12):
    """Full multilabel error inspection: ROC/PR curves, score distributions,
    confusion breakdown, and calibration diagrams."""
    plot_multilabel_roc_pr(y_true, y_proba)
    plot_multilabel_score_distributions(
        y_true, y_proba, threshold=threshold, max_labels=max_labels_grid
    )
    plot_multilabel_confusion_breakdown(y_true, y_proba, threshold=threshold)
    plot_multilabel_calibration(y_true, y_proba, max_labels=max_labels_grid)


# Many-label (200+) aggregate views — no per-label subplots


def plot_multilabel_metric_distribution(y_true, y_proba, bins=20):
    """Histograms of per-label AUC-ROC and AP for the many-label case."""
    labels, _, _, supported, aucs, aps = _multilabel_inputs(y_true, y_proba)

    auc_vals = np.array([aucs[j] for j in supported])
    ap_vals = np.array([aps[j] for j in supported])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Per-label metric distribution  ({len(supported)} labels with both classes)",
        fontsize=12,
        fontweight="bold",
    )

    for ax_, vals, metric, color in [
        (ax1, auc_vals, "AUC-ROC", "tab:blue"),
        (ax2, ap_vals, "AP", "tab:orange"),
    ]:
        ax_.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor="white")
        ax_.axvline(
            np.mean(vals),
            color="black",
            lw=1.5,
            ls="--",
            label=f"mean={np.mean(vals):.3f}",
        )
        ax_.axvline(
            np.median(vals),
            color="crimson",
            lw=1.5,
            ls=":",
            label=f"median={np.median(vals):.3f}",
        )
        ax_.set(
            xlabel=metric, ylabel="Number of labels", title=f"{metric} distribution"
        )
        ax_.legend(fontsize=9)

    fig.tight_layout()
    plt.show()


def plot_multilabel_metric_ranked(y_true, y_proba, metric="auc", top_n=20, bottom_n=20):
    """Horizontal bar chart of top-N best and bottom-N worst labels by AUC or AP."""
    labels, _, _, supported, aucs, aps = _multilabel_inputs(y_true, y_proba)

    scores = aucs if metric.lower() == "auc" else aps
    score_series = pd.Series({labels[j]: scores[j] for j in supported}).sort_values(
        ascending=False
    )

    n_total = len(score_series)
    actual_top = min(top_n, n_total)
    actual_bot = min(bottom_n, n_total - actual_top)

    # Sort each group ascending so the extreme values are at the chart edges
    top = score_series.head(actual_top).sort_values(ascending=True)
    bot = score_series.tail(actual_bot).sort_values(ascending=True)
    show = pd.concat([bot, top])

    colors = ["tab:red"] * len(bot) + ["tab:green"] * len(top)

    fig, ax = plt.subplots(figsize=(8, max(4, len(show) * 0.32)))
    ax.barh(range(len(show)), show.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show.index, fontsize=8)

    sep = len(bot) - 0.5
    ax.axhline(sep, color="gray", lw=1.5, ls="--", alpha=0.8)
    ax.text(
        show.min(),
        sep + 0.15,
        f"↑ worst {actual_bot}   |   best {actual_top} ↑",
        fontsize=8,
        color="gray",
        va="bottom",
    )

    mean_val = score_series.mean()
    ax.axvline(
        mean_val, color="black", lw=1.2, ls=":", alpha=0.7, label=f"mean={mean_val:.3f}"
    )

    metric_name = "AUC-ROC" if metric.lower() == "auc" else "AP"
    ax.set(
        xlabel=metric_name,
        title=f"Top-{actual_top} / Bottom-{actual_bot} labels by {metric_name}"
        f"  (n={n_total} total)",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_multilabel_pr_scatter(y_true, y_proba, threshold=0.5, annotate_n=10):
    """Precision vs Recall scatter — one point per label, sized by support, coloured by F1."""
    labels, y_true_arr, y_proba_arr, supported, *_ = _multilabel_inputs(y_true, y_proba)
    y_pred_arr = (y_proba_arr >= threshold).astype(int)

    records = []
    for j in supported:
        tp = int((y_pred_arr[:, j] & y_true_arr[:, j]).sum())
        fp = int((y_pred_arr[:, j] & (1 - y_true_arr[:, j])).sum())
        fn = int(((1 - y_pred_arr[:, j]) & y_true_arr[:, j]).sum())
        support = int(y_true_arr[:, j].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        records.append(
            dict(label=labels[j], precision=prec, recall=rec, f1=f1, support=support)
        )

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Iso-F1 reference contours
    for f1_val in [0.2, 0.4, 0.6, 0.8]:
        rec_range = np.linspace(f1_val / (2 - f1_val) + 1e-4, 1, 200)
        prec_range = f1_val * rec_range / (2 * rec_range - f1_val)
        mask = (prec_range >= 0) & (prec_range <= 1)
        ax.plot(
            rec_range[mask], prec_range[mask], color="gray", lw=0.8, ls="--", alpha=0.35
        )
        if mask.sum() > 10:
            mid = mask.sum() // 2
            ax.text(
                rec_range[mask][mid],
                prec_range[mask][mid],
                f"F1={f1_val}",
                fontsize=7,
                color="gray",
                alpha=0.6,
            )

    scatter = ax.scatter(
        df["recall"],
        df["precision"],
        c=df["f1"],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=np.clip(df["support"], 5, 300) * 0.9,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("F1 score")

    for _, row in df.nsmallest(annotate_n, "f1").iterrows():
        ax.annotate(
            row["label"],
            (row["recall"], row["precision"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="darkred",
            alpha=0.85,
        )

    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(-0.05, 1.05),
        ylim=(-0.05, 1.05),
        title=f"Precision–Recall per label  (threshold={threshold},"
        f" n={len(df)} labels)",
    )
    fig.tight_layout()
    plt.show()


def plot_multilabel_calibration_summary(y_true, y_proba, bins=10):
    """ECE distribution across labels — compact calibration view for 200+ labels."""
    labels, y_true_arr, y_proba_arr, supported, *_ = _multilabel_inputs(y_true, y_proba)

    eces = []
    bin_edges = np.linspace(0, 1, bins + 1)
    for j in supported:
        p = y_proba_arr[:, j]
        t = y_true_arr[:, j].astype(float)
        n = len(p)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            ece += mask.sum() / n * abs(p[mask].mean() - t[mask].mean())
        eces.append(ece)

    eces = np.array(eces)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(eces, bins=20, color="tab:purple", alpha=0.8, edgecolor="white")
    ax.axvline(
        np.mean(eces),
        color="black",
        lw=1.5,
        ls="--",
        label=f"mean ECE={np.mean(eces):.3f}",
    )
    ax.axvline(
        np.median(eces),
        color="crimson",
        lw=1.5,
        ls=":",
        label=f"median ECE={np.median(eces):.3f}",
    )
    ax.set(
        xlabel="ECE",
        ylabel="Number of labels",
        title=f"Expected Calibration Error distribution  ({len(eces)} labels)",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_multilabel_errors_large(
    y_true, y_proba, threshold=0.5, top_n=20, annotate_n=10
):
    """Aggregate multilabel error analysis (200+ labels): metric histograms, ranked bars, PR scatter, ECE."""
    plot_multilabel_metric_distribution(y_true, y_proba)
    plot_multilabel_metric_ranked(
        y_true, y_proba, metric="auc", top_n=top_n, bottom_n=top_n
    )
    plot_multilabel_pr_scatter(
        y_true, y_proba, threshold=threshold, annotate_n=annotate_n
    )
    plot_multilabel_calibration_summary(y_true, y_proba)


def plot_corr_cirle(X, pca):
    features = X.columns
    loadings = pca.components_.T
    loadings = loadings * np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots(figsize=(10, 10))

    circle = plt.Circle((0, 0), 1, color="gray", fill=False)
    ax.add_artist(circle)

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


def plot_location_map(
    data_train,
    class_colors=CLASS_COLORS,
    ax=None,
    save_file=None,
):
    # Create figure/axes only if not provided
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    # Add map background
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, zorder=1)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    longitude = data_train["metadata"]["longitude"]
    latitude = data_train["metadata"]["latitude"]

    classes = data_train["y_class"].idxmax(axis=1)

    for cls in classes.unique()[::-1]:
        idx = classes[classes == cls].index

        ax.scatter(
            longitude.loc[idx],
            latitude.loc[idx],
            alpha=0.7,
            c=class_colors[cls],
            s=15,
            zorder=3,
            transform=ccrs.PlateCarree(),
        )

    ax.set_title("Recording Locations for Classes")

    # World extent
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # Pantanal rectangle
    rect = Rectangle(
        (-57.6, -21.6),
        1.7,
        5.1,
        linewidth=2,
        edgecolor="yellow",
        facecolor="none",
        zorder=4,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(rect)

    fig.tight_layout()

    if save_file:
        fig.savefig(save_file)

    if created_fig:
        plt.show()

    return ax


def plot_location_map_species(
    data_train,
    species,
    ax=None,
    save_file=None,
):
    # Create figure/axes only if not provided
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    # Add map background
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, zorder=1)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    longitude = data_train["metadata"]["longitude"]
    latitude = data_train["metadata"]["latitude"]

    species_labels = data_train["y_primary"].idxmax(axis=1)

    # Select requested species
    idx = species_labels[species_labels == species].index

    ax.scatter(
        longitude.loc[idx],
        latitude.loc[idx],
        alpha=0.7,
        s=15,
        zorder=3,
        transform=ccrs.PlateCarree(),
    )

    ax.set_title(f"Recording Locations for Species: {species}")

    # World extent
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # Pantanal rectangle
    rect = Rectangle(
        (-57.6, -21.6),
        1.7,
        5.1,
        linewidth=2,
        edgecolor="yellow",
        facecolor="none",
        zorder=4,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(rect)

    fig.tight_layout()

    if save_file:
        fig.savefig(save_file)

    if created_fig:
        plt.show()

    return ax

def plot_scree(explained_variance_ratio, title=None, ax=None, save_file=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    bars = ax.bar(range(1, n + 1), explained_variance_ratio, label="Individual")
    ax.plot(range(1, n + 1), cumulative, marker="o", color="tab:red", label="Cumulative")

    for bar, ratio in zip(bars, explained_variance_ratio):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{ratio:.1%}",
            ha="center", va="bottom", fontsize=7,
        )
    ax.axhline(0.95, linestyle="--", color="gray", linewidth=0.8)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(title if title else "Scree Plot")
    ax.legend()

    if save_file:
        plt.savefig(save_file)

    return ax
