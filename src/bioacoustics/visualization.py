import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display


def plot_spectrogram(S_db, frequencies, times, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    img = ax.imshow(
        S_db,
        aspect="auto",
        origin="lower",
        extent=(
            times[0],
            times[-1],
            frequencies[0],
            frequencies[-1],
        ),
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    fig.colorbar(img, ax=ax, format="%+2.0f dB", shrink=0.8)

    if title is not None:
        ax.set_title(title)
