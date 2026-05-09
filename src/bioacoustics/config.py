from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
TRAIN_METADATA_FILE = Path("train.csv")
TRAIN_SOUNDSCAPES_METADATA_FILE = Path("train_soundscapes_labels.csv")
TAXONOMY_FILE = Path("taxonomy.csv")

TRAIN_AUDIO_DIR = Path("train_audio")
TRAIN_SOUNDSCAPES_AUDIO_DIR = Path("train_soundscapes")

TEST_SOUNDSCAPES_AUDIO_DIR = Path("test_soundscapes")

RESULTS_DIR = ROOT / "results"

SR = 32000  # 32kHz

# STFT / Framing
# these two determine time-frequency resolution tradeoff:
# if n_fft large: better frequency resolution and worse temporal resolution
# if hop_length small: better temporal precision, but more computation and redundancy
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = N_FFT
WINDOW = "hann"
CENTER = True

# Derived quantities
FRAME_RATE = SR / HOP_LENGTH
FRAME_DURATION = HOP_LENGTH / SR
FFT_BIN_HZ = SR / N_FFT

# Mel representation
N_MELS = 80
FMIN = 0
FMAX = SR // 2
N_SELECTED_BANDS = 20

# MFCC
N_MFCC = 20

# Spectral rolloff
ROLLOFF_PERCENT = 0.85

# Spectral contrast
N_CONTRAST_BANDS = 6

# Statistics
PERCENTILES = [10, 25, 50, 75, 90]
PER_BAND_PERCENTILES = [10, 50, 90]

# Onset detection
ONSET_BACKTRACK = False

# Numerical stability
EPS = 1e-9

FRAME_KWARGS: Dict[str, Any] = dict(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window=WINDOW,
    center=CENTER,
)

STFT_KWARGS = dict(
    **FRAME_KWARGS,
)

MEL_KWARGS = dict(
    sr=SR,
    n_mels=N_MELS,
    fmin=FMIN,
    fmax=FMAX,
    **FRAME_KWARGS,
)

MFCC_KWARGS: Dict[str, Any] = dict(
    n_mfcc=N_MFCC,
    **MEL_KWARGS,
)

SPECTRAL_CENTROID_KWARGS: Dict[str, Any] = dict(
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
)

SPECTRAL_BANDWIDTH_KWARGS: Dict[str, Any] = dict(
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
)

SPECTRAL_ROLLOFF_KWARGS: Dict[str, Any] = dict(
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    roll_percent=ROLLOFF_PERCENT,
)

SPECTRAL_CONTRAST_KWARGS: Dict[str, Any] = dict(
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_bands=N_CONTRAST_BANDS,
)

SPECTRAL_FLATNESS_KWARGS: Dict[str, Any] = dict(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window=WINDOW,
    center=CENTER,
)

RMS_KWARGS: Dict[str, Any] = dict(
    frame_length=N_FFT,
    hop_length=HOP_LENGTH,
    center=CENTER,
)

ZCR_KWARGS: Dict[str, Any] = dict(
    frame_length=N_FFT,
    hop_length=HOP_LENGTH,
    center=CENTER,
)

CHROMA_STFT_KWARGS: Dict[str, Any] = dict(
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_chroma=12,
)

ONSET_KWARGS: Dict[str, Any] = dict(
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    backtrack=ONSET_BACKTRACK,
)

DELTA_KWARGS: Dict[str, Any] = dict(
    width=9,
    order=1,
    mode="interp",
)
