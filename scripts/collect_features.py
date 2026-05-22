"""
Run full feature extraction for train audio and train soundscapes.

Usage:
    nohup uv run python scripts/collect_features.py > local/feature_collection.log 2>&1 &

Monitor progress:
    tail -f local/feature_collection.log
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bioacoustics.data import load_metadata, save_results
from bioacoustics.preprocessing import prepare_data

NB_EXAMPLES = None  # None = full dataset
SAVE_DATA = True
COLLECT_TRAIN = False
COLLECT_SOUNDSCAPES = True

start = time.time()
print("Loading metadata...", flush=True)
df_train, df_train_soundscapes, df_taxonomy = load_metadata()
print(f"  train audio:      {len(df_train)} recordings", flush=True)
print(f"  train soundscapes: {len(df_train_soundscapes)} segments", flush=True)

if NB_EXAMPLES is not None:
    import numpy as np

    rng = np.random.default_rng(42)
    train_idx = rng.integers(0, len(df_train), NB_EXAMPLES) if COLLECT_TRAIN else None
    train_soundscapes_idx = (
        rng.integers(0, len(df_train_soundscapes), NB_EXAMPLES)
        if COLLECT_SOUNDSCAPES
        else None
    )
else:
    train_idx = train_soundscapes_idx = None

if COLLECT_TRAIN:
    print("\n[train] Extracting features for train audio...", flush=True)
    data_train = prepare_data(df_train, df_taxonomy, sample_idx=train_idx)
    elapsed = time.time() - start
    print(
        f"  done in {elapsed / 60:.1f} min — shape: {data_train['X'].shape}", flush=True
    )

if COLLECT_SOUNDSCAPES:
    print("\n[soundscapes] Extracting features for train soundscapes...", flush=True)
    data_train_soundscapes = prepare_data(
        df_train_soundscapes, df_taxonomy, sample_idx=train_soundscapes_idx
    )
    elapsed = time.time() - start
    print(
        f"  done in {elapsed / 60:.1f} min — shape: {data_train_soundscapes['X'].shape}",
        flush=True,
    )

if SAVE_DATA:
    print("\nSaving results...", flush=True)
    if COLLECT_TRAIN:
        save_results(data_train, "data_train", "features/collected")
    if COLLECT_SOUNDSCAPES:
        save_results(
            data_train_soundscapes, "data_train_soundscapes", "features/collected"
        )
    print("  saved to results/features/collected/", flush=True)

print(f"\nTotal time: {(time.time() - start) / 60:.1f} min", flush=True)
