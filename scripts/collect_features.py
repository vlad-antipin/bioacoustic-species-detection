"""
Run full feature extraction for train audio and train soundscapes.
Designed to be launched as a background process so the Jupyter kernel stays free.

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

start = time.time()
print("Loading metadata...", flush=True)
df_train, df_train_soundscapes, df_taxonomy = load_metadata()
print(f"  train audio:      {len(df_train)} recordings", flush=True)
print(f"  train soundscapes: {len(df_train_soundscapes)} segments", flush=True)

if NB_EXAMPLES is not None:
    import numpy as np
    np.random.seed(42)
    train_idx = np.random.randint(0, len(df_train), NB_EXAMPLES)
    train_soundscapes_idx = np.random.randint(0, len(df_train_soundscapes), NB_EXAMPLES)
else:
    train_idx = train_soundscapes_idx = None

# print("\n[1/2] Extracting features for train audio...", flush=True)
# data_train = prepare_data(df_train, df_taxonomy, sample_idx=train_idx)
# elapsed = time.time() - start
# print(f"  done in {elapsed/60:.1f} min — shape: {data_train['X'].shape}", flush=True)

print("\n[2/2] Extracting features for train soundscapes...", flush=True)
data_train_soundscapes = prepare_data(
    df_train_soundscapes, df_taxonomy, sample_idx=train_soundscapes_idx
)
elapsed = time.time() - start
print(f"  done in {elapsed/60:.1f} min — shape: {data_train_soundscapes['X'].shape}", flush=True)

if SAVE_DATA:
    print("\nSaving results...", flush=True)
    # save_results(data_train, "features/collected", "data_train")
    save_results(data_train_soundscapes, "features/collected", "data_train_soundscapes")
    print("  saved to results/features/collected/", flush=True)

print(f"\nTotal time: {(time.time() - start)/60:.1f} min", flush=True)
