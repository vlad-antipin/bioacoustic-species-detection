"""
Collect Perch embeddings for train audio and train soundscapes.

Usage:
    nohup uv run python scripts/collect_embeddings.py > local/embedding_collection.log 2>&1 &

Monitor progress:
    tail -f local/embedding_collection.log

Output saved to results/features/embeddings/:
    data_train.pkl, data_train_soundscapes.pkl
    Each dict has keys: X (embeddings DataFrame), y_primary, y_class, metadata.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from perch_hoplite.zoo import model_configs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bioacoustics.config import SR
from bioacoustics.data import load_audio, load_metadata, is_soundscape, save_results
from bioacoustics.preprocessing import get_labels

MODEL_NAME = "perch_v2_cpu"   # use 'perch_v2' if a GPU is available
CHUNK_SECONDS = 5             # perch expects 5-second waveforms at 32 kHz
NB_EXAMPLES = None            # None = full dataset
SAVE_DATA = True
OUT_DIR = "features/embeddings"

def chunk_audio(audio: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    """Split waveform into non-overlapping chunks, zero-padding the last one."""
    chunks = []
    for start in range(0, max(len(audio), chunk_samples), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)
        if start + chunk_samples >= len(audio):
            break
    return chunks


def embed_audio(model, audio: np.ndarray, chunk_samples: int) -> np.ndarray:
    """Embed a waveform by mean-pooling per-chunk embeddings."""
    chunks = chunk_audio(audio, chunk_samples)
    embeddings = []
    for chunk in chunks:
        outputs = model.embed(chunk.astype(np.float32))
        emb = outputs.embeddings
        embeddings.append(emb.reshape(-1, emb.shape[-1]).mean(axis=0))  # collapse all leading axes → (dim,)
    return np.stack(embeddings).mean(axis=0)


def collect_embeddings(df: pd.DataFrame, df_taxonomy, model, chunk_samples: int):
    y_class, y_primary = get_labels(df, df_taxonomy)

    rows = []
    for _, sample in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
        audio = load_audio(sample)
        try:
            emb = embed_audio(model, audio, chunk_samples)
        except Exception as exc:
            print(f"  WARNING: failed for {sample.name!r}: {exc}", flush=True)
            emb = np.full(rows[0].shape if rows else (1024,), np.nan)
        rows.append(emb)

    X = pd.DataFrame(rows, index=df.index)

    mask = ~X.isna().all(axis=1)
    X = X[mask]
    y_primary = y_primary[mask]
    y_class = y_class[mask]

    metadata = None if is_soundscape(df) else df.drop(columns=["primary_label"])
    return {"X": X, "y_primary": y_primary, "y_class": y_class, "metadata": metadata}


start = time.time()
chunk_samples = CHUNK_SECONDS * SR

print("Loading metadata...", flush=True)
df_train, df_train_soundscapes, df_taxonomy = load_metadata()
print(f"  train audio:       {len(df_train)} recordings", flush=True)
print(f"  train soundscapes: {len(df_train_soundscapes)} segments", flush=True)

if NB_EXAMPLES is not None:
    rng = np.random.default_rng(42)
    train_idx = rng.integers(0, len(df_train), NB_EXAMPLES)
    soundscapes_idx = rng.integers(0, len(df_train_soundscapes), NB_EXAMPLES)
    df_train = df_train.iloc[train_idx]
    df_train_soundscapes = df_train_soundscapes.iloc[soundscapes_idx]

print(f"\nLoading Perch model '{MODEL_NAME}'...", flush=True)

model = model_configs.load_model_by_name(MODEL_NAME)
print(f"  model loaded in {time.time() - start:.1f}s", flush=True)

# getting embeddings for all train audio is too slow
print("Skipping train audio")
# print("\n[1/2] Embedding train audio...", flush=True)
# data_train = collect_embeddings(df_train, df_taxonomy, model, chunk_samples)
# elapsed = time.time() - start
# print(f"  done in {elapsed/60:.1f} min — shape: {data_train['X'].shape}", flush=True)

print("\n[2/2] Embedding train soundscapes...", flush=True)
data_train_soundscapes = collect_embeddings(df_train_soundscapes, df_taxonomy, model, chunk_samples)
elapsed = time.time() - start
print(f"  done in {elapsed/60:.1f} min — shape: {data_train_soundscapes['X'].shape}", flush=True)

if SAVE_DATA:
    print("\nSaving results...", flush=True)
    # save_results(data_train, "data_train", OUT_DIR)
    save_results(data_train_soundscapes, "data_train_soundscapes", OUT_DIR)
    print(f"  saved to results/{OUT_DIR}/", flush=True)

print(f"\nTotal time: {(time.time() - start)/60:.1f} min", flush=True)
