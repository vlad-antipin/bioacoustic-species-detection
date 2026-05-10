from pathlib import Path

import pandas as pd
import librosa

import pickle

from .config import (
    DATA_DIR,
    TRAIN_METADATA_FILE,
    TRAIN_SOUNDSCAPES_METADATA_FILE,
    TAXONOMY_FILE,
    TRAIN_AUDIO_DIR,
    TRAIN_SOUNDSCAPES_AUDIO_DIR,
    TEST_SOUNDSCAPES_AUDIO_DIR,
    RESULTS_DIR,
    SR
)


def load_metadata():
    df_train = pd.read_csv(DATA_DIR / TRAIN_METADATA_FILE)
    df_train_soundscapes = pd.read_csv(DATA_DIR / TRAIN_SOUNDSCAPES_METADATA_FILE)
    df_taxonomy = pd.read_csv(DATA_DIR / TAXONOMY_FILE)
    
    df_train.set_index(["filename"], inplace=True)
    df_train_soundscapes.set_index(["filename", "start", "end"], inplace=True)

    return df_train, df_train_soundscapes, df_taxonomy


def load_train_audio(filename):
    audio, _ = librosa.load(DATA_DIR / TRAIN_AUDIO_DIR / filename, sr=SR)
    return audio


def hms_to_seconds(time_str):
    hms = time_str.split(":")
    return sum([int(time) * secs for time, secs in zip(hms, [3600, 60, 1])])


def load_soundscape(filename, start: str, end: str, train=True):
    soundscapes_dir = (
        TRAIN_SOUNDSCAPES_AUDIO_DIR if train else TEST_SOUNDSCAPES_AUDIO_DIR
    )
    start_seconds = hms_to_seconds(start)
    end_seconds = hms_to_seconds(end)
    audio, _ = librosa.load(
        DATA_DIR / soundscapes_dir / filename,
        sr=SR,
        offset=start_seconds,
        duration=end_seconds - start_seconds,
    )

    return audio


def is_soundscape(data: pd.Series | pd.DataFrame): #type: ignore
    if isinstance(data, pd.DataFrame):
        row: pd.Series   = data.iloc[0]
    elif isinstance(data, pd.Series) :
        row = data
    else:
        raise TypeError("Not supported format")
    return isinstance(row.name, tuple)
        


def load_audio(row: pd.Series, train=True):
    if is_soundscape(row):
        return load_soundscape(*row.name, train=train)  #type: ignore
    else:
        return load_train_audio(row.name)


def save_results(result, out_dir, fname):
    with open(RESULTS_DIR / out_dir / f"{fname}.pkl", "wb") as file:
        pickle.dump(result, file)


def load_results(out_dir, fname):
    with open(RESULTS_DIR / out_dir / f"{fname}.pkl", "rb") as file:
        return pickle.load(file)