from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from sklearn.preprocessing import MultiLabelBinarizer

DATA_DIR = Path("../data")
TRAIN_METADATA_FILE = Path("train.csv")
TRAIN_SOUNDSCAPES_METADATA_FILE = Path("train_soundscapes_labels.csv")
TAXONOMY_FILE = Path("taxonomy.csv")

TRAIN_AUDIO_DIR = Path("train_audio")
TRAIN_SOUNDSCAPES_AUDIO_DIR = Path("train_soundscapes")

TEST_SOUNDSCAPES_AUDIO_DIR = Path("test_soundscapes")

SR = 35000  # 32kHz


def load_metadata():
    df_train = pd.read_csv(DATA_DIR / TRAIN_METADATA_FILE)
    df_train_soundscapes = pd.read_csv(DATA_DIR / TRAIN_SOUNDSCAPES_METADATA_FILE)
    df_taxonomy = pd.read_csv(DATA_DIR / TAXONOMY_FILE)

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


def is_soundscape(row: pd.Series | pd.DataFrame):
    assert "filename" in row
    return "start" in row and "end" in row


def load_audio(row: pd.Series, train=True):
    if is_soundscape(row):
        return load_soundscape(row["filename"], row["start"], row["end"], train=train)
    else:
        return load_train_audio(row["filename"])


def get_labels(df, df_taxonomy):
    
    class_encoder = MultiLabelBinarizer()
    primary_encoder = MultiLabelBinarizer()

    class_encoder.fit(df_taxonomy["class_name"].apply(lambda x: [x]))
    primary_encoder.fit(df_taxonomy["primary_label"].apply(lambda x: [x]))

    primary_to_class = df_taxonomy.set_index("primary_label")["class_name"]
    
    # TODO: and secondary labels? - completely ignore them?
    if is_soundscape(df):
        y_class = class_encoder.transform(
            df["primary_label"]
            .apply(lambda x: x.split(";"))
            .apply(
                lambda x: list({primary_to_class[primary_label] for primary_label in x})
            )
        )

        y_primary = primary_encoder.transform(
            df["primary_label"].apply(lambda x: x.split(";"))
        )
    else:
        y_class = class_encoder.transform(df["class_name"].apply(lambda x: [x]))
        y_primary = primary_encoder.transform(df["primary_label"].apply(lambda x: [x]))

    return y_class, y_primary
