import warnings
from enum import Enum, auto

import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split


class FitMode(Enum):
    TRAIN_TO_SOUNDSCAPE = auto()
    MIX_TO_SOUNDSCAPE = auto()
    SOUNDSCAPE_TO_SOUNDSCAPE = auto()
    TRAIN_TO_TRAIN = auto()
    MIX_TO_MIX = auto()


class LabelType(Enum):
    CLASS = "y_class"
    SPECIES = "y_primary"


def ignore_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.multiclass")

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")


def split_soundscapes(data_soundscapes, label_type, test_size=0.2, random_state=42):
    """Split soundscapes so that train and validation sets don't contain samples from the same file"""
    X = data_soundscapes["X"]
    y = data_soundscapes[label_type.value]
    # First level of MultiIndex is file identifier
    file_ids = X.index.get_level_values(0).unique()

    train_files, test_files = train_test_split(
        file_ids, test_size=test_size, random_state=random_state
    )

    train_mask = X.index.get_level_values(0).isin(train_files)
    test_mask = X.index.get_level_values(0).isin(test_files)
    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    return X_train, X_test, y_train, y_test


def split_data(
    data_train, data_soundscapes, fit_mode, label_type, test_size=0.2, random_state=42
):

    if fit_mode == FitMode.TRAIN_TO_TRAIN:
        X = data_train["X"]
        y = data_train[label_type.value]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    elif fit_mode == FitMode.SOUNDSCAPE_TO_SOUNDSCAPE:
        X_train, X_test, y_train, y_test = split_soundscapes(
            data_soundscapes, label_type, test_size=test_size, random_state=random_state
        )
    elif fit_mode == FitMode.TRAIN_TO_SOUNDSCAPE:
        X_train = data_train["X"]
        X_test = data_soundscapes["X"]
        y_train = data_train[label_type.value]
        y_test = data_soundscapes[label_type.value]

    elif fit_mode == FitMode.MIX_TO_SOUNDSCAPE:
        # TODO: smarter mixing

        # Split soundscapes into a train portion and a held-out test portion
        X_soundscape_train, X_test, y_soundscape_train, y_test = split_soundscapes(
            data_soundscapes, label_type, test_size=test_size, random_state=random_state
        )

        # Mix the soundscape train portion with the regular train data, shuffle
        X_mixed = pd.concat([data_train["X"], X_soundscape_train]).sample(
            frac=1, random_state=random_state
        )
        y_mixed = pd.concat([data_train[label_type.value], y_soundscape_train])
        y_mixed = y_mixed.loc[X_mixed.index]

        X_train = X_mixed
        y_train = y_mixed
    elif fit_mode == FitMode.MIX_TO_MIX:
        X_mixed = pd.concat([data_train["X"], data_soundscapes["X"]]).sample(
            frac=1, random_state=random_state
        )
        y_mixed = pd.concat(
            [data_train[label_type.value], data_soundscapes[label_type.value]]
        )
        y_mixed = y_mixed.loc[X_mixed.index]  # align by index

        X_train, X_test, y_train, y_test = train_test_split(
            X_mixed, y_mixed, test_size=test_size, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown fit mode: {fit_mode}")

    return X_train, X_test, y_train, y_test
