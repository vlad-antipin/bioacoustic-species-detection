import warnings
from enum import Enum, auto

import pandas as pd
import numpy as np

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

from tqdm_joblib import tqdm_joblib


class FitMode(Enum):
    TRAIN_TO_SOUNDSCAPE = auto()
    MIX_TO_SOUNDSCAPE = auto()
    SOUNDSCAPE_TO_SOUNDSCAPE = auto()
    TRAIN_TO_TRAIN = auto()
    MIX_TO_MIX = auto()


class Classifier(Enum):
    RF = auto()
    LR = auto()
    XGBOOST = auto()


def ignore_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.multiclass")
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")


def split_soundscapes(data_soundscapes, test_size=0.2, random_state=42):
    """Split soundscapes so that train and validation sets don't contain samples from the same file"""
    X = data_soundscapes["X"]
    y_class = data_soundscapes["y_class"]
    y_primary = data_soundscapes["y_primary"]
    # First level of MultiIndex is file identifier
    file_ids = X.index.get_level_values(0).unique()

    train_files, test_files = train_test_split(
        file_ids, test_size=test_size, random_state=random_state
    )

    train_mask = X.index.get_level_values(0).isin(train_files)
    test_mask = X.index.get_level_values(0).isin(test_files)
    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    y_class_train = y_class.loc[train_mask]
    y_class_test = y_class.loc[test_mask]
    y_primary_train = y_primary.loc[train_mask]
    y_primary_test = y_primary.loc[test_mask]

    return X_train, X_test, y_class_train, y_class_test, y_primary_train, y_primary_test


def split_data(data_train, data_soundscapes, fit_mode, test_size=0.2, random_state=42):

    if fit_mode == FitMode.TRAIN_TO_TRAIN:
        X = data_train["X"]
        y_class = data_train["y_class"]
        y_primary = data_train["y_primary"]
        (
            X_train,
            X_test,
            y_class_train,
            y_class_test,
            y_primary_train,
            y_primary_test,
        ) = train_test_split(
            X, y_class, y_primary, test_size=test_size, random_state=random_state
        )
    elif fit_mode == FitMode.SOUNDSCAPE_TO_SOUNDSCAPE:
        (
            X_train,
            X_test,
            y_class_train,
            y_class_test,
            y_primary_train,
            y_primary_test,
        ) = split_soundscapes(
            data_soundscapes, test_size=test_size, random_state=random_state
        )
    elif fit_mode == FitMode.TRAIN_TO_SOUNDSCAPE:
        X_train = data_train["X"]
        X_test = data_soundscapes["X"]
        y_class_train = data_train["y_class"]
        y_class_test = data_soundscapes["y_class"]
        y_primary_train = data_train["y_primary"]
        y_primary_test = data_soundscapes["y_primary"]

    elif fit_mode == FitMode.MIX_TO_SOUNDSCAPE:
        # TODO: smarter mixing

        # Split soundscapes into a train portion and a held-out test portion
        (
            X_soundscape_train,
            X_test,
            y_class_soundscape_train,
            y_class_test,
            y_primary_soundscape_train,
            y_primary_test,
        ) = split_soundscapes(
            data_soundscapes, test_size=test_size, random_state=random_state
        )

        # Mix the soundscape train portion with the regular train data, shuffle
        X_mixed = pd.concat([data_train["X"], X_soundscape_train]).sample(
            frac=1, random_state=random_state
        )
        y_class_mixed = pd.concat([data_train["y_class"], y_class_soundscape_train])
        y_class_mixed = y_class_mixed.loc[X_mixed.index]

        y_primary_mixed = pd.concat(
            [data_train["y_primary"], y_primary_soundscape_train]
        )
        y_primary_mixed = y_primary_mixed.loc[X_mixed.index]

        X_train = X_mixed
        y_class_train = y_class_mixed
        y_primary_train = y_primary_mixed

    elif fit_mode == FitMode.MIX_TO_MIX:
        X_mixed = pd.concat([data_train["X"], data_soundscapes["X"]]).sample(
            frac=1, random_state=random_state
        )
        y_class_mixed = pd.concat([data_train["y_class"], data_soundscapes["y_class"]])
        y_class_mixed = y_class_mixed.loc[X_mixed.index]  # align by index

        y_primary_mixed = pd.concat(
            [data_train["y_primary"], data_soundscapes["y_primary"]]
        )
        y_primary_mixed = y_primary_mixed.loc[X_mixed.index]

        (
            X_train,
            X_test,
            y_class_train,
            y_class_test,
            y_primary_train,
            y_primary_test,
        ) = train_test_split(
            X_mixed,
            y_class_mixed,
            y_primary_mixed,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown fit mode: {fit_mode}")

    return X_train, X_test, y_class_train, y_class_test, y_primary_train, y_primary_test


def get_prediction_pipeline(classifier: Classifier):
    if classifier == Classifier.LR:
        clf = LogisticRegression(
            solver="liblinear", max_iter=1000, class_weight="balanced"
        )
    elif classifier == Classifier.RF:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, class_weight="balanced"
        )
    elif classifier == Classifier.XGBOOST:
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # otherwise MFCC and RMS are incomparable
            (
                "clf",
                OneVsRestClassifier(clf, n_jobs=-2),
            ),
        ]
    )
    return pipeline


class HierarchicalMixtureOfExperts:
    def __init__(self, n_experts, classifier) -> None:
        self.expert_pipelines = []
        for _ in range(n_experts):
            self.expert_pipelines.append(get_prediction_pipeline(classifier))
            self.n_experts = n_experts

    def fit(self, X, y_parent, y_child):
        assert self.n_experts == len(y_parent.columns), (
            "Number of experts doesn't match the number of parent labels"
        )
        self.n_children = y_child.shape[1]
        for i, parent in enumerate(y_parent.columns):
            expert_mask = y_parent[parent] == 1
            # TODO: restrain only to species that actually belong to this class
            with tqdm_joblib(
                desc=f"Training expert for {parent.ljust(10)}", total=self.n_children
            ):
                self.expert_pipelines[i].fit(X[expert_mask], y_child[expert_mask])

    def predict_proba(self, X, y_parent_proba):
        # TODO: check whether it's right
        y_child_per_expert = np.empty((self.n_experts, len(X), self.n_children))
        for i, expert_pipeline in enumerate(self.expert_pipelines):
            y_child_per_expert[i] = expert_pipeline.predict_proba(X)
        y_child_mixture = (y_child_per_expert * y_parent_proba.T[..., None]).sum(axis=0)
        # TODO: correct proba normalization
        return y_child_mixture

    def predict(self, X, y_parent_proba, threshold=0.5):
        y_proba = self.predict_proba(X, y_parent_proba)
        return (y_proba >= threshold).astype(int)
