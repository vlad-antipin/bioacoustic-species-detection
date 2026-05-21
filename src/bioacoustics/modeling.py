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

from scipy.ndimage import gaussian_filter1d

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


def split_soundscapes(
    data_soundscapes, test_size=0.2, random_state=42, rare_first=True
):
    """Split soundscapes so train/test don't share files.

    Can use a greedy rare-species-first strategy: files containing the rarest
    species are assigned to the test set first so that as many species as
    possible have at least one positive test example, avoiding NaN in macro
    metrics.  Remaining test slots are filled at random.
    """
    X = data_soundscapes["X"]
    y_class = data_soundscapes["y_class"]
    y_primary = data_soundscapes["y_primary"]

    if not rare_first:
        # First level of MultiIndex is file identifier
        file_ids = X.index.get_level_values(0).unique()

        train_files, test_files = train_test_split(
            file_ids, test_size=test_size, random_state=random_state
        )

        train_mask = X.index.get_level_values(0).isin(train_files)
        test_mask = X.index.get_level_values(0).isin(test_files)
    else:
        file_col = X.index.get_level_values(0)
        file_ids = file_col.unique()
        n_test = max(1, round(len(file_ids) * test_size))

        # Per-file species presence: (n_files, n_species) boolean DataFrame
        file_species = y_primary.groupby(level=0).any()

        # How many files each species appears in
        species_file_count = file_species.sum(axis=0)

        # Rarity score per file = minimum file-count across its species
        # (lower score = file contains rarer species → prioritise for test)
        def _rarity(fid):
            present = file_species.loc[fid]
            counts = species_file_count[present]
            return counts.min() if len(counts) > 0 else np.inf

        sorted_files = sorted(file_ids, key=_rarity)

        covered = set()
        test_files = []
        pool = []

        for fid in sorted_files:
            species_here = set(file_species.columns[file_species.loc[fid]])
            if len(test_files) < n_test and species_here - covered:
                test_files.append(fid)
                covered |= species_here
            else:
                pool.append(fid)

        # Fill remaining test slots at random
        rng = np.random.default_rng(random_state)
        rng.shuffle(pool)
        test_files.extend(pool[: n_test - len(test_files)])

        test_set = set(test_files)
        train_mask = ~file_col.isin(test_set)
        test_mask = file_col.isin(test_set)

    return (
        X.loc[train_mask],
        X.loc[test_mask],
        y_class.loc[train_mask],
        y_class.loc[test_mask],
        y_primary.loc[train_mask],
        y_primary.loc[test_mask],
    )


def split_data(
    data_train,
    data_soundscapes,
    fit_mode,
    test_size=0.2,
    random_state=42,
    rare_first=True,
):

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
            data_soundscapes,
            test_size=test_size,
            random_state=random_state,
            rare_first=rare_first,
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
            data_soundscapes,
            test_size=test_size,
            random_state=random_state,
            rare_first=rare_first,
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


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier that automatically sets scale_pos_weight from the binary labels OvR passes."""

    def fit(self, X, y, **kwargs):
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        self.set_params(scale_pos_weight=neg / pos if pos > 0 else 1.0)
        return super().fit(X, y, **kwargs)


def get_prediction_pipeline(classifier: Classifier):
    if classifier == Classifier.LR:
        clf = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            max_iter=1000,
            class_weight="balanced",
            l1_ratio=0.5,
        )
    elif classifier == Classifier.RF:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, class_weight="balanced"
        )
    elif classifier == Classifier.XGBOOST:
        clf = BalancedXGBClassifier(
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
        self.n_experts = n_experts
        self.child_masks = []
        for _ in range(n_experts):
            self.expert_pipelines.append(get_prediction_pipeline(classifier))

    def fit(self, X, y_parent, y_child):
        assert self.n_experts == len(y_parent.columns), (
            "Number of experts doesn't match the number of parent labels"
        )
        self.child_masks = []
        self.n_children = y_child.shape[1]
        for i, parent in enumerate(y_parent.columns):
            expert_mask = y_parent[parent] == 1
            # restrain only to species that actually belong to this parent class
            child_mask = y_child[expert_mask].any(axis=0)
            self.child_masks.append(child_mask)

            with tqdm_joblib(
                desc=f"Training expert for {parent.ljust(10)}", total=sum(child_mask)
            ):
                self.expert_pipelines[i].fit(
                    X[expert_mask], y_child[expert_mask].loc[:, child_mask]
                )

    def predict_proba(self, X, y_parent_proba):
        if isinstance(y_parent_proba, pd.DataFrame):
            y_parent_proba = y_parent_proba.values
        # mixture formula assumes that parent probabilities sum to 1 per sample
        y_parent_proba = y_parent_proba / y_parent_proba.sum(axis=1, keepdims=True)

        y_child_per_expert = np.zeros((self.n_experts, len(X), self.n_children))
        for i, expert_pipeline in enumerate(self.expert_pipelines):
            y_child_expert = expert_pipeline.predict_proba(X)
            if y_child_expert.shape[1] == 2:
                # only one child for a parent -> binary classification
                y_child_expert = y_child_expert[:, 1]
            y_child_per_expert[i, :, self.child_masks[i]] = y_child_expert.T

        y_child_mixture = (y_child_per_expert * y_parent_proba.T[..., None]).sum(axis=0)
        # TODO: correct proba normalization?
        return y_child_mixture

    def predict(self, X, y_parent_proba, threshold=0.5):
        # TODO: tune threshold
        y_proba = self.predict_proba(X, y_parent_proba)
        return (y_proba >= threshold).astype(int)


def get_feature_importance(
    pipeline,
    class_names=None,
    named_step="clf",
):
    model = pipeline.named_steps[named_step]
    feature_names = pipeline[:-1].get_feature_names_out()
    importances = np.array(
        [
            est.feature_importances_
            if hasattr(est, "feature_importances_")
            else est.coef_[0]
            for est in model.estimators_
            if hasattr(est, "feature_importances_") or hasattr(est, "coef_")
        ]
    )

    df = pd.DataFrame(importances, columns=feature_names)
    if class_names is not None and len(df) == len(class_names):
        df.index = class_names
    return df


def smooth_group(group, sigma=2):
    return group.apply(lambda col: gaussian_filter1d(col.values, sigma=sigma))


def smooth_proba(y, sigma=2):
    ''' Temporal Gaussian smoothing on probabilities'''
    y_smooth = (
        y.sort_index()
        .groupby(level=0, group_keys=True)
        .apply(lambda g: smooth_group(g, sigma=sigma))
    )
    return y_smooth
