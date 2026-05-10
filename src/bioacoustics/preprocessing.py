from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from tqdm.auto import tqdm

from .data import is_soundscape, load_audio
from .features import get_features


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

    y_class = pd.DataFrame(
        y_class,  # type: ignore
        columns=class_encoder.classes_,
        index=df.index,
    )

    y_primary = pd.DataFrame(
        y_primary,  # type: ignore
        columns=primary_encoder.classes_,
        index=df.index,
    )

    return y_class, y_primary


def prepare_data(df: pd.DataFrame, df_taxonomy, sample_idx=None):

    if sample_idx is not None:
        df = df.iloc[sample_idx]

    y_class, y_primary = get_labels(df, df_taxonomy)

    features = [
        get_features(load_audio(sample))
        for _, sample in tqdm(df.iterrows(), total=len(df), desc="Extracting features")
    ]
    X = pd.DataFrame(features, index=df.index)

    mask = ~X.isna().all(axis=1)

    X = X[mask]
    y_primary = y_primary[mask]
    y_class = y_class[mask]

    if is_soundscape(df):
        metadata = None
    else:
        metadata = df.drop(columns=["primary_label"])

    return {"X": X, "y_primary": y_primary, "y_class": y_class, "metadata": metadata}
