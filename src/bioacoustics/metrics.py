import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
    hamming_loss,
    label_ranking_average_precision_score,
)

import pandas as pd

from .modeling import smooth_proba


def evaluate_multilabel_model(
    model, X_test, y_test, y_parent_proba=None, smooth_sigma=0
) -> pd.DataFrame:
    """Evaluate a multilabel model and return results as a DataFrame.

    Macro metrics are computed twice when some classes have zero test support:
    once over all classes and once over supported classes only (zero-support
    classes make roc_auc_score return NaN and inflate F1 zeros unfairly).
    The BirdClef competition metric is macro ROC-AUC over supported classes.

    Returns a DataFrame indexed by metric name. Use .to_latex() for export.
    """
    if y_parent_proba is None:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    else:
        y_pred = model.predict(X_test, y_parent_proba)
        y_proba = model.predict_proba(X_test, y_parent_proba)

    if smooth_sigma:
        y_proba_df = pd.DataFrame(y_proba, index=X_test.index)
        y_proba = np.asarray(smooth_proba(y_proba_df, sigma=smooth_sigma).loc[X_test.index])

    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    support = y_test_arr.sum(axis=0) > 0
    n_zero = int((~support).sum())
    n_supported = int(support.sum())
    n_total = y_test_arr.shape[1]

    y_test_sup = y_test_arr[:, support]
    y_pred_sup = y_pred_arr[:, support]
    y_proba_sup = y_proba[:, support]

    def _mac(fn, *args_sup, **kw):
        """Compute macro metric over all classes and over supported classes."""
        all_val = fn(y_test_arr, y_pred_arr if "pred" in fn.__name__ else y_proba, **kw)
        sup_val = fn(y_test_sup, args_sup[0], **kw)
        return all_val, sup_val

    macro_f1_all = f1_score(y_test_arr, y_pred_arr, average="macro", zero_division=0)
    macro_f1_sup = f1_score(y_test_sup, y_pred_sup, average="macro", zero_division=0)
    micro_f1     = f1_score(y_test_arr, y_pred_arr, average="micro", zero_division=0)
    hl           = hamming_loss(y_test_arr, y_pred_arr)

    macro_auc_all = (
        roc_auc_score(y_test_arr, y_proba, average="macro") if n_zero == 0 else np.nan
    )
    macro_auc_sup = roc_auc_score(y_test_sup, y_proba_sup, average="macro")
    micro_auc     = roc_auc_score(y_test_arr, y_proba, average="micro")

    macro_ap_all = average_precision_score(y_test_arr, y_proba, average="macro")
    macro_ap_sup = average_precision_score(y_test_sup, y_proba_sup, average="macro")
    micro_ap     = average_precision_score(y_test_arr, y_proba, average="micro")

    lrap = label_ranking_average_precision_score(y_test_arr, y_proba)

    # NaN in the "supported" column marks metrics that have no per-class variant
    nan = np.nan
    rows = {
        "Macro F1":      (macro_f1_all,  macro_f1_sup),
        "Micro F1":      (micro_f1,      nan),
        "Hamming loss":  (hl,            nan),
        "Macro ROC AUC": (macro_auc_all, macro_auc_sup),
        "Micro ROC AUC": (micro_auc,     nan),
        "Macro AP":      (macro_ap_all,  macro_ap_sup),
        "Micro AP":      (micro_ap,      nan),
        "LRAP":          (lrap,          nan),
    }

    if n_zero > 0:
        df = pd.DataFrame.from_dict(rows, orient="index", columns=["all_classes", "supported_classes"])
        df.attrs["n_supported"] = n_supported
        df.attrs["n_total"] = n_total
    else:
        df = pd.DataFrame(
            {k: v[0] for k, v in rows.items()}, index=["value"]
        ).T
    df.index.name = "metric"

    if n_zero > 0:
        print(
            f"{n_zero}/{n_total} classes have zero test support"
            " — excluded from macro metrics only."
        )

    if y_test_arr.shape[1] <= 10:
        print("\n" + " CLASSIFICATION REPORT ".center(60, "="))
        print(classification_report(y_test, y_pred, target_names=list(y_test.columns)))

    print("\n" + " THRESHOLD-BASED METRICS ".center(60, "="))
    print(
        f"Macro F1:     {macro_f1_all:.4f}"
        + (f"  over {n_supported} classes: {macro_f1_sup:.4f}" if n_zero > 0 else "")
    )
    print(f"Micro F1:     {micro_f1:.4f}")
    print(f"Hamming loss: {hl:.4f}")

    print("\n" + " RANKING & PROBABILITY METRICS ".center(60, "="))
    print(
        f"Macro ROC AUC: {macro_auc_all if n_zero == 0 else 'nan   '}"
        + (f"  over {n_supported} classes: {macro_auc_sup:.4f}" if n_zero > 0 else "")
    )
    print(f"Micro ROC AUC: {micro_auc:.4f}")
    print(
        f"Macro AP:      {macro_ap_all:.4f}"
        + (f"  over {n_supported} classes: {macro_ap_sup:.4f}" if n_zero > 0 else "")
    )
    print(f"Micro AP:      {micro_ap:.4f}")
    print(f"LRAP:          {lrap:.4f}")

    return df
