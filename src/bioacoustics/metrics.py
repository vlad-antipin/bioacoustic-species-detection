import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
    hamming_loss,
    label_ranking_average_precision_score,
)


def evaluate_multilabel_model(model, X_test, y_test, y_parent_proba=None):
    """
    Macro - metric computed per class then averaged, giving each label
    equal weight regardless of frequency.

    Micro - metric computed globally by aggregating all true positives,
    false positives, and false negatives across labels.

    Hamming loss - fraction of incorrectly predicted label assignments
    across all samples and labels.

    LRAP (Label Ranking Average Precision) - evaluates how well true
    labels are ranked above others for each sample; averages the rank
    quality over true labels and samples.

    Macro metrics are computed only over classes that have at least one
    positive sample in the test set; zero-support classes would make
    roc_auc_score return NaN and contribute uninformative zeros to F1.
    """

    if y_parent_proba is None:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    else:
        y_pred = model.predict(X_test, y_parent_proba)
        y_proba = model.predict_proba(X_test, y_parent_proba)

    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    # Classes with at least one positive sample in the test set
    support = y_test_arr.sum(axis=0) > 0
    n_zero = int((~support).sum())
    n_total = y_test_arr.shape[1]

    if n_zero > 0:
        print(
            f"{n_zero}/{n_total} classes have zero test support"
            " — excluded from macro metrics only."
        )

    y_test_macro = y_test_arr[:, support]
    y_pred_macro = y_pred_arr[:, support]
    y_proba_macro = y_proba[:, support]

    if y_test.shape[1] <= 10:
        print("\n" + " CLASSIFICATION REPORT ".center(60, "="))
        print(classification_report(y_test, y_pred, target_names=list(y_test.columns)))

    print("\n" + " THRESHOLD-BASED METRICS ".center(60, "="))
    print(
        f"Macro F1:     {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
        + (
            f"  over {support.sum()} classes: {f1_score(y_test_macro, y_pred_macro, average='macro', zero_division=0):.4f}"
            if n_zero > 0
            else ""
        )
    )
    print(
        f"Micro F1:     {f1_score(y_test_arr, y_pred_arr, average='micro', zero_division=0):.4f}"
    )
    print(f"Hamming loss: {hamming_loss(y_test_arr, y_pred_arr):.4f}")

    print("\n" + " RANKING & PROBABILITY METRICS ".center(60, "="))
    print(
        "Macro ROC AUC: "
        + (
            f"{roc_auc_score(y_test, y_proba, average='macro'):.4f}"
            if n_zero == 0
            else "nan   "
        )
        + (
            f"  over {support.sum()} classes: {roc_auc_score(y_test_macro, y_proba_macro, average='macro'):.4f}"
            if n_zero > 0
            else ""
        )
    )
    print(f"Micro ROC AUC: {roc_auc_score(y_test_arr, y_proba, average='micro'):.4f}")

    print(
        f"Macro AP:      {average_precision_score(y_test, y_proba, average='macro'):.4f}"
        + (
            f"  over {support.sum()} classes: {average_precision_score(y_test_macro, y_proba_macro, average='macro'):.4f}"
            if n_zero > 0
            else ""
        )
    )
    print(
        f"Micro AP:      {average_precision_score(y_test_arr, y_proba, average='micro'):.4f}"
    )

    print(
        f"LRAP:          {label_ranking_average_precision_score(y_test_arr, y_proba):.4f}"
    )
