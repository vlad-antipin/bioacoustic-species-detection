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
    """

    # TODO: handle correctly missing labels
    if y_parent_proba is None:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    else:
        y_pred = model.predict(X_test, y_parent_proba)
        y_proba = model.predict_proba(X_test, y_parent_proba)

    if y_test.shape[1] <= 10:
        print("\n" + " CLASSIFICATION REPORT ".center(60, "="))
        print(classification_report(y_test, y_pred, target_names=y_test.columns))

    print("\n" + " THRESHOLD-BASED METRICS ".center(60, "="))
    print(f"Macro F1:   {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Micro F1:   {f1_score(y_test, y_pred, average='micro'):.4f}")
    print(f"Hamming loss: {hamming_loss(y_test, y_pred):.4f}")

    print("\n" + " RANKING & PROBABILITY METRICS ".center(60, "="))
    print(f"Macro ROC AUC: {roc_auc_score(y_test, y_proba, average='macro'):.4f}")
    print(f"Micro ROC AUC: {roc_auc_score(y_test, y_proba, average='micro'):.4f}")

    print(
        f"Macro AP:      {average_precision_score(y_test, y_proba, average='macro'):.4f}"
    )
    print(
        f"Micro AP:      {average_precision_score(y_test, y_proba, average='micro'):.4f}"
    )

    print(
        f"LRAP:          {label_ranking_average_precision_score(y_test, y_proba):.4f}"
    )
