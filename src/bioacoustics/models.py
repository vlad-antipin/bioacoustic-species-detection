import warnings
from sklearn.exceptions import UndefinedMetricWarning

def ignore_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn.multiclass"
    )


    warnings.filterwarnings(
        "ignore",
        category=UndefinedMetricWarning
    )

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn.metrics"
    )