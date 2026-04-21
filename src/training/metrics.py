from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def summarize_classification(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class_precision": {label: float(value) for label, value in zip(label_names, precision)},
        "per_class_recall": {label: float(value) for label, value in zip(label_names, recall)},
        "per_class_f1": {label: float(value) for label, value in zip(label_names, f1)},
        "per_class_support": {label: int(value) for label, value in zip(label_names, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def classification_report_frame(report_dict: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(report_dict).T

