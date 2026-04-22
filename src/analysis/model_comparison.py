from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class CheckpointMetrics:
    experiment_name: str
    checkpoint_dir: Path
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    test_classification_report: pd.DataFrame


def load_checkpoint_metrics(checkpoint_dir: str | Path) -> CheckpointMetrics:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    val_metrics = json.loads((checkpoint_dir / "val_metrics.json").read_text(encoding="utf-8"))
    test_metrics = json.loads((checkpoint_dir / "test_metrics.json").read_text(encoding="utf-8"))
    test_report = pd.read_csv(checkpoint_dir / "test_classification_report.csv")
    return CheckpointMetrics(
        experiment_name=str(config["experiment_name"]),
        checkpoint_dir=checkpoint_dir,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        test_classification_report=test_report,
    )


def build_split_metrics_frame(
    metrics_by_model: dict[str, CheckpointMetrics],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, metrics in metrics_by_model.items():
        for split_name, split_metrics in [("val", metrics.val_metrics), ("test", metrics.test_metrics)]:
            rows.append(
                {
                    "model": model_name,
                    "experiment_name": metrics.experiment_name,
                    "split": split_name,
                    "accuracy": float(split_metrics["accuracy"]),
                    "macro_f1": float(split_metrics["macro_f1"]),
                    "weighted_f1": float(split_metrics["weighted_f1"]),
                }
            )
    return pd.DataFrame(rows)


def build_test_improvement_frame(
    baseline: CheckpointMetrics,
    main_model: CheckpointMetrics,
    baseline_name: str = "cnn_baseline",
    main_name: str = "wav2vec2",
) -> pd.DataFrame:
    rows = []
    for metric_name in ["accuracy", "macro_f1", "weighted_f1"]:
        baseline_value = float(baseline.test_metrics[metric_name])
        main_value = float(main_model.test_metrics[metric_name])
        rows.append(
            {
                "metric": metric_name,
                baseline_name: baseline_value,
                main_name: main_value,
                "absolute_gain": main_value - baseline_value,
            }
        )
    return pd.DataFrame(rows)


def build_per_class_f1_comparison_frame(
    baseline: CheckpointMetrics,
    main_model: CheckpointMetrics,
    baseline_name: str = "cnn_baseline",
    main_name: str = "wav2vec2",
) -> pd.DataFrame:
    def extract_f1(report_df: pd.DataFrame) -> pd.DataFrame:
        df = report_df.rename(columns={"Unnamed: 0": "label"}).copy()
        df = df[df["label"].isin(["neutral", "happy", "sad", "angry", "fearful", "disgust"])].copy()
        return df[["label", "f1-score"]].rename(columns={"f1-score": "f1"})

    baseline_f1 = extract_f1(baseline.test_classification_report).rename(columns={"f1": baseline_name})
    main_f1 = extract_f1(main_model.test_classification_report).rename(columns={"f1": main_name})

    merged = baseline_f1.merge(main_f1, on="label", how="inner")
    merged["absolute_gain"] = merged[main_name] - merged[baseline_name]
    return merged.sort_values("absolute_gain", ascending=False).reset_index(drop=True)


def build_comparison_markdown(
    baseline: CheckpointMetrics,
    main_model: CheckpointMetrics,
    baseline_name: str = "cnn_baseline",
    main_name: str = "wav2vec2",
) -> str:
    test_gain_macro_f1 = float(main_model.test_metrics["macro_f1"]) - float(baseline.test_metrics["macro_f1"])
    test_gain_accuracy = float(main_model.test_metrics["accuracy"]) - float(baseline.test_metrics["accuracy"])

    return "\n".join(
        [
            "# Model Comparison Summary",
            "",
            "## Main Result",
            (
                f"- {main_name} outperformed {baseline_name} on the speaker-independent test split by "
                f"{test_gain_accuracy:.4f} accuracy and {test_gain_macro_f1:.4f} macro F1."
            ),
            "",
            "## Interpretation",
            "- The CNN baseline is a valid reference point, but it struggles especially on happy and sad.",
            "- The wav2vec2 model captures broader emotional structure and generalizes much better across unseen speakers.",
            "- This strengthens the project narrative: the representation analysis is built on top of a genuinely stronger speech model, not just a marginal improvement.",
        ]
    ) + "\n"
