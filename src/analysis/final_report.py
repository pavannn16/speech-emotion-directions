from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class FinalArtifacts:
    checkpoint_dir: Path
    analysis_dir: Path
    label_names: list[str]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    test_predictions: pd.DataFrame
    test_classification_report: pd.DataFrame
    layerwise_metrics: pd.DataFrame
    centroid_cosine_matrix: pd.DataFrame
    projection_summary: pd.DataFrame
    steering_summary: pd.DataFrame


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_final_artifacts(
    checkpoint_dir: str | Path,
    analysis_dir: str | Path,
) -> FinalArtifacts:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    analysis_dir = Path(analysis_dir).resolve()

    label_mapping = _load_json(checkpoint_dir / "label_mapping.json")
    label_names = list(label_mapping["labels"])

    return FinalArtifacts(
        checkpoint_dir=checkpoint_dir,
        analysis_dir=analysis_dir,
        label_names=label_names,
        val_metrics=_load_json(checkpoint_dir / "val_metrics.json"),
        test_metrics=_load_json(checkpoint_dir / "test_metrics.json"),
        test_predictions=pd.read_csv(checkpoint_dir / "test_predictions.csv"),
        test_classification_report=pd.read_csv(checkpoint_dir / "test_classification_report.csv"),
        layerwise_metrics=pd.read_csv(analysis_dir / "layerwise_centroid_metrics.csv"),
        centroid_cosine_matrix=pd.read_csv(analysis_dir / "centroid_cosine_matrix.csv", index_col=0),
        projection_summary=pd.read_csv(analysis_dir / "projection_summary_test_centered.csv"),
        steering_summary=pd.read_csv(analysis_dir / "steering_summary.csv"),
    )


def build_overall_metrics_frame(artifacts: FinalArtifacts) -> pd.DataFrame:
    rows = []
    for split_name, metrics in [("val", artifacts.val_metrics), ("test", artifacts.test_metrics)]:
        rows.append(
            {
                "split": split_name,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
            }
        )
    return pd.DataFrame(rows)


def build_confusion_matrix_frame(artifacts: FinalArtifacts) -> pd.DataFrame:
    matrix = artifacts.test_metrics["confusion_matrix"]
    return pd.DataFrame(matrix, index=artifacts.label_names, columns=artifacts.label_names)


def build_best_layer_summary(artifacts: FinalArtifacts) -> dict[str, Any]:
    layerwise = artifacts.layerwise_metrics
    val_rows = layerwise[layerwise["split"] == "val"].sort_values(["macro_f1", "accuracy"], ascending=False)
    best_row = val_rows.iloc[0].to_dict()
    final_layer_index = int(layerwise["layer_index"].max())
    final_row = layerwise[(layerwise["split"] == "test") & (layerwise["layer_index"] == final_layer_index)].iloc[0]
    return {
        "best_val_layer_index": int(best_row["layer_index"]),
        "best_val_macro_f1": float(best_row["macro_f1"]),
        "best_val_accuracy": float(best_row["accuracy"]),
        "final_layer_index": final_layer_index,
        "final_test_macro_f1": float(final_row["macro_f1"]),
        "final_test_accuracy": float(final_row["accuracy"]),
    }


def build_projection_alignment_frame(
    artifacts: FinalArtifacts,
    reference_label: str = "neutral",
) -> pd.DataFrame:
    projection_df = artifacts.projection_summary.copy()
    reference_direction_column = f"proj_to_{reference_label}_minus_neutral"
    direction_columns = [
        column
        for column in projection_df.columns
        if column.startswith("proj_to_") and column.endswith("_minus_neutral")
        and column != reference_direction_column
    ]

    rows: list[dict[str, Any]] = []
    for row in projection_df.itertuples(index=False):
        true_label = str(row.true_label)
        values = {column: float(getattr(row, column)) for column in direction_columns}

        if true_label == reference_label:
            max_non_neutral = max(values.values()) if values else 0.0
            rows.append(
                {
                    "true_label": true_label,
                    "count": int(row.count),
                    "expected_direction": "all_non_neutral_negative",
                    "top_direction": "all_non_neutral_negative" if max_non_neutral < 0 else "not_all_negative",
                    "top_score": max_non_neutral,
                    "matched_expectation": bool(max_non_neutral < 0),
                }
            )
            continue

        expected_direction = f"proj_to_{true_label}_minus_neutral"
        top_direction = max(values, key=values.get)
        rows.append(
            {
                "true_label": true_label,
                "count": int(row.count),
                "expected_direction": expected_direction.replace("proj_to_", "").replace("_minus_neutral", ""),
                "top_direction": top_direction.replace("proj_to_", "").replace("_minus_neutral", ""),
                "top_score": float(values[top_direction]),
                "expected_score": float(values.get(expected_direction, 0.0)),
                "matched_expectation": bool(top_direction == expected_direction),
            }
        )

    return pd.DataFrame(rows)


def build_steering_summary_frame(artifacts: FinalArtifacts) -> pd.DataFrame:
    steering = artifacts.steering_summary.copy()
    steering["alpha"] = steering["alpha"].astype(float)
    best_rows = []

    for target_label, group in steering.groupby("target_label", sort=False):
        preferred = group[group["alpha"] == 0.5]
        if preferred.empty:
            preferred = group.sort_values("delta_target_prob_all", ascending=False).head(1)
        row = preferred.iloc[0].to_dict()
        row["target_label"] = target_label
        best_rows.append(row)

    return pd.DataFrame(best_rows)


def build_takeaways_markdown(artifacts: FinalArtifacts) -> str:
    overall = build_overall_metrics_frame(artifacts)
    best_layer = build_best_layer_summary(artifacts)
    projection_alignment = build_projection_alignment_frame(artifacts)
    steering = build_steering_summary_frame(artifacts)

    val_row = overall[overall["split"] == "val"].iloc[0]
    test_row = overall[overall["split"] == "test"].iloc[0]
    aligned_count = int(projection_alignment["matched_expectation"].sum())
    total_projection_rows = int(len(projection_alignment))
    mean_steering_delta = float(steering["delta_target_prob_all"].mean())

    lines = [
        "# Final Results Summary",
        "",
        "## Core Performance",
        f"- Validation accuracy: {val_row['accuracy']:.4f}",
        f"- Validation macro F1: {val_row['macro_f1']:.4f}",
        f"- Test accuracy: {test_row['accuracy']:.4f}",
        f"- Test macro F1: {test_row['macro_f1']:.4f}",
        "",
        "## Representation Findings",
        (
            f"- Best nearest-centroid validation layer: {best_layer['best_val_layer_index']} "
            f"(macro F1={best_layer['best_val_macro_f1']:.4f})."
        ),
        (
            f"- Final-layer nearest-centroid test performance: accuracy={best_layer['final_test_accuracy']:.4f}, "
            f"macro F1={best_layer['final_test_macro_f1']:.4f}."
        ),
        (
            f"- Projection alignment after actor+statement centering matched expectations for "
            f"{aligned_count}/{total_projection_rows} summary rows."
        ),
        "",
        "## Steering-Style Intervention",
        f"- Mean delta in target probability at the selected alpha values: {mean_steering_delta:.4f}.",
        "- These interventions operate on classifier embeddings, so they support a representation-level claim rather than a generative-behavior claim.",
    ]

    return "\n".join(lines) + "\n"
