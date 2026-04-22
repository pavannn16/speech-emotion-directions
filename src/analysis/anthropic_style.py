from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.analysis.emotion_vectors import (
    build_direction_vectors,
    center_within_groups,
    compute_class_centroids,
    cosine_centroid_predict,
    normalize_rows,
    project_onto_directions,
)


def build_projection_probability_frame(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    directions: np.ndarray,
    label_names: list[str],
    reference_label: str = "neutral",
) -> pd.DataFrame:
    metadata = metadata.reset_index(drop=True).copy()
    projections = project_onto_directions(embeddings, directions)
    reference_idx = label_names.index(reference_label)

    rows: list[dict[str, Any]] = []
    for sample_idx in range(len(metadata)):
        meta = metadata.iloc[sample_idx]
        true_label = str(meta["final_label"])
        for target_idx, target_label in enumerate(label_names):
            if target_idx == reference_idx:
                continue
            rows.append(
                {
                    "sample_index": sample_idx,
                    "split": str(meta["split"]),
                    "actor_id": str(meta["actor_id"]),
                    "statement_code": str(meta["statement_code"]),
                    "repetition_code": str(meta.get("repetition_code", "")),
                    "intensity": str(meta["intensity"]),
                    "true_label": true_label,
                    "target_label": target_label,
                    "projection": float(projections[sample_idx, target_idx]),
                    "target_probability": float(probabilities[sample_idx, target_idx]),
                    "is_true_target": bool(true_label == target_label),
                }
            )
    return pd.DataFrame(rows)


def summarize_projection_probability_correlations(
    projection_probability_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_label, group in projection_probability_df.groupby("target_label", sort=False):
        all_corr = group["projection"].corr(group["target_probability"], method="pearson")
        true_subset = group[group["is_true_target"]]
        true_corr = true_subset["projection"].corr(true_subset["target_probability"], method="pearson")
        rows.append(
            {
                "target_label": target_label,
                "num_samples_all": int(len(group)),
                "pearson_all": float(all_corr) if pd.notna(all_corr) else np.nan,
                "num_samples_true_subset": int(len(true_subset)),
                "pearson_true_subset": float(true_corr) if pd.notna(true_corr) else np.nan,
                "mean_projection_all": float(group["projection"].mean()),
                "mean_probability_all": float(group["target_probability"].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_centering_strategies(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    split_names: pd.Series | np.ndarray,
    actor_ids: pd.Series | np.ndarray,
    statement_codes: pd.Series | np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    split_array = np.asarray(split_names)
    actor_array = pd.Series(actor_ids).astype(str).to_numpy()
    statement_array = pd.Series(statement_codes).astype(str).to_numpy()
    train_mask = split_array == "train"
    num_labels = len(label_names)

    actor_centered = center_within_groups(embeddings, actor_array)
    actor_statement_centered = center_within_groups(actor_centered, statement_array)

    variants = {
        "raw": embeddings,
        "actor_centered": actor_centered,
        "actor_statement_centered": actor_statement_centered,
    }

    rows: list[dict[str, Any]] = []
    for variant_name, variant_embeddings in variants.items():
        centroids = compute_class_centroids(
            embeddings=variant_embeddings[train_mask],
            label_ids=label_ids[train_mask],
            num_labels=num_labels,
        )
        for split_name in ["val", "test"]:
            split_mask = split_array == split_name
            preds = cosine_centroid_predict(variant_embeddings[split_mask], centroids)
            rows.append(
                {
                    "variant": variant_name,
                    "split": split_name,
                    "accuracy": float(accuracy_score(label_ids[split_mask], preds)),
                    "macro_f1": float(f1_score(label_ids[split_mask], preds, average="macro")),
                    "weighted_f1": float(f1_score(label_ids[split_mask], preds, average="weighted")),
                }
            )
    return pd.DataFrame(rows)


def build_same_context_displacement_frame(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    directions: np.ndarray,
    label_names: list[str],
    reference_label: str = "neutral",
) -> pd.DataFrame:
    metadata = metadata.reset_index(drop=True).copy()
    metadata["row_index"] = np.arange(len(metadata))
    normalized_directions = normalize_rows(directions)
    reference_idx = label_names.index(reference_label)
    context_columns = ["split", "actor_id", "statement_code", "repetition_code"]

    reference_embeddings: dict[tuple[str, str, str, str], np.ndarray] = {}
    for context_values, group in metadata[metadata["final_label"] == reference_label].groupby(context_columns, dropna=False):
        row_indices = group["row_index"].to_numpy(dtype=np.int64)
        reference_embeddings[tuple(map(str, context_values))] = embeddings[row_indices].mean(axis=0)

    rows: list[dict[str, Any]] = []
    for row in metadata.itertuples(index=False):
        true_label = str(row.final_label)
        if true_label == reference_label:
            continue

        context_key = tuple(map(str, [row.split, row.actor_id, row.statement_code, row.repetition_code]))
        reference_embedding = reference_embeddings.get(context_key)
        if reference_embedding is None:
            continue

        target_idx = label_names.index(true_label)
        displacement = embeddings[int(row.row_index)] - reference_embedding
        normalized_displacement = normalize_rows(displacement[None, :])[0]
        projections = displacement @ normalized_directions.T
        top_idx = int(np.argmax(projections))

        rows.append(
            {
                "sample_index": int(row.row_index),
                "split": str(row.split),
                "actor_id": str(row.actor_id),
                "statement_code": str(row.statement_code),
                "repetition_code": str(row.repetition_code),
                "intensity": str(row.intensity),
                "true_label": true_label,
                "expected_projection": float(projections[target_idx]),
                "top_direction": label_names[top_idx],
                "top_projection": float(projections[top_idx]),
                "matched_expected_direction": bool(top_idx == target_idx),
                "cosine_to_expected_direction": float(normalized_displacement @ normalized_directions[target_idx]),
                "displacement_norm": float(np.linalg.norm(displacement)),
            }
        )

    return pd.DataFrame(rows)


def summarize_same_context_displacements(displacement_df: pd.DataFrame) -> pd.DataFrame:
    if displacement_df.empty:
        return pd.DataFrame(
            columns=[
                "true_label",
                "count",
                "mean_expected_projection",
                "mean_cosine_to_expected_direction",
                "match_rate",
            ]
        )

    rows: list[dict[str, Any]] = []
    for true_label, group in displacement_df.groupby("true_label", sort=False):
        rows.append(
            {
                "true_label": true_label,
                "count": int(len(group)),
                "mean_expected_projection": float(group["expected_projection"].mean()),
                "mean_cosine_to_expected_direction": float(group["cosine_to_expected_direction"].mean()),
                "match_rate": float(group["matched_expected_direction"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_intensity_projection_frame(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    directions: np.ndarray,
    label_names: list[str],
    reference_label: str = "neutral",
) -> pd.DataFrame:
    metadata = metadata.reset_index(drop=True).copy()
    projections = project_onto_directions(embeddings, directions)
    rows: list[dict[str, Any]] = []

    for sample_idx, row in metadata.iterrows():
        true_label = str(row["final_label"])
        if true_label == reference_label:
            continue
        target_idx = label_names.index(true_label)
        rows.append(
            {
                "sample_index": int(sample_idx),
                "split": str(row["split"]),
                "actor_id": str(row["actor_id"]),
                "statement_code": str(row["statement_code"]),
                "repetition_code": str(row.get("repetition_code", "")),
                "true_label": true_label,
                "intensity": str(row["intensity"]),
                "expected_projection": float(projections[sample_idx, target_idx]),
            }
        )

    return pd.DataFrame(rows)


def summarize_intensity_projections(intensity_projection_df: pd.DataFrame) -> pd.DataFrame:
    if intensity_projection_df.empty:
        return pd.DataFrame(
            columns=[
                "true_label",
                "intensity",
                "mean_expected_projection",
                "median_expected_projection",
                "count",
            ]
        )

    grouped = (
        intensity_projection_df.groupby(["true_label", "intensity"], as_index=False)
        .agg(
            mean_expected_projection=("expected_projection", "mean"),
            median_expected_projection=("expected_projection", "median"),
            count=("expected_projection", "count"),
        )
    )
    return grouped


def build_paired_intensity_delta_frame(
    intensity_projection_df: pd.DataFrame,
) -> pd.DataFrame:
    pair_keys = ["split", "actor_id", "statement_code", "repetition_code", "true_label"]
    rows: list[dict[str, Any]] = []

    for context_values, group in intensity_projection_df.groupby(pair_keys, dropna=False):
        normal = group[group["intensity"] == "normal"]
        strong = group[group["intensity"] == "strong"]
        if normal.empty or strong.empty:
            continue

        normal_value = float(normal.iloc[0]["expected_projection"])
        strong_value = float(strong.iloc[0]["expected_projection"])
        rows.append(
            {
                "split": str(context_values[0]),
                "actor_id": str(context_values[1]),
                "statement_code": str(context_values[2]),
                "repetition_code": str(context_values[3]),
                "true_label": str(context_values[4]),
                "normal_projection": normal_value,
                "strong_projection": strong_value,
                "strong_minus_normal": strong_value - normal_value,
            }
        )

    return pd.DataFrame(rows)


def summarize_paired_intensity_deltas(paired_delta_df: pd.DataFrame) -> pd.DataFrame:
    if paired_delta_df.empty:
        return pd.DataFrame(
            columns=[
                "true_label",
                "count",
                "mean_strong_minus_normal",
                "median_strong_minus_normal",
                "positive_rate",
            ]
        )

    rows: list[dict[str, Any]] = []
    for true_label, group in paired_delta_df.groupby("true_label", sort=False):
        rows.append(
            {
                "true_label": true_label,
                "count": int(len(group)),
                "mean_strong_minus_normal": float(group["strong_minus_normal"].mean()),
                "median_strong_minus_normal": float(group["strong_minus_normal"].median()),
                "positive_rate": float((group["strong_minus_normal"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_train_directions_with_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    label_ids: np.ndarray,
    label_names: list[str],
    reference_label: str = "neutral",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_array = metadata["split"].to_numpy()
    train_mask = split_array == "train"
    actor_centered = center_within_groups(embeddings, metadata["actor_id"].astype(str))
    actor_statement_centered = center_within_groups(actor_centered, metadata["statement_code"].astype(str))
    centroids = compute_class_centroids(
        actor_statement_centered[train_mask],
        label_ids[train_mask],
        num_labels=len(label_names),
    )
    directions = build_direction_vectors(centroids, label_names.index(reference_label))
    return actor_statement_centered, centroids, directions
