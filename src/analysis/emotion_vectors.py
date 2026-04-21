from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class LoadedEmbeddingArtifacts:
    metadata: pd.DataFrame
    layer_embeddings: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    true_label_ids: np.ndarray
    pred_label_ids: np.ndarray
    summary: dict


def load_embedding_artifacts(output_dir: str | Path) -> LoadedEmbeddingArtifacts:
    output_dir = Path(output_dir).resolve()
    metadata = pd.read_csv(output_dir / "embedding_metadata.csv")
    arrays = np.load(output_dir / "embedding_arrays.npz")
    summary = json.loads((output_dir / "embedding_summary.json").read_text(encoding="utf-8"))
    return LoadedEmbeddingArtifacts(
        metadata=metadata,
        layer_embeddings=arrays["layer_embeddings"],
        logits=arrays["logits"],
        probabilities=arrays["probabilities"],
        true_label_ids=arrays["true_label_ids"],
        pred_label_ids=arrays["pred_label_ids"],
        summary=summary,
    )


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def compute_class_centroids(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    num_labels: int,
) -> np.ndarray:
    centroids = []
    for label_id in range(num_labels):
        mask = label_ids == label_id
        if not np.any(mask):
            raise ValueError(f"No examples found for label id {label_id}")
        centroids.append(embeddings[mask].mean(axis=0))
    return np.stack(centroids, axis=0)


def pairwise_cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(vectors)
    return normalized @ normalized.T


def cosine_centroid_predict(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    similarities = normalize_rows(embeddings) @ normalize_rows(centroids).T
    return similarities.argmax(axis=1).astype(np.int64)


def evaluate_layerwise_centroid_classifier(
    layer_embeddings: np.ndarray,
    split_names: Iterable[str],
    label_ids: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    split_array = np.asarray(list(split_names))
    label_ids = np.asarray(label_ids, dtype=np.int64)
    train_mask = split_array == "train"
    num_labels = len(label_names)

    records: list[dict[str, float | int | str]] = []
    for layer_idx in range(layer_embeddings.shape[1]):
        train_embeddings = layer_embeddings[train_mask, layer_idx]
        train_labels = label_ids[train_mask]
        centroids = compute_class_centroids(train_embeddings, train_labels, num_labels)

        for split_name in ["train", "val", "test"]:
            split_mask = split_array == split_name
            split_embeddings = layer_embeddings[split_mask, layer_idx]
            split_labels = label_ids[split_mask]
            split_preds = cosine_centroid_predict(split_embeddings, centroids)
            records.append(
                {
                    "layer_index": layer_idx,
                    "split": split_name,
                    "accuracy": float(accuracy_score(split_labels, split_preds)),
                    "macro_f1": float(f1_score(split_labels, split_preds, average="macro")),
                    "weighted_f1": float(f1_score(split_labels, split_preds, average="weighted")),
                }
            )

    return pd.DataFrame(records)


def center_within_groups(embeddings: np.ndarray, group_ids: Iterable[str]) -> np.ndarray:
    centered = embeddings.copy()
    group_array = np.asarray(list(group_ids))
    for group_id in np.unique(group_array):
        mask = group_array == group_id
        centered[mask] = centered[mask] - centered[mask].mean(axis=0, keepdims=True)
    return centered


def build_direction_vectors(centroids: np.ndarray, reference_index: int) -> np.ndarray:
    return centroids - centroids[reference_index]


def project_onto_directions(embeddings: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return embeddings @ normalize_rows(directions).T


def summarize_projection_means(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    directions: np.ndarray,
    label_names: list[str],
    direction_names: list[str] | None = None,
) -> pd.DataFrame:
    direction_names = direction_names or label_names
    projections = project_onto_directions(embeddings, directions)
    rows = []
    for label_id, label_name in enumerate(label_names):
        mask = label_ids == label_id
        row: dict[str, float | int | str] = {
            "true_label": label_name,
            "count": int(mask.sum()),
        }
        for direction_idx, direction_name in enumerate(direction_names):
            row[f"proj_to_{direction_name}"] = float(projections[mask, direction_idx].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def linear_classifier_probabilities(
    embeddings: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
) -> np.ndarray:
    logits = embeddings @ classifier_weight.T + classifier_bias
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)


def evaluate_direction_steering(
    embeddings: np.ndarray,
    true_label_ids: np.ndarray,
    direction_vectors: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    label_names: list[str],
    target_label_ids: Iterable[int],
    alphas: Iterable[float],
) -> pd.DataFrame:
    base_probs = linear_classifier_probabilities(embeddings, classifier_weight, classifier_bias)
    rows: list[dict[str, float | str]] = []

    for target_label_id in target_label_ids:
        direction = direction_vectors[target_label_id]
        target_mask = true_label_ids == target_label_id
        for alpha in alphas:
            steered_embeddings = embeddings + (alpha * direction[None, :])
            steered_probs = linear_classifier_probabilities(steered_embeddings, classifier_weight, classifier_bias)
            steered_preds = steered_probs.argmax(axis=1)

            rows.append(
                {
                    "target_label": label_names[target_label_id],
                    "alpha": float(alpha),
                    "mean_target_prob_all": float(steered_probs[:, target_label_id].mean()),
                    "delta_target_prob_all": float(
                        (steered_probs[:, target_label_id] - base_probs[:, target_label_id]).mean()
                    ),
                    "pred_as_target_rate_all": float((steered_preds == target_label_id).mean()),
                    "mean_target_prob_true_subset": float(steered_probs[target_mask, target_label_id].mean()),
                    "delta_target_prob_true_subset": float(
                        (steered_probs[target_mask, target_label_id] - base_probs[target_mask, target_label_id]).mean()
                    ),
                    "pred_as_target_rate_true_subset": float((steered_preds[target_mask] == target_label_id).mean()),
                }
            )

    return pd.DataFrame(rows)
