"""Advanced analysis functions for direction-only classification, causal ablation,
emotion arithmetic, cross-dataset transfer, layer-wise steering, and sparse
autoencoder interpretability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.analysis.emotion_vectors import (
    build_direction_vectors,
    center_within_groups,
    compute_class_centroids,
    linear_classifier_probabilities,
    normalize_rows,
    project_onto_directions,
)


# ---------------------------------------------------------------------------
# 1. Direction-only classification
# ---------------------------------------------------------------------------


def direction_classify(
    embeddings: np.ndarray,
    directions: np.ndarray,
    reference_idx: int,
) -> np.ndarray:
    """Classify samples by largest projection onto emotion directions.

    For the reference class (neutral) we use the *negative* maximum —
    a sample is classified as neutral when it projects least toward any
    non-neutral direction.
    """
    projections = project_onto_directions(embeddings, directions)
    # Exclude reference column (all zeros) before argmax
    non_ref_mask = np.ones(projections.shape[1], dtype=bool)
    non_ref_mask[reference_idx] = False
    non_ref_projections = projections[:, non_ref_mask]
    non_ref_indices = np.where(non_ref_mask)[0]

    # A sample is neutral if its max projection onto any emotion direction
    # is below a threshold (median of train max-projections works well),
    # but a simpler and parameter-free approach: compare the max non-ref
    # projection against zero.  If the strongest pull is negative the
    # sample sits on the neutral side of every direction.
    max_non_ref = non_ref_projections.max(axis=1)
    best_non_ref_local = non_ref_projections.argmax(axis=1)
    best_non_ref_global = non_ref_indices[best_non_ref_local]

    preds = np.where(max_non_ref > 0, best_non_ref_global, reference_idx)
    return preds.astype(np.int64)


def evaluate_direction_classifier(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    directions: np.ndarray,
    label_names: list[str],
    reference_label: str = "neutral",
    split_name: str = "test",
) -> dict[str, Any]:
    """Return accuracy, macro-F1, and per-class report for direction-only classifier."""
    reference_idx = label_names.index(reference_label)
    preds = direction_classify(embeddings, directions, reference_idx)
    return {
        "split": split_name,
        "accuracy": float(accuracy_score(label_ids, preds)),
        "macro_f1": float(f1_score(label_ids, preds, average="macro")),
        "weighted_f1": float(f1_score(label_ids, preds, average="weighted")),
        "classification_report": classification_report(
            label_ids, preds, target_names=label_names, output_dict=True,
        ),
        "predictions": preds,
    }


def build_direction_classifier_comparison(
    results: list[dict[str, Any]],
    trained_metrics: dict[str, float],
) -> pd.DataFrame:
    """Build a comparison DataFrame between direction-only and trained classifiers."""
    rows = []
    for r in results:
        rows.append({
            "method": "direction_only",
            "split": r["split"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
        })
    rows.append({
        "method": "trained_head",
        "split": "test",
        "accuracy": trained_metrics.get("accuracy", 0.0),
        "macro_f1": trained_metrics.get("macro_f1", 0.0),
        "weighted_f1": trained_metrics.get("weighted_f1", 0.0),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Causal ablation
# ---------------------------------------------------------------------------


def ablate_direction_component(
    embeddings: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Remove the component along *direction* from each embedding.

    z' = z - (z . d_hat) * d_hat
    """
    d_hat = direction / np.linalg.norm(direction, keepdims=True).clip(1e-12)
    projections = embeddings @ d_hat[:, None]  # (N, 1)
    return embeddings - projections * d_hat[None, :]


def ablate_all_directions(
    embeddings: np.ndarray,
    directions: np.ndarray,
    reference_idx: int,
) -> np.ndarray:
    """Sequentially remove all non-reference direction components."""
    result = embeddings.copy()
    for idx in range(directions.shape[0]):
        if idx == reference_idx:
            continue
        if np.linalg.norm(directions[idx]) < 1e-12:
            continue
        result = ablate_direction_component(result, directions[idx])
    return result


def evaluate_ablation(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    directions: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    label_names: list[str],
    reference_idx: int,
) -> pd.DataFrame:
    """Compare classifier accuracy on original vs single-direction-ablated
    vs all-directions-ablated embeddings."""
    rows: list[dict[str, Any]] = []

    # Baseline (no ablation)
    base_probs = linear_classifier_probabilities(embeddings, classifier_weight, classifier_bias)
    base_preds = base_probs.argmax(axis=1)
    rows.append({
        "ablation": "none",
        "accuracy": float(accuracy_score(label_ids, base_preds)),
        "macro_f1": float(f1_score(label_ids, base_preds, average="macro")),
        "weighted_f1": float(f1_score(label_ids, base_preds, average="weighted")),
    })

    # Single-direction ablation
    for idx, name in enumerate(label_names):
        if idx == reference_idx:
            continue
        if np.linalg.norm(directions[idx]) < 1e-12:
            continue
        ablated = ablate_direction_component(embeddings, directions[idx])
        probs = linear_classifier_probabilities(ablated, classifier_weight, classifier_bias)
        preds = probs.argmax(axis=1)
        rows.append({
            "ablation": f"remove_{name}",
            "accuracy": float(accuracy_score(label_ids, preds)),
            "macro_f1": float(f1_score(label_ids, preds, average="macro")),
            "weighted_f1": float(f1_score(label_ids, preds, average="weighted")),
        })

    # All-direction ablation
    all_ablated = ablate_all_directions(embeddings, directions, reference_idx)
    probs = linear_classifier_probabilities(all_ablated, classifier_weight, classifier_bias)
    preds = probs.argmax(axis=1)
    rows.append({
        "ablation": "remove_all",
        "accuracy": float(accuracy_score(label_ids, preds)),
        "macro_f1": float(f1_score(label_ids, preds, average="macro")),
        "weighted_f1": float(f1_score(label_ids, preds, average="weighted")),
    })

    return pd.DataFrame(rows)


def per_class_ablation_impact(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    directions: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    label_names: list[str],
    reference_idx: int,
) -> pd.DataFrame:
    """For each emotion, measure the F1 drop when its own direction is removed
    vs when other directions are removed."""
    base_probs = linear_classifier_probabilities(embeddings, classifier_weight, classifier_bias)
    base_preds = base_probs.argmax(axis=1)
    base_report = classification_report(label_ids, base_preds, target_names=label_names, output_dict=True)

    rows: list[dict[str, Any]] = []
    for ablated_idx, ablated_name in enumerate(label_names):
        if ablated_idx == reference_idx or np.linalg.norm(directions[ablated_idx]) < 1e-12:
            continue
        ablated = ablate_direction_component(embeddings, directions[ablated_idx])
        probs = linear_classifier_probabilities(ablated, classifier_weight, classifier_bias)
        preds = probs.argmax(axis=1)
        report = classification_report(label_ids, preds, target_names=label_names, output_dict=True)
        for eval_name in label_names:
            rows.append({
                "ablated_direction": ablated_name,
                "evaluated_class": eval_name,
                "base_f1": float(base_report[eval_name]["f1-score"]),
                "ablated_f1": float(report[eval_name]["f1-score"]),
                "f1_delta": float(report[eval_name]["f1-score"] - base_report[eval_name]["f1-score"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Emotion arithmetic / blending
# ---------------------------------------------------------------------------


def blend_emotions(
    neutral_centroid: np.ndarray,
    directions: np.ndarray,
    blend_weights: dict[int, float],
) -> np.ndarray:
    """Create a blended embedding: neutral + sum(w_i * direction_i)."""
    blended = neutral_centroid.copy()
    for idx, weight in blend_weights.items():
        blended = blended + weight * directions[idx]
    return blended


def evaluate_blends(
    neutral_centroid: np.ndarray,
    directions: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    label_names: list[str],
    reference_idx: int,
    blend_configs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Evaluate a list of blend configurations.

    Each config has a 'name' and 'weights' dict mapping label_idx -> weight.
    Returns predicted probabilities for each blend.
    """
    rows: list[dict[str, Any]] = []
    for config in blend_configs:
        blended = blend_emotions(neutral_centroid, directions, config["weights"])
        probs = linear_classifier_probabilities(
            blended[None, :], classifier_weight, classifier_bias,
        )[0]
        row: dict[str, Any] = {"blend_name": config["name"]}
        for idx, name in enumerate(label_names):
            row[f"prob_{name}"] = float(probs[idx])
        row["predicted_class"] = label_names[int(probs.argmax())]
        rows.append(row)
    return pd.DataFrame(rows)


def interpolation_path(
    neutral_centroid: np.ndarray,
    direction_a: np.ndarray,
    direction_b: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    label_names: list[str],
    steps: int = 11,
) -> pd.DataFrame:
    """Interpolate from direction_a to direction_b through neutral.

    alpha goes from -1 (pure A) through 0 (neutral) to +1 (pure B).
    """
    alphas = np.linspace(-1.0, 1.0, steps)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        if alpha <= 0:
            embedding = neutral_centroid + abs(alpha) * direction_a
        else:
            embedding = neutral_centroid + alpha * direction_b
        probs = linear_classifier_probabilities(
            embedding[None, :], classifier_weight, classifier_bias,
        )[0]
        row: dict[str, Any] = {"alpha": float(alpha)}
        for idx, name in enumerate(label_names):
            row[f"prob_{name}"] = float(probs[idx])
        row["predicted_class"] = label_names[int(probs.argmax())]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Cross-dataset direction transfer
# ---------------------------------------------------------------------------


CREMAD_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_cremad_filename(filename: str) -> dict[str, str]:
    """Parse a CREMA-D filename like '1001_DFA_ANG_XX.wav'."""
    parts = filename.replace(".wav", "").split("_")
    return {
        "actor_id": parts[0],
        "sentence": parts[1],
        "emotion_code": parts[2],
        "intensity_code": parts[3] if len(parts) > 3 else "XX",
    }


def build_cremad_metadata(audio_files: list[str]) -> pd.DataFrame:
    """Build a metadata DataFrame for CREMA-D files."""
    rows = []
    for f in audio_files:
        parsed = parse_cremad_filename(f)
        emotion = CREMAD_EMOTION_MAP.get(parsed["emotion_code"])
        if emotion is None:
            continue
        rows.append({
            "file_name": f,
            "actor_id": parsed["actor_id"],
            "sentence": parsed["sentence"],
            "emotion_code": parsed["emotion_code"],
            "final_label": emotion,
            "intensity_code": parsed["intensity_code"],
        })
    return pd.DataFrame(rows)


def evaluate_transfer_directions(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    directions: np.ndarray,
    label_names: list[str],
    reference_label: str = "neutral",
) -> dict[str, Any]:
    """Evaluate RAVDESS-derived directions on external dataset embeddings."""
    reference_idx = label_names.index(reference_label)
    preds = direction_classify(embeddings, directions, reference_idx)
    return {
        "accuracy": float(accuracy_score(label_ids, preds)),
        "macro_f1": float(f1_score(label_ids, preds, average="macro")),
        "weighted_f1": float(f1_score(label_ids, preds, average="weighted")),
        "classification_report": classification_report(
            label_ids, preds, target_names=label_names, output_dict=True,
        ),
        "predictions": preds,
    }


# ---------------------------------------------------------------------------
# 5. Layer-wise steering propagation (helper utilities)
# ---------------------------------------------------------------------------


def build_layerwise_directions(
    layer_embeddings: np.ndarray,
    label_ids: np.ndarray,
    split_mask: np.ndarray,
    label_names: list[str],
    reference_idx: int,
) -> np.ndarray:
    """Build direction vectors for each layer.

    Returns shape (num_layers, num_labels, hidden_size).
    """
    num_layers = layer_embeddings.shape[1]
    num_labels = len(label_names)
    hidden_size = layer_embeddings.shape[2]
    all_directions = np.zeros((num_layers, num_labels, hidden_size), dtype=np.float32)
    for layer_idx in range(num_layers):
        emb = layer_embeddings[split_mask, layer_idx]
        ids = label_ids[split_mask]
        centroids = compute_class_centroids(emb, ids, num_labels)
        all_directions[layer_idx] = build_direction_vectors(centroids, reference_idx)
    return all_directions


def evaluate_layerwise_steering(
    layer_embeddings: np.ndarray,
    label_ids: np.ndarray,
    layerwise_directions: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    label_names: list[str],
    reference_idx: int,
    target_label_ids: list[int],
    injection_layers: list[int],
    alpha: float = 0.5,
) -> pd.DataFrame:
    """For each injection layer, steer at that layer's embedding, then
    measure how the final-layer classifier responds."""
    # We use the final layer embeddings as the base for classification
    final_layer_idx = layer_embeddings.shape[1] - 1
    final_embeddings = layer_embeddings[:, final_layer_idx]
    base_probs = linear_classifier_probabilities(final_embeddings, classifier_weight, classifier_bias)

    rows: list[dict[str, Any]] = []
    for injection_layer in injection_layers:
        for target_id in target_label_ids:
            direction = layerwise_directions[injection_layer, target_id]
            if np.linalg.norm(direction) < 1e-12:
                continue

            # Steer at final layer using direction from injection_layer
            steered = final_embeddings + alpha * direction[None, :]
            steered_probs = linear_classifier_probabilities(steered, classifier_weight, classifier_bias)

            rows.append({
                "injection_layer": injection_layer,
                "target_label": label_names[target_id],
                "alpha": alpha,
                "mean_delta_target_prob": float(
                    (steered_probs[:, target_id] - base_probs[:, target_id]).mean()
                ),
                "mean_target_prob_steered": float(steered_probs[:, target_id].mean()),
                "pred_as_target_rate": float((steered_probs.argmax(axis=1) == target_id).mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Sparse autoencoder
# ---------------------------------------------------------------------------


def train_sparse_autoencoder_numpy(
    embeddings: np.ndarray,
    dictionary_size: int = 2048,
    sparsity_coeff: float = 1e-3,
    learning_rate: float = 1e-3,
    num_epochs: int = 200,
    batch_size: int = 128,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Train a simple sparse autoencoder with L1 penalty using numpy.

    Architecture: x -> encoder (linear + ReLU) -> z (sparse code) -> decoder (linear) -> x_hat
    Loss: ||x - x_hat||^2 + sparsity_coeff * ||z||_1

    Returns dict with 'encoder_weight', 'encoder_bias', 'decoder_weight', 'decoder_bias',
    'loss_history'.
    """
    rng = np.random.RandomState(seed)
    input_size = embeddings.shape[1]

    # Xavier initialization
    encoder_weight = rng.randn(dictionary_size, input_size).astype(np.float32) * np.sqrt(2.0 / input_size)
    encoder_bias = np.zeros(dictionary_size, dtype=np.float32)
    decoder_weight = rng.randn(input_size, dictionary_size).astype(np.float32) * np.sqrt(2.0 / dictionary_size)
    decoder_bias = np.zeros(input_size, dtype=np.float32)

    n_samples = embeddings.shape[0]
    loss_history = []

    for epoch in range(num_epochs):
        indices = rng.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            x = embeddings[batch_idx]
            bs = x.shape[0]

            # Forward
            pre_act = x @ encoder_weight.T + encoder_bias  # (bs, dict_size)
            z = np.maximum(pre_act, 0)  # ReLU
            x_hat = z @ decoder_weight.T + decoder_bias  # (bs, input_size)

            # Loss
            recon_error = x - x_hat
            recon_loss = (recon_error ** 2).sum() / bs
            sparsity_loss = sparsity_coeff * np.abs(z).sum() / bs
            total_loss = recon_loss + sparsity_loss

            # Backward (manual gradients)
            d_x_hat = -2.0 * recon_error / bs  # (bs, input_size)
            d_decoder_weight = d_x_hat.T @ z  # (input_size, dict_size)
            d_decoder_bias = d_x_hat.sum(axis=0)  # (input_size,)
            d_z = d_x_hat @ decoder_weight  # (bs, dict_size)
            d_z += sparsity_coeff * np.sign(z) / bs  # L1 gradient

            # ReLU backward
            d_pre_act = d_z * (pre_act > 0).astype(np.float32)
            d_encoder_weight = d_pre_act.T @ x  # (dict_size, input_size)
            d_encoder_bias = d_pre_act.sum(axis=0)  # (dict_size,)

            # Update
            encoder_weight -= learning_rate * d_encoder_weight
            encoder_bias -= learning_rate * d_encoder_bias
            decoder_weight -= learning_rate * d_decoder_weight
            decoder_bias -= learning_rate * d_decoder_bias

            epoch_loss += total_loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(float(avg_loss))

    return {
        "encoder_weight": encoder_weight,
        "encoder_bias": encoder_bias,
        "decoder_weight": decoder_weight,
        "decoder_bias": decoder_bias,
        "loss_history": np.array(loss_history),
    }


def encode_with_sae(
    embeddings: np.ndarray,
    encoder_weight: np.ndarray,
    encoder_bias: np.ndarray,
) -> np.ndarray:
    """Encode embeddings through the SAE encoder (ReLU activation)."""
    return np.maximum(embeddings @ encoder_weight.T + encoder_bias, 0)


def analyze_sae_features(
    activations: np.ndarray,
    label_ids: np.ndarray,
    label_names: list[str],
    top_k: int = 20,
) -> pd.DataFrame:
    """Find features most selective for each emotion class.

    For each feature, compute the mean activation per class, then rank
    by selectivity (ratio of top-class mean to overall mean).
    """
    n_features = activations.shape[1]
    num_labels = len(label_names)

    # Compute per-class mean activation for each feature
    class_means = np.zeros((num_labels, n_features), dtype=np.float32)
    for label_id in range(num_labels):
        mask = label_ids == label_id
        if mask.sum() > 0:
            class_means[label_id] = activations[mask].mean(axis=0)

    overall_mean = activations.mean(axis=0)
    # Fraction of samples where each feature is active
    activation_rate = (activations > 0).mean(axis=0)

    rows: list[dict[str, Any]] = []
    for label_id, label_name in enumerate(label_names):
        # Selectivity: class_mean / overall_mean (higher = more selective)
        selectivity = np.where(
            overall_mean > 1e-8,
            class_means[label_id] / overall_mean,
            0.0,
        )
        top_features = np.argsort(selectivity)[::-1][:top_k]
        for rank, feat_idx in enumerate(top_features):
            rows.append({
                "emotion": label_name,
                "rank": rank + 1,
                "feature_index": int(feat_idx),
                "class_mean_activation": float(class_means[label_id, feat_idx]),
                "overall_mean_activation": float(overall_mean[feat_idx]),
                "selectivity_ratio": float(selectivity[feat_idx]),
                "activation_rate": float(activation_rate[feat_idx]),
            })
    return pd.DataFrame(rows)


def sae_feature_emotion_heatmap(
    activations: np.ndarray,
    label_ids: np.ndarray,
    label_names: list[str],
    top_k_per_class: int = 5,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Build a heatmap matrix of (selected_features x emotions).

    Returns (heatmap_matrix, feature_indices, label_names).
    """
    n_features = activations.shape[1]
    num_labels = len(label_names)

    class_means = np.zeros((num_labels, n_features), dtype=np.float32)
    for label_id in range(num_labels):
        mask = label_ids == label_id
        if mask.sum() > 0:
            class_means[label_id] = activations[mask].mean(axis=0)

    overall_mean = activations.mean(axis=0)

    # Select top-k most selective features per class
    selected_features: list[int] = []
    for label_id in range(num_labels):
        selectivity = np.where(
            overall_mean > 1e-8,
            class_means[label_id] / overall_mean,
            0.0,
        )
        top_k = np.argsort(selectivity)[::-1][:top_k_per_class]
        for f in top_k:
            if f not in selected_features:
                selected_features.append(int(f))

    # Build heatmap: normalize each feature's class means to [0, 1]
    heatmap = class_means[:, selected_features].T  # (n_selected, num_labels)
    row_maxes = heatmap.max(axis=1, keepdims=True)
    heatmap = np.where(row_maxes > 1e-8, heatmap / row_maxes, 0.0)

    return heatmap, selected_features, label_names
