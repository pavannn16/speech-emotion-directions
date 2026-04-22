from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from src.data.dataset import EmotionLabelEncoder, RavdessWav2VecDataset, Wav2VecCollator, load_project_metadata
from src.models.wav2vec_classifier import Wav2VecEmotionClassifier
from src.training.train_wav2vec import get_best_available_device, move_batch_to_device, resolve_project_path


@dataclass
class LoadedCheckpoint:
    model: Wav2VecEmotionClassifier
    feature_extractor: AutoFeatureExtractor
    label_encoder: EmotionLabelEncoder
    config: dict[str, Any]
    checkpoint_dir: Path


@dataclass
class ExtractedEmbeddings:
    metadata: pd.DataFrame
    layer_embeddings: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    true_label_ids: np.ndarray
    pred_label_ids: np.ndarray


def load_trained_checkpoint(
    checkpoint_dir: str | Path,
    device: torch.device | None = None,
) -> LoadedCheckpoint:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    label_mapping = json.loads((checkpoint_dir / "label_mapping.json").read_text(encoding="utf-8"))

    label_encoder = EmotionLabelEncoder(labels=list(label_mapping["labels"]))
    model = Wav2VecEmotionClassifier(
        backbone_name=config["backbone_name"],
        num_labels=len(label_encoder.labels),
        dropout=float(config.get("dropout", 0.2)),
        freeze_feature_encoder=False,
    )

    state_dict = torch.load(checkpoint_dir / "model_state.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    target_device = device or get_best_available_device()
    model = model.to(target_device)
    model.eval()

    feature_extractor_dir = checkpoint_dir / "feature_extractor"
    if feature_extractor_dir.exists():
        feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_dir)
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(config["backbone_name"])

    return LoadedCheckpoint(
        model=model,
        feature_extractor=feature_extractor,
        label_encoder=label_encoder,
        config=config,
        checkpoint_dir=checkpoint_dir,
    )


def build_extraction_dataloader(
    metadata_path: str | Path,
    feature_extractor: AutoFeatureExtractor,
    label_encoder: EmotionLabelEncoder,
    sample_rate: int,
    batch_size: int,
    num_workers: int = 0,
    raw_audio_root: str | Path | None = None,
) -> DataLoader:
    metadata = load_project_metadata(metadata_path, raw_audio_root=raw_audio_root)
    dataset = RavdessWav2VecDataset(metadata, label_encoder=label_encoder, sample_rate=sample_rate)
    collator = Wav2VecCollator(feature_extractor=feature_extractor, sample_rate=sample_rate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )


def extract_embeddings(
    checkpoint: LoadedCheckpoint,
    metadata_path: str | Path,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: torch.device | None = None,
    raw_audio_root: str | Path | None = None,
) -> ExtractedEmbeddings:
    metadata_path = resolve_project_path(metadata_path)
    target_device = device or next(checkpoint.model.parameters()).device
    sample_rate = int(checkpoint.config["sample_rate"])
    eval_batch_size = int(batch_size or checkpoint.config.get("eval_batch_size", 16))
    loader_workers = int(num_workers if num_workers is not None else checkpoint.config.get("num_workers", 0))

    dataloader = build_extraction_dataloader(
        metadata_path=metadata_path,
        feature_extractor=checkpoint.feature_extractor,
        label_encoder=checkpoint.label_encoder,
        sample_rate=sample_rate,
        batch_size=eval_batch_size,
        num_workers=loader_workers,
        raw_audio_root=raw_audio_root,
    )

    metadata_rows: list[dict[str, Any]] = []
    layer_embeddings_batches: list[np.ndarray] = []
    logits_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    true_label_batches: list[np.ndarray] = []
    pred_label_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="extract", leave=False):
            metadata = batch["metadata"]
            model_inputs = move_batch_to_device(batch, target_device)
            outputs = checkpoint.model(
                input_values=model_inputs["input_values"],
                attention_mask=model_inputs.get("attention_mask"),
                output_hidden_states=True,
            )

            pooled_layers = [
                checkpoint.model.pool_hidden_states(hidden_state, model_inputs.get("attention_mask"))
                for hidden_state in (outputs.hidden_states or ())
            ]
            pooled_layers_tensor = torch.stack(pooled_layers, dim=1).cpu()
            logits = outputs.logits.cpu()
            probabilities = torch.softmax(logits, dim=-1)
            pred_ids = probabilities.argmax(dim=-1)
            true_ids = model_inputs["labels"].cpu()

            for meta, true_id, pred_id in zip(metadata, true_ids.tolist(), pred_ids.tolist()):
                row = dict(meta)
                row["true_label_id"] = int(true_id)
                row["true_label"] = checkpoint.label_encoder.decode(int(true_id))
                row["pred_label_id"] = int(pred_id)
                row["pred_label"] = checkpoint.label_encoder.decode(int(pred_id))
                metadata_rows.append(row)

            layer_embeddings_batches.append(pooled_layers_tensor.numpy().astype(np.float32))
            logits_batches.append(logits.numpy().astype(np.float32))
            probability_batches.append(probabilities.numpy().astype(np.float32))
            true_label_batches.append(true_ids.numpy().astype(np.int64))
            pred_label_batches.append(pred_ids.numpy().astype(np.int64))

    return ExtractedEmbeddings(
        metadata=pd.DataFrame(metadata_rows),
        layer_embeddings=np.concatenate(layer_embeddings_batches, axis=0),
        logits=np.concatenate(logits_batches, axis=0),
        probabilities=np.concatenate(probability_batches, axis=0),
        true_label_ids=np.concatenate(true_label_batches, axis=0),
        pred_label_ids=np.concatenate(pred_label_batches, axis=0),
    )


def save_extracted_embeddings(
    embeddings: ExtractedEmbeddings,
    output_dir: str | Path,
    checkpoint: LoadedCheckpoint,
    metadata_path: str | Path,
) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings.metadata.to_csv(output_dir / "embedding_metadata.csv", index=False)
    np.savez_compressed(
        output_dir / "embedding_arrays.npz",
        layer_embeddings=embeddings.layer_embeddings,
        logits=embeddings.logits,
        probabilities=embeddings.probabilities,
        true_label_ids=embeddings.true_label_ids,
        pred_label_ids=embeddings.pred_label_ids,
    )

    summary = {
        "checkpoint_dir": str(checkpoint.checkpoint_dir),
        "metadata_path": str(resolve_project_path(metadata_path)),
        "num_samples": int(embeddings.layer_embeddings.shape[0]),
        "num_layers": int(embeddings.layer_embeddings.shape[1]),
        "hidden_size": int(embeddings.layer_embeddings.shape[2]),
        "num_labels": int(len(checkpoint.label_encoder.labels)),
        "label_names": checkpoint.label_encoder.labels,
    }
    with (output_dir / "embedding_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return output_dir


def extract_and_save_embeddings(
    checkpoint_dir: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: torch.device | None = None,
    raw_audio_root: str | Path | None = None,
) -> Path:
    checkpoint = load_trained_checkpoint(checkpoint_dir, device=device)
    embeddings = extract_embeddings(
        checkpoint=checkpoint,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        raw_audio_root=raw_audio_root,
    )
    return save_extracted_embeddings(
        embeddings=embeddings,
        output_dir=output_dir,
        checkpoint=checkpoint,
        metadata_path=metadata_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pooled wav2vec2 embeddings from a trained checkpoint.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint directory from training.")
    parser.add_argument("--metadata-path", type=Path, required=True, help="RAVDESS metadata CSV path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save extracted embeddings.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override extraction batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data loader workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = extract_and_save_embeddings(
        checkpoint_dir=args.checkpoint_dir,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Saved extracted embeddings to {output_dir}")


if __name__ == "__main__":
    main()
