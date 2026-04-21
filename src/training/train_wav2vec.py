from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup

from src.data.dataset import (
    EmotionLabelEncoder,
    RavdessWav2VecDataset,
    Wav2VecCollator,
    compute_class_weights,
    load_project_metadata,
)
from src.models.wav2vec_classifier import Wav2VecEmotionClassifier
from src.training.metrics import classification_report_frame, summarize_classification
from src.utils.config import load_yaml_config
from src.utils.paths import PROJECT_ROOT
from src.utils.seed import set_seed


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_best_available_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if key == "metadata":
            continue
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def maybe_limit_samples(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return df
    return df.head(limit).reset_index(drop=True)


def build_dataloaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], EmotionLabelEncoder, AutoFeatureExtractor]:
    label_encoder = EmotionLabelEncoder()
    metadata_path = resolve_project_path(config["metadata_path"])
    sample_rate = int(config["sample_rate"])
    feature_extractor = AutoFeatureExtractor.from_pretrained(config["backbone_name"])

    train_df = maybe_limit_samples(load_project_metadata(metadata_path, split="train"), config.get("max_train_samples"))
    val_df = maybe_limit_samples(load_project_metadata(metadata_path, split="val"), config.get("max_val_samples"))
    test_df = maybe_limit_samples(load_project_metadata(metadata_path, split="test"), config.get("max_test_samples"))

    train_dataset = RavdessWav2VecDataset(train_df, label_encoder=label_encoder, sample_rate=sample_rate)
    val_dataset = RavdessWav2VecDataset(val_df, label_encoder=label_encoder, sample_rate=sample_rate)
    test_dataset = RavdessWav2VecDataset(test_df, label_encoder=label_encoder, sample_rate=sample_rate)

    collator = Wav2VecCollator(feature_extractor=feature_extractor, sample_rate=sample_rate)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=collator,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=int(config["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=collator,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=int(config["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=collator,
        ),
    }

    return dataloaders, label_encoder, feature_extractor


def build_model(config: dict[str, Any], label_encoder: EmotionLabelEncoder) -> Wav2VecEmotionClassifier:
    return Wav2VecEmotionClassifier(
        backbone_name=config["backbone_name"],
        num_labels=len(label_encoder.labels),
        dropout=float(config["dropout"]),
        freeze_feature_encoder=bool(config.get("freeze_feature_encoder", False)),
    )


def evaluate_model(
    model: Wav2VecEmotionClassifier,
    dataloader: DataLoader,
    label_encoder: EmotionLabelEncoder,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            metadata = batch["metadata"]
            model_inputs = move_batch_to_device(batch, device)
            labels = model_inputs["labels"]
            outputs = model(
                input_values=model_inputs["input_values"],
                attention_mask=model_inputs.get("attention_mask"),
                labels=labels,
                class_weights=class_weights,
                output_hidden_states=False,
            )

            probs = torch.softmax(outputs.logits, dim=-1).cpu()
            preds = probs.argmax(dim=-1).tolist()
            true = labels.cpu().tolist()

            all_true.extend(true)
            all_pred.extend(preds)

            for meta, pred_id, true_id, prob_vec in zip(metadata, preds, true, probs.tolist()):
                row = dict(meta)
                row["true_label_id"] = int(true_id)
                row["pred_label_id"] = int(pred_id)
                row["true_label"] = label_encoder.decode(true_id)
                row["pred_label"] = label_encoder.decode(pred_id)
                for label_name, probability in zip(label_encoder.labels, prob_vec):
                    row[f"prob_{label_name}"] = float(probability)
                rows.append(row)

    metrics = summarize_classification(all_true, all_pred, label_encoder.labels)
    predictions_df = pd.DataFrame(rows)
    return metrics, predictions_df


def train_one_epoch(
    model: Wav2VecEmotionClassifier,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    device: torch.device,
    gradient_accumulation_steps: int,
    class_weights: torch.Tensor | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="train", leave=False), start=1):
        model_inputs = move_batch_to_device(batch, device)
        outputs = model(
            input_values=model_inputs["input_values"],
            attention_mask=model_inputs.get("attention_mask"),
            labels=model_inputs["labels"],
            class_weights=class_weights,
            output_hidden_states=False,
        )

        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        running_loss += float(loss.item()) * gradient_accumulation_steps

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    remaining = len(dataloader) % gradient_accumulation_steps
    if remaining != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return running_loss / max(len(dataloader), 1)


def save_training_artifacts(
    output_dir: Path,
    model: Wav2VecEmotionClassifier,
    feature_extractor: AutoFeatureExtractor,
    config: dict[str, Any],
    label_encoder: EmotionLabelEncoder,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    test_predictions: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_state.pt")
    feature_extractor.save_pretrained(output_dir / "feature_extractor")

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "labels": label_encoder.labels,
                "label_to_id": label_encoder.label_to_id,
                "id_to_label": {str(key): value for key, value in label_encoder.id_to_label.items()},
            },
            handle,
            indent=2,
        )

    with (output_dir / "val_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(val_metrics, handle, indent=2)

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    classification_report_frame(test_metrics["classification_report"]).to_csv(
        output_dir / "test_classification_report.csv",
        index=True,
    )


def run_training(config: dict[str, Any], dry_run: bool = False) -> None:
    set_seed(int(config["seed"]))

    dataloaders, label_encoder, feature_extractor = build_dataloaders(config)
    device = get_best_available_device()

    train_df = load_project_metadata(resolve_project_path(config["metadata_path"]), split="train")
    class_weights = None
    if bool(config.get("use_class_weights", False)):
        class_weights = compute_class_weights(train_df, label_encoder).to(device)

    model = build_model(config, label_encoder).to(device)

    if dry_run:
        print("Dry run completed successfully.")
        print(f"device={device}")
        print(f"train_size={len(dataloaders['train'].dataset)}")
        print(f"val_size={len(dataloaders['val'].dataset)}")
        print(f"test_size={len(dataloaders['test'].dataset)}")
        print(f"labels={label_encoder.labels}")
        if class_weights is not None:
            print(f"class_weights={class_weights.tolist()}")
        return

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    total_train_steps = len(dataloaders["train"]) * int(config["num_epochs"])
    warmup_steps = int(total_train_steps * float(config.get("warmup_ratio", 0.1)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_train_steps, 1),
    )

    best_macro_f1 = -1.0
    best_state = None
    patience = int(config.get("early_stopping_patience", 4))
    patience_counter = 0
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))

    for epoch in range(int(config["num_epochs"])):
        freeze_epochs = int(config.get("freeze_feature_encoder_epochs", 0))
        model.set_feature_encoder_frozen(epoch < freeze_epochs)

        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            class_weights=class_weights,
        )
        val_metrics, _ = evaluate_model(
            model,
            dataloaders["val"],
            label_encoder,
            device,
            class_weights=class_weights,
        )
        val_macro_f1 = float(val_metrics["macro_f1"])
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, _ = evaluate_model(
        model,
        dataloaders["val"],
        label_encoder,
        device,
        class_weights=class_weights,
    )
    test_metrics, test_predictions = evaluate_model(
        model,
        dataloaders["test"],
        label_encoder,
        device,
        class_weights=class_weights,
    )

    output_dir = resolve_project_path(config["output_dir"])
    save_training_artifacts(
        output_dir=output_dir,
        model=model,
        feature_extractor=feature_extractor,
        config=config,
        label_encoder=label_encoder,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
    )

    print(f"Saved model and evaluation artifacts to {output_dir}")
    print(f"Final val macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Final test macro F1: {test_metrics['macro_f1']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune wav2vec2 on speaker-independent RAVDESS.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/wav2vec.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets and model, then exit without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    run_training(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
