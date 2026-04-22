from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


PROJECT_LABELS = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]


class EmotionLabelEncoder:
    def __init__(self, labels: list[str] | None = None) -> None:
        self.labels = labels or PROJECT_LABELS
        self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    def encode(self, label: str) -> int:
        return self.label_to_id[label]

    def decode(self, label_id: int) -> str:
        return self.id_to_label[int(label_id)]


def resolve_ravdess_audio_path(
    file_path: str | Path,
    actor_id: str,
    file_name: str,
    raw_audio_root: str | Path | None = None,
) -> Path:
    original_path = Path(file_path)
    if original_path.exists():
        return original_path.resolve()

    if raw_audio_root is not None:
        raw_audio_root = Path(raw_audio_root)
        actor_folder = f"Actor_{str(actor_id).zfill(2)}"
        candidates = [
            raw_audio_root / actor_folder / file_name,
            raw_audio_root / file_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

    return original_path


def rebase_metadata_audio_paths(
    metadata: pd.DataFrame,
    raw_audio_root: str | Path | None = None,
) -> pd.DataFrame:
    if raw_audio_root is None or "file_path" not in metadata.columns:
        return metadata

    rebased = metadata.copy()
    rebased_paths: list[str] = []
    missing_rows: list[tuple[str, str, str]] = []

    for row in rebased.itertuples(index=False):
        resolved_path = resolve_ravdess_audio_path(
            file_path=getattr(row, "file_path"),
            actor_id=getattr(row, "actor_id"),
            file_name=getattr(row, "file_name"),
            raw_audio_root=raw_audio_root,
        )
        rebased_paths.append(str(resolved_path))
        if not resolved_path.exists():
            missing_rows.append((str(getattr(row, "actor_id")), str(getattr(row, "file_name")), str(resolved_path)))

    rebased["file_path"] = rebased_paths

    if missing_rows:
        preview = ", ".join(
            f"actor={actor_id} file={file_name} path={resolved_path}"
            for actor_id, file_name, resolved_path in missing_rows[:5]
        )
        raise FileNotFoundError(
            "Could not resolve one or more RAVDESS audio files in the current runtime. "
            f"Examples: {preview}"
        )

    return rebased


def load_project_metadata(
    metadata_path: str | Path,
    split: str | None = None,
    raw_audio_root: str | Path | None = None,
) -> pd.DataFrame:
    metadata_path = Path(metadata_path)
    df = pd.read_csv(metadata_path)
    df = df[df["keep_for_project"].astype(bool)].copy()

    df["actor_id"] = df["actor_id"].astype(str).str.zfill(2)
    df["statement_code"] = df["statement_code"].astype(str)

    if split is not None:
        df = df[df["split"] == split].copy()

    df = rebase_metadata_audio_paths(df.reset_index(drop=True), raw_audio_root=raw_audio_root)
    return df.reset_index(drop=True)


def compute_class_weights(metadata: pd.DataFrame, label_encoder: EmotionLabelEncoder) -> torch.Tensor:
    counts = metadata["final_label"].value_counts()
    total = counts.sum()
    weights = []
    for label in label_encoder.labels:
        label_count = float(counts.get(label, 0))
        weight = total / (len(label_encoder.labels) * label_count)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def load_audio_array(audio_path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    waveform, original_sr = sf.read(audio_path)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    waveform = waveform.astype(np.float32)

    if original_sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=sample_rate)

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform = waveform / peak

    return waveform.astype(np.float32)


class RavdessWav2VecDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        label_encoder: EmotionLabelEncoder,
        sample_rate: int = 16_000,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        audio = load_audio_array(row["file_path"], sample_rate=self.sample_rate)
        label = self.label_encoder.encode(row["final_label"])

        return {
            "audio": audio,
            "label": label,
            "sample_index": index,
            "file_name": row["file_name"],
            "file_path": row["file_path"],
            "actor_id": row["actor_id"],
            "statement_code": row["statement_code"],
            "statement": row["statement"],
            "emotion": row["emotion"],
            "final_label": row["final_label"],
            "intensity": row["intensity"],
            "split": row["split"],
            "duration_seconds": float(row["duration_seconds"]),
        }


@dataclass
class Wav2VecCollator:
    feature_extractor: Any
    sample_rate: int = 16_000

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        audio_arrays = [item["audio"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        metadata = [
            {
                "file_name": item["file_name"],
                "file_path": item["file_path"],
                "actor_id": item["actor_id"],
                "statement_code": item["statement_code"],
                "statement": item["statement"],
                "emotion": item["emotion"],
                "final_label": item["final_label"],
                "intensity": item["intensity"],
                "split": item["split"],
                "duration_seconds": item["duration_seconds"],
            }
            for item in batch
        ]

        encoded = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        encoded["metadata"] = metadata
        return encoded
