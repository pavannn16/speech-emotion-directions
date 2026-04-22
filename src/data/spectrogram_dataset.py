from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.dataset import EmotionLabelEncoder, load_audio_array


@dataclass
class SpectrogramConfig:
    sample_rate: int = 16_000
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    fmin: float = 0.0
    fmax: float | None = 8_000.0
    target_frames: int = 600


def waveform_to_log_mel_spectrogram(
    waveform: np.ndarray,
    config: SpectrogramConfig,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=lambda x: np.max(x) if np.size(x) else 1.0)

    current_frames = int(log_mel.shape[1])
    if current_frames < config.target_frames:
        pad_width = config.target_frames - current_frames
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
    elif current_frames > config.target_frames:
        log_mel = log_mel[:, : config.target_frames]

    mean = float(log_mel.mean())
    std = float(log_mel.std())
    if std > 0:
        log_mel = (log_mel - mean) / std
    else:
        log_mel = log_mel - mean

    return log_mel.astype(np.float32)


class RavdessSpectrogramDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        label_encoder: EmotionLabelEncoder,
        spectrogram_config: SpectrogramConfig,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.spectrogram_config = spectrogram_config

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        waveform = load_audio_array(row["file_path"], sample_rate=self.spectrogram_config.sample_rate)
        features = waveform_to_log_mel_spectrogram(waveform, self.spectrogram_config)
        label = self.label_encoder.encode(row["final_label"])

        return {
            "features": torch.from_numpy(features).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.long),
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


def spectrogram_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    features = torch.stack([item["features"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
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
    return {"features": features, "labels": labels, "metadata": metadata}
