from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CnnEmotionOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    embedding: torch.Tensor


class CnnEmotionClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        dropout: float = 0.3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> CnnEmotionOutput:
        encoded = self.encoder(features)
        embedding = self.embedding_head(encoded)
        logits = self.classifier(embedding)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=class_weights)

        return CnnEmotionOutput(
            loss=loss,
            logits=logits,
            embedding=embedding,
        )
