from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


@dataclass
class Wav2VecEmotionOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    pooled_output: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None


class Wav2VecEmotionClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_labels: int,
        dropout: float = 0.2,
        freeze_feature_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

        if freeze_feature_encoder:
            self.set_feature_encoder_frozen(True)

    def set_feature_encoder_frozen(self, frozen: bool) -> None:
        if hasattr(self.backbone, "feature_extractor"):
            for param in self.backbone.feature_extractor.parameters():
                param.requires_grad = not frozen

    def _get_feature_attention_mask(
        self,
        sequence_length: int,
        attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        if hasattr(self.backbone, "_get_feature_vector_attention_mask"):
            return self.backbone._get_feature_vector_attention_mask(sequence_length, attention_mask).to(device)

        return None

    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked_hidden = hidden_states * mask
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return masked_hidden.sum(dim=1) / lengths

    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        feature_mask = self._get_feature_attention_mask(
            sequence_length=hidden_states.shape[1],
            attention_mask=attention_mask,
            device=hidden_states.device,
        )
        return self._mean_pool(hidden_states, feature_mask)

    def extract_all_layer_pooled_outputs(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        pooled_outputs = []
        for hidden in outputs.hidden_states:
            pooled_outputs.append(self.pool_hidden_states(hidden, attention_mask))
        return pooled_outputs

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> Wav2VecEmotionOutput:
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        pooled_output = self.pool_hidden_states(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=class_weights)

        return Wav2VecEmotionOutput(
            loss=loss,
            logits=logits,
            pooled_output=pooled_output,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )
