"""Stage-1 image classifier: no mask or region target participates in training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from e1.model import QwenE1


@dataclass
class ClassificationOutput:
    anomaly_logits: torch.Tensor
    loss: torch.Tensor | None = None
    classification_loss: torch.Tensor | None = None


class QwenE1Classification(QwenE1):
    """Reuse the visual feature adapter, but remove every pixel/region objective.

    The mask and region heads are frozen and disconnected from the classification
    path.  Global average pooled visual features condition the virtual prompt, so
    classification gradients cannot reward an all-normal segmentation mask.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for parameter in self.decoder.mask_head.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.region_head.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        sample_labels: torch.Tensor | None = None,
    ) -> ClassificationOutput:
        features, image_embeds, deepstack_features = self._vision_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        # PatchDecoder returns mask/region outputs, but those frozen heads are
        # not used here.  Only its multi-layer feature projections and fusion
        # blocks receive gradients from the image-level classification loss.
        _, _, fused = self.decoder(features)
        virtual_prompt = self.prompt_learner.forward_global(fused)
        vocabulary_logits = self._language_logits(inputs, image_embeds, virtual_prompt, deepstack_features)
        yes = torch.logsumexp(vocabulary_logits[:, self.yes_token_ids], dim=1)
        no = torch.logsumexp(vocabulary_logits[:, self.no_token_ids], dim=1)
        anomaly_logits = yes - no

        classification = None
        if sample_labels is not None:
            classification = F.binary_cross_entropy_with_logits(
                anomaly_logits, sample_labels.to(anomaly_logits.dtype)
            )
        return ClassificationOutput(
            anomaly_logits=anomaly_logits,
            loss=classification,
            classification_loss=classification,
        )
