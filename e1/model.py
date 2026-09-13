"""E1：冻结 Qwen2.5-VL，只训练多层 patch decoder 与条件虚拟 prompt。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import get_vision_window_index


@dataclass
class E1Output:
    anomaly_logits: torch.Tensor
    mask_logits: torch.Tensor
    region_logits: torch.Tensor
    loss: torch.Tensor | None = None
    classification_loss: torch.Tensor | None = None
    segmentation_loss: torch.Tensor | None = None
    region_loss: torch.Tensor | None = None
    focal_loss: torch.Tensor | None = None
    dice_loss: torch.Tensor | None = None
    dice_loss_nonempty: torch.Tensor | None = None
    dice_loss_empty: torch.Tensor | None = None
    dice_nonempty_count: int = 0
    dice_empty_count: int = 0
    mask_probability_mean: torch.Tensor | None = None
    predicted_area: torch.Tensor | None = None


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class PatchDecoder(nn.Module):
    def __init__(self, vision_hidden: int, n_layers: int, channels: int = 256) -> None:
        super().__init__()
        if channels % n_layers:
            raise ValueError("decoder channels 必须能被 feature layer 数整除")
        branch = channels // n_layers
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(vision_hidden, branch, 1),
                nn.GroupNorm(8, branch),
                nn.GELU(),
            )
            for _ in range(n_layers)
        ])
        self.fusion = nn.Sequential(
            ResidualConvBlock(channels),
            ResidualConvBlock(channels),
        )
        self.mask_head = nn.Conv2d(channels, 1, 1)
        self.region_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 10),  # 9 个位置 + none
        )

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError("vision feature 数量与 decoder projection 不一致")
        fused = torch.cat(
            [projection(feature.float()) for projection, feature in zip(self.projections, features)], dim=1
        )
        fused = self.fusion(fused)
        return self.mask_head(fused), self.region_head(fused), fused


class ConditionalPromptLearner(nn.Module):
    def __init__(self, decoder_channels: int, text_hidden: int, n_virtual_tokens: int = 4) -> None:
        super().__init__()
        self.n_virtual_tokens = n_virtual_tokens
        self.base_prompt = nn.Parameter(torch.zeros(n_virtual_tokens, text_hidden))
        nn.init.normal_(self.base_prompt, std=0.02)
        self.conditioner = nn.Sequential(
            nn.Linear(decoder_channels, 512),
            nn.GELU(),
            nn.Linear(512, n_virtual_tokens * text_hidden),
        )

    def forward(self, fused: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(mask_logits).flatten(2)
        values = fused.flatten(2)
        pooled = (values * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)
        return self._condition(pooled)

    def forward_global(self, fused: torch.Tensor) -> torch.Tensor:
        """Classification-only prompt without mask-conditioned spatial pooling."""
        return self._condition(fused.float().mean(dim=(-2, -1)))

    def _condition(self, pooled: torch.Tensor) -> torch.Tensor:
        conditioned = self.conditioner(pooled).view(
            pooled.shape[0], self.n_virtual_tokens, -1
        )
        return conditioned + self.base_prompt.unsqueeze(0)


def segmentation_loss_components(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_nonempty_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    # Reductions span 448 * 448 pixels.  Keep the loss math in FP32 even when
    # the decoder forward runs under autocast: an FP16 reduction/gradient can
    # overflow on an otherwise finite example before GradScaler unscales it.
    logits = logits.float()
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha = 0.25 * targets + 0.75 * (1 - targets)
    focal = (alpha * (1 - pt).pow(2) * bce).mean()
    intersection = (probabilities * targets).flatten(1).sum(1)
    denominator = probabilities.flatten(1).sum(1) + targets.flatten(1).sum(1)
    dice_per_example = 1 - (2 * intersection + 1) / (denominator + 1)
    nonempty = targets.flatten(1).sum(1) > 0
    if dice_nonempty_only:
        dice = dice_per_example[nonempty].mean() if nonempty.any() else logits.float().sum() * 0.0
    else:
        dice = dice_per_example.mean()
    return focal + dice, focal, dice, int(nonempty.sum().item()), int((~nonempty).sum().item())


def dice_loss_by_mask(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean Dice loss for non-empty and empty GT masks separately."""
    probabilities = torch.sigmoid(logits.float())
    targets = targets.float()
    intersection = (probabilities * targets).flatten(1).sum(1)
    denominator = probabilities.flatten(1).sum(1) + targets.flatten(1).sum(1)
    values = 1 - (2 * intersection + 1) / (denominator + 1)
    nonempty = targets.flatten(1).sum(1) > 0
    nonempty_loss = values[nonempty].mean() if nonempty.any() else values.sum() * 0.0
    empty_loss = values[~nonempty].mean() if (~nonempty).any() else values.sum() * 0.0
    return nonempty_loss, empty_loss


def focal_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return segmentation_loss_components(logits, targets)[0]


class QwenE1(nn.Module):
    """Qwen 基座全冻；梯度只流向 decoder 与 prompt_learner。"""

    def __init__(
        self,
        base_model: nn.Module,
        yes_token_ids: list[int],
        no_token_ids: list[int],
        feature_layers: tuple[int, ...] = (7, 15, 23, 31),
        decoder_channels: int = 288,
        n_virtual_tokens: int = 4,
        lambda_seg: float = 0.2,
        lambda_region: float = 0.1,
        dice_nonempty_only: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.feature_layers = feature_layers
        self.lambda_seg = lambda_seg
        self.lambda_region = lambda_region
        self.dice_nonempty_only = bool(dice_nonempty_only)
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        vision_config = base_model.config.vision_config
        text_config = base_model.config.text_config
        self.decoder = PatchDecoder(
            vision_hidden=vision_config.hidden_size,
            n_layers=len(feature_layers),
            channels=decoder_channels,
        )
        self.prompt_learner = ConditionalPromptLearner(
            decoder_channels, text_config.hidden_size, n_virtual_tokens
        )
        self.register_buffer("yes_token_ids", torch.tensor(yes_token_ids, dtype=torch.long))
        self.register_buffer("no_token_ids", torch.tensor(no_token_ids, dtype=torch.long))

    @property
    def visual(self):
        return self.base_model.model.visual

    def train(self, mode: bool = True):
        super().train(mode)
        # 冻结基座始终 eval；只有新增模块进入 train。
        self.base_model.eval()
        self.decoder.train(mode)
        self.prompt_learner.train(mode)
        return self

    def _restore_spatial(self, hidden: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        visual = self.visual
        # Qwen3-VL has no sliding-window vision attention. Its block outputs are
        # already ordered patch tokens after spatial merging.
        if not hasattr(visual, "window_size"):
            maps = []
            offset = 0
            for shape in grid_thw.tolist():
                temporal, height, width = (int(value) for value in shape)
                # Hooks on Qwen3 blocks see the pre-merger sequence. The merger
                # is only for the frozen LLM image tokens; decoder keeps the
                # full HxW patch map so no spatial information is discarded.
                count = temporal * height * width
                part = hidden[offset:offset + count]
                offset += count
                merge = visual.spatial_merge_size
                feature = part.reshape(temporal, height // merge, width // merge,
                                       merge, merge, -1)
                feature = feature.permute(0, 1, 3, 2, 4, 5).reshape(
                    temporal, height, width, -1).mean(0)
                maps.append(feature.permute(2, 0, 1).contiguous())
            if offset != len(hidden):
                raise ValueError('Vision patch count does not match image_grid_thw')
            return torch.stack(maps)
        window_index, _ = get_vision_window_index(
            grid_thw,
            spatial_merge_size=visual.spatial_merge_size,
            window_size=visual.window_size,
            patch_size=visual.patch_size,
        )
        unit = visual.spatial_merge_unit
        grouped = hidden.reshape(hidden.shape[0] // unit, unit, hidden.shape[-1])
        restored = grouped[torch.argsort(window_index)].reshape_as(hidden)
        parts = torch.split(restored, grid_thw.prod(-1).tolist())
        maps = []
        for part, shape in zip(parts, grid_thw.tolist()):
            temporal, height, width = (int(value) for value in shape)
            feature = part.reshape(temporal, height, width, -1).mean(0)
            maps.append(feature.permute(2, 0, 1).contiguous())
        spatial_shapes = {tuple(value.shape[-2:]) for value in maps}
        if len(spatial_shapes) != 1:
            raise ValueError("E1 batch 要求固定 direct448 网格")
        return torch.stack(maps)

    def _vision_features(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer_index in self.feature_layers:
            def capture(_module, _inputs, output, index=layer_index):
                captured[index] = output
            handles.append(self.visual.blocks[layer_index].register_forward_hook(capture))
        try:
            with torch.no_grad():
                vision_outputs = self.visual(
                    pixel_values.to(dtype=self.visual.dtype), grid_thw=grid_thw
                )
        finally:
            for handle in handles:
                handle.remove()
        features = [self._restore_spatial(captured[index], grid_thw) for index in self.feature_layers]
        deepstack = getattr(vision_outputs, "deepstack_features", None)
        if deepstack is not None:
            deepstack = [value.detach() for value in deepstack]
        return features, vision_outputs.pooler_output.detach(), deepstack

    def _language_logits(
        self,
        inputs: dict[str, torch.Tensor],
        image_embeds: torch.Tensor,
        virtual_prompt: torch.Tensor,
        deepstack_features=None,
    ) -> torch.Tensor:
        model = self.base_model.model
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        embeddings = model.get_input_embeddings()(input_ids)
        image_mask, _ = model.get_placeholder_mask(
            input_ids, inputs_embeds=embeddings, image_features=image_embeds
        )
        embeddings = embeddings.masked_scatter(
            image_mask, image_embeds.to(embeddings.device, embeddings.dtype)
        )
        position_kwargs = dict(
            input_ids=input_ids,
            image_grid_thw=inputs["image_grid_thw"],
            video_grid_thw=None,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
            second_per_grid_ts=None,
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )
        try:
            position_ids = model.compute_3d_position_ids(**position_kwargs)
        except TypeError as error:
            if "second_per_grid_ts" not in str(error):
                raise
            position_kwargs.pop("second_per_grid_ts")
            position_ids = model.compute_3d_position_ids(**position_kwargs)
        if position_ids is None:
            raise RuntimeError("processor 未提供构造 Qwen MRoPE 所需的 mm_token_type_ids")
        n_virtual = virtual_prompt.shape[1]
        prompt_positions = torch.arange(n_virtual, device=embeddings.device)
        prompt_positions = prompt_positions.view(1, 1, -1).expand(3, embeddings.shape[0], -1)
        position_ids = torch.cat([prompt_positions, position_ids + n_virtual], dim=-1)
        embeddings = torch.cat([virtual_prompt.to(embeddings.dtype), embeddings], dim=1)
        prompt_mask = torch.ones(
            attention_mask.shape[0], n_virtual,
            device=attention_mask.device, dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        language_kwargs = dict(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        if deepstack_features is not None and len(deepstack_features):
            visual_pos_masks = image_mask[..., 0] if image_mask.ndim == 3 else image_mask
            prompt_false = torch.zeros(
                visual_pos_masks.shape[0], n_virtual,
                device=visual_pos_masks.device, dtype=visual_pos_masks.dtype,
            )
            language_kwargs["visual_pos_masks"] = torch.cat([prompt_false, visual_pos_masks], dim=1)
            language_kwargs["deepstack_visual_embeds"] = deepstack_features
        outputs = model.language_model(**language_kwargs)
        return self.base_model.lm_head(outputs.last_hidden_state[:, -1, :]).float()

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        sample_labels: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        region_labels: torch.Tensor | None = None,
    ) -> E1Output:
        features, image_embeds, deepstack_features = self._vision_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        lowres_mask, region_logits, fused = self.decoder(features)
        mask_logits = F.interpolate(
            lowres_mask, size=(448, 448), mode="bilinear", align_corners=False
        )
        virtual_prompt = self.prompt_learner(fused, lowres_mask)
        vocabulary_logits = self._language_logits(inputs, image_embeds, virtual_prompt, deepstack_features)
        yes = torch.logsumexp(vocabulary_logits[:, self.yes_token_ids], dim=1)
        no = torch.logsumexp(vocabulary_logits[:, self.no_token_ids], dim=1)
        anomaly_logits = yes - no

        total = classification = segmentation = region = None
        focal = dice = dice_nonempty = dice_empty = None
        dice_nonempty_count = dice_empty_count = 0
        if sample_labels is not None:
            classification = F.binary_cross_entropy_with_logits(
                anomaly_logits, sample_labels.to(anomaly_logits.dtype)
            )
            total = classification
        if masks is not None:
            segmentation, focal, dice, dice_nonempty_count, dice_empty_count = segmentation_loss_components(
                mask_logits, masks, dice_nonempty_only=self.dice_nonempty_only
            )
            dice_nonempty, dice_empty = dice_loss_by_mask(mask_logits, masks)
            total = self.lambda_seg * segmentation if total is None else total + self.lambda_seg * segmentation
        if region_labels is not None:
            region = F.cross_entropy(region_logits.float(), region_labels.long())
            total = self.lambda_region * region if total is None else total + self.lambda_region * region
        return E1Output(
            anomaly_logits=anomaly_logits,
            mask_logits=mask_logits,
            region_logits=region_logits,
            loss=total,
            classification_loss=classification,
            segmentation_loss=segmentation,
            region_loss=region,
            focal_loss=focal,
            dice_loss=dice,
            dice_loss_nonempty=dice_nonempty,
            dice_loss_empty=dice_empty,
            dice_nonempty_count=dice_nonempty_count,
            dice_empty_count=dice_empty_count,
            mask_probability_mean=torch.sigmoid(mask_logits).mean().detach(),
            predicted_area=(torch.sigmoid(mask_logits) >= 0.5).float().mean().detach(),
        )

    def segmentation_only(self, inputs: dict[str, torch.Tensor], masks: torch.Tensor) -> E1Output:
        """Use frozen visual features and a supervised mask decoder only."""
        if tuple(masks.shape[-2:]) != (448, 448) or not (masks.flatten(1).sum(1) > 0).all():
            raise ValueError('Segmentation training requires nonempty 448x448 anomaly masks')
        features, _, _ = self._vision_features(inputs["pixel_values"], inputs["image_grid_thw"])
        lowres, region, _ = self.decoder(features)
        logits = F.interpolate(lowres, size=(448, 448), mode="bilinear", align_corners=False)
        loss, focal, dice, n, e = segmentation_loss_components(logits, masks, dice_nonempty_only=True)
        return E1Output(anomaly_logits=logits.flatten(1).mean(1), mask_logits=logits,
                        region_logits=region, loss=loss, segmentation_loss=loss,
                        focal_loss=focal, dice_loss=dice, dice_nonempty_count=n,
                        dice_empty_count=e, mask_probability_mean=torch.sigmoid(logits).mean(),
                        predicted_area=(torch.sigmoid(logits)>=0.5).float().mean())

    def trainable_parameter_summary(self) -> dict[str, int]:
        result = {"decoder": 0, "prompt_learner": 0, "base_model": 0}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("decoder"):
                result["decoder"] += parameter.numel()
            elif name.startswith("prompt_learner"):
                result["prompt_learner"] += parameter.numel()
            else:
                result["base_model"] += parameter.numel()
        return result
