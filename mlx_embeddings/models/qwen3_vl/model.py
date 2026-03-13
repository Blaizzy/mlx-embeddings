from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask
from mlx_vlm.models.qwen3_vl import Model as Qwen3VLBackbone
from mlx_vlm.models.qwen3_vl import LanguageModel as Qwen3VLLanguageModel
from mlx_vlm.models.qwen3_vl import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_vl import VisionModel as Qwen3VLVisionModel

from ..base import BaseModelArgs, BaseModelOutput, normalize_embeddings



def build_qwen3_vl_config(vlm_config: Dict[str, Any]) -> ModelConfig:
    base_config = dict(vlm_config)
    base_config["model_type"] = "qwen3_vl"

    config = ModelConfig.from_dict(base_config)
    if isinstance(config.text_config, dict):
        config.text_config = TextConfig.from_dict(config.text_config)
    if isinstance(config.vision_config, dict):
        config.vision_config = VisionConfig.from_dict(config.vision_config)
    return config


def last_non_padding_token(
    hidden_states: mx.array,
    attention_mask: Optional[mx.array],
) -> mx.array:
    if attention_mask is None:
        return hidden_states[:, -1]

    last_one_positions = mx.argmax(attention_mask[:, ::-1], axis=1)
    token_positions = attention_mask.shape[1] - last_one_positions - 1
    batch_positions = mx.arange(hidden_states.shape[0])
    return hidden_states[batch_positions, token_positions]


def compute_qwen3_vl_hidden_states(
    model: Qwen3VLBackbone,
    input_ids: mx.array,
    attention_mask: Optional[mx.array] = None,
    position_ids: Optional[mx.array] = None,
    inputs_embeds: Optional[mx.array] = None,
    pixel_values: Optional[mx.array] = None,
    image_grid_thw: Optional[mx.array] = None,
    video_grid_thw: Optional[mx.array] = None,
    cache=None,
) -> mx.array:
    if input_ids is None:
        raise ValueError("`input_ids` is required for Qwen3-VL embedding and reranking.")

    visual_pos_masks = None
    deepstack_visual_embeds = None

    if inputs_embeds is None:
        input_embeddings = model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mask=attention_mask,
        )
        inputs_embeds = input_embeddings.inputs_embeds
        visual_pos_masks = input_embeddings.visual_pos_masks
        deepstack_visual_embeds = input_embeddings.deepstack_visual_embeds

    language_model = model.language_model
    if pixel_values is not None:
        language_model._rope_deltas = None
        language_model._position_ids = None

    rope_mask = attention_mask
    model_mask = attention_mask
    if attention_mask is not None and attention_mask.ndim == 2:
        seq_length = attention_mask.shape[-1]
        causal_mask = create_causal_mask(seq_length)
        valid_tokens = attention_mask.astype(mx.bool_)
        key_mask = mx.expand_dims(valid_tokens, axis=(1, 2))
        query_mask = mx.expand_dims(valid_tokens, axis=(1, 3))
        model_mask = causal_mask[None, None, :, :] & key_mask & query_mask
    if attention_mask is not None and attention_mask.shape[-1] != input_ids.shape[-1]:
        rope_mask = None

    if position_ids is None and (rope_mask is None or rope_mask.ndim == 2):
        if (
            language_model._position_ids is not None
            and language_model._position_ids.shape[-1] >= input_ids.shape[-1]
        ):
            position_ids = language_model._position_ids[:, :, : input_ids.shape[-1]]
        else:
            position_ids, rope_deltas = language_model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=rope_mask,
            )
            language_model._rope_deltas = rope_deltas
            language_model._position_ids = position_ids

    return language_model.model(
        input_ids,
        inputs_embeds=inputs_embeds,
        mask=model_mask,
        cache=cache,
        position_ids=position_ids,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )


@dataclass
class ModelOutput(BaseModelOutput):
    logits: Optional[mx.array] = None
    scores: Optional[mx.array] = None


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Dict[str, Any]
    vision_config: Dict[str, Any]
    vlm_config: Dict[str, Any]
    model_type: str = "qwen3_vl"
    normalize: bool = True
    yes_token_id: int = 9693
    no_token_id: int = 2152

    @classmethod
    def from_dict(cls, params):
        text_config_raw = params.get("text_config", {})
        vision_config_raw = params.get("vision_config", {})

        text_config = asdict(TextConfig.from_dict(text_config_raw)) if text_config_raw else {}
        vision_config = asdict(VisionConfig.from_dict(vision_config_raw)) if vision_config_raw else {}

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            vlm_config=dict(params),
            model_type=params.get("model_type", "qwen3_vl"),
            normalize=params.get("normalize", True),
            yes_token_id=params.get("yes_token_id", 9693),
            no_token_id=params.get("no_token_id", 2152),
        )


class Model(nn.Module):
    LanguageModel = Qwen3VLLanguageModel
    VisionModel = Qwen3VLVisionModel

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = Qwen3VLBackbone(build_qwen3_vl_config(config.vlm_config))

    def get_input_embeddings(self):
        return self.model.language_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model.language_model.model = decoder

    def get_decoder(self):
        return self.model.language_model.model

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.vision_tower

    def get_image_features(
        self,
        pixel_values: mx.array,
        image_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        return self.model.vision_tower(pixel_values, image_grid_thw)[0]

    def get_video_features(
        self,
        pixel_values: mx.array,
        video_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        return self.model.vision_tower(pixel_values, video_grid_thw)[0]

    def get_binary_weight(self) -> mx.array:
        if hasattr(self.model.language_model, "lm_head"):
            lm_head_weight = self.model.language_model.lm_head.weight
        else:
            lm_head_weight = self.model.language_model.model.embed_tokens.weight

        return (
            lm_head_weight[self.config.yes_token_id]
            - lm_head_weight[self.config.no_token_id]
        )

    def __call__(
        self,
        input_ids: mx.array = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values=None,
        inputs_embeds: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        cache=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ) -> ModelOutput:
        del kwargs, cache_position, logits_to_keep

        if pixel_values is None:
            pixel_values = pixel_values_videos
        if cache is None:
            cache = past_key_values

        hidden_states = compute_qwen3_vl_hidden_states(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache=cache,
        )

        pooled = last_non_padding_token(hidden_states, attention_mask)
        text_embeds = (
            normalize_embeddings(pooled) if self.config.normalize else pooled
        )
        logits = mx.sum(pooled * self.get_binary_weight(), axis=-1)
        scores = mx.sigmoid(logits)

        return ModelOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            text_embeds=text_embeds,
            logits=logits,
            scores=scores,
        )

    def compute_scores(self, inputs: Dict[str, mx.array]) -> mx.array:
        return self(**inputs).scores

    def embed(self, inputs, processor, **kwargs) -> mx.array:
        model_inputs = processor.prepare_embedding_inputs(inputs, **kwargs)
        return self(**model_inputs).text_embeds

    def rerank(self, inputs, processor, **kwargs) -> mx.array:
        model_inputs = processor.prepare_reranker_inputs(inputs, **kwargs)
        if model_inputs is None:
            return mx.array([])
        return self(**model_inputs).scores

    def process(self, inputs, processor, **kwargs):
        if processor is None or not hasattr(processor, "prepare_model_inputs"):
            raise ValueError(
                "Qwen3-VL high-level processing requires the custom Qwen3-VL processor."
            )

        if isinstance(inputs, dict) and "documents" in inputs:
            return self.rerank(inputs, processor, **kwargs)
        return self.embed(inputs, processor, **kwargs)

    def sanitize(self, weights):
        sanitized = {}
        language_prefix = "model.language_model."
        language_model_prefix = f"{language_prefix}model."
        vision_prefix = "model.vision_tower."

        for key, value in weights.items():
            mapped_key = key

            if key.startswith(language_model_prefix):
                mapped_key = key
            elif key.startswith(language_prefix):
                mapped_key = key.replace(language_prefix, language_model_prefix, 1)
            elif key.startswith("model.visual."):
                mapped_key = key.replace("model.visual.", vision_prefix, 1)
            elif key.startswith(vision_prefix):
                mapped_key = key
            elif key.startswith("model.lm_head"):
                mapped_key = key.replace(
                    "model.lm_head", "model.language_model.lm_head", 1
                )
            elif key.startswith("lm_head"):
                mapped_key = key.replace(
                    "lm_head", "model.language_model.lm_head", 1
                )

            if mapped_key.startswith(vision_prefix):
                vision_key = mapped_key[len(vision_prefix) :]
                for sanitized_key, sanitized_value in self.model.vision_tower.sanitize(
                    {vision_key: value}
                ).items():
                    sanitized[f"{vision_prefix}{sanitized_key}"] = sanitized_value
            else:
                sanitized[mapped_key] = value

        return sanitized
