from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_causal_mask
from mlx_vlm.models.qwen3_vl import LanguageModel as Qwen3VLLanguageModel
from mlx_vlm.models.qwen3_vl import Model as Qwen3VLBackbone
from mlx_vlm.models.qwen3_vl import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_vl import VisionModel as Qwen3VLVisionModel

from ..base import BaseModelArgs, BaseModelOutput, normalize_embeddings


def _patched_deepstack_process(
    self,
    hidden_states: mx.array,
    visual_pos_masks: mx.array,
    visual_embeds: mx.array,
) -> mx.array:
    """Fixed version of mlx-vlm's Qwen3-VL ``_deepstack_process``.

    Upstream passes the full concatenated ``visual_embeds`` (all samples)
    into each sample's ``batch_result.at[batch_indices].add(...)``, which
    only broadcasts when the batch has one image. This version slices
    ``visual_embeds`` per sample using the running offset of image-token
    counts so it works for multi-image batches.
    """
    batch_size = hidden_states.shape[0]
    updated = []
    offset = 0
    for b in range(batch_size):
        batch_mask = visual_pos_masks[b]
        batch_hidden = hidden_states[b]
        batch_indices = mx.array(np.where(batch_mask)[0], dtype=mx.uint32)
        n = int(batch_indices.shape[0])
        if n == 0:
            updated.append(batch_hidden)
            continue
        batch_result = mx.array(batch_hidden)
        batch_result = batch_result.at[batch_indices].add(
            visual_embeds[offset : offset + n]
        )
        offset += n
        updated.append(batch_result)
    return mx.stack(updated, axis=0)


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
        raise ValueError(
            "`input_ids` is required for Qwen3-VL embedding and reranking."
        )

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

        text_config = (
            asdict(TextConfig.from_dict(text_config_raw)) if text_config_raw else {}
        )
        vision_config = (
            asdict(VisionConfig.from_dict(vision_config_raw))
            if vision_config_raw
            else {}
        )

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            vlm_config=dict(params),
            model_type=params.get("model_type", "qwen3_vl"),
            normalize=params.get("normalize", True),
            yes_token_id=params.get("yes_token_id", 9693),
            no_token_id=params.get("no_token_id", 2152),
        )


class Model(Qwen3VLBackbone):
    LanguageModel = Qwen3VLLanguageModel
    VisionModel = Qwen3VLVisionModel

    def __init__(self, config: ModelArgs):
        self.args = config
        super().__init__(build_qwen3_vl_config(config.vlm_config))
        # Fix upstream mlx-vlm Qwen3-VL bug (as of 0.4.4): _deepstack_process
        # indexes the full concatenated visual_embeds at each batch sample's
        # image positions, which is only correct for batch_size=1. Patch the
        # instance with a version that slices visual_embeds per sample.
        lm_inner = self.language_model.model
        lm_inner._deepstack_process = _patched_deepstack_process.__get__(
            lm_inner, type(lm_inner)
        )

    @property
    def visual(self):
        return self.vision_tower

    def get_image_features(
        self,
        pixel_values: mx.array,
        image_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        return self.vision_tower(pixel_values, image_grid_thw)[0]

    def get_video_features(
        self,
        pixel_values: mx.array,
        video_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        return self.vision_tower(pixel_values, video_grid_thw)[0]

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        # Work around an mlx-vlm bug (as of 0.4.4): Qwen3-VL's
        # get_input_embeddings assigns `mx.eval(deepstack_image_embeds)` to
        # `deepstack_visual_embeds`, but mx.eval returns None — so multi-scale
        # deepstack features are silently dropped, costing ~0.1 cosine on the
        # final image embedding. If they came back None but we actually have
        # images, re-run the vision tower just to grab the deepstack list.
        feats = super().get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values, **kwargs
        )
        if (
            pixel_values is not None
            and feats.deepstack_visual_embeds is None
            and getattr(self.config.vision_config, "deepstack_visual_indexes", None)
        ):
            image_grid_thw = kwargs.get("image_grid_thw")
            video_grid_thw = kwargs.get("video_grid_thw")
            grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
            dtype = self.vision_tower.patch_embed.proj.weight.dtype
            _, deepstack = self.vision_tower(pixel_values.astype(dtype), grid_thw)
            feats.deepstack_visual_embeds = deepstack
        return feats

    def get_binary_logits(self, pooled: mx.array) -> mx.array:
        if hasattr(self.language_model, "lm_head"):
            token_logits = self.language_model.lm_head(pooled)
        else:
            token_logits = self.language_model.model.embed_tokens.as_linear(pooled)

        return (
            token_logits[:, self.args.yes_token_id]
            - token_logits[:, self.args.no_token_id]
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
            model=self,
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
        text_embeds = normalize_embeddings(pooled) if self.args.normalize else pooled
        logits = self.get_binary_logits(pooled)
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
