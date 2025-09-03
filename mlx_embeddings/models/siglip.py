import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import ViTModelOutput, normalize_embeddings


@dataclass
class TextConfig:
    vocab_size: int = 32000
    max_position_embeddings: int = 64
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    layer_norm_eps: float = 1e-6
    projection_size: Optional[int] = None
    model_type: str = "siglip_text_model"

    def __post_init__(self):
        if self.projection_size is None:
            self.projection_size = self.hidden_size


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    layer_norm_eps: float = 1e-6
    model_type: str = "siglip_vision_model"
    vision_use_head: bool = True
    # SigLIP2 parameters
    num_patches: Optional[int] = None  # For SigLIP2, defaults to 256
    max_num_patches: Optional[int] = None  # For naflex variants


@dataclass
class ModelArgs:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "siglip"
    output_hidden_states: bool = False
    output_attentions: bool = False
    use_return_dict: bool = True
    num_labels: int = 0

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class MHA(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.in_proj = nn.Linear(dims, dims * 3, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, queries: mx.array, keys: mx.array, values: mx.array, mask=None):
        B, L, D = queries.shape

        qkv = self.in_proj(keys)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, x, mask=None):
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        # Process attention mask for multi-head attention if provided
        if mask is not None:
            if mask.ndim == 2:
                # mask shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                # mask shape: (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                mask = mask[:, None, :, :]
            # For boolean masks, convert to additive mask
            if mask.dtype == mx.bool_:
                # Convert boolean mask to additive mask (True -> 0.0, False -> -inf)
                mask = mx.where(mask, 0.0, -mx.inf)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs, approx: str = "none"):
        super().__init__()
        self.activation_fn = nn.GELU(approx=approx)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, approx: str = "none"):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config, approx=approx)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:

        r = self.self_attn(self.layer_norm1(x), mask)
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r


class Encoder(nn.Module):
    def __init__(self, config: ModelArgs, approx: str = "none"):
        super().__init__()
        self.layers = [
            EncoderLayer(config, approx=approx) for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        encoder_states = (x,) if output_hidden_states else None
        for l in self.layers:
            x = l(x, mask=mask)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

        return (x, encoder_states)


class VisionEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # For SigLIP2, use num_patches if provided, otherwise calculate from image_size
        if config.num_patches is not None:
            self.num_patches = config.num_patches
        else:
            self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def interpolate_pos_encoding(
        self, embeddings: mx.array, height: int, width: int
    ) -> mx.array:
        # TODO: Implement this
        raise NotImplementedError(
            "Interpolation of positional encodings is not implemented for SigLIP"
        )

    def __call__(
        self,
        x: mx.array,
        interpolate_pos_encoding: bool = False,
        pixel_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        _, _, height, width = x.shape
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.transpose(patch_embeddings, (0, 3, 1, 2))
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=2, end_axis=3)
        patch_embeddings = mx.transpose(patch_embeddings, (0, 2, 1))

        # Handle variable sequence length for SigLIP2 naflex variants
        batch_size, seq_len, embed_dim = patch_embeddings.shape

        # If we have fewer patches than expected, pad to num_positions
        if seq_len < self.num_positions:
            padding_size = self.num_positions - seq_len
            padding = mx.zeros((batch_size, padding_size, embed_dim))
            patch_embeddings = mx.concatenate([patch_embeddings, padding], axis=1)
        elif seq_len > self.num_positions:
            # Truncate if we have more patches than expected
            patch_embeddings = patch_embeddings[:, : self.num_positions, :]

        position_ids = mx.array(np.arange(self.num_positions)[None, :])
        embeddings = patch_embeddings
        if interpolate_pos_encoding:
            embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings += self.position_embedding(position_ids)

        return embeddings


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: VisionConfig, approx: str = "none"):
        super().__init__()

        self.probe = mx.ones((1, 1, config.hidden_size))
        self.attention = MHA(config.hidden_size, config.num_attention_heads, bias=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config, approx=approx)

    def __call__(self, hidden_state):
        batch_size = hidden_state.shape[0]
        # Repeat the probe for each item in the batch
        # mx.repeat only takes one axis at a time, so we need to do it sequentially
        probe = mx.repeat(self.probe, batch_size, axis=0)

        hidden_state = self.attention(probe, hidden_state, hidden_state)

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config, approx="precise")
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.use_head = (
            True if not hasattr(config, "vision_use_head") else config.vision_use_head
        )

        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config, approx="precise")

    def __call__(
        self,
        pixel_values: mx.array,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        pixel_attention_mask: Optional[mx.array] = None,
        spatial_shapes: Optional[mx.array] = None,
    ) -> mx.array:
        x = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            pixel_attention_mask=pixel_attention_mask,
        )

        # For SigLIP2, we accept pixel_attention_mask but don't process it yet
        # This maintains API compatibility while keeping the original behavior
        attention_mask = None

        x, encoder_outputs = self.encoder(
            x=x, output_hidden_states=output_hidden_states, mask=attention_mask
        )

        x = self.post_layernorm(x)
        pooler_output = self.head(x) if self.use_head else None

        if output_hidden_states:
            return x, pooler_output, encoder_outputs[1:]
        else:
            return x, pooler_output


class SiglipVisionModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in ["siglip_vision_model"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.vision_model = SiglipVisionTransformer(config)

    def __call__(
        self,
        pixel_values: mx.array,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        pixel_attention_mask: Optional[mx.array] = None,
        spatial_shapes: Optional[mx.array] = None,
    ) -> mx.array:
        return self.vision_model(
            pixel_values,
            output_hidden_states,
            interpolate_pos_encoding,
            pixel_attention_mask,
            spatial_shapes,
        )

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            if "patch_embedding.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights


class SiglipTextEmbeddings(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        )
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = mx.array(np.arange(seq_length)[None, :])

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights


class SiglipTextTransformer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = Encoder(config, approx="precise")
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.head = nn.Linear(embed_dim, config.projection_size)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:

        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])
        x = self.embeddings(input_ids, position_ids)
        x, encoder_states = self.encoder(x, output_hidden_states, attention_mask)
        x = self.final_layer_norm(x)

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        pooled_output = x[:, -1, :]
        pooled_output = self.head(pooled_output)

        if output_hidden_states:
            return x, pooled_output, encoder_states[1:]
        else:
            return x, pooled_output


class SiglipTextModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in ["siglip_text_model"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.text_model = SiglipTextTransformer(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        return self.text_model(
            input_ids, attention_mask, position_ids, output_hidden_states
        )


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        vision_config = config.vision_config
        self.vision_model = SiglipVisionModel(vision_config)

        if config.num_labels > 0:
            # Classifier head
            self.classifier = (
                nn.Linear(config.vision_config.hidden_size, config.num_labels)
                if config.num_labels > 0
                else nn.Identity()
            )
        else:
            text_config = config.text_config
            self.text_model = SiglipTextModel(text_config)
            self.logit_scale = mx.zeros((1,))
            self.logit_bias = mx.zeros((1,))

    def get_text_features(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:

        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = text_outputs[1]
        if output_hidden_states:
            return pooled_output, text_outputs[2]
        else:
            return pooled_output

    def get_image_features(
        self,
        pixel_values: Optional[mx.array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        pixel_attention_mask: Optional[mx.array] = None,
        spatial_shapes: Optional[mx.array] = None,
    ) -> mx.array:

        if pixel_values is not None:
            dtype = (
                self.vision_model.vision_model.embeddings.patch_embedding.weight.dtype
            )
            if not isinstance(pixel_values, mx.array):
                pixel_values = mx.array(pixel_values)

            # MLX Conv2d expects NHWC, but processors usually output NCHW.
            # If the input looks like NCHW, we transpose it.
            if pixel_values.ndim == 4 and pixel_values.shape[1] == 3:
                pixel_values = pixel_values.transpose(0, 2, 3, 1)

            pixel_values = pixel_values.astype(dtype)

        # Use SiglipModel's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

        pooled_output = vision_outputs[1]

        return pooled_output

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        pixel_attention_mask: Optional[mx.array] = None,
        spatial_shapes: Optional[mx.array] = None,
    ) -> mx.array:
        if pixel_values is not None:
            dtype = (
                self.vision_model.vision_model.embeddings.patch_embedding.weight.dtype
            )
            if not isinstance(pixel_values, mx.array):
                pixel_values = mx.array(pixel_values)

            # MLX Conv2d expects NHWC, but processors usually output NCHW.
            # If the input looks like NCHW, we transpose it.
            if pixel_values.ndim == 4 and pixel_values.shape[1] == 3:
                pixel_values = pixel_values.transpose(0, 2, 3, 1)

            pixel_values = pixel_values.astype(dtype)

        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

        image_embeds = vision_outputs[1]

        # classifier head
        logits = None
        if self.config.num_labels > 0:
            # average pool the patch tokens
            image_embeds_mean = mx.mean(image_embeds, axis=1)
            # apply classifier
            logits = self.classifier(image_embeds_mean)

            return ViTModelOutput(
                logits=logits,
                image_embeds=image_embeds,
                vision_model_output=vision_outputs,
            )

        else:
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=output_hidden_states,
            )

            text_embeds = text_outputs[1]

            # normalized features
            image_embeds = normalize_embeddings(image_embeds)
            text_embeds = normalize_embeddings(text_embeds)

            # cosine similarity as logits
            logits_per_text = mx.matmul(text_embeds, image_embeds.T)

            # Apply scale and bias
            logits_per_text = (
                logits_per_text * mx.exp(self.logit_scale) + self.logit_bias
            )

            logits_per_image = logits_per_text.T

            return ViTModelOutput(
                logits_per_text=logits_per_text,
                logits_per_image=logits_per_image,
                text_embeds=text_embeds,
                image_embeds=image_embeds,
                text_model_output=text_outputs,
                vision_model_output=vision_outputs,
            )

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if k.startswith("text_model") and not k.startswith("text_model.text_model"):
                sanitized_weights["text_model." + k] = v
            elif k.startswith("vision_model") and not k.startswith(
                "vision_model.vision_model"
            ):
                if "in_proj_bias" in k:
                    k = k.replace("in_proj_bias", "in_proj.bias")
                if "in_proj_weight" in k:
                    k = k.replace("in_proj_weight", "in_proj.weight")
                sanitized_weights["vision_model." + k] = v
            else:
                sanitized_weights[k] = v

        if hasattr(self.text_model, "sanitize"):
            sanitized_weights = self.text_model.sanitize(sanitized_weights)
        if hasattr(self.vision_model, "sanitize"):
            sanitized_weights = self.vision_model.sanitize(sanitized_weights)

        return sanitized_weights
