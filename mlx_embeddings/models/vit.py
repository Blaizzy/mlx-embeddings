import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    num_labels: int = 1000

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

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, config.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def __call__(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {num_channels}")
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"Input image size ({height}*{width}) doesn't match expected size ({self.image_size}*{self.image_size})")
        x = self.projection(pixel_values)
        return mx.reshape(mx.transpose(x, [0, 2, 3, 1]), (batch_size, -1, x.shape[1]))

class ViTEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.cls_token = mx.random.normal((1, 1, config.hidden_size))
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = mx.random.normal((1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = mx.repeat(self.cls_token, batch_size, axis=0)
        embeddings = mx.concatenate([cls_tokens, embeddings], axis=1)
        embeddings = embeddings + self.position_embeddings
        return self.dropout(embeddings)

class ViTSelfAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = mx.reshape(x, new_x_shape)
        return mx.transpose(x, [0, 2, 1, 3])

    def __call__(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = mx.matmul(query_layer, mx.transpose(key_layer, [0, 1, 3, 2]))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = mx.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = mx.matmul(attention_probs, value_layer)
        context_layer = mx.transpose(context_layer, [0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = mx.reshape(context_layer, new_context_layer_shape)
        return context_layer

class ViTSelfOutput(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor

class ViTAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def __call__(self, hidden_states):
        self_output = self.attention(hidden_states)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class ViTIntermediate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class ViTOutput(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class ViTLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size)
        self.layernorm_after = nn.LayerNorm(config.hidden_size)

    def __call__(self, hidden_states):
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return layer_output

class ViTEncoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.layer = [ViTLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states

class ViTPooler(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def __call__(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.pooler = ViTPooler(config)

    def __call__(self, pixel_values):
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = self.layernorm(encoder_outputs)
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "embeddings.patch_embeddings.projection.weight" in k:
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