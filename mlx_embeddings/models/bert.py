import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, compute_similarity


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    num_hidden_layers: int
    num_attention_heads: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    vocab_size: int = 30522


class BertEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = mx.arange(seq_length, dtype=mx.int32)[None, :]
        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.reshape(new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def __call__(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = mx.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = mx.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = mx.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def __call__(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def __call__(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.layer = [BertLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
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
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None, return_dict: Optional[bool] = True,):
        embedding_output = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (pooled_output, sequence_output) 

        return {
            "embeddings": pooled_output,
            "last_hidden_state": sequence_output,
        }

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights

class ModelForSentenceSimilarity(Model):
    """
    Computes similarity scores between input sequences and reference sentences.
    """
    def __init__(self, config):
        super().__init__(config)
    
    def __call__(
        self,
        input_ids,
        reference_input_ids: Optional[mx.array] = None,  # Shape: [num_references, seq_len]
        attention_mask: Optional[mx.array] = None,
        reference_attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        similarity_scores: Optional[mx.array] = None,  # Shape: [batch_size, num_references]
        return_dict: Optional[bool] = True,
    ):
        # Get embeddings for input batch
        batch_outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        batch_embeddings = batch_outputs["embeddings"]  # [batch_size, hidden_size]
        
        if reference_input_ids is not None:
        
            # Get embeddings for reference sentences
            ref_outputs = super().__call__(
                input_ids=reference_input_ids,
                attention_mask=reference_attention_mask,
                return_dict=True
            )
            reference_embeddings = ref_outputs["embeddings"]  # [num_references, hidden_size]
            
            # Compute similarities between batch and references
            similarities = compute_similarity(
                batch_embeddings,  # [batch_size, hidden_size]
                reference_embeddings  # [num_references, hidden_size]
            )  # Result: [batch_size, num_references]
            
            loss = None 
            ### can remove all this if no training is planned
            if similarity_scores is not None:
                # MSE loss between computed similarities and target scores
                # similarity_scores should be shape [batch_size, num_references]
                loss = nn.losses.mse_loss(similarities, similarity_scores)
        
        else :
            similarities = None
            loss = None
            
        if not return_dict:
            return (loss, similarities, batch_embeddings)
            
        return {
            "loss": loss,
            "similarities": similarities,  # [batch_size, num_references]
            "embeddings": batch_embeddings,  # [batch_size, hidden_size]
        }

class ModelForSentenceTransformers(ModelForSentenceSimilarity):
    """
    Extends ModelForSentenceSimilarity to provide embeddings for input sequences.
    This class is only meant to align with the ModernBERT model. Could just replace ModelForSentenceSimilarity.
    """
    def __init__(self, config: ModelArgs):
        super().__init__(config)

