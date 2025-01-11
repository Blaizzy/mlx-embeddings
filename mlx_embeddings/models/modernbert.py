import math
from dataclasses import dataclass
from typing import Optional, Dict, Literal, Any

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, compute_similarity, apply_rotary_pos_emb 

### NOTE :  removed all the attention_outputs (eager mode), may add it back later
### given no flash attention 2, padded/unpadded was also removed
### TODO: 
# review the padding strategy
# review compiling opportunities

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    ### hidden_activation : str = "gelu" ## should not be necessary, we'd just apply gelu.
    max_position_embeddings: Optional[int] = None
    norm_eps: float = 1e-05
    norm_bias : bool = False
    global_rope_theta : float = 160000.0
    attention_bias: bool = False
    attention_dropout : float =0.0
    global_attn_every_n_layers : int =3
    local_attention : int =128
    local_rope_theta: float = 10000
    embedding_dropout : float =0.0
    mlp_bias: bool = False
    mlp_dropout : float = 0.0
    initializer_range=0.02 # relevant for MLX?
    initializer_cutoff_factor=2.0 # relevant for MLX?
    pad_token_id=50283
    eos_token_id=50282
    bos_token_id=50281
    cls_token_id=50281
    sep_token_id=50282
    output_hidden_states: bool = False 
    use_return_dict: bool = True 
    # output_attentions: bool = False # not relevant if we only use sdpa
    # deterministic_flash_attn=False ## for torch only, to remove???
    # reference_compile=None ## for torch only, to remove???

    ### pipeline args, mostly for classification (unrelated to this project but keeping for now - feel free to remove)
    decoder_bias=True,
    classifier_pooling: Literal["cls", "mean"] = "cls"
    classifier_dropout=0.0 
    classifier_bias=False
    # classifier_activation="gelu"
    sparse_prediction=True ### True seems a more appropriate value for MLM
    sparse_pred_ignore_index=-100 
    is_regression: Optional[bool] = None
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[int, str]] = None

    @property
    def num_labels(self) -> int:
        """
        Number of labels is determined by:
        - For zero-shot classification: length of label_candidates
        - For regression or binary with sigmoid: 1
        - For classification: length of id2label mapping
        """
        
        if self.is_regression:
            return 1
        
        # if self.pipeline_config.get("binary_sigmoid", False):
        #     return 1
            
        if self.id2label is None:
            raise ValueError(
                "id2label mapping must be provided for categorical classification. "
                "For regression or binary classification with sigmoid output, "
                "set is_regression=True or binary_sigmoid=True in pipeline_config."
            )
            
        return len(self.id2label)


class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int =2048, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
    
    ### property decorator to make a method behave like an attribute and avoid a flag for missing parameters
    ### the flip side is that the value is recalculated at every forward pass 
    ### TBC for training
    @property
    def inv_freq(self):
        return 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=mx.int32) / self.dim)) # [ dim/2 ]

    def __call__(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = mx.expand_dims(self.inv_freq, [0, 2]) # [1, dim/2, 1]
        inv_freq_expanded = mx.broadcast_to(
            inv_freq_expanded,
            [position_ids.shape[0], inv_freq_expanded.shape[1], 1]
        ) # [bs, dim/2, 1]

        position_ids_expanded = mx.expand_dims(position_ids.astype(mx.float32), 1) # [bs, 1, seq_len]
    
        # Computing position embeddings
        freqs = mx.matmul(inv_freq_expanded, position_ids_expanded) # [bs, dim/2, seq_len]
        freqs = mx.transpose(freqs, [0, 2, 1]) # [bs, seq_len, dim/2]
        
        # Duplicating frequencies
        emb = mx.concatenate([freqs, freqs], axis=-1) # [bs, seq_len, dim]
        
        # Computing sin and cos
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        
        return cos.astype(x.dtype), sin.astype(x.dtype)


class ModernBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config: ModelArgs):
        super().__init__() 
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias) 
        self.drop = nn.Dropout(p=config.embedding_dropout)

    def __call__(self, input_ids):
        embeddings = self.tok_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings


class ModernBertMLP(nn.Module):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces class BertIntermediate`
    and class SelfOutput with a single module that has similar functionality.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size *2, bias=config.mlp_bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=config.mlp_dropout)
        self.Wo = nn.Linear(int(config.intermediate_size), config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states):
        x = self.Wi(hidden_states)
        # Implementing chunk operation
        split_dim = x.shape[-1] // 2
        input, gate = x[:, :, :split_dim], x[:, :, split_dim:] ### I need to understand this better : https://arxiv.org/pdf/2002.05202v1
        return self.Wo(self.drop(self.act(input) * gate))


class ModernBertAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.
    For now, only supports the Scaled Dot-Product Attention (SDPA) implementation.
    """
    def __init__(self, config: ModelArgs, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads ({config.num_attention_heads})"
            )
        
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        max_position_embeddings = config.max_position_embeddings
        if self.local_attention != (-1, -1):
            if config.local_rope_theta is not None:
                rope_theta = config.local_rope_theta
            max_position_embeddings = config.local_attention

        self.rotary_emb = ModernBertRotaryEmbedding(
            dim=self.head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta
        )

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(p=config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()
        
    def __call__(
            self, 
            hidden_states, 
            attention_mask = None,
            sliding_window_mask = None,
            position_ids=None,
            # output_attentions: Optional[bool] = False, ### is not used with sdpa (only with eager mode),
            **kwargs
        ):
        qkv = self.Wqkv(hidden_states)
        bs = hidden_states.shape[0]
        qkv = mx.reshape(qkv, (bs, -1, 3, self.num_heads, self.head_dim))

        # Get attention outputs using SDPA
        cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
        qkv = mx.transpose(qkv, [0, 3, 2, 1, 4])  # [batch_size, nheads, 3, seqlen, headdim]
        query, key, value = mx.split(qkv, indices_or_sections=3, axis=2)
        query = query.squeeze(2) 
        key = key.squeeze(2)
        value = value.squeeze(2)

        # Applying rotary embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Handling local attention if needed
        if self.local_attention != (-1, -1):
            attention_mask = sliding_window_mask

        # Computing attention using MLX's SDPA
        scale = 1.0 / math.sqrt(query.shape[-1])
        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value,
            scale=scale,
            mask=attention_mask
        )
        
        # Reshaping and apply output projection
        attn_output = mx.transpose(attn_output, [0, 2, 1, 3])
        attn_output = mx.reshape(attn_output, (bs, -1, self.all_head_size))
        
        # Applying output projection and dropout
        hidden_states = self.Wo(attn_output)
        hidden_states = self.out_drop(hidden_states)

        return (hidden_states,)


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp = ModernBertMLP(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def __call__(
            self, 
            hidden_states , 
            attention_mask =None, 
            sliding_window_mask = None,
            position_ids  = None,
            # output_attentions: Optional[bool] = False, ## should not be used with sdpa
    ):
        normalized_hidden_states = self.attn_norm(hidden_states)
        attention_output = self.attn( 
            normalized_hidden_states, 
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            # output_attentions=output_attentions
        )
        hidden_states = hidden_states + attention_output[0]
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_output

        return (hidden_states,)   # removed attention outputs


class ModernBertModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [
            ModernBertEncoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False ### TBC

    def get_input_embeddings(self) -> ModernBertEmbeddings:
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def __call__(
            self, 
            input_ids, 
            attention_mask = None, # shape: (batch_size, seq_len) see below
            sliding_window_mask = None,
            position_ids = None,
            # output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ):
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions # should not be used with sdpa
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = () if output_hidden_states else None
        # all_attentions = () if output_attentions else None # should not be used with sdpa

        # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask) ###  TODO : review padding strategy before removing this line

        batch_size, seq_len = input_ids.shape[:2]

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len)) ### updated with _update_attention_mask() below

        if position_ids is None:
            position_ids = mx.arange(seq_len, dtype=mx.int32)[None, :]

        # get attention mask and sliding window mask
        attention_mask, sliding_window_mask = self._update_attention_mask(
            attention_mask=attention_mask,
            # output_attentions=False ### should not be used with sdpa
        )

        hidden_states = self.embeddings(input_ids)

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # if self.gradient_checkpointing and self.training:
            #     ### already covered by trainer, delete after confirming (can delete if no training planned)
            #     layer_outputs = mx.checkpoint(
            #         encoder_layer.__call__,
            #         hidden_states,
            #         attention_mask,
            #         sliding_window_mask,
            #         position_ids,
            #         # output_attentions,
            #     )
            # else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                # output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
        
        hidden_states = self.final_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            # "attentions": all_attentions,
        }
    
    def _update_attention_mask(self, attention_mask): 
        
        # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        attention_mask = attention_mask[:, None, None, :]
        
        # Create the causal mask for global attention
        # (1, 1, seq_len, seq_len)
        seq_len = attention_mask.shape[-1]
        global_attention_mask = mx.broadcast_to(attention_mask, (attention_mask.shape[0], 1, seq_len, seq_len))
        
        # Create position indices for sliding window
        rows = mx.arange(seq_len)
        rows = rows[None, :]  # (1, seq_len)
        # Calculate position-wise distances
        distance = mx.abs(rows - rows.T)  # (seq_len, seq_len)
        
        # Create sliding window mask using mx.where
        window_mask = mx.where(
            distance <= (self.config.local_attention // 2),
            mx.ones_like(distance),
            mx.zeros_like(distance)
        )
        
        # Expand dimensions using None indexing
        window_mask = window_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
            
        # Broadcast to match batch size
        window_mask = mx.broadcast_to(window_mask, global_attention_mask.shape)
        
        # Creating sliding window attention mask
        # Replacing non-window positions with large negative value
        sliding_window_mask = mx.where(
            window_mask,
            global_attention_mask,
            float('-inf') ## if not broadcasted for some reason : float('-inf') * mx.ones_like(global_attention_mask)
        )
    
        return global_attention_mask, sliding_window_mask


### below are classes for specific pipelines
### I removed pipelines that do not seem relevant for this project (text-classification, masked-lm, token-classification, training)
### but I continue to work on these other features there, if you want to add them back : https://github.com/pappitti/modernbert-mlx

class Model(nn.Module):
    """
    Computes pooled, unnormalized embeddings for input sequences using a ModernBERT model.

    Note : sanitization is a hack to align with other models here while downloading weights 
    with the maskedlm config from HF (original modelBert model).
    The decoder.bias is ignored here

    In practice, most embeddings models are sentence transformers, and there is a dedicated class for that below.
    It comes with a different sanitization method to align with the sentence transformers weights.
    """ 
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)

    def __call__(
        self, 
        input_ids : mx.array, 
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        # output_hidden_states: Optional[bool] = None, # not needed, only last_hidden_state is returned
        return_dict: Optional[bool] = True,
    ):
        
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape 
            attention_mask = mx.ones((batch_size, seq_len)) ### updated via _update_attention_mask() in the model

        # Get embeddings and encoder outputs as before
        encoder_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=None, # Not needed, only last_hidden_state is returned
            return_dict=return_dict,
        )
        hidden_state = encoder_outputs["last_hidden_state"] if isinstance(encoder_outputs, dict) else encoder_outputs[0] 
        
        # Do pooling here (unlike BERT)
        if self.config.classifier_pooling == "cls":
            pooled = hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":                
            attention_mask = mx.expand_dims(attention_mask, -1)
            pooled = mx.sum(hidden_state * attention_mask, axis=1) / mx.sum(attention_mask, axis=1)

        if not return_dict:
            return (pooled, hidden_state) 

        return {
            "embeddings": pooled,
            "last_hidden_states": hidden_state,
        }
    
    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if k in ["head.norm.weight", "head.dense.weight", "decoder.bias"]:
                ### this is the hack
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
        reference_input_ids : Optional[mx.array] = None,  # Shape: [num_references, seq_len]
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
            position_ids=position_ids, ### ?
            return_dict=True
        )
        batch_embeddings = batch_outputs["embeddings"]  # [batch_size, hidden_size]

        if reference_input_ids is not None:
        
            # Get embeddings for reference sentences
            ref_outputs = super().__call__(
                input_ids=reference_input_ids,
                attention_mask=reference_attention_mask,
                position_ids=position_ids, ### ?
                return_dict=True
            )
            reference_embeddings = ref_outputs["embeddings"]  # [num_references, hidden_size]
            
            # Compute similarities between batch and references
            similarities = compute_similarity(
                batch_embeddings,  # [batch_size, hidden_size]
                reference_embeddings  # [num_references, hidden_size]
            )  # Result: [batch_size, num_references]
            
            loss = None ### can remove all this if no training is planned
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
    This class sanitizes typical sentence transformers weights to align with the ModernBERT model.
    """
    def __init__(self, config: ModelArgs):
        super().__init__(config)

    def sanitize(self, weights):
        """Convert sentence transformer weights to ModernBERT format."""
        sanitized_weights = {}
        
        for k, v in weights.items():
            new_key = "model." + k
            sanitized_weights[new_key] = v
        return sanitized_weights


