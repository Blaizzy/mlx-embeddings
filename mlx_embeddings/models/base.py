import inspect
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class BaseModelOutput:
    last_hidden_state: Optional[mx.array] = None
    pooler_output: Optional[mx.array] = None
    text_embeds: Optional[mx.array] = None  # mean pooled and normalized embeddings
    hidden_states: Optional[List[mx.array]] = None
    logits: Optional[mx.array] = None  # token-level or sequence classification logits


@dataclass
class ViTModelOutput:
    logits: Optional[mx.array] = None
    text_embeds: Optional[mx.array] = None
    image_embeds: Optional[mx.array] = None
    logits_per_text: Optional[mx.array] = None
    logits_per_image: Optional[mx.array] = None
    text_model_output: Optional[mx.array] = None
    vision_model_output: Optional[mx.array] = None


def normalize_embeddings(embeddings, p=2, axis=-1, keepdims=True, eps=1e-9):
    return embeddings / mx.maximum(
        mx.linalg.norm(embeddings, ord=p, axis=axis, keepdims=keepdims), eps
    )
