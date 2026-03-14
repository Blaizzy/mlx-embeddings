from abc import ABC, abstractmethod
from typing import List, Optional, Union

import mlx.core as mx
from PIL import Image
from transformers import BatchEncoding, BatchFeature, ProcessorMixin


class BaseColVisionProcessor(ABC, ProcessorMixin):
    """
    Base class for visual retriever processors.
    Ported from PyTorch to MLX from:
    - https://github.com/illuin-tech/colpali/blob/main/colpali_engine/utils/processing_utils.py

    Removed the methods: get_topk_plaid, create_plaid_index, get_n_patches
    """

    @abstractmethod
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def score(
        self,
        qs: List[mx.array],
        ps: List[mx.array],
        **kwargs,
    ) -> mx.array:
        pass

    @staticmethod
    def score_single_vector(
        qs: List[mx.array],
        ps: List[mx.array],
    ) -> mx.array:
        """
        Compute the dot product score for the given single-vector query and passage embeddings using MLX.
        """
        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        qs_stacked = mx.stack(qs)
        ps_stacked = mx.stack(ps)

        scores = mx.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        assert scores.shape[0] == len(
            qs
        ), f"Expected {len(qs)} scores, got {scores.shape[0]}"
        return scores.astype(mx.float32)

    @staticmethod
    def score_multi_vector(
        qs: Union[mx.array, List[mx.array]],
        ps: Union[mx.array, List[mx.array]],
        batch_size: int = 128,
    ) -> mx.array:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`) using MLX.
        """
        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        def pad_to_max(arrays):
            max_len = max(a.shape[0] for a in arrays)
            emb_dim = arrays[0].shape[1]
            padded = []
            for a in arrays:
                pad_width = max_len - a.shape[0]
                if pad_width > 0:
                    pad = mx.zeros((pad_width, emb_dim), dtype=a.dtype)
                    padded.append(mx.concatenate([a, pad], axis=0))
                else:
                    padded.append(a)
            return mx.stack(padded)

        scores_list = []
        for i in range(0, len(qs), batch_size):
            qs_batch = pad_to_max(qs[i : i + batch_size])
            scores_batch = []
            for j in range(0, len(ps), batch_size):
                ps_batch = pad_to_max(ps[j : j + batch_size])
                # einsum: (b,n,d),(c,s,d)->(b,c,n,s)
                sim = mx.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                maxsim = mx.max(sim, axis=3)  # max over s
                summed = mx.sum(maxsim, axis=2)  # sum over n
                scores_batch.append(summed)
            scores_batch = mx.concatenate(scores_batch, axis=1)
            scores_list.append(scores_batch)
        scores = mx.concatenate(scores_list, axis=0)
        assert scores.shape[0] == len(
            qs
        ), f"Expected {len(qs)} scores, got {scores.shape[0]}"
        return scores.astype(mx.float32)
