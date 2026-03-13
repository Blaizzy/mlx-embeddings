import mlx.core as mx
from transformers.tokenization_utils_base import BatchEncoding

from mlx_embeddings.tokenizer_utils import (
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
    TokenizerWrapper,
)


class MockTokenizer:
    def __init__(self, with_batch_encode_plus=False):
        self.vocab = {"hello": 0}
        self.call_args = []
        self.batch_encode_plus_args = []
        self._with_batch_encode_plus = with_batch_encode_plus

    def decode(self, token_ids):
        return "".join("x" for _ in token_ids)

    def _build_batch(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return BatchEncoding(
            data={
                "input_ids": [[len(text)] for text in texts],
                "attention_mask": [[1] for _ in texts],
            },
            tensor_type=kwargs.get("return_tensors"),
        )

    def __call__(self, texts, *args, **kwargs):
        self.call_args.append((texts, args, kwargs))
        return self._build_batch(texts, **kwargs)

    def batch_encode_plus(self, texts, *args, **kwargs):
        if not self._with_batch_encode_plus:
            raise AttributeError("batch_encode_plus is not available")
        self.batch_encode_plus_args.append((texts, args, kwargs))
        return self._build_batch(texts, **kwargs)


class TestTokenizerWrapper:
    def test_detokenizer_access(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer)

        assert wrapper.detokenizer is not None
        assert isinstance(wrapper.detokenizer, NaiveStreamingDetokenizer)

    def test_attribute_forwarding(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer)

        assert wrapper.vocab == tokenizer.vocab

    def test_custom_detokenizer_class(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer, SPMStreamingDetokenizer)

        assert isinstance(wrapper.detokenizer, SPMStreamingDetokenizer)

    def test_call_forwards_to_underlying_tokenizer(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer)

        result = wrapper(
            ["hello", "world"],
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=32,
        )

        assert tokenizer.call_args == [
            (
                ["hello", "world"],
                (),
                {
                    "return_tensors": "mlx",
                    "padding": True,
                    "truncation": True,
                    "max_length": 32,
                },
            )
        ]
        assert result["input_ids"].tolist() == [[5], [5]]
        assert result["attention_mask"].tolist() == [[1], [1]]

    def test_batch_encode_plus_falls_back_to_callable_interface(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer)

        result = wrapper.batch_encode_plus(
            ["hello", "mlx"],
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=32,
        )

        assert tokenizer.call_args == [
            (
                ["hello", "mlx"],
                (),
                {
                    "return_tensors": "mlx",
                    "padding": True,
                    "truncation": True,
                    "max_length": 32,
                },
            )
        ]
        assert result["input_ids"].tolist() == [[5], [3]]
        assert result["attention_mask"].tolist() == [[1], [1]]

    def test_batch_encode_plus_uses_underlying_method_when_available(self):
        tokenizer = MockTokenizer(with_batch_encode_plus=True)
        wrapper = TokenizerWrapper(tokenizer)

        result = wrapper.batch_encode_plus(["hello"], return_tensors="mlx")

        assert tokenizer.batch_encode_plus_args == [
            (["hello"], (), {"return_tensors": "mlx"})
        ]
        assert tokenizer.call_args == []
        assert isinstance(result["input_ids"], type(mx.array([1])))
