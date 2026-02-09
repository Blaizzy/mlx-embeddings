"""Tests for Architecture enum and _resolve_model_type function."""

from mlx_embeddings.utils import Architecture, ARCHITECTURE_REMAPPING, _resolve_model_type


class TestArchitectureEnum:
    """Tests for the Architecture enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert Architecture.JINA_FOR_RANKING.value == "JinaForRanking"
        assert Architecture.MODERN_BERT_FOR_MASKED_LM.value == "ModernBertForMaskedLM"
        assert (
            Architecture.MODERN_BERT_FOR_SEQUENCE_CLASSIFICATION.value
            == "ModernBertForSequenceClassification"
        )
        assert Architecture.MODERN_BERT_MODEL.value == "ModernBertModel"

    def test_from_string_valid(self):
        """Test from_string with valid architecture names."""
        arch = Architecture.from_string("JinaForRanking")
        assert arch == Architecture.JINA_FOR_RANKING

        arch = Architecture.from_string("ModernBertForMaskedLM")
        assert arch == Architecture.MODERN_BERT_FOR_MASKED_LM

        arch = Architecture.from_string("ModernBertForSequenceClassification")
        assert arch == Architecture.MODERN_BERT_FOR_SEQUENCE_CLASSIFICATION

        arch = Architecture.from_string("ModernBertModel")
        assert arch == Architecture.MODERN_BERT_MODEL

    def test_from_string_invalid(self):
        """Test from_string with invalid architecture names."""
        arch = Architecture.from_string("NonExistentArchitecture")
        assert arch is None

        arch = Architecture.from_string("")
        assert arch is None

        arch = Architecture.from_string("SomeRandomString")
        assert arch is None

    def test_architecture_remapping(self):
        """Test that ARCHITECTURE_REMAPPING uses enum values."""
        assert Architecture.JINA_FOR_RANKING in ARCHITECTURE_REMAPPING
        assert ARCHITECTURE_REMAPPING[Architecture.JINA_FOR_RANKING] == "jina_reranker"


class TestResolveModelType:
    """Tests for _resolve_model_type function with architecture normalization."""

    def test_architectures_none(self):
        """Test with architectures=None."""
        config = {"model_type": "qwen3"}
        result = _resolve_model_type(config)
        assert result == "qwen3"

    def test_architectures_missing(self):
        """Test with missing architectures field."""
        config = {"model_type": "bert"}
        result = _resolve_model_type(config)
        assert result == "bert"

    def test_architectures_string(self):
        """Test with architectures as a string (should be normalized to list)."""
        config = {"architectures": "JinaForRanking", "model_type": "qwen3"}
        result = _resolve_model_type(config)
        assert result == "jina_reranker"

    def test_architectures_list(self):
        """Test with architectures as a list."""
        config = {"architectures": ["JinaForRanking"], "model_type": "qwen3"}
        result = _resolve_model_type(config)
        assert result == "jina_reranker"

    def test_architectures_tuple(self):
        """Test with architectures as a tuple."""
        config = {"architectures": ("JinaForRanking",), "model_type": "qwen3"}
        result = _resolve_model_type(config)
        assert result == "jina_reranker"

    def test_architectures_set(self):
        """Test with architectures as a set."""
        config = {"architectures": {"JinaForRanking"}, "model_type": "qwen3"}
        result = _resolve_model_type(config)
        assert result == "jina_reranker"

    def test_architectures_set_mixed_types(self):
        """Test with architectures as a set containing mixed types."""
        # Should filter to strings before sorting, avoiding TypeError
        config = {"architectures": {"JinaForRanking", 123, None}, "model_type": "qwen3"}
        result = _resolve_model_type(config)
        assert result == "jina_reranker"
        
        # Set with only non-string values should fall back to model_type
        config = {"architectures": {123, 456}, "model_type": "bert"}
        result = _resolve_model_type(config)
        assert result == "bert"

    def test_architectures_empty_list(self):
        """Test with empty architectures list (should fall back to model_type)."""
        config = {"architectures": [], "model_type": "xlm_roberta"}
        result = _resolve_model_type(config)
        assert result == "xlm_roberta"

    def test_architectures_invalid_type(self):
        """Test with invalid architectures type (should fall back to model_type)."""
        config = {"architectures": 123, "model_type": "modernbert"}
        result = _resolve_model_type(config)
        assert result == "modernbert"

        config = {"architectures": {"key": "value"}, "model_type": "bert"}
        result = _resolve_model_type(config)
        assert result == "bert"

    def test_architectures_no_match(self):
        """Test with architectures that don't match ARCHITECTURE_REMAPPING."""
        config = {"architectures": ["SomeOtherArch"], "model_type": "bert"}
        result = _resolve_model_type(config)
        assert result == "bert"

    def test_architectures_multiple_with_match(self):
        """Test with multiple architectures where one matches."""
        config = {
            "architectures": ["UnknownArch", "JinaForRanking"],
            "model_type": "qwen3",
        }
        # Should match the first recognized architecture in the remapping
        # Since UnknownArch is not in the enum, it should skip to JinaForRanking
        result = _resolve_model_type(config)
        assert result == "jina_reranker"

    def test_model_type_with_hyphen(self):
        """Test that model_type with hyphens is converted to underscores."""
        config = {"model_type": "xlm-roberta"}
        result = _resolve_model_type(config)
        assert result == "xlm_roberta"

    def test_model_type_empty_string(self):
        """Test with empty model_type."""
        config = {"model_type": ""}
        result = _resolve_model_type(config)
        assert result == ""

    def test_model_type_missing(self):
        """Test with missing model_type (should return empty string)."""
        config = {}
        result = _resolve_model_type(config)
        assert result == ""

    def test_modern_bert_architectures(self):
        """Test ModernBert architecture variants."""
        config = {"architectures": ["ModernBertForMaskedLM"], "model_type": "modernbert"}
        result = _resolve_model_type(config)
        # ModernBertForMaskedLM is in the enum but not in ARCHITECTURE_REMAPPING
        # so it should fall back to model_type
        assert result == "modernbert"

        config = {
            "architectures": ["ModernBertForSequenceClassification"],
            "model_type": "modernbert",
        }
        result = _resolve_model_type(config)
        assert result == "modernbert"

        config = {"architectures": ["ModernBertModel"], "model_type": "modernbert"}
        result = _resolve_model_type(config)
        assert result == "modernbert"

    def test_priority_architecture_over_model_type(self):
        """Test that architecture-based routing takes precedence over model_type."""
        # Even though model_type is "qwen3", JinaForRanking should route to jina_reranker
        config = {"architectures": ["JinaForRanking"], "model_type": "qwen3"}
        result = _resolve_model_type(config)
        assert result == "jina_reranker"
