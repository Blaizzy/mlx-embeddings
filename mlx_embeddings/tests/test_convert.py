"""Tests for convert CLI and quantization mode (q_mode) support."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from mlx_embeddings.convert import configure_parser


class TestConfigureParser:
    """Tests for the CLI argument parser configuration."""

    def setup_method(self):
        self.parser = configure_parser()

    def test_q_mode_default(self):
        args = self.parser.parse_args(["--hf-path", "test/model"])
        assert args.q_mode == "affine"

    def test_q_mode_valid_choices(self):
        for mode in ("affine", "mxfp4", "nvfp4", "mxfp8"):
            args = self.parser.parse_args(
                ["--hf-path", "test/model", "--q-mode", mode]
            )
            assert args.q_mode == mode

    def test_q_mode_invalid_choice(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args(["--hf-path", "test/model", "--q-mode", "bogus"])

    def test_quantize_flag(self):
        args = self.parser.parse_args(["--hf-path", "test/model", "-q"])
        assert args.quantize is True

    def test_default_bits_and_group_size(self):
        args = self.parser.parse_args(["--hf-path", "test/model"])
        assert args.q_bits == 4
        assert args.q_group_size == 64

    def test_all_convert_args_present(self):
        args = self.parser.parse_args([
            "--hf-path", "test/model",
            "--mlx-path", "/tmp/out",
            "-q",
            "--q-group-size", "32",
            "--q-bits", "8",
            "--q-mode", "mxfp8",
            "--dtype", "bfloat16",
            "--upload-repo", "user/repo",
            "-d",
        ])
        assert args.hf_path == "test/model"
        assert args.mlx_path == "/tmp/out"
        assert args.quantize is True
        assert args.q_group_size == 32
        assert args.q_bits == 8
        assert args.q_mode == "mxfp8"
        assert args.dtype == "bfloat16"
        assert args.upload_repo == "user/repo"
        assert args.dequantize is True


class TestQuantizeModeDefaults:
    """Tests for the defaults_for_mode logic inside quantize_model."""

    def _call_defaults_for_mode(self, mode, group_size, bits):
        """Extract and call defaults_for_mode via quantize_model internals.

        We test by calling quantize_model with a mock model and inspecting
        the config output for the effective values.
        """
        import copy

        import mlx.core as mx
        import mlx.nn as nn

        from mlx_embeddings.utils import quantize_model

        linear = nn.Linear(128, 64)
        model = nn.Sequential(linear)
        config = {"model_type": "test"}

        weights, qconfig = quantize_model(
            model, config, group_size, bits, mode=mode
        )
        return qconfig["quantization"]["group_size"], qconfig["quantization"]["bits"], qconfig["quantization"]["mode"]

    def test_affine_defaults(self):
        gs, bits, mode = self._call_defaults_for_mode("affine", 64, 4)
        assert gs == 64
        assert bits == 4
        assert mode == "affine"

    def test_mxfp4_defaults(self):
        gs, bits, mode = self._call_defaults_for_mode("mxfp4", 0, 0)
        assert gs == 32
        assert bits == 4
        assert mode == "mxfp4"

    def test_nvfp4_defaults(self):
        gs, bits, mode = self._call_defaults_for_mode("nvfp4", 0, 0)
        assert gs == 16
        assert bits == 4
        assert mode == "nvfp4"

    def test_mxfp8_defaults(self):
        gs, bits, mode = self._call_defaults_for_mode("mxfp8", 0, 0)
        assert gs == 32
        assert bits == 8
        assert mode == "mxfp8"

    def test_user_override_group_size(self):
        gs, bits, mode = self._call_defaults_for_mode("affine", 128, 4)
        assert gs == 128
        assert bits == 4

    def test_user_override_bits(self):
        gs, bits, mode = self._call_defaults_for_mode("affine", 64, 8)
        assert gs == 64
        assert bits == 8

    def test_unsupported_mode_raises(self):
        import mlx.nn as nn

        from mlx_embeddings.utils import quantize_model

        model = nn.Sequential(nn.Linear(128, 64))
        with pytest.raises(ValueError, match="Unsupported quantization mode"):
            quantize_model(model, {}, 64, 4, mode="bogus")

    def test_config_records_mode(self):
        _, _, mode = self._call_defaults_for_mode("nvfp4", 0, 0)
        assert mode == "nvfp4"


class TestConvertQModePassthrough:
    """Verify that convert() passes q_mode through to quantize_model."""

    @patch("glob.glob", return_value=[])
    @patch("mlx_embeddings.utils.save_weights")
    @patch("mlx_embeddings.utils.save_config")
    @patch("mlx_embeddings.utils.quantize_model")
    @patch("mlx_embeddings.utils.fetch_from_hub")
    @patch("mlx_embeddings.utils.get_model_path")
    def test_convert_passes_q_mode(
        self,
        mock_get_model_path,
        mock_fetch,
        mock_quantize,
        mock_save_config,
        mock_save_weights,
        mock_glob,
    ):
        from pathlib import Path

        import mlx.core as mx
        import mlx.nn as nn

        mock_get_model_path.return_value = Path("/fake/path")

        mock_model = MagicMock(spec=nn.Module)
        mock_model.parameters.return_value = {"w": mx.zeros((4, 4))}
        mock_fetch.return_value = (mock_model, {"model_type": "test"}, MagicMock())

        mock_quantize.return_value = (
            {"w": mx.zeros((4, 4))},
            {"model_type": "test", "quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}},
        )

        from mlx_embeddings.utils import convert

        convert(
            hf_path="test/model",
            mlx_path="/tmp/out",
            quantize=True,
            q_mode="mxfp4",
        )

        mock_quantize.assert_called_once()
        call_kwargs = mock_quantize.call_args
        assert call_kwargs.kwargs.get("mode") == "mxfp4" or (
            len(call_kwargs.args) > 4 and call_kwargs.args[4] == "mxfp4"
        )
