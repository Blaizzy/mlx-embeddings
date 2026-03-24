import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_uses_version_module():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert re.search(
        r"dynamic\s*=\s*\[[^\]]*\"version\"",
        pyproject,
        re.DOTALL,
    )
    assert re.search(
        r'version\s*=\s*\{\s*attr\s*=\s*"mlx_embeddings\.version\.__version__"\s*\}',
        pyproject,
    )
