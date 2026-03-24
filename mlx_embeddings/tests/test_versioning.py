from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_uses_internal_version_module():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "version" in pyproject["project"]["dynamic"]
    assert (
        pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        == "mlx_embeddings._version.__version__"
    )
