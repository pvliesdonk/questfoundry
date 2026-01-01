"""Test version information."""

from questfoundry import __version__


def test_version_is_valid() -> None:
    """Version string follows semver format."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_version_matches_pyproject() -> None:
    """Version matches pyproject.toml."""
    import tomllib
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    assert __version__ == pyproject["project"]["version"]
