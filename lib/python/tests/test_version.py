"""Tests for version module."""

from questfoundry.version import __version__, __version_info__, get_version


def test_version_string_exists():
    """Test that version string is defined."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_version_info_tuple():
    """Test that version info is a tuple."""
    assert isinstance(__version_info__, tuple)
    assert len(__version_info__) == 3
    assert all(isinstance(x, int) for x in __version_info__)


def test_get_version_returns_string():
    """Test get_version returns the version string."""
    version = get_version()
    assert isinstance(version, str)
    assert version == __version__


def test_version_format():
    """Test version follows semantic versioning format."""
    # Should be in format X.Y.Z or X.Y.Z-suffix
    parts = __version__.split("-")[0].split(".")
    assert len(parts) == 3, "Version must be in MAJOR.MINOR.PATCH format"
    assert all(p.isdigit() for p in parts), "All version parts must be integers"
