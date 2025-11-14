"""Version information for QuestFoundry-Py."""

__version__ = "2.0.0"
__version_info__ = (2, 0, 0)


def get_version() -> str:
    """Get the version string.

    Returns:
        The version string in MAJOR.MINOR.PATCH format.

    Example:
        >>> get_version()
        '2.0.0'
    """
    return __version__
