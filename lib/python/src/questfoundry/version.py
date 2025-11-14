"""Version information for QuestFoundry-Py."""

__version_info__ = (0, 6, 0)
__version__ = ".".join(map(str, __version_info__))


def get_version() -> str:
    """Get the version string.

    Returns:
        The version string in MAJOR.MINOR.PATCH format.

    Example:
        >>> get_version()
        '2.0.0'
    """
    return __version__
