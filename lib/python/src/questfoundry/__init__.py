"""QuestFoundry Python Library - Layer 6"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("questfoundry-py")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

from .logging_config import get_logger, setup_logging  # noqa: F401
from .models import Artifact, HookCard, TUBrief  # noqa: F401
from .validators import validate_instance, validate_schema  # noqa: F401

__all__ = [
    "__version__",
    "Artifact",
    "HookCard",
    "TUBrief",
    "validate_schema",
    "validate_instance",
    "setup_logging",
    "get_logger",
]
