"""Type stub extensions for the logging module to add trace method."""

import logging
from typing import Any

# Extend the Logger class to include the trace method
class Logger(logging.Logger):
    def trace(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
