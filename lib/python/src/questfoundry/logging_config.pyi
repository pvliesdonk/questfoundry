"""Type stubs for logging_config module."""

import logging
from typing import Literal

TRACE: int

class TraceFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "%",
        include_module: bool = True,
    ) -> None: ...
    def format(self, record: logging.LogRecord) -> str: ...

def setup_logging(
    level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    format_string: str | None = None,
    include_module: bool = True,
    handlers: list[logging.Handler] | None = None,
) -> None: ...
def get_logger(name: str) -> logging.Logger: ...
