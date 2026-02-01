"""Export format handlers (Twee, HTML, JSON)."""

from __future__ import annotations

from questfoundry.export.base import (
    ExportChoice,
    ExportCodeword,
    ExportCodexEntry,
    ExportContext,
    ExportEntity,
    Exporter,
    ExportIllustration,
    ExportPassage,
)
from questfoundry.export.context import build_export_context
from questfoundry.export.json_exporter import JsonExporter
from questfoundry.export.twee_exporter import TweeExporter

_EXPORTERS: dict[str, type[JsonExporter | TweeExporter]] = {
    "json": JsonExporter,
    "twee": TweeExporter,
}


def get_exporter(format_name: str) -> JsonExporter | TweeExporter:
    """Get an exporter instance by format name.

    Args:
        format_name: Export format (e.g., "json", "twee", "html").

    Returns:
        Exporter instance.

    Raises:
        ValueError: If the format is not supported.
    """
    cls = _EXPORTERS.get(format_name)
    if cls is None:
        supported = ", ".join(sorted(_EXPORTERS))
        msg = f"Unknown export format '{format_name}'. Supported: {supported}"
        raise ValueError(msg)
    return cls()


__all__ = [
    "ExportChoice",
    "ExportCodeword",
    "ExportCodexEntry",
    "ExportContext",
    "ExportEntity",
    "ExportIllustration",
    "ExportPassage",
    "Exporter",
    "JsonExporter",
    "TweeExporter",
    "build_export_context",
    "get_exporter",
]
