"""Deterministic metadata header shared across all SHIP exporters.

Implements R-3.6: every export must carry a metadata block with
``pipeline_version``, ``graph_snapshot_hash``, ``format_version``, and
``generation_timestamp`` so downstream auditors can pin a delivered
artefact to the exact source graph and code version that produced it.

Each exporter declares its own ``format_version`` constant and embeds
the metadata in a format-appropriate way (JSON top-level field, Twee
metadata passage, HTML ``<meta>`` tags, PDF sidecar JSON).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.export.base import ExportContext

_PACKAGE_NAME = "questfoundry"
_UNKNOWN_VERSION = "0.0.0+unknown"


@dataclass(frozen=True)
class ExportMetadata:
    """R-3.6 metadata header carried by every SHIP export."""

    pipeline_version: str
    graph_snapshot_hash: str
    format_version: str
    generation_timestamp: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def get_pipeline_version() -> str:
    """Resolve the installed questfoundry version, or a stable sentinel.

    Returns ``_UNKNOWN_VERSION`` when the package is not installed
    (e.g. running from a checkout without ``pip install -e .``); the
    sentinel is stable so the metadata block stays well-formed and
    snapshot hashes remain reproducible across runs.
    """
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _UNKNOWN_VERSION


def compute_graph_snapshot_hash(context: ExportContext) -> str:
    """Deterministic content hash of the ExportContext.

    Hashes a canonical JSON serialization of the context (sorted keys,
    no whitespace) so the same graph produces the same hash across
    runs, machines, and Python versions. Used to fingerprint the
    source state of an export.
    """
    payload = json.dumps(
        asdict(context),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_export_metadata(
    context: ExportContext,
    format_version: str,
    *,
    timestamp: str | None = None,
) -> ExportMetadata:
    """Assemble an ``ExportMetadata`` for the given context and format.

    Args:
        context: Source ExportContext (its content drives the snapshot hash).
        format_version: Per-exporter schema version (e.g. ``"1.0.0"``).
        timestamp: Optional override for the generation timestamp; tests
            inject a fixed value to make their assertions stable. Defaults
            to ``datetime.now(UTC).isoformat()``.

    Returns:
        ExportMetadata with all four R-3.6 fields populated.
    """
    return ExportMetadata(
        pipeline_version=get_pipeline_version(),
        graph_snapshot_hash=compute_graph_snapshot_hash(context),
        format_version=format_version,
        generation_timestamp=timestamp or datetime.now(UTC).isoformat(),
    )
