"""Per-format export validation (SHIP Phase 4 / R-4.1 to R-4.4).

A technical integrity check — not a quality check (R-4.3). Each
validator parses the produced file and confirms internal consistency:
every choice link resolves to a passage that actually exists, the
file is loadable, the metadata block is present, etc.

Validators raise :class:`ExportValidationError` on failure with a
specific human-readable message naming the broken reference (R-4.4).
``ShipStage._phase_4_validate()`` runs the validator matching the
chosen format and re-raises as :class:`ShipStageError` so SHIP halts
before delivering the bundle (R-4.2).
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

log = get_logger(__name__)


class ExportValidationError(ValueError):
    """Raised when a per-format integrity check fails (R-4.2 / R-4.4).

    The message names the specific broken reference so the user can
    fix the upstream cause without re-running the entire pipeline.
    """


# ---------------------------------------------------------------------------
# Twee
# ---------------------------------------------------------------------------

# Top-of-line passage header, e.g. ":: Start [start]" or ":: castle".
# We capture the bare name (first whitespace-or-bracket-delimited token).
_TWEE_HEADER_RE = re.compile(r"^::\s*([^\s\[]+)")
# Plain link form: [[label->target]] or [[target]].
_TWEE_LINK_RE = re.compile(r"\[\[([^\]]+?)(?:->([^\]]+))?\]\]")
# Macro form inside <<link>>: <<goto "target">>.
_TWEE_GOTO_RE = re.compile(r'<<goto\s+"([^"]+)"\s*>>')

# Twee headers that are SugarCube infrastructure, not navigable passages.
_TWEE_RESERVED_HEADERS = {
    "StoryTitle",
    "StoryData",
    "StoryInit",
    "StoryArtDirection",
    "StoryMetadata",
}


def validate_twee(path: Path) -> None:
    """Verify Twee link reachability — every link target exists AND
    every passage is reachable from ``Start`` via choice links.

    The spec (Phase 4 Operations) requires both: ``Parse the Twee
    file; verify every :: passage_id is reachable via choice links;
    verify no broken links``. Codex/StoryArtDirection/StoryMetadata
    passages are intentionally unlinked metadata sidecars (R-3.6 +
    DRESS) and are exempt from the reachability check.

    Raises:
        ExportValidationError: If a link references an undefined passage,
            if the file lacks a ``Start`` passage, or if a navigable
            passage is defined but never reached from ``Start``.
    """
    text = path.read_text(encoding="utf-8")
    passage_names: set[str] = set()
    for line in text.splitlines():
        match = _TWEE_HEADER_RE.match(line)
        if match:
            passage_names.add(match.group(1))

    if not passage_names:
        msg = f"Twee export {path.name} contains no passage headers (no `:: <name>` lines)"
        raise ExportValidationError(msg)
    if "Start" not in passage_names:
        msg = f"Twee export {path.name} is missing a `:: Start` passage"
        raise ExportValidationError(msg)

    # Build per-passage outbound link sets, scanning each passage body
    # (everything between its `::` header and the next `::` header).
    outlinks = _twee_outlinks_per_passage(text)

    # Broken-target check: every referenced name must exist.
    referenced: set[str] = set()
    for targets in outlinks.values():
        referenced.update(targets)
    broken = sorted(
        ref for ref in referenced if ref not in passage_names and ref not in _TWEE_RESERVED_HEADERS
    )
    if broken:
        msg = (
            f"Twee export {path.name} has {len(broken)} broken link target(s): "
            f"{', '.join(repr(b) for b in broken[:5])}"
            f"{' …' if len(broken) > 5 else ''}. "
            f"Each target must match a `:: <name>` header in the file."
        )
        raise ExportValidationError(msg)

    # Reachability BFS from Start. Passages defined as metadata
    # sidecars (codex, art direction, story metadata) are intentionally
    # unlinked — exclude them from the reachable-set requirement.
    reachable = _bfs_reachable("Start", outlinks)
    must_reach = {
        name for name in passage_names if name not in _TWEE_RESERVED_HEADERS and name != "Codex"
    }
    orphans = sorted(must_reach - reachable)
    if orphans:
        msg = (
            f"Twee export {path.name} has {len(orphans)} passage(s) defined but "
            f"unreachable from `:: Start`: "
            f"{', '.join(repr(o) for o in orphans[:5])}"
            f"{' …' if len(orphans) > 5 else ''}. "
            f"Every navigable passage must be reachable via choice links."
        )
        raise ExportValidationError(msg)


def _twee_outlinks_per_passage(text: str) -> dict[str, set[str]]:
    """Map each passage name to the set of names it links to.

    Splits the file at `::` headers and scans each body for both link
    forms ([[…]] and <<goto "…">>).
    """
    lines = text.splitlines()
    headers: list[tuple[int, str]] = []  # (line index, name)
    for i, line in enumerate(lines):
        match = _TWEE_HEADER_RE.match(line)
        if match:
            headers.append((i, match.group(1)))

    outlinks: dict[str, set[str]] = {}
    for idx, (line_no, name) in enumerate(headers):
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        body = "\n".join(lines[line_no + 1 : end])
        targets: set[str] = set()
        for match in _TWEE_LINK_RE.finditer(body):
            targets.add((match.group(2) or match.group(1)).strip())
        for match in _TWEE_GOTO_RE.finditer(body):
            targets.add(match.group(1).strip())
        outlinks[name] = targets
    return outlinks


def _bfs_reachable(start: str, outlinks: dict[str, set[str]]) -> set[str]:
    """Standard BFS over the link graph starting at ``start``."""
    seen: set[str] = {start}
    frontier: list[str] = [start]
    while frontier:
        node = frontier.pop()
        for target in outlinks.get(node, ()):
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    return seen


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

# Required top-level keys. Schema is defined by ``json_exporter.JsonExporter``;
# additive changes are allowed (R-3.4) so we only check the always-present set.
_JSON_REQUIRED_KEYS = {"_metadata", "title", "passages", "choices"}
_JSON_REQUIRED_METADATA_KEYS = {
    "pipeline_version",
    "graph_snapshot_hash",
    "format_version",
    "generation_timestamp",
}


def validate_json(path: Path) -> None:
    """Verify JSON loadable, schema-shaped, and internally consistent.

    Raises:
        ExportValidationError: On parse failure, missing required keys,
            or any choice referencing a passage id that is not in the
            passages list.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        msg = f"JSON export {path.name} could not be parsed: {e}"
        raise ExportValidationError(msg) from e

    if not isinstance(data, dict):
        msg = f"JSON export {path.name} top-level is not an object"
        raise ExportValidationError(msg)

    missing_keys = sorted(_JSON_REQUIRED_KEYS - data.keys())
    if missing_keys:
        msg = f"JSON export {path.name} missing required top-level key(s): {missing_keys}"
        raise ExportValidationError(msg)

    metadata = data["_metadata"]
    if not isinstance(metadata, dict):
        msg = f"JSON export {path.name} `_metadata` is not an object"
        raise ExportValidationError(msg)
    missing_meta = sorted(_JSON_REQUIRED_METADATA_KEYS - metadata.keys())
    if missing_meta:
        msg = (
            f"JSON export {path.name} `_metadata` missing required key(s): {missing_meta}. "
            f"R-3.6 requires every export to carry a complete provenance block."
        )
        raise ExportValidationError(msg)

    passage_ids = {p["id"] for p in data["passages"] if isinstance(p, dict) and "id" in p}
    broken = sorted(
        c["to_passage"]
        for c in data["choices"]
        if isinstance(c, dict) and c.get("to_passage") not in passage_ids
    )
    if broken:
        msg = (
            f"JSON export {path.name} has {len(broken)} choice(s) targeting "
            f"undefined passage(s): {', '.join(repr(b) for b in broken[:5])}"
            f"{' …' if len(broken) > 5 else ''}."
        )
        raise ExportValidationError(msg)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


class _HtmlChoiceCollector(HTMLParser):
    """Collect passage div ids and choice anchor data-target values.

    Used to verify that every choice points at an existing passage div
    without spinning up a headless browser (R-4.3 says technical
    integrity, not visual rendering).
    """

    def __init__(self) -> None:
        super().__init__()
        self.passage_ids: set[str] = set()
        self.choice_targets: list[str] = []
        self.has_body = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "body":
            self.has_body = True
        if tag == "div":
            attr_dict = dict(attrs)
            pid = attr_dict.get("id")
            if attr_dict.get("class") == "passage" and pid:
                self.passage_ids.add(pid)
        if tag == "a":
            attr_dict = dict(attrs)
            classes = (attr_dict.get("class") or "").split()
            target = attr_dict.get("data-target")
            if "choice" in classes and target:
                self.choice_targets.append(target)


def validate_html(path: Path) -> None:
    """Verify HTML parses, has a body, and every choice resolves.

    Raises:
        ExportValidationError: If the file fails to parse, lacks a
            ``<body>``, has no passage divs, or has a ``data-target``
            attribute on a ``<a class="choice">`` that doesn't match
            an existing passage div id.
    """
    text = path.read_text(encoding="utf-8")
    parser = _HtmlChoiceCollector()
    try:
        parser.feed(text)
    except Exception as e:  # html.parser is permissive; surface anything that escapes
        msg = f"HTML export {path.name} failed to parse: {e}"
        raise ExportValidationError(msg) from e

    if not parser.has_body:
        msg = f"HTML export {path.name} has no <body> element"
        raise ExportValidationError(msg)
    if not parser.passage_ids:
        msg = f'HTML export {path.name} has no passage divs (`<div class="passage">`)'
        raise ExportValidationError(msg)

    broken = sorted({t for t in parser.choice_targets if t not in parser.passage_ids})
    if broken:
        msg = (
            f"HTML export {path.name} has {len(broken)} choice(s) with broken "
            f"data-target attribute(s): {', '.join(repr(b) for b in broken[:5])}"
            f"{' …' if len(broken) > 5 else ''}. "
            f"Each value must equal a passage div id."
        )
        raise ExportValidationError(msg)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def validate_pdf(path: Path) -> None:
    """Verify the PDF sidecar is present and the page map is internally consistent.

    Per R-4.3 we check the sidecar (which carries every passage_id →
    page_number assignment) rather than parsing PDF internals. The
    sidecar is the canonical source of "turn to page X" targets, so
    every value must be in the range ``1..N`` where N is the number
    of distinct page numbers.

    Raises:
        ExportValidationError: Sidecar missing, malformed, or carrying
            an out-of-range page number.
    """
    # with_name (not with_suffix) — multi-dot suffix warns under Python 3.12
    # and raises under 3.13. Mirror the construction in pdf_exporter._write_pdf_sidecar.
    sidecar = path.with_name(path.name + ".map.json")
    if not sidecar.exists():
        msg = (
            f"PDF export {path.name} is missing its sidecar {sidecar.name} "
            f"(required by R-3.6 + #1336 for provenance and pagination debugging)."
        )
        raise ExportValidationError(msg)

    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        msg = f"PDF sidecar {sidecar.name} could not be parsed: {e}"
        raise ExportValidationError(msg) from e

    page_map = data.get("page_map")
    if not isinstance(page_map, dict) or not page_map:
        msg = f"PDF sidecar {sidecar.name} `page_map` is missing or empty"
        raise ExportValidationError(msg)

    # Valid range is 1..N where N is the total number of passages
    # (entries in page_map). A bijective gamebook maps every passage
    # to a distinct page number in that range; using len(set(values))
    # would shrink N in the presence of duplicates and produce a
    # confusing out-of-range message instead of the duplicate one.
    pages = list(page_map.values())
    n_pages = len(page_map)
    out_of_range = sorted(
        f"{pid} → {pg}"
        for pid, pg in page_map.items()
        if not isinstance(pg, int) or pg < 1 or pg > n_pages
    )
    if out_of_range:
        msg = (
            f"PDF sidecar {sidecar.name} has {len(out_of_range)} page number(s) "
            f"outside the valid range 1..{n_pages}: "
            f"{', '.join(out_of_range[:5])}"
            f"{' …' if len(out_of_range) > 5 else ''}."
        )
        raise ExportValidationError(msg)
    if len(pages) != len(set(pages)):
        msg = (
            f"PDF sidecar {sidecar.name} has duplicate page numbers — "
            f"{len(pages)} entries, only {len(set(pages))} distinct pages."
        )
        raise ExportValidationError(msg)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

VALIDATORS: dict[str, Callable[[Path], None]] = {
    "twee": validate_twee,
    "json": validate_json,
    "html": validate_html,
    "pdf": validate_pdf,
}


def validate_export(format_name: str, path: Path) -> None:
    """Dispatch to the validator matching ``format_name``.

    Raises:
        ExportValidationError: Forwarded from the per-format validator.
        KeyError: If ``format_name`` has no registered validator (a
            programmer error — every supported format must have one).
    """
    validator = VALIDATORS[format_name]
    validator(path)
