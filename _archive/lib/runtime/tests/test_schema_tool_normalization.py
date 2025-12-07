from __future__ import annotations

from questfoundry.runtime.core.schema_tool_generator import (
    _normalize_artifact_input,
    _normalize_tu_brief_checkpoint,
)


def test_normalize_tu_brief_checkpoint_passthrough_valid_hh_mm() -> None:
    """Already-valid HH:MM values should be returned unchanged."""
    assert _normalize_tu_brief_checkpoint("01:30") == "01:30"
    assert _normalize_tu_brief_checkpoint("23:59") == "23:59"


def test_normalize_tu_brief_checkpoint_from_plain_minutes() -> None:
    """Plain minute counts are converted to HH:MM."""
    assert _normalize_tu_brief_checkpoint("45") == "00:45"
    assert _normalize_tu_brief_checkpoint("90") == "01:30"


def test_normalize_tu_brief_checkpoint_from_mm_ss() -> None:
    """MM:SS-style values are interpreted as minutes:seconds."""
    # Common pattern seen in logs: "45:00" intended as 45 minutes
    assert _normalize_tu_brief_checkpoint("45:00") == "00:45"


def test_normalize_artifact_input_codex_pack_coverage_report_string() -> None:
    """String coverage_report is wrapped into a minimal dict structure."""
    data = {"coverage_report": "Concise summary of coverage."}
    normalized = _normalize_artifact_input("codex_pack", data)
    assert isinstance(normalized["coverage_report"], dict)
    assert normalized["coverage_report"]["summary"] == "Concise summary of coverage."
