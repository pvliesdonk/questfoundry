from __future__ import annotations

from questfoundry.runtime.tools.validation_tools import EvaluateQualityBar, ValidateArtifact


def test_validate_artifact_success():
    tool = ValidateArtifact()
    result = tool._run(artifact_type="phrasing_patterns", content={})
    assert result["valid"] is True


def test_validate_artifact_failure():
    tool = ValidateArtifact()
    result = tool._run(artifact_type="phrasing_patterns", content="not-an-object")
    assert result["valid"] is False
    assert result["errors"]


def test_evaluate_quality_bar_skips_checks():
    tool = EvaluateQualityBar()
    result = tool._run(gate_id="integrity", artifacts={})
    assert result["overall_status"] == "pass"
    assert result["checks"]
    assert result["checks"][0]["status"] == "skipped"
