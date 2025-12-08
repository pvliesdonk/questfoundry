"""Tests for Gatekeeper quality bar evaluation tools."""

from __future__ import annotations

import json

import pytest

from questfoundry.runtime.tools.gatekeeper import (
    CreateGatecheckReport,
    EvaluateAccessibility,
    EvaluateDeterminism,
    EvaluateGateways,
    EvaluateIntegrity,
    EvaluateNonlinearity,
    EvaluatePresentation,
    EvaluateReachability,
    EvaluateStyle,
)


class TestEvaluationTools:
    """Tests for quality bar evaluation tools."""

    @pytest.fixture
    def state_with_artifact(self) -> dict:
        """Create a state with a test artifact."""
        return {
            "hot_store": {
                "topology_001": {
                    "title": "Mystery Mansion",
                    "scenes": [
                        {"id": "entry", "title": "Entry Hall", "choices": ["library", "kitchen"]},
                        {"id": "library", "title": "Library", "choices": ["entry", "secret_room"]},
                        {"id": "kitchen", "title": "Kitchen", "choices": ["entry"]},
                        {"id": "secret_room", "title": "Secret Room", "choices": ["library"]},
                    ],
                    "gates": [
                        {"target": "secret_room", "condition": "has_key"}
                    ]
                }
            },
            "metadata": {},
        }

    def test_evaluate_integrity_missing_artifact(self) -> None:
        """Evaluate integrity returns error for missing artifact."""
        tool = EvaluateIntegrity()
        tool.state = {"hot_store": {}}

        result = json.loads(tool._run("nonexistent"))
        assert result["passed"] is False
        assert "not found" in result["issues"][0]

    def test_evaluate_integrity_with_artifact(self, state_with_artifact: dict) -> None:
        """Evaluate integrity returns criteria for valid artifact."""
        tool = EvaluateIntegrity()
        tool.state = state_with_artifact

        result = json.loads(tool._run("topology_001"))
        assert result["bar"] == "integrity"
        assert result["artifact_id"] == "topology_001"
        assert "artifact_content" in result
        assert "evaluation_criteria" in result
        assert len(result["evaluation_criteria"]) > 0

    def test_evaluate_reachability_with_artifact(self, state_with_artifact: dict) -> None:
        """Evaluate reachability returns criteria for valid artifact."""
        tool = EvaluateReachability()
        tool.state = state_with_artifact

        result = json.loads(tool._run("topology_001"))
        assert result["bar"] == "reachability"
        assert any("reach" in c.lower() for c in result["evaluation_criteria"])

    def test_evaluate_nonlinearity_with_artifact(self, state_with_artifact: dict) -> None:
        """Evaluate nonlinearity returns criteria for valid artifact."""
        tool = EvaluateNonlinearity()
        tool.state = state_with_artifact

        result = json.loads(tool._run("topology_001"))
        assert result["bar"] == "nonlinearity"
        assert any("path" in c.lower() for c in result["evaluation_criteria"])

    def test_evaluate_gateways_with_artifact(self, state_with_artifact: dict) -> None:
        """Evaluate gateways returns criteria for valid artifact."""
        tool = EvaluateGateways()
        tool.state = state_with_artifact

        result = json.loads(tool._run("topology_001"))
        assert result["bar"] == "gateways"
        assert any("gate" in c.lower() for c in result["evaluation_criteria"])

    def test_evaluate_style_with_artifact(self, state_with_artifact: dict) -> None:
        """Evaluate style returns criteria for valid artifact."""
        tool = EvaluateStyle()
        tool.state = state_with_artifact

        result = json.loads(tool._run("topology_001"))
        assert result["bar"] == "style"
        assert any("voice" in c.lower() or "tone" in c.lower() for c in result["evaluation_criteria"])

    def test_evaluate_determinism_with_artifact(self, state_with_artifact: dict) -> None:
        """Evaluate determinism returns criteria for valid artifact."""
        tool = EvaluateDeterminism()
        tool.state = state_with_artifact

        result = json.loads(tool._run("topology_001"))
        assert result["bar"] == "determinism"
        assert any("consistent" in c.lower() or "same" in c.lower() for c in result["evaluation_criteria"])

    def test_evaluate_presentation_with_artifact(self, state_with_artifact: dict) -> None:
        """Evaluate presentation returns criteria for valid artifact."""
        tool = EvaluatePresentation()
        tool.state = state_with_artifact

        result = json.loads(tool._run("topology_001"))
        assert result["bar"] == "presentation"
        assert any("format" in c.lower() or "structure" in c.lower() for c in result["evaluation_criteria"])

    def test_evaluate_accessibility_with_artifact(self, state_with_artifact: dict) -> None:
        """Evaluate accessibility returns criteria for valid artifact."""
        tool = EvaluateAccessibility()
        tool.state = state_with_artifact

        result = json.loads(tool._run("topology_001"))
        assert result["bar"] == "accessibility"
        assert any("warning" in c.lower() or "alt text" in c.lower() for c in result["evaluation_criteria"])

    def test_no_state_returns_error(self) -> None:
        """Tools return error when no state is set."""
        tool = EvaluateIntegrity()
        tool.state = None

        result = json.loads(tool._run("anything"))
        assert "error" in result


class TestCreateGatecheckReport:
    """Tests for the GatecheckReport creation tool."""

    @pytest.fixture
    def state(self) -> dict:
        """Create an empty state for report creation."""
        return {
            "hot_store": {},
            "metadata": {},
        }

    def test_create_passing_report(self, state: dict) -> None:
        """Create a passing gatecheck report."""
        tool = CreateGatecheckReport()
        tool.state = state
        tool.role_id = "gatekeeper"

        result = json.loads(tool._run(
            target_artifact="topology_001",
            bars_checked=["integrity", "reachability"],
            status="passed",
            bar_results={"integrity": "PASS", "reachability": "PASS"},
            issues=[],
            recommendations=[],
        ))

        assert result["success"] is True
        assert "report_id" in result
        assert result["status"] == "passed"
        assert state["hot_store"][result["report_id"]]["status"] == "passed"

    def test_create_failing_report(self, state: dict) -> None:
        """Create a failing gatecheck report with issues."""
        tool = CreateGatecheckReport()
        tool.state = state
        tool.role_id = "gatekeeper"

        result = json.loads(tool._run(
            target_artifact="scene_001",
            bars_checked=["style", "accessibility"],
            status="failed",
            bar_results={"style": "FAIL - inconsistent tone", "accessibility": "PASS"},
            issues=["Inconsistent narrative voice between paragraphs 2 and 5"],
            recommendations=["Review with Narrator for tone consistency"],
        ))

        assert result["success"] is True
        assert result["status"] == "failed"
        assert result["issues_count"] == 1

        report = state["hot_store"][result["report_id"]]
        assert report["status"] == "failed"
        assert len(report["issues"]) == 1
        assert len(report["recommendations"]) == 1

    def test_invalid_quality_bar(self, state: dict) -> None:
        """Invalid quality bar names are rejected."""
        tool = CreateGatecheckReport()
        tool.state = state
        tool.role_id = "gatekeeper"

        result = json.loads(tool._run(
            target_artifact="topology_001",
            bars_checked=["invalid_bar", "also_invalid"],
            status="passed",
        ))

        assert "error" in result
        assert "invalid_bar" in result["error"]

    def test_report_numbering(self, state: dict) -> None:
        """Reports are numbered incrementally."""
        tool = CreateGatecheckReport()
        tool.state = state
        tool.role_id = "gatekeeper"

        # Create first report
        result1 = json.loads(tool._run(
            target_artifact="artifact_a",
            bars_checked=["integrity"],
            status="passed",
        ))

        # Create second report
        result2 = json.loads(tool._run(
            target_artifact="artifact_b",
            bars_checked=["style"],
            status="failed",
            issues=["Issue found"],
        ))

        # Verify different IDs
        assert result1["report_id"] != result2["report_id"]
        assert len(state["hot_store"]) == 2

    def test_waiver_reason_included(self, state: dict) -> None:
        """Waiver reason is stored when provided."""
        tool = CreateGatecheckReport()
        tool.state = state
        tool.role_id = "gatekeeper"

        result = json.loads(tool._run(
            target_artifact="topology_001",
            bars_checked=["accessibility"],
            status="waived",
            waiver_reason="Time constraint - will address in next iteration",
        ))

        report = state["hot_store"][result["report_id"]]
        assert report["status"] == "waived"
        assert "waiver_reason" in report
        assert "Time constraint" in report["waiver_reason"]

    def test_no_state_returns_error(self) -> None:
        """Tool returns error when no state is set."""
        tool = CreateGatecheckReport()
        tool.state = None

        result = json.loads(tool._run(
            target_artifact="anything",
            bars_checked=["integrity"],
            status="passed",
        ))

        assert "error" in result


class TestAllQualityBars:
    """Test that all 8 quality bars have evaluation tools."""

    def test_all_bars_have_tools(self) -> None:
        """Verify all 8 quality bars have corresponding evaluation tools."""
        from questfoundry.generated.models.enums import QualityBar

        tool_bar_mapping = {
            EvaluateIntegrity: QualityBar.INTEGRITY,
            EvaluateReachability: QualityBar.REACHABILITY,
            EvaluateNonlinearity: QualityBar.NONLINEARITY,
            EvaluateGateways: QualityBar.GATEWAYS,
            EvaluateStyle: QualityBar.STYLE,
            EvaluateDeterminism: QualityBar.DETERMINISM,
            EvaluatePresentation: QualityBar.PRESENTATION,
            EvaluateAccessibility: QualityBar.ACCESSIBILITY,
        }

        # Verify we have a tool for each quality bar
        assert len(tool_bar_mapping) == len(QualityBar)

        # Verify each tool reports the correct bar name
        state = {"hot_store": {"test": {"data": "value"}}}
        for ToolClass, bar in tool_bar_mapping.items():
            tool = ToolClass()
            tool.state = state
            result = json.loads(tool._run("test"))
            assert result["bar"] == bar.value, f"{ToolClass.__name__} should evaluate {bar.value}"
