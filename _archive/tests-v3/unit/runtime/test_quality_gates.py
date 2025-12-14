"""Tests for quality gate enforcement."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.runtime.quality_gates import (
    QualityGateValidator,
    TransitionValidationResult,
    ValidationResult,
    create_quality_gate_validator,
)


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_basic_result(self):
        """Test basic validation result."""
        result = ValidationResult(
            criterion_id="integrity",
            passed=True,
            score=1.0,
            feedback="All checks passed",
        )

        assert result.criterion_id == "integrity"
        assert result.passed is True
        assert result.score == 1.0
        assert result.feedback == "All checks passed"
        assert result.issues == []

    def test_failed_result_with_issues(self):
        """Test failed result with issues."""
        result = ValidationResult(
            criterion_id="integrity",
            passed=False,
            blocking=True,
            score=0.5,
            feedback="Some checks failed",
            issues=["Dangling link found", "Missing return path"],
        )

        assert not result.passed
        assert result.blocking is True
        assert len(result.issues) == 2


class TestTransitionValidationResult:
    """Tests for TransitionValidationResult model."""

    def test_successful_transition(self):
        """Test successful transition result."""
        result = TransitionValidationResult(
            can_transition=True,
            results=[
                ValidationResult(criterion_id="integrity", passed=True),
                ValidationResult(criterion_id="style", passed=True),
            ],
            guidance="All criteria passed.",
        )

        assert result.can_transition is True
        assert len(result.results) == 2
        assert len(result.blocking_failures) == 0

    def test_blocked_transition(self):
        """Test blocked transition result."""
        result = TransitionValidationResult(
            can_transition=False,
            results=[
                ValidationResult(criterion_id="integrity", passed=False, blocking=True),
                ValidationResult(criterion_id="style", passed=True),
            ],
            blocking_failures=["integrity"],
            guidance="Address blocking issues.",
        )

        assert result.can_transition is False
        assert "integrity" in result.blocking_failures


class TestQualityGateValidator:
    """Tests for QualityGateValidator."""

    @pytest.fixture
    def mock_studio(self):
        """Create a mock studio with quality criteria."""
        studio = MagicMock()

        # Mock artifact type with lifecycle
        section_type = MagicMock()
        lifecycle = MagicMock()

        # Mock transition with requires_validation
        transition = MagicMock()
        transition.from_state = "approved"
        transition.to_state = "cold"
        transition.requires_validation = ["integrity", "style"]

        lifecycle.transitions = [transition]
        section_type.lifecycle = lifecycle

        studio.artifact_types = {"section": section_type}

        # Mock quality criteria - use explicit dict-like behavior
        integrity_criterion = MagicMock()
        integrity_criterion.id = "integrity"
        integrity_criterion.name = "Integrity"
        integrity_criterion.description = "Check structural integrity"
        integrity_criterion.enforcement = "llm"
        integrity_criterion.blocking = "gate"
        integrity_criterion.check = {"type": "llm_rubric"}  # Actual dict
        integrity_criterion.failure_guidance = "Fix dangling links"
        # Disable model_dump to use __dict__ fallback
        integrity_criterion.model_dump = None

        style_criterion = MagicMock()
        style_criterion.id = "style"
        style_criterion.name = "Style"
        style_criterion.description = "Check style consistency"
        style_criterion.enforcement = "llm"
        style_criterion.blocking = "advisory"
        style_criterion.check = {"type": "llm_rubric"}  # Actual dict
        style_criterion.failure_guidance = "Improve style"
        style_criterion.model_dump = None

        studio.quality_criteria = {
            "integrity": integrity_criterion,
            "style": style_criterion,
        }

        return studio

    @pytest.fixture
    def validator(self, mock_studio):
        """Create a validator with mock studio."""
        return QualityGateValidator(studio=mock_studio)

    def test_get_transition_criteria(self, validator):
        """Test getting criteria for a transition."""
        criteria = validator.get_transition_criteria("section", "approved", "cold")

        assert "integrity" in criteria
        assert "style" in criteria

    def test_get_transition_criteria_no_validation(self, validator):
        """Test transition with no validation requirements."""
        criteria = validator.get_transition_criteria("section", "draft", "review")

        assert criteria == []

    def test_get_transition_criteria_unknown_type(self, validator):
        """Test with unknown artifact type."""
        criteria = validator.get_transition_criteria("unknown", "draft", "review")

        assert criteria == []

    def test_get_criterion_definition(self, validator):
        """Test getting a criterion definition."""
        criterion = validator.get_criterion_definition("integrity")

        assert criterion is not None
        assert criterion["id"] == "integrity"
        assert criterion["enforcement"] == "llm"
        assert criterion["blocking"] == "gate"

    def test_get_criterion_definition_unknown(self, validator):
        """Test with unknown criterion."""
        criterion = validator.get_criterion_definition("unknown")

        assert criterion is None


class TestQualityGateValidatorAsync:
    """Async tests for QualityGateValidator."""

    @pytest.fixture
    def mock_studio(self):
        """Create a mock studio with quality criteria."""
        studio = MagicMock()

        # Mock artifact type
        section_type = MagicMock()
        lifecycle = MagicMock()

        transition = MagicMock()
        transition.from_state = "approved"
        transition.to_state = "cold"
        transition.requires_validation = ["integrity"]

        lifecycle.transitions = [transition]
        section_type.lifecycle = lifecycle

        studio.artifact_types = {"section": section_type}

        # Mock criterion - blocking gate with actual dict for check
        integrity_criterion = MagicMock()
        integrity_criterion.id = "integrity"
        integrity_criterion.name = "Integrity"
        integrity_criterion.description = "Check integrity"
        integrity_criterion.enforcement = "llm"
        integrity_criterion.blocking = "gate"
        integrity_criterion.check = {"type": "llm_rubric", "rubric": {}}  # Actual dict
        integrity_criterion.failure_guidance = "Fix issues"
        integrity_criterion.model_dump = None  # Use __dict__ fallback

        studio.quality_criteria = {"integrity": integrity_criterion}

        return studio

    @pytest.fixture
    def validator(self, mock_studio):
        return QualityGateValidator(studio=mock_studio)

    @pytest.fixture
    def mock_llm_pass(self):
        """LLM that returns passing result."""
        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(
            content='{"criterion_id": "integrity", "passed": true, "score": 0.9, "feedback": "Good", "issues": []}'
        )
        return llm

    @pytest.fixture
    def mock_llm_fail(self):
        """LLM that returns failing result."""
        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(
            content='{"criterion_id": "integrity", "passed": false, "score": 0.3, "feedback": "Issues found", "issues": ["Dangling link"]}'
        )
        return llm

    @pytest.mark.asyncio
    async def test_validate_transition_no_criteria(self, validator):
        """Test validation when no criteria required."""
        artifact = {"type": "section", "data": {"content": "test"}}

        result = await validator.validate_transition(
            artifact=artifact,
            artifact_type="section",
            from_state="draft",
            to_state="review",
        )

        assert result.can_transition is True
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_validate_transition_passes(self, validator, mock_llm_pass):
        """Test validation that passes."""
        artifact = {"type": "section", "data": {"content": "test"}}

        result = await validator.validate_transition(
            artifact=artifact,
            artifact_type="section",
            from_state="approved",
            to_state="cold",
            llm=mock_llm_pass,
        )

        assert result.can_transition is True
        assert len(result.results) == 1
        assert result.results[0].passed is True

    @pytest.mark.asyncio
    async def test_validate_transition_blocked(self, validator, mock_llm_fail):
        """Test validation that blocks transition."""
        artifact = {"type": "section", "data": {"content": "test"}}

        result = await validator.validate_transition(
            artifact=artifact,
            artifact_type="section",
            from_state="approved",
            to_state="cold",
            llm=mock_llm_fail,
        )

        assert result.can_transition is False
        assert "integrity" in result.blocking_failures
        assert "blocking criteria failed" in result.guidance.lower()

    @pytest.mark.asyncio
    async def test_validate_transition_no_llm_deferred(self, validator):
        """Test validation defers when no LLM provided."""
        artifact = {"type": "section", "data": {"content": "test"}}

        result = await validator.validate_transition(
            artifact=artifact,
            artifact_type="section",
            from_state="approved",
            to_state="cold",
            llm=None,
        )

        # Should pass by default when LLM unavailable
        assert result.can_transition is True
        assert result.results[0].feedback == "LLM validation deferred (no LLM provided)"


class TestQualityGateValidatorAdvisory:
    """Tests for advisory (non-blocking) criteria."""

    @pytest.fixture
    def mock_studio_advisory(self):
        """Create studio with advisory criteria."""
        studio = MagicMock()

        section_type = MagicMock()
        lifecycle = MagicMock()

        transition = MagicMock()
        transition.from_state = "draft"
        transition.to_state = "review"
        transition.requires_validation = ["style"]

        lifecycle.transitions = [transition]
        section_type.lifecycle = lifecycle

        studio.artifact_types = {"section": section_type}

        # Advisory criterion (doesn't block) with actual dict for check
        style_criterion = MagicMock()
        style_criterion.id = "style"
        style_criterion.name = "Style"
        style_criterion.description = "Check style"
        style_criterion.enforcement = "llm"
        style_criterion.blocking = "advisory"  # Not gate!
        style_criterion.check = {"type": "llm_rubric", "rubric": {}}  # Actual dict
        style_criterion.failure_guidance = "Improve style"
        style_criterion.model_dump = None  # Use __dict__ fallback

        studio.quality_criteria = {"style": style_criterion}

        return studio

    @pytest.mark.asyncio
    async def test_advisory_failure_doesnt_block(self, mock_studio_advisory):
        """Test that advisory criterion failure doesn't block transition."""
        validator = QualityGateValidator(studio=mock_studio_advisory)

        # LLM that returns failure
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='{"criterion_id": "style", "passed": false, "score": 0.4, "feedback": "Style needs work", "issues": ["Inconsistent tone"]}'
        )

        artifact = {"type": "section"}

        result = await validator.validate_transition(
            artifact=artifact,
            artifact_type="section",
            from_state="draft",
            to_state="review",
            llm=mock_llm,
        )

        # Should still allow transition despite failure
        assert result.can_transition is True
        assert "style" in result.advisory_failures
        assert "style" not in result.blocking_failures


class TestCreateQualityGateValidator:
    """Tests for factory function."""

    def test_create_validator(self):
        """Test factory creates validator."""
        studio = MagicMock()
        validator = create_quality_gate_validator(studio)

        assert isinstance(validator, QualityGateValidator)
        assert validator.studio is studio

    def test_create_validator_no_studio(self):
        """Test factory with no studio."""
        validator = create_quality_gate_validator()

        assert isinstance(validator, QualityGateValidator)
        assert validator.studio is None
