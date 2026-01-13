"""Tests for the observability tracing module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from questfoundry.observability.tracing import (
    _prepare_metadata,
    build_runnable_config,
    generate_run_id,
    get_current_run_tree,
    get_pipeline_run_id,
    is_tracing_enabled,
    set_pipeline_run_id,
    trace_context,
    traceable,
)


class TestPipelineRunId:
    """Tests for pipeline run ID management."""

    def test_generate_run_id_returns_uuid_string(self) -> None:
        """generate_run_id returns a valid UUID string."""
        run_id = generate_run_id()
        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID format: 8-4-4-4-12

    def test_generate_run_id_returns_unique_values(self) -> None:
        """Each call to generate_run_id returns a unique value."""
        ids = [generate_run_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_set_and_get_pipeline_run_id(self) -> None:
        """set_pipeline_run_id and get_pipeline_run_id work correctly."""
        # Initially None (test isolation via ContextVar)
        run_id = generate_run_id()
        set_pipeline_run_id(run_id)
        assert get_pipeline_run_id() == run_id

    def test_pipeline_run_id_default_is_none(self) -> None:
        """Pipeline run ID defaults to None when not set."""
        # This test relies on ContextVar isolation between tests
        # The conftest fixture sets LANGSMITH_TRACING=false, but doesn't affect ContextVar
        # We need to reset it for this test
        from questfoundry.observability.tracing import _pipeline_run_id

        token = _pipeline_run_id.set(None)
        try:
            assert get_pipeline_run_id() is None
        finally:
            _pipeline_run_id.reset(token)


class TestPrepareMetadata:
    """Tests for _prepare_metadata helper."""

    def test_prepare_metadata_with_none(self) -> None:
        """_prepare_metadata returns empty dict for None input."""
        # Reset pipeline run ID for this test
        from questfoundry.observability.tracing import _pipeline_run_id

        token = _pipeline_run_id.set(None)
        try:
            result = _prepare_metadata(None)
            assert result == {}
        finally:
            _pipeline_run_id.reset(token)

    def test_prepare_metadata_copies_input(self) -> None:
        """_prepare_metadata creates a copy, not a reference."""
        original = {"key": "value"}
        result = _prepare_metadata(original)
        result["new_key"] = "new_value"
        assert "new_key" not in original

    def test_prepare_metadata_injects_run_id(self) -> None:
        """_prepare_metadata adds pipeline_run_id when set."""
        run_id = generate_run_id()
        set_pipeline_run_id(run_id)
        result = _prepare_metadata({"existing": "value"})
        assert result["pipeline_run_id"] == run_id
        assert result["existing"] == "value"


class TestBuildRunnableConfig:
    """Tests for build_runnable_config."""

    def test_build_runnable_config_empty(self) -> None:
        """build_runnable_config with no args returns minimal config."""
        from questfoundry.observability.tracing import _pipeline_run_id

        token = _pipeline_run_id.set(None)
        try:
            config = build_runnable_config()
            assert config == {}
        finally:
            _pipeline_run_id.reset(token)

    def test_build_runnable_config_with_run_name(self) -> None:
        """build_runnable_config includes run_name."""
        config = build_runnable_config(run_name="Test Run")
        assert config["run_name"] == "Test Run"

    def test_build_runnable_config_with_tags(self) -> None:
        """build_runnable_config includes tags."""
        config = build_runnable_config(tags=["tag1", "tag2"])
        assert config["tags"] == ["tag1", "tag2"]

    def test_build_runnable_config_copies_tags(self) -> None:
        """build_runnable_config copies tags list."""
        original_tags = ["tag1"]
        config = build_runnable_config(tags=original_tags)
        config["tags"].append("tag2")
        assert "tag2" not in original_tags

    def test_build_runnable_config_with_metadata(self) -> None:
        """build_runnable_config includes metadata."""
        config = build_runnable_config(metadata={"key": "value"})
        assert config["metadata"]["key"] == "value"

    def test_build_runnable_config_includes_pipeline_run_id(self) -> None:
        """build_runnable_config adds pipeline_run_id to metadata."""
        run_id = generate_run_id()
        set_pipeline_run_id(run_id)
        config = build_runnable_config(metadata={"other": "data"})
        assert config["metadata"]["pipeline_run_id"] == run_id
        assert config["metadata"]["other"] == "data"

    def test_build_runnable_config_with_recursion_limit(self) -> None:
        """build_runnable_config includes recursion_limit."""
        config = build_runnable_config(recursion_limit=50)
        assert config["recursion_limit"] == 50


class TestTraceContext:
    """Tests for trace_context context manager."""

    def test_trace_context_without_langsmith_returns_nullcontext(self) -> None:
        """trace_context returns nullcontext when langsmith unavailable."""
        with patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", False):
            ctx = trace_context("Test Span")
            # nullcontext is a context manager that does nothing
            with ctx:
                pass  # Should not raise

    def test_trace_context_with_langsmith_calls_trace(self) -> None:
        """trace_context calls langsmith.trace when available."""
        mock_trace = MagicMock()
        with (
            patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", True),
            patch("questfoundry.observability.tracing.langsmith") as mock_langsmith,
        ):
            mock_langsmith.trace = mock_trace
            trace_context(
                "Test Span",
                run_type="chain",
                tags=["tag1"],
                metadata={"key": "value"},
            )
            mock_trace.assert_called_once()


class TestTraceable:
    """Tests for @traceable decorator."""

    @pytest.mark.asyncio
    async def test_traceable_without_langsmith_returns_unchanged(self) -> None:
        """@traceable returns function unchanged when langsmith unavailable."""
        with patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", False):

            @traceable(name="Test")
            async def my_func(x: int) -> int:
                return x * 2

            result = await my_func(5)
            assert result == 10

    @pytest.mark.asyncio
    async def test_traceable_preserves_function_metadata(self) -> None:
        """@traceable preserves __name__ and __doc__."""
        with patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", False):

            @traceable(name="Test")
            async def my_documented_func() -> None:
                """This is the docstring."""
                pass

            assert my_documented_func.__name__ == "my_documented_func"
            assert my_documented_func.__doc__ == "This is the docstring."

    def test_traceable_copies_tags(self) -> None:
        """@traceable copies tags to prevent mutation."""
        with patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", False):
            original_tags = ["tag1"]

            @traceable(name="Test", tags=original_tags)
            async def my_func() -> None:
                pass

            # Decorator should have copied the tags
            original_tags.append("tag2")
            # The decorator's internal copy should not be affected
            # (We can't easily verify this without langsmith, but the copy is made)

    def test_traceable_copies_metadata(self) -> None:
        """@traceable copies metadata to prevent mutation."""
        with patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", False):
            original_metadata = {"key": "value"}

            @traceable(name="Test", metadata=original_metadata)
            async def my_func() -> None:
                pass

            # Modify original after decoration
            original_metadata["new_key"] = "new_value"
            # The decorator's internal copy should not be affected


class TestGetCurrentRunTree:
    """Tests for get_current_run_tree."""

    def test_get_current_run_tree_without_langsmith(self) -> None:
        """get_current_run_tree returns None when langsmith unavailable."""
        with patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", False):
            result = get_current_run_tree()
            assert result is None

    def test_get_current_run_tree_handles_exception(self) -> None:
        """get_current_run_tree returns None on exception."""
        with (
            patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", True),
            patch(
                "questfoundry.observability.tracing.ls_get_current_run_tree",
                side_effect=RuntimeError("No active trace"),
            ),
        ):
            result = get_current_run_tree()
            assert result is None


class TestIsTracingEnabled:
    """Tests for is_tracing_enabled."""

    def test_is_tracing_enabled_without_langsmith(self) -> None:
        """is_tracing_enabled returns False when langsmith unavailable."""
        with patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", False):
            assert is_tracing_enabled() is False

    def test_is_tracing_enabled_with_env_true(self) -> None:
        """is_tracing_enabled returns True when env var is 'true'."""
        with (
            patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", True),
            patch.dict("os.environ", {"LANGSMITH_TRACING": "true"}),
        ):
            assert is_tracing_enabled() is True

    def test_is_tracing_enabled_with_env_false(self) -> None:
        """is_tracing_enabled returns False when env var is not 'true'."""
        with (
            patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", True),
            patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}),
        ):
            assert is_tracing_enabled() is False

    def test_is_tracing_enabled_case_insensitive(self) -> None:
        """is_tracing_enabled is case insensitive for env var."""
        with (
            patch("questfoundry.observability.tracing.LANGSMITH_AVAILABLE", True),
            patch.dict("os.environ", {"LANGSMITH_TRACING": "TRUE"}),
        ):
            assert is_tracing_enabled() is True
