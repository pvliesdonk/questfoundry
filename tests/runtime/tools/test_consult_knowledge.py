"""
Tests for consult_knowledge tool.

This tool retrieves full knowledge content from the studio knowledge base,
implementing the menu+consult pattern.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.models.base import KnowledgeEntry
from questfoundry.runtime.models.enums import KnowledgeLayer
from questfoundry.runtime.tools import TOOL_IMPLEMENTATIONS, ToolContext, ToolValidationError
from questfoundry.runtime.tools.consult_knowledge import ConsultKnowledgeTool


@pytest.fixture
def mock_knowledge_entries() -> dict[str, KnowledgeEntry]:
    """Create mock knowledge entries for testing."""
    return {
        "spoiler_hygiene": KnowledgeEntry(
            id="spoiler_hygiene",
            name="Spoiler Hygiene",
            layer=KnowledgeLayer.MUST_KNOW,
            summary="Never reveal future plot points.",
            content={
                "type": "inline",
                "text": "## Spoiler Hygiene\n\n### Core Principle\n\nNever reveal future plot points.\n\n### Examples\n\nWRONG: The detective will discover the butler did it.\nCORRECT: The detective examines the clues carefully.",
            },
            related_entries=["sources_of_truth", "diegetic_gates"],
        ),
        "runtime_guidelines": KnowledgeEntry(
            id="runtime_guidelines",
            name="Runtime Guidelines",
            layer=KnowledgeLayer.MUST_KNOW,
            summary="Guidelines for runtime behavior.",
            content={
                "type": "inline",
                "text": "## Runtime Guidelines\n\n### Tool Usage\n\nAlways use tools for actions.\n\n### Error Handling\n\nHandle errors gracefully.",
            },
        ),
        "empty_entry": KnowledgeEntry(
            id="empty_entry",
            name="Empty Entry",
            layer=KnowledgeLayer.SHOULD_KNOW,
            summary="Entry with no content.",
            content=None,
        ),
        "structured_entry": KnowledgeEntry(
            id="structured_entry",
            name="Structured Entry",
            layer=KnowledgeLayer.ROLE_SPECIFIC,
            summary="Entry with structured data.",
            content={
                "type": "structured",
                "data": {"key": "value", "nested": {"inner": "data"}},
            },
        ),
        "corpus_entry": KnowledgeEntry(
            id="corpus_entry",
            name="Corpus Entry",
            layer=KnowledgeLayer.LOOKUP,
            summary="Entry pointing to corpus.",
            content={
                "type": "corpus",
                "corpus_ref": {"store_ref": "reference_library", "path_pattern": "guides/**"},
            },
        ),
    }


@pytest.fixture
def mock_studio(mock_knowledge_entries: dict[str, KnowledgeEntry]) -> MagicMock:
    """Create mock studio with knowledge entries."""
    studio = MagicMock()
    studio.knowledge = mock_knowledge_entries
    return studio


@pytest.fixture
def tool_context(mock_studio: MagicMock) -> ToolContext:
    """Create tool context with mock studio."""
    context = MagicMock(spec=ToolContext)
    context.studio = mock_studio
    context.agent_id = "test_agent"
    context.broker = None
    return context


@pytest.fixture
def consult_knowledge_tool(tool_context: ToolContext) -> ConsultKnowledgeTool:
    """Create ConsultKnowledgeTool instance for testing."""
    # Get the tool definition from domain
    definition = MagicMock()
    definition.id = "consult_knowledge"
    definition.name = "Consult Knowledge"
    definition.input_schema = MagicMock()
    definition.input_schema.properties = {
        "entry_id": {"type": "string"},
        "section": {"type": "string"},
    }
    definition.input_schema.required = ["entry_id"]

    tool = ConsultKnowledgeTool(definition=definition, context=tool_context)
    return tool


class TestConsultKnowledgeToolRegistration:
    """Tests for tool registration."""

    def test_tool_is_registered(self):
        """consult_knowledge should be registered in TOOL_IMPLEMENTATIONS."""
        assert "consult_knowledge" in TOOL_IMPLEMENTATIONS

    def test_tool_class_is_correct(self):
        """Registered tool should be ConsultKnowledgeTool."""
        assert TOOL_IMPLEMENTATIONS["consult_knowledge"] is ConsultKnowledgeTool


class TestConsultKnowledgeNoEntryId:
    """Tests for calling without entry_id."""

    async def test_no_entry_id_returns_available_entries(
        self, consult_knowledge_tool: ConsultKnowledgeTool
    ):
        """Calling without entry_id should list available entries."""
        result = await consult_knowledge_tool.execute({})

        assert result.success is True
        assert "available_entries" in result.data
        assert "spoiler_hygiene" in result.data["available_entries"]
        assert "runtime_guidelines" in result.data["available_entries"]

    async def test_no_entry_id_includes_hint(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Response should include hint about how to use the tool."""
        result = await consult_knowledge_tool.execute({})

        assert result.success is True
        assert "hint" in result.data
        assert "entry_id" in result.data["hint"].lower()


class TestConsultKnowledgeSuccess:
    """Tests for successful knowledge retrieval."""

    async def test_retrieves_entry_by_id(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Should retrieve full content for valid entry_id."""
        result = await consult_knowledge_tool.execute({"entry_id": "spoiler_hygiene"})

        assert result.success is True
        assert result.data["entry_id"] == "spoiler_hygiene"
        assert result.data["name"] == "Spoiler Hygiene"
        assert result.data["layer"] == "must_know"
        assert "Spoiler Hygiene" in result.data["content"]

    async def test_includes_related_entries(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Should include related_entries in response."""
        result = await consult_knowledge_tool.execute({"entry_id": "spoiler_hygiene"})

        assert result.success is True
        assert "related_entries" in result.data
        assert "sources_of_truth" in result.data["related_entries"]
        assert "diegetic_gates" in result.data["related_entries"]

    async def test_empty_related_entries(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Entry without related entries should return empty list."""
        result = await consult_knowledge_tool.execute({"entry_id": "runtime_guidelines"})

        assert result.success is True
        assert result.data["related_entries"] == []


class TestConsultKnowledgeSectionExtraction:
    """Tests for section extraction from content."""

    async def test_extracts_specific_section(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Should extract content from specified section."""
        result = await consult_knowledge_tool.execute(
            {
                "entry_id": "runtime_guidelines",
                "section": "Tool Usage",
            }
        )

        assert result.success is True
        assert "Tool Usage" in result.data["content"]
        assert "Error Handling" not in result.data["content"]

    async def test_section_not_found_returns_full_content(
        self, consult_knowledge_tool: ConsultKnowledgeTool
    ):
        """Section not found should return full content with warning."""
        result = await consult_knowledge_tool.execute(
            {
                "entry_id": "spoiler_hygiene",
                "section": "Nonexistent Section",
            }
        )

        assert result.success is True
        assert "warning" in result.data
        assert "Nonexistent Section" in result.data["warning"]
        # Should still have full content
        assert "Core Principle" in result.data["content"]


class TestConsultKnowledgeNotFound:
    """Tests for entry not found."""

    async def test_unknown_entry_returns_error(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Unknown entry_id should return error."""
        result = await consult_knowledge_tool.execute({"entry_id": "nonexistent"})

        assert result.success is False
        assert "nonexistent" in result.error
        assert "available_entries" in result.data


class TestConsultKnowledgeContentTypes:
    """Tests for different content types."""

    async def test_inline_content(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Inline content should return text directly."""
        result = await consult_knowledge_tool.execute({"entry_id": "spoiler_hygiene"})

        assert result.success is True
        assert "Core Principle" in result.data["content"]

    async def test_structured_content(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Structured content should return JSON representation."""
        result = await consult_knowledge_tool.execute({"entry_id": "structured_entry"})

        assert result.success is True
        assert "key" in result.data["content"]
        assert "value" in result.data["content"]

    async def test_corpus_content(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Corpus content should indicate to use consult_corpus tool."""
        result = await consult_knowledge_tool.execute({"entry_id": "corpus_entry"})

        assert result.success is True
        assert "corpus" in result.data["content"].lower()
        assert "consult_corpus" in result.data["content"]

    async def test_empty_content(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Entry without content should indicate no content available."""
        result = await consult_knowledge_tool.execute({"entry_id": "empty_entry"})

        assert result.success is True
        assert "No content" in result.data["content"]


class TestConsultKnowledgeValidation:
    """Tests for input validation."""

    async def test_entry_id_must_be_string(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """entry_id must be a string."""
        with pytest.raises(ToolValidationError, match="entry_id.*must be a string"):
            consult_knowledge_tool.validate_input({"entry_id": 123})

    async def test_section_must_be_string(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """section must be a string."""
        with pytest.raises(ToolValidationError, match="section.*must be a string"):
            consult_knowledge_tool.validate_input(
                {
                    "entry_id": "test",
                    "section": ["invalid"],
                }
            )

    async def test_valid_input_passes(self, consult_knowledge_tool: ConsultKnowledgeTool):
        """Valid input should pass validation."""
        # Should not raise
        consult_knowledge_tool.validate_input(
            {
                "entry_id": "spoiler_hygiene",
                "section": "Core Principle",
            }
        )


class TestConsultKnowledgeIntegration:
    """Integration tests with real domain data."""

    @pytest.fixture
    def domain_v4_path(self):
        """Return path to domain-v4 for integration tests."""
        from pathlib import Path

        return Path(__file__).parent.parent.parent.parent / "domain-v4"

    async def test_with_real_domain(self, domain_v4_path):
        """Test with actual domain-v4 knowledge entries."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)
        if not result.success:
            pytest.skip(f"Could not load domain: {result.errors}")

        studio = result.studio

        # Should have knowledge entries
        assert len(studio.knowledge) > 0
        assert "spoiler_hygiene" in studio.knowledge

        # Create tool context with real studio
        context = MagicMock(spec=ToolContext)
        context.studio = studio
        context.agent_id = "test_agent"
        context.broker = None

        definition = MagicMock()
        definition.id = "consult_knowledge"
        definition.name = "Consult Knowledge"
        definition.input_schema = MagicMock()
        definition.input_schema.properties = {}
        definition.input_schema.required = []

        tool = ConsultKnowledgeTool(definition=definition, context=context)

        # Test retrieval
        result = await tool.execute({"entry_id": "spoiler_hygiene"})
        assert result.success is True
        assert "Spoiler" in result.data["content"]
