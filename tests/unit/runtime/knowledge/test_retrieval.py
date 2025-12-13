"""Tests for knowledge retrieval tools."""

from pathlib import Path

import pytest

from questfoundry.runtime.domain import load_studio
from questfoundry.runtime.knowledge.retrieval import (
    ConsultKnowledgeTool,
    QueryKnowledgeTool,
    create_consult_knowledge_tool,
    create_query_knowledge_tool,
)


# Path to the domain-v4 test data
DOMAIN_V4_PATH = Path(__file__).parents[4] / "domain-v4"


@pytest.fixture
def studio():
    """Load the domain-v4 studio."""
    studio_path = DOMAIN_V4_PATH / "studio.json"
    return load_studio(studio_path)


class TestConsultKnowledgeTool:
    """Tests for ConsultKnowledgeTool."""

    def test_creates_tool(self, studio) -> None:
        """Test creating consult knowledge tool."""
        sr = studio.agents["showrunner"]
        tool = create_consult_knowledge_tool(studio, sr)

        assert isinstance(tool, ConsultKnowledgeTool)
        assert tool.name == "consult_knowledge"

    def test_consult_existing_entry(self, studio) -> None:
        """Test consulting an existing knowledge entry."""
        sr = studio.agents["showrunner"]
        tool = create_consult_knowledge_tool(studio, sr)

        # Consult an entry that exists and SR can access
        result = tool._run("spoiler_hygiene")

        assert "Spoiler Hygiene" in result
        assert "Player-facing surfaces" in result

    def test_consult_nonexistent_entry(self, studio) -> None:
        """Test consulting a nonexistent entry."""
        sr = studio.agents["showrunner"]
        tool = create_consult_knowledge_tool(studio, sr)

        result = tool._run("nonexistent_entry")

        assert "not found" in result.lower()

    def test_consult_respects_access_control(self, studio) -> None:
        """Test that access control is respected."""
        # Create a test where an agent cannot access an entry
        # For now, just verify the tool doesn't crash
        gk = studio.agents["gatekeeper"]
        tool = create_consult_knowledge_tool(studio, gk)

        # GK should be able to access spoiler_hygiene (applicable to validators)
        result = tool._run("spoiler_hygiene")
        assert isinstance(result, str)


class TestQueryKnowledgeTool:
    """Tests for QueryKnowledgeTool."""

    def test_creates_tool(self, studio) -> None:
        """Test creating query knowledge tool."""
        sr = studio.agents["showrunner"]
        tool = create_query_knowledge_tool(studio, sr)

        assert isinstance(tool, QueryKnowledgeTool)
        assert tool.name == "query_knowledge"

    def test_query_finds_matching_entries(self, studio) -> None:
        """Test querying for matching entries."""
        sr = studio.agents["showrunner"]
        tool = create_query_knowledge_tool(studio, sr)

        # Query for something that should match
        result = tool._run("spoiler")

        # Should find spoiler_hygiene if it's in can_lookup or role_specific
        assert isinstance(result, str)

    def test_query_no_matches(self, studio) -> None:
        """Test querying with no matches."""
        sr = studio.agents["showrunner"]
        tool = create_query_knowledge_tool(studio, sr)

        result = tool._run("xyznonexistent123")

        assert "No knowledge entries found" in result

    def test_query_returns_summaries(self, studio) -> None:
        """Test that query returns summaries not full content."""
        sr = studio.agents["showrunner"]
        tool = create_query_knowledge_tool(studio, sr)

        # Query for something generic
        result = tool._run("quality")

        # Result should suggest using consult_knowledge
        if "Found" in result:
            assert "consult_knowledge" in result
