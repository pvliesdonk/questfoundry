"""Tests for the MyST directive parser."""

from questfoundry.compiler.parser import (
    DirectiveType,
    parse_myst_string,
)


def test_parse_role_meta() -> None:
    """Test parsing a role-meta directive."""
    content = """
# Showrunner

:::{role-meta}
id: showrunner
abbr: SR
archetype: Product Owner
agency: high
mandate: "Manage by Exception"
:::

The Showrunner is the primary interface.
"""
    result = parse_myst_string(content)

    assert not result.has_errors
    assert len(result.directives) == 1

    directive = result.directives[0]
    assert directive.type == DirectiveType.ROLE_META
    assert directive["id"] == "showrunner"
    assert directive["abbr"] == "SR"
    assert directive["agency"] == "high"


def test_parse_loop_directives() -> None:
    """Test parsing loop-related directives."""
    content = """
# Story Spark

:::{loop-meta}
id: story_spark
name: "Story Spark"
trigger: user_request
entry_point: showrunner
:::

## Workflow

:::{graph-node}
id: showrunner
role: showrunner
timeout: 300
:::

:::{graph-node}
id: plotwright
role: plotwright
:::

:::{graph-edge}
source: showrunner
target: plotwright
condition: "intent.status == 'completed'"
:::
"""
    result = parse_myst_string(content)

    assert not result.has_errors
    assert len(result.directives) == 4

    loop_meta = result.get_directive(DirectiveType.LOOP_META)
    assert loop_meta is not None
    assert loop_meta["id"] == "story_spark"

    nodes = result.get_directives(DirectiveType.GRAPH_NODE)
    assert len(nodes) == 2

    edges = result.get_directives(DirectiveType.GRAPH_EDGE)
    assert len(edges) == 1
    assert edges[0]["source"] == "showrunner"


def test_parse_enum_type() -> None:
    """Test parsing enum type directive."""
    content = """
:::{enum-type}
id: HookType
values:
  - narrative: "Changes to story content"
  - scene: "New or modified scenes"
  - factual: "Canon facts"
:::
"""
    result = parse_myst_string(content)

    assert not result.has_errors
    directive = result.directives[0]
    assert directive.type == DirectiveType.ENUM_TYPE
    assert directive["id"] == "HookType"
    assert len(directive["values"]) == 3


def test_parse_unknown_directive() -> None:
    """Test that unknown directives produce errors."""
    content = """
:::{unknown-directive}
foo: bar
:::
"""
    result = parse_myst_string(content)

    assert result.has_errors
    assert "unknown-directive" in result.errors[0].lower()


def test_parse_prose_sections() -> None:
    """Test extraction of prose sections."""
    content = """
# Main Title

This is the intro.

## Section One

Content for section one.

## Section Two

Content for section two.
"""
    result = parse_myst_string(content)

    assert "Main Title" in result.prose_sections
    assert "Section One" in result.prose_sections
    assert "section one" in result.prose_sections["Section One"].lower()
