"""Tests for book binder functionality"""

import tempfile
from pathlib import Path

import pytest

from questfoundry.export import BookBinder, ViewArtifact
from questfoundry.models.artifact import Artifact


@pytest.fixture
def sample_view():
    """Create a sample view with artifacts"""
    artifacts = [
        Artifact(
            type="hook_card",
            data={
                "header": {"short_name": "Test Hook"},
                "description": "A test hook for the adventure",
                "trigger": "scene_start",
            },
            metadata={"id": "HOOK-001", "temperature": "cold", "player_safe": True},
        ),
        Artifact(
            type="canon_pack",
            data={
                "name": "Test Canon",
                "content": "Important canon information",
                "tags": ["lore", "background"],
            },
            metadata={"id": "CANON-001", "temperature": "cold", "player_safe": True},
        ),
        Artifact(
            type="codex_entry",
            data={
                "title": "Ancient Ruin",
                "text": "The ruins date back centuries...",
                "category": "locations",
            },
            metadata={"id": "CODEX-001", "temperature": "cold", "player_safe": True},
        ),
    ]

    return ViewArtifact(
        view_id="VIEW-TEST-001",
        snapshot_id="SNAP-001",
        artifacts=artifacts,
        metadata={"version": "1.0"},
    )


@pytest.fixture
def binder():
    """Create a BookBinder instance"""
    return BookBinder()


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_render_html_basic(binder, sample_view):
    """Test basic HTML rendering"""
    html = binder.render_html(sample_view)

    assert html is not None
    assert len(html) > 0

    # Check for HTML structure
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html
    assert "<body>" in html

    # Check for title
    assert "View: VIEW-TEST-001" in html

    # Check for view metadata
    assert "VIEW-TEST-001" in html
    assert "SNAP-001" in html

    # Check for artifacts
    assert "Test Hook" in html
    assert "Test Canon" in html
    assert "Ancient Ruin" in html


def test_render_html_with_custom_title(binder, sample_view):
    """Test HTML rendering with custom title"""
    html = binder.render_html(sample_view, title="My Custom Title")

    assert "My Custom Title" in html
    assert "<title>My Custom Title</title>" in html


def test_render_markdown_basic(binder, sample_view):
    """Test basic Markdown rendering"""
    markdown = binder.render_markdown(sample_view)

    assert markdown is not None
    assert len(markdown) > 0

    # Check for Markdown structure
    assert "# View: VIEW-TEST-001" in markdown

    # Check for view metadata
    assert "VIEW-TEST-001" in markdown
    assert "SNAP-001" in markdown

    # Check for artifacts
    assert "Test Hook" in markdown
    assert "Test Canon" in markdown
    assert "Ancient Ruin" in markdown

    # Check for artifact sections
    assert "## Test Hook" in markdown or "## hook_card" in markdown


def test_render_markdown_with_custom_title(binder, sample_view):
    """Test Markdown rendering with custom title"""
    markdown = binder.render_markdown(sample_view, title="My Custom Title")

    assert "# My Custom Title" in markdown


def test_render_markdown_without_metadata(binder, sample_view):
    """Test Markdown rendering without metadata"""
    markdown = binder.render_markdown(sample_view, include_metadata=False)

    # Title should still be present
    assert "# View: VIEW-TEST-001" in markdown

    # But metadata section should be absent
    assert "View ID:" not in markdown or "---" not in markdown


def test_save_html(binder, sample_view, temp_output_dir):
    """Test saving HTML to file"""
    html = binder.render_html(sample_view)
    output_path = temp_output_dir / "output.html"

    saved_path = binder.save_html(html, output_path)

    assert saved_path.exists()
    assert saved_path == output_path

    # Verify content
    with open(saved_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert content == html
    assert "Test Hook" in content


def test_save_markdown(binder, sample_view, temp_output_dir):
    """Test saving Markdown to file"""
    markdown = binder.render_markdown(sample_view)
    output_path = temp_output_dir / "output.md"

    saved_path = binder.save_markdown(markdown, output_path)

    assert saved_path.exists()
    assert saved_path == output_path

    # Verify content
    with open(saved_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert content == markdown
    assert "Test Hook" in content


def test_save_creates_directory(binder, sample_view, temp_output_dir):
    """Test saving creates parent directories"""
    html = binder.render_html(sample_view)
    output_path = temp_output_dir / "nested" / "dir" / "output.html"

    assert not output_path.parent.exists()

    saved_path = binder.save_html(html, output_path)

    assert saved_path.exists()
    assert saved_path.parent.exists()


def test_artifact_sorting(sample_view):
    """Test artifact sorting"""
    binder = BookBinder(sort_artifacts=True)
    artifacts = binder._prepare_artifacts(sample_view.artifacts)

    # Should be sorted by type then ID
    types = [a.type for a in artifacts]
    assert types == sorted(types)


def test_no_artifact_sorting(sample_view):
    """Test disabling artifact sorting"""
    # Reverse the artifacts
    reversed_artifacts = list(reversed(sample_view.artifacts))
    sample_view.artifacts = reversed_artifacts

    binder = BookBinder(sort_artifacts=False)
    artifacts = binder._prepare_artifacts(sample_view.artifacts)

    # Should maintain original order
    assert artifacts == reversed_artifacts


def test_extract_artifact_header(binder):
    """Test header extraction from various artifact structures"""
    # Simple name field
    artifact1 = Artifact(
        type="test",
        data={"name": "Test Name"},
        metadata={"id": "TEST-001"},
    )
    assert binder._extract_artifact_header(artifact1) == "Test Name"

    # Title field
    artifact2 = Artifact(
        type="test",
        data={"title": "Test Title"},
        metadata={"id": "TEST-002"},
    )
    assert binder._extract_artifact_header(artifact2) == "Test Title"

    # Nested header structure
    artifact3 = Artifact(
        type="test",
        data={"header": {"short_name": "Short Name"}},
        metadata={"id": "TEST-003"},
    )
    assert binder._extract_artifact_header(artifact3) == "Short Name"

    # No header
    artifact4 = Artifact(
        type="test",
        data={"other_field": "value"},
        metadata={"id": "TEST-004"},
    )
    assert binder._extract_artifact_header(artifact4) is None


def test_html_escape(binder):
    """Test HTML escaping"""
    assert binder._escape_html("<script>") == "&lt;script&gt;"
    assert binder._escape_html("A & B") == "A &amp; B"
    assert binder._escape_html('"quoted"') == "&quot;quoted&quot;"


def test_render_empty_view(binder):
    """Test rendering view with no artifacts"""
    empty_view = ViewArtifact(
        view_id="VIEW-EMPTY",
        snapshot_id="SNAP-001",
        artifacts=[],
        metadata={},
    )

    # HTML should still render
    html = binder.render_html(empty_view)
    assert "VIEW-EMPTY" in html
    assert "Artifacts: 0" in html

    # Markdown should still render
    markdown = binder.render_markdown(empty_view)
    assert "VIEW-EMPTY" in markdown
    assert "Artifacts: 0" in markdown


def test_render_nested_data(binder):
    """Test rendering artifacts with nested data structures"""
    artifact = Artifact(
        type="complex",
        data={
            "name": "Complex Artifact",
            "nested": {
                "level1": "value1",
                "level2": {"deep": "value2"},
            },
            "list": ["item1", "item2", "item3"],
        },
        metadata={"id": "COMPLEX-001"},
    )

    view = ViewArtifact(
        view_id="VIEW-001",
        snapshot_id="SNAP-001",
        artifacts=[artifact],
        metadata={},
    )

    # Should handle nested structures
    html = binder.render_html(view)
    assert "Complex Artifact" in html
    assert "level1" in html
    assert "value1" in html

    markdown = binder.render_markdown(view)
    assert "Complex Artifact" in markdown
    assert "level1" in markdown
    assert "item1" in markdown


def test_custom_html_template(sample_view):
    """Test using a custom HTML template"""
    custom_template = "<html><body><h1>CUSTOM</h1>{content}</body></html>"
    binder = BookBinder(html_template=custom_template)

    html = binder.render_html(sample_view)

    assert "<h1>CUSTOM</h1>" in html
    assert "Test Hook" in html
