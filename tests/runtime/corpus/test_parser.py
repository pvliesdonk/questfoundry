"""Tests for corpus file parser."""

from pathlib import Path
from textwrap import dedent

from questfoundry.runtime.corpus.parser import (
    CorpusFrontmatter,
    _extract_sections,
    parse_corpus_file,
)


class TestCorpusFrontmatter:
    """Tests for CorpusFrontmatter."""

    def test_from_dict_complete(self) -> None:
        """Test parsing complete frontmatter."""
        data = {
            "title": "Test Title",
            "summary": "This is a test summary that is long enough.",
            "topics": ["topic-one", "topic-two", "topic-three"],
            "cluster": "prose-and-language",
        }
        fm = CorpusFrontmatter.from_dict(data)

        assert fm.title == "Test Title"
        assert fm.summary == "This is a test summary that is long enough."
        assert fm.topics == ["topic-one", "topic-two", "topic-three"]
        assert fm.cluster == "prose-and-language"

    def test_from_dict_missing_fields(self) -> None:
        """Test parsing with missing fields."""
        data = {"title": "Only Title"}
        fm = CorpusFrontmatter.from_dict(data)

        assert fm.title == "Only Title"
        assert fm.summary == ""
        assert fm.topics == []
        assert fm.cluster == ""

    def test_validate_valid(self) -> None:
        """Test validation of valid frontmatter."""
        fm = CorpusFrontmatter(
            title="Valid Title Here",
            summary="This is a valid summary that meets the minimum length requirement.",
            topics=["topic-one", "topic-two", "topic-three"],
            cluster="genre-conventions",
        )
        errors = fm.validate()
        assert errors == []

    def test_validate_missing_title(self) -> None:
        """Test validation catches missing title."""
        fm = CorpusFrontmatter(
            title="",
            summary="This is a valid summary that meets the minimum length requirement.",
            topics=["topic-one", "topic-two", "topic-three"],
            cluster="genre-conventions",
        )
        errors = fm.validate()
        assert "Missing required field: title" in errors

    def test_validate_short_title(self) -> None:
        """Test validation catches short title."""
        fm = CorpusFrontmatter(
            title="Hi",
            summary="This is a valid summary that meets the minimum length requirement.",
            topics=["topic-one", "topic-two", "topic-three"],
            cluster="genre-conventions",
        )
        errors = fm.validate()
        assert any("Title too short" in e for e in errors)

    def test_validate_short_summary(self) -> None:
        """Test validation catches short summary."""
        fm = CorpusFrontmatter(
            title="Valid Title Here",
            summary="Too short",
            topics=["topic-one", "topic-two", "topic-three"],
            cluster="genre-conventions",
        )
        errors = fm.validate()
        assert any("Summary too short" in e for e in errors)

    def test_validate_long_summary(self) -> None:
        """Test validation catches long summary."""
        fm = CorpusFrontmatter(
            title="Valid Title Here",
            summary="x" * 350,
            topics=["topic-one", "topic-two", "topic-three"],
            cluster="genre-conventions",
        )
        errors = fm.validate()
        assert any("Summary too long" in e for e in errors)

    def test_validate_few_topics(self) -> None:
        """Test validation catches too few topics."""
        fm = CorpusFrontmatter(
            title="Valid Title Here",
            summary="This is a valid summary that meets the minimum length requirement.",
            topics=["topic-one"],
            cluster="genre-conventions",
        )
        errors = fm.validate()
        assert any("Too few topics" in e for e in errors)

    def test_validate_invalid_cluster(self) -> None:
        """Test validation catches invalid cluster."""
        fm = CorpusFrontmatter(
            title="Valid Title Here",
            summary="This is a valid summary that meets the minimum length requirement.",
            topics=["topic-one", "topic-two", "topic-three"],
            cluster="invalid-cluster",
        )
        errors = fm.validate()
        assert any("Invalid cluster" in e for e in errors)


class TestExtractSections:
    """Tests for section extraction."""

    def test_extract_single_section(self) -> None:
        """Test extracting a single section."""
        content = dedent("""
            # Main Heading

            Some content here.
            More content.
        """).strip()

        sections = _extract_sections(content)

        assert len(sections) == 1
        assert sections[0].heading == "Main Heading"
        assert sections[0].level == 1
        assert "Some content here" in sections[0].content

    def test_extract_multiple_sections(self) -> None:
        """Test extracting multiple sections."""
        content = dedent("""
            # First Heading

            First content.

            ## Second Heading

            Second content.

            ### Third Heading

            Third content.
        """).strip()

        sections = _extract_sections(content)

        assert len(sections) == 3
        assert sections[0].heading == "First Heading"
        assert sections[0].level == 1
        assert sections[1].heading == "Second Heading"
        assert sections[1].level == 2
        assert sections[2].heading == "Third Heading"
        assert sections[2].level == 3

    def test_extract_nested_sections(self) -> None:
        """Test that nested structure is captured."""
        content = dedent("""
            # Parent

            Parent content.

            ## Child 1

            Child 1 content.

            ## Child 2

            Child 2 content.
        """).strip()

        sections = _extract_sections(content)

        assert len(sections) == 3
        assert sections[0].level == 1
        assert sections[1].level == 2
        assert sections[2].level == 2

    def test_empty_sections_skipped(self) -> None:
        """Test that empty sections are skipped."""
        content = dedent("""
            # Heading with content

            Some content.

            # Empty heading

            # Another with content

            More content.
        """).strip()

        sections = _extract_sections(content)

        # Should skip the empty section
        assert len(sections) == 2
        assert sections[0].heading == "Heading with content"
        assert sections[1].heading == "Another with content"

    def test_line_numbers_correct(self) -> None:
        """Test that line numbers are recorded correctly."""
        content = "# First\n\nContent.\n\n# Second\n\nMore."

        sections = _extract_sections(content)

        assert sections[0].line_start == 1
        assert sections[1].line_start == 5


class TestParseCorpusFile:
    """Tests for parse_corpus_file function."""

    def test_parse_valid_file(self, tmp_path: Path) -> None:
        """Test parsing a valid corpus file."""
        content = dedent("""
            ---
            title: Test Document Title
            summary: This is a test summary that is long enough to pass validation.
            topics:
              - topic-one
              - topic-two
              - topic-three
            cluster: prose-and-language
            ---

            # Main Section

            This is the main content.

            ## Subsection

            More content here.
        """).strip()

        file_path = tmp_path / "test.md"
        file_path.write_text(content)

        result = parse_corpus_file(file_path)

        assert result is not None
        assert result.path == file_path
        assert result.frontmatter.title == "Test Document Title"
        assert result.frontmatter.cluster == "prose-and-language"
        assert len(result.frontmatter.topics) == 3
        assert len(result.sections) == 2
        assert result.content_hash != ""

    def test_parse_missing_frontmatter(self, tmp_path: Path) -> None:
        """Test parsing file without frontmatter returns None."""
        content = "# Just a heading\n\nNo frontmatter here."

        file_path = tmp_path / "no_frontmatter.md"
        file_path.write_text(content)

        result = parse_corpus_file(file_path)

        assert result is None

    def test_parse_invalid_yaml(self, tmp_path: Path) -> None:
        """Test parsing file with invalid YAML returns None."""
        content = dedent("""
            ---
            title: [invalid yaml
            ---

            # Content
        """).strip()

        file_path = tmp_path / "bad_yaml.md"
        file_path.write_text(content)

        result = parse_corpus_file(file_path)

        assert result is None

    def test_parse_nonexistent_file(self, tmp_path: Path) -> None:
        """Test parsing nonexistent file returns None."""
        result = parse_corpus_file(tmp_path / "nonexistent.md")
        assert result is None

    def test_content_hash_changes(self, tmp_path: Path) -> None:
        """Test that content hash changes with file content."""
        base_content = dedent("""\
            ---
            title: Hash Test Document
            summary: This is a test for content hashing functionality.
            topics:
              - hash
              - test
              - content
            cluster: prose-and-language
            ---

            # Content

            {}
        """)

        file_path = tmp_path / "hash_test.md"

        # First version
        file_path.write_text(base_content.format("Version 1"))
        result1 = parse_corpus_file(file_path)

        # Second version
        file_path.write_text(base_content.format("Version 2"))
        result2 = parse_corpus_file(file_path)

        assert result1 is not None
        assert result2 is not None
        assert result1.content_hash != result2.content_hash
