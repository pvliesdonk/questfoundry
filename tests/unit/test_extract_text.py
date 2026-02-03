"""Tests for providers.content.extract_text."""

from __future__ import annotations

from questfoundry.providers.content import extract_text


class TestExtractText:
    """Test extract_text handles all provider content formats."""

    def test_string_passthrough(self) -> None:
        assert extract_text("hello world") == "hello world"

    def test_empty_string(self) -> None:
        assert extract_text("") == ""

    def test_gemini_single_text_block(self) -> None:
        content = [{"type": "text", "text": "Hello from Gemini"}]
        assert extract_text(content) == "Hello from Gemini"

    def test_gemini_text_block_with_extras(self) -> None:
        content = [
            {
                "type": "text",
                "text": "Hello from Gemini",
                "extras": {"signature": "abc123..."},
            }
        ]
        assert extract_text(content) == "Hello from Gemini"

    def test_gemini_multiple_text_blocks(self) -> None:
        content = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]
        assert extract_text(content) == "First part\nSecond part"

    def test_non_text_blocks_ignored(self) -> None:
        content = [
            {"type": "image", "data": "..."},
            {"type": "text", "text": "Only text"},
        ]
        assert extract_text(content) == "Only text"

    def test_empty_list_falls_back(self) -> None:
        result = extract_text([])
        assert result == "[]"

    def test_list_without_text_blocks_falls_back(self) -> None:
        content = [{"type": "image", "data": "..."}]
        assert extract_text(content) == str(content)

    def test_unexpected_type_falls_back(self) -> None:
        result = extract_text(42)  # type: ignore[arg-type]
        assert result == "42"

    def test_block_with_non_string_text_skipped(self) -> None:
        content = [
            {"type": "text", "text": 123},
            {"type": "text", "text": "valid"},
        ]
        assert extract_text(content) == "valid"
