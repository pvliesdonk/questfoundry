"""Tests for export internationalization module."""

from __future__ import annotations

from questfoundry.export.i18n import (
    get_language_name,
    get_output_language_instruction,
    get_ui_strings,
)


class TestGetLanguageName:
    def test_known_language(self) -> None:
        assert get_language_name("en") == "English"
        assert get_language_name("nl") == "Dutch"
        assert get_language_name("de") == "German"

    def test_case_insensitive(self) -> None:
        assert get_language_name("EN") == "English"
        assert get_language_name("Nl") == "Dutch"

    def test_unknown_returns_code(self) -> None:
        assert get_language_name("xx") == "xx"
        assert get_language_name("tlh") == "tlh"


class TestGetUiStrings:
    def test_english_strings(self) -> None:
        ui = get_ui_strings("en")
        assert ui["the_end"] == "The End"
        assert ui["codex"] == "Codex"
        assert ui["cover_alt"] == "Cover illustration"
        assert ui["continue"] == "continue"

    def test_dutch_strings(self) -> None:
        ui = get_ui_strings("nl")
        assert ui["the_end"] == "Einde"
        assert ui["codex"] == "Codex"
        assert ui["cover_alt"] == "Omslagillustratie"
        assert ui["continue"] == "verder"

    def test_german_strings(self) -> None:
        ui = get_ui_strings("de")
        assert ui["the_end"] == "Ende"
        assert ui["codex"] == "Kodex"

    def test_french_strings(self) -> None:
        ui = get_ui_strings("fr")
        assert ui["the_end"] == "Fin"

    def test_case_insensitive(self) -> None:
        ui = get_ui_strings("NL")
        assert ui["the_end"] == "Einde"

    def test_unknown_falls_back_to_english(self) -> None:
        ui = get_ui_strings("xx")
        assert ui["the_end"] == "The End"
        assert ui["codex"] == "Codex"


class TestGetOutputLanguageInstruction:
    def test_english_returns_empty(self) -> None:
        assert get_output_language_instruction("en") == ""

    def test_english_case_insensitive(self) -> None:
        assert get_output_language_instruction("EN") == ""

    def test_dutch_returns_instruction(self) -> None:
        result = get_output_language_instruction("nl")
        assert "Dutch" in result
        assert "All IDs" in result  # Keep IDs in English

    def test_german_returns_instruction(self) -> None:
        result = get_output_language_instruction("de")
        assert "German" in result

    def test_unknown_uses_code_as_name(self) -> None:
        result = get_output_language_instruction("xx")
        assert "xx" in result
        assert result != ""
