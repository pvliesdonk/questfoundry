"""Internationalization for export UI strings.

Maps ISO 639-1 language codes to player-facing UI strings used in
exported HTML and Twee files. Falls back to English for unknown codes.
"""

from __future__ import annotations

# ISO 639-1 code → full language name (for LLM prompt instructions).
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "nl": "Dutch",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
}

# UI strings per language for export chrome (buttons, labels, headings).
UI_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "the_end": "The End",
        "codex": "Codex",
        "cover_alt": "Cover illustration",
        "continue": "continue",
    },
    "nl": {
        "the_end": "Einde",
        "codex": "Codex",
        "cover_alt": "Omslagillustratie",
        "continue": "verder",
    },
    "de": {
        "the_end": "Ende",
        "codex": "Kodex",
        "cover_alt": "Titelbild",
        "continue": "weiter",
    },
    "fr": {
        "the_end": "Fin",
        "codex": "Codex",
        "cover_alt": "Illustration de couverture",
        "continue": "continuer",
    },
    "es": {
        "the_end": "Fin",
        "codex": "Códice",
        "cover_alt": "Ilustración de portada",
        "continue": "continuar",
    },
    "it": {
        "the_end": "Fine",
        "codex": "Codex",
        "cover_alt": "Illustrazione di copertina",
        "continue": "continua",
    },
    "pt": {
        "the_end": "Fim",
        "codex": "Códice",
        "cover_alt": "Ilustração de capa",
        "continue": "continuar",
    },
    "ja": {
        "the_end": "終わり",
        "codex": "コデックス",
        "cover_alt": "表紙イラスト",
        "continue": "続ける",
    },
    "ko": {
        "the_end": "끝",
        "codex": "코덱스",
        "cover_alt": "표지 삽화",
        "continue": "계속",
    },
    "zh": {
        "the_end": "终",
        "codex": "法典",
        "cover_alt": "封面插图",
        "continue": "继续",
    },
    "ru": {
        "the_end": "Конец",
        "codex": "Кодекс",
        "cover_alt": "Обложка",
        "continue": "продолжить",
    },
    "pl": {
        "the_end": "Koniec",
        "codex": "Kodeks",
        "cover_alt": "Ilustracja okładki",
        "continue": "kontynuuj",
    },
    "sv": {
        "the_end": "Slut",
        "codex": "Kodex",
        "cover_alt": "Omslagsbild",
        "continue": "fortsätt",
    },
    "da": {
        "the_end": "Slut",
        "codex": "Kodeks",
        "cover_alt": "Forsideillustration",
        "continue": "fortsæt",
    },
    "no": {
        "the_end": "Slutt",
        "codex": "Kodeks",
        "cover_alt": "Forsideillustrasjon",
        "continue": "fortsett",
    },
}


def get_language_name(code: str) -> str:
    """Get full language name from ISO 639-1 code.

    Args:
        code: ISO 639-1 language code (e.g., "en", "nl").

    Returns:
        Full language name, or the code itself if unknown.
    """
    return LANGUAGE_NAMES.get(code.lower(), code)


def get_ui_strings(language: str) -> dict[str, str]:
    """Get UI strings for a language, falling back to English.

    Args:
        language: ISO 639-1 language code.

    Returns:
        Dictionary of UI string keys to localized values.
    """
    return {**UI_STRINGS["en"], **UI_STRINGS.get(language.lower(), {})}


def get_output_language_instruction(language: str) -> str:
    """Build a prompt instruction for output language.

    Returns an empty string for English (no instruction needed).
    For other languages, returns an instruction telling the LLM to
    write player-facing content in the target language.

    Args:
        language: ISO 639-1 language code.

    Returns:
        Language instruction string, or empty string for English.
    """
    if language.lower() == "en":
        return ""
    name = get_language_name(language)
    return (
        f"## Output Language: {name}\n"
        f"Write ALL narrative content in {name}, including:\n"
        f"- Entity concepts, notes, and descriptions\n"
        f"- Dilemma questions, answers, and explanations\n"
        f"- Path names and descriptions\n"
        f"- Beat summaries\n"
        f"- Consequence descriptions and narrative effects\n"
        f"- Choice labels\n"
        f"- Any prose or dialogue\n"
        f"\n"
        f"Keep these in English:\n"
        f"- All IDs (entity_id, path_id, dilemma_id, beat_id, etc.)\n"
        f"- Field names and JSON/YAML keys\n"
        f"- Enum values (retained, cut, major, minor, advances, reveals, commits, complicates)\n"
        f"- Entity categories (character, location, object, faction)\n"
    )
