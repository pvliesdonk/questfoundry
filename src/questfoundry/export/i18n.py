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
    return UI_STRINGS.get(language.lower(), UI_STRINGS["en"])


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
        f"Write all narrative content in {name}. "
        f"Keep structural IDs, field names, and enums in English."
    )
