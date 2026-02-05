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
        # PDF gamebook-specific strings
        "instructions_title": "How to Play",
        "go_to": "go to",
        "if_you_have": "If you have",
        "codeword_checklist": "Codeword Checklist",
        "requires_codeword": "Requires",
    },
    "nl": {
        "the_end": "Einde",
        "codex": "Codex",
        "cover_alt": "Omslagillustratie",
        "continue": "verder",
        "instructions_title": "Hoe te spelen",
        "go_to": "ga naar",
        "if_you_have": "Als je hebt",
        "codeword_checklist": "Codewoorden Checklist",
        "requires_codeword": "Vereist",
    },
    "de": {
        "the_end": "Ende",
        "codex": "Kodex",
        "cover_alt": "Titelbild",
        "continue": "weiter",
        "instructions_title": "Spielanleitung",
        "go_to": "gehe zu",
        "if_you_have": "Wenn du hast",
        "codeword_checklist": "Codewort-Checkliste",
        "requires_codeword": "Erfordert",
    },
    "fr": {
        "the_end": "Fin",
        "codex": "Codex",
        "cover_alt": "Illustration de couverture",
        "continue": "continuer",
        "instructions_title": "Comment jouer",
        "go_to": "allez au",
        "if_you_have": "Si vous avez",
        "codeword_checklist": "Liste des mots de passe",
        "requires_codeword": "Nécessite",
    },
    "es": {
        "the_end": "Fin",
        "codex": "Códice",
        "cover_alt": "Ilustración de portada",
        "continue": "continuar",
        "instructions_title": "Cómo jugar",
        "go_to": "ve a",
        "if_you_have": "Si tienes",
        "codeword_checklist": "Lista de palabras clave",
        "requires_codeword": "Requiere",
    },
    "it": {
        "the_end": "Fine",
        "codex": "Codex",
        "cover_alt": "Illustrazione di copertina",
        "continue": "continua",
        "instructions_title": "Come giocare",
        "go_to": "vai a",
        "if_you_have": "Se hai",
        "codeword_checklist": "Lista delle parole chiave",
        "requires_codeword": "Richiede",
    },
    "pt": {
        "the_end": "Fim",
        "codex": "Códice",
        "cover_alt": "Ilustração de capa",
        "continue": "continuar",
        "instructions_title": "Como jogar",
        "go_to": "vá para",
        "if_you_have": "Se você tem",
        "codeword_checklist": "Lista de palavras-chave",
        "requires_codeword": "Requer",
    },
    "ja": {
        "the_end": "終わり",
        "codex": "コデックス",
        "cover_alt": "表紙イラスト",
        "continue": "続ける",
        "instructions_title": "遊び方",
        "go_to": "へ進む",
        "if_you_have": "もし持っていれば",
        "codeword_checklist": "コードワードチェックリスト",
        "requires_codeword": "必要",
    },
    "ko": {
        "the_end": "끝",
        "codex": "코덱스",
        "cover_alt": "표지 삽화",
        "continue": "계속",
        "instructions_title": "플레이 방법",
        "go_to": "으로 이동",
        "if_you_have": "가지고 있다면",
        "codeword_checklist": "코드워드 체크리스트",
        "requires_codeword": "필요",
    },
    "zh": {
        "the_end": "终",
        "codex": "法典",
        "cover_alt": "封面插图",
        "continue": "继续",
        "instructions_title": "游戏说明",
        "go_to": "转到",
        "if_you_have": "如果你有",
        "codeword_checklist": "密码词清单",
        "requires_codeword": "需要",
    },
    "ru": {
        "the_end": "Конец",
        "codex": "Кодекс",
        "cover_alt": "Обложка",
        "continue": "продолжить",
        "instructions_title": "Как играть",
        "go_to": "перейти к",
        "if_you_have": "Если у вас есть",  # noqa: RUF001
        "codeword_checklist": "Список кодовых слов",
        "requires_codeword": "Требуется",
    },
    "pl": {
        "the_end": "Koniec",
        "codex": "Kodeks",
        "cover_alt": "Ilustracja okładki",
        "continue": "kontynuuj",
        "instructions_title": "Jak grać",
        "go_to": "idź do",
        "if_you_have": "Jeśli masz",
        "codeword_checklist": "Lista haseł",
        "requires_codeword": "Wymaga",
    },
    "sv": {
        "the_end": "Slut",
        "codex": "Kodex",
        "cover_alt": "Omslagsbild",
        "continue": "fortsätt",
        "instructions_title": "Hur man spelar",
        "go_to": "gå till",
        "if_you_have": "Om du har",
        "codeword_checklist": "Kodordslista",
        "requires_codeword": "Kräver",
    },
    "da": {
        "the_end": "Slut",
        "codex": "Kodeks",
        "cover_alt": "Forsideillustration",
        "continue": "fortsæt",
        "instructions_title": "Sådan spiller du",
        "go_to": "gå til",
        "if_you_have": "Hvis du har",
        "codeword_checklist": "Kodeordsliste",
        "requires_codeword": "Kræver",
    },
    "no": {
        "the_end": "Slutt",
        "codex": "Kodeks",
        "cover_alt": "Forsideillustrasjon",
        "continue": "fortsett",
        "instructions_title": "Hvordan spille",
        "go_to": "gå til",
        "if_you_have": "Hvis du har",
        "codeword_checklist": "Kodeordsliste",
        "requires_codeword": "Krever",
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
    For other supported languages, returns an instruction telling the LLM to
    write player-facing content in the target language.

    Unknown language codes return empty string to prevent prompt injection
    from unsanitized user input.

    Args:
        language: ISO 639-1 language code.

    Returns:
        Language instruction string, or empty string for English/unknown codes.
    """
    lang_lower = language.lower()
    if lang_lower == "en":
        return ""
    # Only allow known languages to prevent prompt injection
    if lang_lower not in LANGUAGE_NAMES:
        return ""
    name = LANGUAGE_NAMES[lang_lower]
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
