"""Agent-to-human communication for interactive mode."""

from typing import Any, Callable

# Type alias for human callback functions
HumanCallback = Callable[[str, dict[str, Any]], str]
"""
Callback function signature for agent-to-human questions.

This callback enables roles to ask humans questions during interactive mode.
In batch/guided mode, roles operate autonomously without human interaction.

Args:
    question: The question text to ask the human
    context: Additional context including:
        - question: The original question (repeated for convenience)
        - context: Domain-specific context dict
        - suggestions: List of suggested answers
        - artifacts: Relevant artifacts
        - role: Name of the role asking

Returns:
    Human's response text

Example:
    >>> def my_callback(question: str, context: dict[str, Any]) -> str:
    ...     suggestions = context.get("suggestions", [])
    ...     print(f"Question: {question}")
    ...     if suggestions:
    ...         print(f"Suggestions: {', '.join(suggestions)}")
    ...     return input("Your answer: ")
    ...
    >>> role = InteractiveRole(provider, human_callback=my_callback)
    >>> answer = role.ask_human("What's the protagonist's name?")
"""


def default_human_callback(question: str, context: dict[str, Any]) -> str:
    """
    Default human callback that prompts via stdin.

    This is a simple reference implementation. Production systems should
    provide their own callback that integrates with their UI (CLI, Web, etc.).

    Args:
        question: The question to ask
        context: Additional context

    Returns:
        Human's response from stdin
    """
    role = context.get("role", "Agent")
    suggestions = context.get("suggestions", [])

    print(f"\n[{role}] {question}")

    if suggestions:
        print("\nSuggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
        print()

    return input("Your answer: ")


def batch_mode_callback(question: str, context: dict[str, Any]) -> str:
    """
    Batch mode callback that returns default/empty response.

    Used when no human interaction is desired (automated workflows).

    Args:
        question: The question (ignored)
        context: Additional context

    Returns:
        Empty string or first suggestion if available
    """
    suggestions = context.get("suggestions", [])

    # If there are suggestions, use the first one
    if suggestions:
        return suggestions[0]

    # Otherwise return empty string
    return ""
