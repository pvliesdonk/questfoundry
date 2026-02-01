"""Tests for LLM JSONL logger."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from questfoundry.observability import LLMLogger

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    logs = tmp_path / "logs"
    logs.mkdir()
    return tmp_path


def test_llm_logger_creates_log_file(temp_project: Path) -> None:
    """Logger creates JSONL file in logs directory."""
    logger = LLMLogger(temp_project)

    assert logger.log_path == temp_project / "logs" / "llm_calls.jsonl"
    assert logger.log_path.parent.exists()


def test_llm_logger_creates_logs_dir(tmp_path: Path) -> None:
    """Logger creates logs directory if missing."""
    LLMLogger(tmp_path)

    assert (tmp_path / "logs").exists()


def test_llm_logger_disabled_does_not_create_dir(tmp_path: Path) -> None:
    """Disabled logger does not create logs directory."""
    logger = LLMLogger(tmp_path, enabled=False)

    assert not (tmp_path / "logs").exists()
    assert logger.enabled is False


def test_llm_logger_disabled_does_not_write(tmp_path: Path) -> None:
    """Disabled logger does not write to file."""
    logger = LLMLogger(tmp_path, enabled=False)
    entry = LLMLogger.create_entry(
        stage="dream",
        model="test-model",
        messages=[],
        content="",
        tokens_used=0,
        finish_reason="stop",
        duration_seconds=0.0,
    )

    logger.log(entry)

    assert not logger.log_path.exists()


def test_create_entry_with_timestamp() -> None:
    """create_entry generates timestamp automatically."""
    entry = LLMLogger.create_entry(
        stage="dream",
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        content="response",
        tokens_used=100,
        finish_reason="stop",
        duration_seconds=1.5,
    )

    assert entry.timestamp is not None
    assert "T" in entry.timestamp  # ISO format


def test_create_entry_with_defaults() -> None:
    """create_entry uses sensible defaults."""
    entry = LLMLogger.create_entry(
        stage="dream",
        model="test-model",
        messages=[],
        content="",
        tokens_used=0,
        finish_reason="stop",
        duration_seconds=0.0,
    )

    assert entry.temperature == 0.7
    assert entry.max_tokens == 4096
    assert entry.input_tokens == 0
    assert entry.output_tokens == 0
    assert entry.phase == ""
    assert entry.error is None
    assert entry.metadata == {}


def test_create_entry_with_granular_tokens() -> None:
    """create_entry accepts input_tokens, output_tokens, and phase."""
    entry = LLMLogger.create_entry(
        stage="seed",
        model="qwen3:4b-instruct-32k",
        messages=[],
        content="",
        tokens_used=150,
        finish_reason="stop",
        duration_seconds=1.0,
        input_tokens=100,
        output_tokens=50,
        phase="serialize",
    )

    assert entry.input_tokens == 100
    assert entry.output_tokens == 50
    assert entry.phase == "serialize"
    assert entry.tokens_used == 150


def test_create_entry_with_metadata() -> None:
    """create_entry accepts arbitrary metadata."""
    entry = LLMLogger.create_entry(
        stage="dream",
        model="test-model",
        messages=[],
        content="",
        tokens_used=0,
        finish_reason="stop",
        duration_seconds=0.0,
        custom_field="value",
        another_field=42,
    )

    assert entry.metadata == {"custom_field": "value", "another_field": 42}


def test_log_writes_jsonl(temp_project: Path) -> None:
    """log() writes JSON line to file."""
    logger = LLMLogger(temp_project)
    entry = LLMLogger.create_entry(
        stage="dream",
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        content="response",
        tokens_used=100,
        finish_reason="stop",
        duration_seconds=1.5,
    )

    logger.log(entry)

    assert logger.log_path.exists()
    with logger.log_path.open() as f:
        line = f.readline()
        data = json.loads(line)

    assert data["stage"] == "dream"
    assert data["model"] == "test-model"
    assert data["tokens_used"] == 100


def test_log_appends_multiple_entries(temp_project: Path) -> None:
    """Multiple log() calls append to same file."""
    logger = LLMLogger(temp_project)

    for i in range(3):
        entry = LLMLogger.create_entry(
            stage=f"stage{i}",
            model="test-model",
            messages=[],
            content="",
            tokens_used=i * 100,
            finish_reason="stop",
            duration_seconds=0.0,
        )
        logger.log(entry)

    with logger.log_path.open() as f:
        lines = f.readlines()

    assert len(lines) == 3

    # Verify each line is valid JSON
    for i, line in enumerate(lines):
        data = json.loads(line)
        assert data["stage"] == f"stage{i}"
        assert data["tokens_used"] == i * 100


def test_read_entries_empty(temp_project: Path) -> None:
    """read_entries returns empty list for non-existent file."""
    logger = LLMLogger(temp_project)

    entries = logger.read_entries()

    assert entries == []


def test_read_entries_roundtrip(temp_project: Path) -> None:
    """Entries can be read back after logging."""
    logger = LLMLogger(temp_project)
    original_entry = LLMLogger.create_entry(
        stage="dream",
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        content="response",
        tokens_used=100,
        finish_reason="stop",
        duration_seconds=1.5,
        temperature=0.5,
    )

    logger.log(original_entry)
    entries = logger.read_entries()

    assert len(entries) == 1
    assert entries[0].stage == "dream"
    assert entries[0].model == "test-model"
    assert entries[0].tokens_used == 100
    assert entries[0].temperature == 0.5


def test_log_preserves_full_content(temp_project: Path) -> None:
    """Log entries preserve full content without truncation."""
    logger = LLMLogger(temp_project)
    long_content = "x" * 10000  # Long content

    entry = LLMLogger.create_entry(
        stage="dream",
        model="test-model",
        messages=[{"role": "user", "content": long_content}],
        content=long_content,
        tokens_used=100,
        finish_reason="stop",
        duration_seconds=1.5,
    )
    logger.log(entry)

    entries = logger.read_entries()
    assert len(entries[0].content) == 10000
    assert entries[0].messages[0]["content"] == long_content


def test_read_entries_roundtrip_with_granular_tokens(temp_project: Path) -> None:
    """Granular token fields survive write/read roundtrip."""
    logger = LLMLogger(temp_project)
    entry = LLMLogger.create_entry(
        stage="seed",
        model="qwen3:4b-instruct-32k",
        messages=[{"role": "user", "content": "test"}],
        content="response",
        tokens_used=150,
        finish_reason="stop",
        duration_seconds=1.0,
        input_tokens=100,
        output_tokens=50,
        phase="serialize",
    )

    logger.log(entry)
    entries = logger.read_entries()

    assert len(entries) == 1
    assert entries[0].input_tokens == 100
    assert entries[0].output_tokens == 50
    assert entries[0].phase == "serialize"


def test_read_entries_backward_compatible(temp_project: Path) -> None:
    """Old JSONL entries without new fields can still be read."""
    import json

    logger = LLMLogger(temp_project)
    # Write an entry missing the new fields (simulates old format)
    old_entry = {
        "timestamp": "2026-01-30T00:00:00+00:00",
        "stage": "dream",
        "model": "test",
        "messages": [],
        "temperature": 0.7,
        "max_tokens": 4096,
        "content": "",
        "tokens_used": 0,
        "finish_reason": "stop",
        "duration_seconds": 0.0,
    }
    with logger.log_path.open("w") as f:
        f.write(json.dumps(old_entry) + "\n")

    entries = logger.read_entries()
    assert len(entries) == 1
    assert entries[0].input_tokens == 0
    assert entries[0].output_tokens == 0
    assert entries[0].phase == ""


def test_log_entry_with_error(temp_project: Path) -> None:
    """Log entries can include error messages."""
    logger = LLMLogger(temp_project)
    entry = LLMLogger.create_entry(
        stage="dream",
        model="test-model",
        messages=[],
        content="",
        tokens_used=0,
        finish_reason="error",
        duration_seconds=0.5,
        error="Connection timeout",
    )

    logger.log(entry)
    entries = logger.read_entries()

    assert entries[0].error == "Connection timeout"
