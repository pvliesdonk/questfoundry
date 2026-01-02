"""Tests for artifact reading, writing, and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from questfoundry.artifacts import (
    ArtifactNotFoundError,
    ArtifactReader,
    ArtifactValidationError,
    ArtifactValidator,
    ArtifactWriter,
    DreamArtifact,
    Scope,
)

# --- DreamArtifact Model Tests ---


def test_dream_artifact_valid_minimal() -> None:
    """Minimal valid DREAM artifact."""
    artifact = DreamArtifact(
        genre="mystery",
        tone=["dark"],
        audience="adult",
        themes=["betrayal"],
    )
    assert artifact.type == "dream"
    assert artifact.version == 1
    assert artifact.genre == "mystery"
    assert artifact.subgenre is None
    assert artifact.scope is None


def test_dream_artifact_valid_full() -> None:
    """Full DREAM artifact with all fields."""
    artifact = DreamArtifact(
        genre="mystery",
        subgenre="noir",
        tone=["dark", "atmospheric"],
        audience="adult",
        themes=["betrayal", "redemption"],
        style_notes="Hard-boiled narration",
        scope=Scope(
            target_word_count=15000,
            estimated_passages=40,
            branching_depth="moderate",
            estimated_playtime_minutes=30,
        ),
    )
    assert artifact.subgenre == "noir"
    assert artifact.scope is not None
    assert artifact.scope.target_word_count == 15000


def test_dream_artifact_invalid_empty_genre() -> None:
    """Empty genre should fail validation."""
    with pytest.raises(ValidationError) as exc_info:
        DreamArtifact(
            genre="",
            tone=["dark"],
            audience="adult",
            themes=["betrayal"],
        )
    assert "genre" in str(exc_info.value)


def test_dream_artifact_invalid_empty_tone() -> None:
    """Empty tone list should fail validation."""
    with pytest.raises(ValidationError) as exc_info:
        DreamArtifact(
            genre="mystery",
            tone=[],
            audience="adult",
            themes=["betrayal"],
        )
    assert "tone" in str(exc_info.value)


def test_dream_artifact_invalid_audience() -> None:
    """Empty audience should fail validation."""
    with pytest.raises(ValidationError) as exc_info:
        DreamArtifact(
            genre="mystery",
            tone=["dark"],
            audience="",  # Empty string not allowed
            themes=["betrayal"],
        )
    assert "audience" in str(exc_info.value)


def test_dream_artifact_invalid_empty_themes() -> None:
    """Empty themes list should fail validation."""
    with pytest.raises(ValidationError) as exc_info:
        DreamArtifact(
            genre="mystery",
            tone=["dark"],
            audience="adult",
            themes=[],
        )
    assert "themes" in str(exc_info.value)


def test_scope_invalid_word_count() -> None:
    """Word count below minimum should fail."""
    with pytest.raises(ValidationError) as exc_info:
        Scope(
            target_word_count=500,  # Below 1000 minimum
            estimated_passages=10,
        )
    assert "target_word_count" in str(exc_info.value)


def test_scope_invalid_branching_depth() -> None:
    """Empty branching depth should fail."""
    with pytest.raises(ValidationError) as exc_info:
        Scope(
            target_word_count=10000,
            estimated_passages=10,
            branching_depth="",  # Empty string not allowed
        )
    assert "branching_depth" in str(exc_info.value)


def test_dream_artifact_accepts_flexible_audience() -> None:
    """Audience accepts any non-empty string for LLM flexibility."""
    artifact = DreamArtifact(
        genre="mystery",
        tone=["dark"],
        audience="adults",  # Not strictly "adult" but valid
        themes=["betrayal"],
    )
    assert artifact.audience == "adults"


def test_scope_accepts_flexible_branching_depth() -> None:
    """Branching depth accepts any non-empty string for LLM flexibility."""
    scope = Scope(
        target_word_count=10000,
        estimated_passages=10,
        branching_depth="extensive",  # Custom value
    )
    assert scope.branching_depth == "extensive"


# --- ArtifactWriter Tests ---


def test_writer_creates_artifact_file(tmp_path: Path) -> None:
    """Writer creates artifact YAML file."""
    writer = ArtifactWriter(tmp_path)
    artifact = DreamArtifact(
        genre="mystery",
        tone=["dark"],
        audience="adult",
        themes=["betrayal"],
    )

    path = writer.write(artifact, "dream")

    assert path.exists()
    assert path == tmp_path / "artifacts" / "dream.yaml"


def test_writer_creates_artifacts_directory(tmp_path: Path) -> None:
    """Writer creates artifacts directory if missing."""
    writer = ArtifactWriter(tmp_path)
    artifact = DreamArtifact(
        genre="mystery",
        tone=["dark"],
        audience="adult",
        themes=["betrayal"],
    )

    writer.write(artifact, "dream")

    assert (tmp_path / "artifacts").is_dir()


def test_writer_writes_dict(tmp_path: Path) -> None:
    """Writer can write raw dictionaries."""
    writer = ArtifactWriter(tmp_path)
    data = {
        "type": "dream",
        "version": 1,
        "genre": "mystery",
        "tone": ["dark"],
        "audience": "adult",
        "themes": ["betrayal"],
    }

    path = writer.write(data, "dream")

    assert path.exists()


def test_writer_excludes_none_values(tmp_path: Path) -> None:
    """Writer excludes None values from output."""
    writer = ArtifactWriter(tmp_path)
    artifact = DreamArtifact(
        genre="mystery",
        tone=["dark"],
        audience="adult",
        themes=["betrayal"],
        subgenre=None,  # Should not appear in output
    )

    path = writer.write(artifact, "dream")
    content = path.read_text()

    assert "subgenre" not in content


# --- ArtifactReader Tests ---


def test_reader_reads_artifact(tmp_path: Path) -> None:
    """Reader reads artifact from YAML file."""
    # Write first
    writer = ArtifactWriter(tmp_path)
    artifact = DreamArtifact(
        genre="mystery",
        tone=["dark"],
        audience="adult",
        themes=["betrayal"],
    )
    writer.write(artifact, "dream")

    # Read back
    reader = ArtifactReader(tmp_path)
    data = reader.read("dream")

    assert data["genre"] == "mystery"
    assert data["tone"] == ["dark"]


def test_reader_read_validated(tmp_path: Path) -> None:
    """Reader validates against Pydantic model."""
    # Write first
    writer = ArtifactWriter(tmp_path)
    artifact = DreamArtifact(
        genre="mystery",
        subgenre="noir",
        tone=["dark", "atmospheric"],
        audience="adult",
        themes=["betrayal"],
    )
    writer.write(artifact, "dream")

    # Read validated
    reader = ArtifactReader(tmp_path)
    loaded = reader.read_validated("dream", DreamArtifact)

    assert isinstance(loaded, DreamArtifact)
    assert loaded.genre == "mystery"
    assert loaded.subgenre == "noir"


def test_reader_not_found(tmp_path: Path) -> None:
    """Reader raises error for missing artifact."""
    reader = ArtifactReader(tmp_path)

    with pytest.raises(ArtifactNotFoundError) as exc_info:
        reader.read("nonexistent")

    assert exc_info.value.stage_name == "nonexistent"


def test_reader_exists(tmp_path: Path) -> None:
    """Reader checks artifact existence."""
    writer = ArtifactWriter(tmp_path)
    artifact = DreamArtifact(
        genre="mystery",
        tone=["dark"],
        audience="adult",
        themes=["betrayal"],
    )
    writer.write(artifact, "dream")

    reader = ArtifactReader(tmp_path)

    assert reader.exists("dream")
    assert not reader.exists("nonexistent")


# --- ArtifactValidator Tests ---


def test_validator_valid_data() -> None:
    """Validator accepts valid data."""
    validator = ArtifactValidator()
    data = {
        "type": "dream",
        "version": 1,
        "genre": "mystery",
        "tone": ["dark"],
        "audience": "adult",
        "themes": ["betrayal"],
    }

    errors = validator.validate(data, "dream")

    assert errors == []


def test_validator_is_valid() -> None:
    """is_valid returns boolean."""
    validator = ArtifactValidator()
    valid_data = {
        "type": "dream",
        "version": 1,
        "genre": "mystery",
        "tone": ["dark"],
        "audience": "adult",
        "themes": ["betrayal"],
    }
    invalid_data = {
        "type": "dream",
        "version": 1,
        "genre": "mystery",
        "tone": [],  # Invalid: empty
        "audience": "adult",
        "themes": ["betrayal"],
    }

    assert validator.is_valid(valid_data, "dream")
    assert not validator.is_valid(invalid_data, "dream")


def test_validator_invalid_data_pydantic() -> None:
    """Validator catches Pydantic validation errors."""
    validator = ArtifactValidator()
    data = {
        "type": "dream",
        "version": 1,
        "genre": "mystery",
        "tone": [],  # Invalid: must have at least 1
        "audience": "adult",
        "themes": ["betrayal"],
    }

    errors = validator.validate(data, "dream")

    assert len(errors) > 0
    assert any("tone" in e for e in errors)


def test_validator_invalid_data_schema() -> None:
    """Validator catches JSON Schema validation errors."""
    validator = ArtifactValidator()
    data = {
        "type": "dream",
        "version": 1,
        "genre": "mystery",
        "tone": ["dark"],
        "audience": "invalid_audience",  # Invalid enum value
        "themes": ["betrayal"],
    }

    errors = validator.validate(data, "dream")

    assert len(errors) > 0


def test_validator_raise_on_error() -> None:
    """Validator raises exception when requested."""
    validator = ArtifactValidator()
    data = {
        "type": "dream",
        "version": 1,
        "genre": "mystery",
        "tone": [],
        "audience": "adult",
        "themes": ["betrayal"],
    }

    with pytest.raises(ArtifactValidationError) as exc_info:
        validator.validate(data, "dream", raise_on_error=True)

    assert exc_info.value.stage_name == "dream"
    assert len(exc_info.value.errors) > 0


def test_validator_missing_schema_graceful() -> None:
    """Validator handles missing schema gracefully."""
    validator = ArtifactValidator()
    data = {"type": "unknown", "version": 1}

    # Should not raise, just skip JSON Schema validation
    errors = validator.validate(data, "unknown_stage")

    # Only Pydantic errors (no model for unknown_stage)
    assert errors == []


# --- Round-trip Tests ---


def test_roundtrip_preserves_data(tmp_path: Path) -> None:
    """Writing then reading preserves all data."""
    original = DreamArtifact(
        genre="mystery",
        subgenre="noir",
        tone=["dark", "atmospheric", "melancholic"],
        audience="adult",
        themes=["betrayal", "redemption"],
        style_notes="Hard-boiled narration in first person.",
        scope=Scope(
            target_word_count=15000,
            estimated_passages=40,
            branching_depth="moderate",
        ),
    )

    writer = ArtifactWriter(tmp_path)
    writer.write(original, "dream")

    reader = ArtifactReader(tmp_path)
    loaded = reader.read_validated("dream", DreamArtifact)

    assert loaded.genre == original.genre
    assert loaded.subgenre == original.subgenre
    assert loaded.tone == original.tone
    assert loaded.audience == original.audience
    assert loaded.themes == original.themes
    assert loaded.style_notes == original.style_notes
    assert loaded.scope is not None
    assert loaded.scope.target_word_count == original.scope.target_word_count
