"""Tests for media workspace utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from questfoundry.models.artifact import Artifact
from questfoundry.utils.media import MediaWorkspace


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_media_workspace_initialization(temp_workspace):
    """Test MediaWorkspace initialization creates directories."""
    workspace = MediaWorkspace(temp_workspace)

    assert workspace.workspace_path == temp_workspace
    assert workspace.images_dir == temp_workspace / "renders"
    assert workspace.audio_dir == temp_workspace / "audio"
    assert workspace.images_dir.exists()
    assert workspace.audio_dir.exists()


def test_save_image(temp_workspace):
    """Test saving image data to workspace."""
    workspace = MediaWorkspace(temp_workspace)
    image_data = b"fake_image_data"
    artifact_id = "IMG-001"

    image_path = workspace.save_image(image_data, artifact_id)

    assert image_path.exists()
    assert image_path.name == "IMG-001.png"
    assert image_path.read_bytes() == image_data


def test_save_image_with_custom_format(temp_workspace):
    """Test saving image with custom format."""
    workspace = MediaWorkspace(temp_workspace)
    image_data = b"fake_image_data"
    artifact_id = "IMG-002"

    image_path = workspace.save_image(image_data, artifact_id, format="jpg")

    assert image_path.exists()
    assert image_path.name == "IMG-002.jpg"
    assert image_path.read_bytes() == image_data


def test_save_image_with_metadata(temp_workspace):
    """Test saving image with metadata."""
    workspace = MediaWorkspace(temp_workspace)
    image_data = b"fake_image_data"
    artifact_id = "IMG-003"
    metadata = {"prompt": "test prompt", "model": "test-model"}

    image_path = workspace.save_image(image_data, artifact_id, metadata=metadata)

    # Check image was saved
    assert image_path.exists()

    # Check metadata was saved
    metadata_path = workspace.images_dir / f"{artifact_id}.metadata.json"
    assert metadata_path.exists()
    saved_metadata = json.loads(metadata_path.read_text())
    assert saved_metadata == metadata


def test_save_audio(temp_workspace):
    """Test saving audio data to workspace."""
    workspace = MediaWorkspace(temp_workspace)
    audio_data = b"fake_audio_data"
    artifact_id = "AUDIO-001"

    audio_path = workspace.save_audio(audio_data, artifact_id)

    assert audio_path.exists()
    assert audio_path.name == "AUDIO-001.mp3"
    assert audio_path.read_bytes() == audio_data


def test_save_audio_with_custom_format(temp_workspace):
    """Test saving audio with custom format."""
    workspace = MediaWorkspace(temp_workspace)
    audio_data = b"fake_audio_data"
    artifact_id = "AUDIO-002"

    audio_path = workspace.save_audio(audio_data, artifact_id, format="wav")

    assert audio_path.exists()
    assert audio_path.name == "AUDIO-002.wav"
    assert audio_path.read_bytes() == audio_data


def test_save_audio_with_metadata(temp_workspace):
    """Test saving audio with metadata."""
    workspace = MediaWorkspace(temp_workspace)
    audio_data = b"fake_audio_data"
    artifact_id = "AUDIO-003"
    metadata = {"voice": "test-voice", "model": "test-model"}

    audio_path = workspace.save_audio(audio_data, artifact_id, metadata=metadata)

    # Check audio was saved
    assert audio_path.exists()

    # Check metadata was saved
    metadata_path = workspace.audio_dir / f"{artifact_id}.metadata.json"
    assert metadata_path.exists()
    saved_metadata = json.loads(metadata_path.read_text())
    assert saved_metadata == metadata


def test_get_image_path(temp_workspace):
    """Test getting image path."""
    workspace = MediaWorkspace(temp_workspace)
    artifact_id = "IMG-004"

    path = workspace.get_image_path(artifact_id)

    assert path == workspace.images_dir / "IMG-004.png"


def test_get_image_path_custom_format(temp_workspace):
    """Test getting image path with custom format."""
    workspace = MediaWorkspace(temp_workspace)
    artifact_id = "IMG-005"

    path = workspace.get_image_path(artifact_id, format="jpg")

    assert path == workspace.images_dir / "IMG-005.jpg"


def test_get_audio_path(temp_workspace):
    """Test getting audio path."""
    workspace = MediaWorkspace(temp_workspace)
    artifact_id = "AUDIO-004"

    path = workspace.get_audio_path(artifact_id)

    assert path == workspace.audio_dir / "AUDIO-004.mp3"


def test_get_audio_path_custom_format(temp_workspace):
    """Test getting audio path with custom format."""
    workspace = MediaWorkspace(temp_workspace)
    artifact_id = "AUDIO-005"

    path = workspace.get_audio_path(artifact_id, format="wav")

    assert path == workspace.audio_dir / "AUDIO-005.wav"


def test_create_artifact_for_image(temp_workspace):
    """Test creating artifact record for image."""
    workspace = MediaWorkspace(temp_workspace)
    artifact_id = "IMG-006"
    image_path = workspace.images_dir / f"{artifact_id}.png"
    image_path.write_bytes(b"fake_data")

    artifact = workspace.create_artifact_for_image(artifact_id, image_path)

    assert isinstance(artifact, Artifact)
    assert artifact.type == "render"
    assert "file_path" in artifact.data
    assert "absolute_path" in artifact.data
    assert artifact.metadata.get("id") == artifact_id


def test_create_artifact_for_image_with_metadata(temp_workspace):
    """Test creating artifact with custom metadata."""
    workspace = MediaWorkspace(temp_workspace)
    artifact_id = "IMG-007"
    image_path = workspace.images_dir / f"{artifact_id}.png"
    image_path.write_bytes(b"fake_data")
    custom_metadata = {"prompt": "test", "width": 1024}

    artifact = workspace.create_artifact_for_image(
        artifact_id,
        image_path,
        artifact_type="illustration",
        metadata=custom_metadata,
    )

    assert artifact.type == "illustration"
    assert artifact.data["prompt"] == "test"
    assert artifact.data["width"] == 1024


def test_create_artifact_for_audio(temp_workspace):
    """Test creating artifact record for audio."""
    workspace = MediaWorkspace(temp_workspace)
    artifact_id = "AUDIO-006"
    audio_path = workspace.audio_dir / f"{artifact_id}.mp3"
    audio_path.write_bytes(b"fake_data")

    artifact = workspace.create_artifact_for_audio(artifact_id, audio_path)

    assert isinstance(artifact, Artifact)
    assert artifact.type == "audio_asset"
    assert "file_path" in artifact.data
    assert "absolute_path" in artifact.data
    assert artifact.metadata.get("id") == artifact_id


def test_create_artifact_for_audio_with_metadata(temp_workspace):
    """Test creating audio artifact with custom metadata."""
    workspace = MediaWorkspace(temp_workspace)
    artifact_id = "AUDIO-007"
    audio_path = workspace.audio_dir / f"{artifact_id}.mp3"
    audio_path.write_bytes(b"fake_data")
    custom_metadata = {"voice": "narrator", "duration": 120}

    artifact = workspace.create_artifact_for_audio(
        artifact_id,
        audio_path,
        artifact_type="narration",
        metadata=custom_metadata,
    )

    assert artifact.type == "narration"
    assert artifact.data["voice"] == "narrator"
    assert artifact.data["duration"] == 120


def test_generate_artifact_id(temp_workspace):
    """Test generating deterministic artifact ID."""
    workspace = MediaWorkspace(temp_workspace)
    content = "test content for hashing"

    id1 = workspace.generate_artifact_id(content)
    id2 = workspace.generate_artifact_id(content)

    # Should be deterministic
    assert id1 == id2
    # Should be 12 characters (truncated SHA256)
    assert len(id1) == 12


def test_generate_artifact_id_with_prefix(temp_workspace):
    """Test generating artifact ID with prefix."""
    workspace = MediaWorkspace(temp_workspace)
    content = "test content"
    prefix = "IMG"

    artifact_id = workspace.generate_artifact_id(content, prefix=prefix)

    assert artifact_id.startswith("IMG_")
    # Total length: prefix (3) + underscore (1) + hash (12)
    assert len(artifact_id) == 16


def test_generate_artifact_id_different_content(temp_workspace):
    """Test that different content produces different IDs."""
    workspace = MediaWorkspace(temp_workspace)

    id1 = workspace.generate_artifact_id("content 1")
    id2 = workspace.generate_artifact_id("content 2")

    assert id1 != id2
