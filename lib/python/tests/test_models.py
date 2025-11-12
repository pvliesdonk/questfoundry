from datetime import datetime

from questfoundry.models import Artifact, HookCard, TUBrief


def test_artifact_creation():
    artifact = Artifact(type="test", data={})
    assert artifact.type == "test"
    assert artifact.data == {}
    assert artifact.metadata == {}


def test_artifact_with_data():
    artifact = Artifact(
        type="test",
        data={"key": "value"},
        metadata={"id": "TEST-001"},
    )
    assert artifact.type == "test"
    assert artifact.data["key"] == "value"
    assert artifact.metadata["id"] == "TEST-001"


def test_artifact_id_property():
    artifact = Artifact(type="test", data={})

    # Test getter with no ID
    assert artifact.artifact_id is None

    # Test setter
    artifact.artifact_id = "TEST-001"
    assert artifact.artifact_id == "TEST-001"
    assert artifact.metadata["id"] == "TEST-001"


def test_artifact_timestamps():
    artifact = Artifact(type="test", data={})

    # Test created timestamp
    now = datetime.now()
    artifact.created = now
    assert artifact.created is not None
    assert isinstance(artifact.created, datetime)
    assert artifact.metadata["created"] == now.isoformat()

    # Test modified timestamp
    artifact.modified = now
    assert artifact.modified is not None
    assert isinstance(artifact.modified, datetime)
    assert artifact.metadata["modified"] == now.isoformat()


def test_artifact_timestamp_from_string():
    """Test that timestamps can be parsed from ISO strings"""
    artifact = Artifact(
        type="test",
        data={},
        metadata={
            "created": "2024-01-15T10:30:00",
            "modified": "2024-01-15T11:00:00",
        },
    )

    assert isinstance(artifact.created, datetime)
    assert isinstance(artifact.modified, datetime)
    assert artifact.created.year == 2024
    assert artifact.created.month == 1
    assert artifact.created.day == 15


def test_artifact_author():
    artifact = Artifact(type="test", data={})

    # Test getter with no author
    assert artifact.author is None

    # Test setter
    artifact.author = "test_user"
    assert artifact.author == "test_user"
    assert artifact.metadata["author"] == "test_user"


def test_artifact_to_dict():
    artifact = Artifact(
        type="test",
        data={"key": "value"},
        metadata={"id": "TEST-001", "author": "test_user"},
    )

    result = artifact.to_dict()
    assert result["type"] == "test"
    assert result["data"]["key"] == "value"
    assert result["metadata"]["id"] == "TEST-001"
    assert result["metadata"]["author"] == "test_user"


def test_artifact_from_dict():
    data = {
        "type": "test",
        "data": {"key": "value"},
        "metadata": {"id": "TEST-001"},
    }

    artifact = Artifact.from_dict(data)
    assert artifact.type == "test"
    assert artifact.data["key"] == "value"
    assert artifact.metadata["id"] == "TEST-001"


def test_artifact_from_dict_minimal():
    """Test from_dict with minimal data"""
    data = {"type": "test"}

    artifact = Artifact.from_dict(data)
    assert artifact.type == "test"
    assert artifact.data == {}
    assert artifact.metadata == {}


def test_artifact_roundtrip():
    """Test that artifact survives to_dict/from_dict roundtrip"""
    original = Artifact(
        type="test",
        data={"nested": {"key": "value"}},
        metadata={"id": "TEST-001", "author": "test_user"},
    )

    dict_form = original.to_dict()
    reconstructed = Artifact.from_dict(dict_form)

    assert reconstructed.type == original.type
    assert reconstructed.data == original.data
    assert reconstructed.metadata == original.metadata


def test_hook_card():
    card = HookCard(data={"title": "Test"})
    assert card.type == "hook_card"


def test_tu_brief():
    brief = TUBrief(data={})
    assert brief.type == "tu_brief"


def test_canon_pack():
    from questfoundry.models import CanonPack

    canon = CanonPack(data={"lore": "Ancient history"})
    assert canon.type == "canon_pack"


def test_gatecheck_report():
    from questfoundry.models import GatecheckReport

    report = GatecheckReport(data={"status": "passed"})
    assert report.type == "gatecheck_report"


def test_codex_entry():
    from questfoundry.models import CodexEntry

    entry = CodexEntry(data={"term": "Magic"})
    assert entry.type == "codex_entry"


def test_all_artifact_types_have_correct_type():
    """Test that all artifact type classes have correct type field"""
    from questfoundry.models import (
        ArtManifest,
        ArtPlan,
        AudioPlan,
        CanonPack,
        CodexEntry,
        Cuelist,
        EditNotes,
        FrontMatter,
        GatecheckReport,
        HookCard,
        LanguagePack,
        PNPlaytestNotes,
        ProjectMetadata,
        RegisterMap,
        ResearchMemo,
        Shotlist,
        StyleAddendum,
        StyleManifest,
        TUBrief,
        ViewLog,
    )

    artifacts = {
        "art_manifest": ArtManifest,
        "art_plan": ArtPlan,
        "audio_plan": AudioPlan,
        "canon_pack": CanonPack,
        "codex_entry": CodexEntry,
        "cuelist": Cuelist,
        "edit_notes": EditNotes,
        "front_matter": FrontMatter,
        "gatecheck_report": GatecheckReport,
        "hook_card": HookCard,
        "language_pack": LanguagePack,
        "pn_playtest_notes": PNPlaytestNotes,
        "project_metadata": ProjectMetadata,
        "register_map": RegisterMap,
        "research_memo": ResearchMemo,
        "shotlist": Shotlist,
        "style_addendum": StyleAddendum,
        "style_manifest": StyleManifest,
        "tu_brief": TUBrief,
        "view_log": ViewLog,
    }

    for expected_type, artifact_class in artifacts.items():
        instance = artifact_class(data={})
        assert instance.type == expected_type, (
            f"{artifact_class.__name__} has wrong type"
        )
