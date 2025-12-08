"""Tests for HotStore - ephemeral in-memory workspace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from questfoundry.runtime.stores import HotStore


class TestHotStoreBasics:
    """Basic artifact operations."""

    def test_create_empty(self) -> None:
        """Create empty hot store."""
        hot = HotStore()
        assert len(hot) == 0
        assert hot.keys() == []

    def test_put_and_get(self) -> None:
        """Store and retrieve artifact."""
        hot = HotStore()
        hot.put("scene_001", {"title": "Opening", "content": "..."})

        artifact = hot.get("scene_001")
        assert artifact is not None
        assert artifact["title"] == "Opening"

    def test_get_missing_returns_none(self) -> None:
        """Get missing key returns None."""
        hot = HotStore()
        assert hot.get("missing") is None

    def test_delete(self) -> None:
        """Delete artifact."""
        hot = HotStore()
        hot.put("temp", "value")
        assert hot.has("temp")

        assert hot.delete("temp")
        assert not hot.has("temp")

    def test_delete_missing_returns_false(self) -> None:
        """Delete missing key returns False."""
        hot = HotStore()
        assert not hot.delete("missing")

    def test_clear(self) -> None:
        """Clear all artifacts."""
        hot = HotStore()
        hot.put("a", 1)
        hot.put("b", 2)
        assert len(hot) == 2

        hot.clear()
        assert len(hot) == 0


class TestDictLikeAccess:
    """Dict-like interface for backwards compatibility."""

    def test_getitem(self) -> None:
        """Dict-like access."""
        hot = HotStore()
        hot.put("key", "value")
        assert hot["key"] == "value"

    def test_setitem(self) -> None:
        """Dict-like assignment."""
        hot = HotStore()
        hot["key"] = "value"
        assert hot.get("key") == "value"

    def test_delitem(self) -> None:
        """Dict-like deletion."""
        hot = HotStore()
        hot["key"] = "value"
        del hot["key"]
        assert "key" not in hot

    def test_contains(self) -> None:
        """Dict-like membership."""
        hot = HotStore()
        hot["key"] = "value"
        assert "key" in hot
        assert "missing" not in hot

    def test_len(self) -> None:
        """Dict-like length."""
        hot = HotStore()
        assert len(hot) == 0
        hot["a"] = 1
        hot["b"] = 2
        assert len(hot) == 2

    def test_iter(self) -> None:
        """Dict-like iteration."""
        hot = HotStore()
        hot["a"] = 1
        hot["b"] = 2
        keys = list(hot)
        assert set(keys) == {"a", "b"}

    def test_items(self) -> None:
        """Dict-like items()."""
        hot = HotStore()
        hot["a"] = 1
        hot["b"] = 2
        items = dict(hot.items())
        assert items == {"a": 1, "b": 2}

    def test_setdefault(self) -> None:
        """Dict-like setdefault()."""
        hot = HotStore()
        result = hot.setdefault("key", "default")
        assert result == "default"
        assert hot["key"] == "default"

        # Existing key not overwritten
        result = hot.setdefault("key", "other")
        assert result == "default"


class TestHooksAndBrief:
    """Hook and brief operations."""

    def test_add_hook(self) -> None:
        """Add hooks."""
        hot = HotStore()
        hot.add_hook({"type": "narrative", "title": "Mystery"})
        hot.add_hook({"type": "mechanical", "title": "Puzzle"})

        hooks = hot.get_hooks()
        assert len(hooks) == 2

    def test_clear_hooks(self) -> None:
        """Clear hooks."""
        hot = HotStore()
        hot.add_hook({"title": "Hook 1"})
        hot.clear_hooks()
        assert hot.get_hooks() == []

    def test_set_brief(self) -> None:
        """Set and get current brief."""
        hot = HotStore()
        brief = {"type": "scene_draft", "target": "scene_001"}
        hot.set_brief(brief)
        assert hot.get_brief() == brief

    def test_clear_brief(self) -> None:
        """Clear current brief."""
        hot = HotStore()
        hot.set_brief({"type": "test"})
        hot.clear_brief()
        assert hot.get_brief() is None


class TestScratch:
    """Scratch space (role working memory)."""

    def test_scratch_put_get(self) -> None:
        """Put and get from scratch."""
        hot = HotStore()
        hot.scratch_put("notes", "Important info")
        assert hot.scratch_get("notes") == "Important info"

    def test_scratch_missing_returns_none(self) -> None:
        """Missing scratch key returns None."""
        hot = HotStore()
        assert hot.scratch_get("missing") is None

    def test_scratch_clear(self) -> None:
        """Clear scratch space."""
        hot = HotStore()
        hot.scratch_put("a", 1)
        hot.scratch_put("b", 2)
        hot.scratch_clear()
        assert hot.scratch_get("a") is None


class TestCheckpointing:
    """Checkpoint and restore."""

    def test_checkpoint_and_restore(self, tmp_path: Path) -> None:
        """Checkpoint and restore hot store."""
        hot = HotStore()
        hot.put("scene", {"title": "Opening"})
        hot.add_hook({"type": "narrative"})
        hot.set_brief({"type": "draft"})
        hot.scratch_put("notes", "Important")

        checkpoint_path = tmp_path / "checkpoint.json"
        hot.checkpoint(checkpoint_path)

        # Restore
        restored = HotStore.from_checkpoint(checkpoint_path)
        assert restored.get("scene") == {"title": "Opening"}
        assert len(restored.get_hooks()) == 1
        assert restored.get_brief() == {"type": "draft"}
        assert restored.scratch_get("notes") == "Important"

    def test_checkpoint_creates_directory(self, tmp_path: Path) -> None:
        """Checkpoint creates parent directories."""
        hot = HotStore()
        hot.put("test", "value")

        checkpoint_path = tmp_path / "subdir" / "deep" / "checkpoint.json"
        hot.checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

    def test_restore_nonexistent_raises(self, tmp_path: Path) -> None:
        """Restore from nonexistent file raises."""
        with pytest.raises(FileNotFoundError):
            HotStore.from_checkpoint(tmp_path / "missing.json")

    def test_dirty_tracking(self) -> None:
        """Track unsaved changes."""
        hot = HotStore()
        # New store starts clean (no changes to save)

        hot.put("key", "value")
        assert hot.is_dirty

    def test_checkpoint_clears_dirty(self, tmp_path: Path) -> None:
        """Checkpoint clears dirty flag."""
        hot = HotStore()
        hot.put("key", "value")
        assert hot.is_dirty

        hot.checkpoint(tmp_path / "checkpoint.json")
        assert not hot.is_dirty
