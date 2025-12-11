"""HotStore - In-memory workspace for active creative work with optional persistence.

The HotStore is the mutable workspace where roles collaborate on drafts.
Content here is not player-safe and may contain spoilers.

Characteristics:
- Lifetime: Session-scoped (in-memory for performance)
- Persistence: Work artifacts (hooks, briefs) can persist to cold_store's artifacts table
- Contents: Artifacts in draft/proposed/in_progress states
- Spoilers: Allowed (not player-safe)

Persistence Model:
- Content artifacts (scenes, acts, chapters) are promoted to cold_store via Lorekeeper
- Work artifacts (hooks, briefs, gatecheck reports) persist in cold_store's artifacts table
- Use load_from_cold_store() at session start to restore work artifacts
- Use save_to_cold_store() at session end to persist work artifacts
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from questfoundry.runtime.stores.cold_store import ColdStore

logger = logging.getLogger(__name__)


class HotStore(BaseModel):
    """Ephemeral in-memory workspace for active creative work.

    Structure from stores.md:
    - artifacts: dict[str, Artifact] - All working artifacts by ID
    - hooks: list[HookCard] - Active hook cards
    - current_brief: Brief | None - Active work order
    - scratch: dict[str, Any] - Role working memory

    Examples
    --------
    Create and use a hot store::

        hot = HotStore()
        hot.put("scene_001", {"title": "Opening", "content": "..."})
        scene = hot.get("scene_001")

    Checkpoint and resume::

        hot.checkpoint(Path("project/.qf/checkpoint.json"))
        # ... crash and restart ...
        hot = HotStore.from_checkpoint(Path("project/.qf/checkpoint.json"))
    """

    artifacts: dict[str, Any] = Field(default_factory=dict)
    hooks: list[Any] = Field(default_factory=list)
    current_brief: Any | None = None
    scratch: dict[str, Any] = Field(default_factory=dict)

    # Internal tracking
    _dirty: bool = False

    model_config = {"extra": "allow"}

    # =========================================================================
    # Artifact Operations
    # =========================================================================

    def get(self, key: str) -> Any | None:
        """Get an artifact by key."""
        return self.artifacts.get(key)

    def put(self, key: str, value: Any) -> None:
        """Store an artifact."""
        self.artifacts[key] = value
        self._dirty = True

    def delete(self, key: str) -> bool:
        """Delete an artifact. Returns True if existed."""
        if key in self.artifacts:
            del self.artifacts[key]
            self._dirty = True
            return True
        return False

    def keys(self) -> list[str]:
        """List all artifact keys."""
        return list(self.artifacts.keys())

    def has(self, key: str) -> bool:
        """Check if artifact exists."""
        return key in self.artifacts

    def clear(self) -> None:
        """Clear all artifacts."""
        self.artifacts.clear()
        self._dirty = True

    # =========================================================================
    # Hook Operations
    # =========================================================================

    def add_hook(self, hook: Any) -> None:
        """Add a hook card."""
        self.hooks.append(hook)
        self._dirty = True

    def get_hooks(self) -> list[Any]:
        """Get all hooks."""
        return list(self.hooks)

    def clear_hooks(self) -> None:
        """Clear all hooks."""
        self.hooks.clear()
        self._dirty = True

    # =========================================================================
    # Brief Operations
    # =========================================================================

    def set_brief(self, brief: Any) -> None:
        """Set the current brief."""
        self.current_brief = brief
        self._dirty = True

    def get_brief(self) -> Any | None:
        """Get the current brief."""
        return self.current_brief

    def clear_brief(self) -> None:
        """Clear the current brief."""
        self.current_brief = None
        self._dirty = True

    # =========================================================================
    # Scratch Operations (Role Working Memory)
    # =========================================================================

    def scratch_get(self, key: str) -> Any | None:
        """Get from scratch space."""
        return self.scratch.get(key)

    def scratch_put(self, key: str, value: Any) -> None:
        """Put to scratch space."""
        self.scratch[key] = value
        self._dirty = True

    def scratch_clear(self) -> None:
        """Clear scratch space."""
        self.scratch.clear()
        self._dirty = True

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def checkpoint(self, path: Path) -> None:
        """Save hot store state to a checkpoint file.

        Parameters
        ----------
        path : Path
            Path to checkpoint file (e.g., project/.qf/checkpoint.json)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize artifacts (handle Pydantic models)
        serialized_artifacts = {}
        for key, value in self.artifacts.items():
            if hasattr(value, "model_dump"):
                serialized_artifacts[key] = value.model_dump()
            else:
                serialized_artifacts[key] = value

        # Serialize hooks
        serialized_hooks = []
        for hook in self.hooks:
            if hasattr(hook, "model_dump"):
                serialized_hooks.append(hook.model_dump())
            else:
                serialized_hooks.append(hook)

        # Serialize brief
        serialized_brief = None
        if self.current_brief is not None:
            if hasattr(self.current_brief, "model_dump"):
                serialized_brief = self.current_brief.model_dump()
            else:
                serialized_brief = self.current_brief

        checkpoint_data = {
            "version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "artifacts": serialized_artifacts,
            "hooks": serialized_hooks,
            "current_brief": serialized_brief,
            "scratch": self.scratch,
        }

        with open(path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        self._dirty = False
        logger.info(f"Hot store checkpointed to {path}")

    @classmethod
    def from_checkpoint(cls, path: Path) -> HotStore:
        """Load hot store from a checkpoint file.

        Parameters
        ----------
        path : Path
            Path to checkpoint file.

        Returns
        -------
        HotStore
            Restored hot store.

        Raises
        ------
        FileNotFoundError
            If checkpoint file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path) as f:
            data = json.load(f)

        hot = cls(
            artifacts=data.get("artifacts", {}),
            hooks=data.get("hooks", []),
            current_brief=data.get("current_brief"),
            scratch=data.get("scratch", {}),
        )
        hot._dirty = False
        logger.info(f"Hot store restored from {path}")
        return hot

    @property
    def is_dirty(self) -> bool:
        """Check if hot store has unsaved changes."""
        return self._dirty

    # =========================================================================
    # Dict-like Access (for backwards compatibility with state["hot_store"])
    # =========================================================================

    def __getitem__(self, key: str) -> Any:
        """Dict-like access to artifacts."""
        return self.artifacts[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like assignment to artifacts."""
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        """Dict-like deletion from artifacts."""
        self.delete(key)

    def __contains__(self, key: str) -> bool:
        """Dict-like membership test."""
        return key in self.artifacts

    def __len__(self) -> int:
        """Number of artifacts."""
        return len(self.artifacts)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over artifact keys (overrides Pydantic's field iteration)."""
        return iter(self.artifacts)

    def items(self) -> Any:
        """Dict-like items()."""
        return self.artifacts.items()

    def values(self) -> Any:
        """Dict-like values()."""
        return self.artifacts.values()

    def get_dict(self, key: str, default: Any = None) -> Any:
        """Get with default (like dict.get)."""
        return self.artifacts.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default value if key doesn't exist."""
        if key not in self.artifacts:
            self.artifacts[key] = default
            self._dirty = True
        return self.artifacts[key]

    # =========================================================================
    # Cold Store Integration (persistent work artifacts)
    # =========================================================================

    # Artifact types that should persist to cold_store's artifacts table
    PERSISTENT_ARTIFACT_TYPES: ClassVar[frozenset[str]] = frozenset({
        "hook_card",
        "brief",
        "gatecheck_report",
        "shotlist",
        "audio_plan",
        "translation_pack",
    })

    def load_from_cold_store(self, cold_store: ColdStore) -> int:
        """Load persistent work artifacts from cold_store.

        Call this at session start to restore hooks, briefs, and other
        work artifacts from the previous session.

        Parameters
        ----------
        cold_store : ColdStore
            The cold store to load from.

        Returns
        -------
        int
            Number of artifacts loaded.
        """
        count = 0

        # Load all work artifacts
        for stored in cold_store.get_all_artifacts():
            # Reconstruct artifact from stored data
            artifact_data = stored.data
            anchor = stored.anchor

            # Store in artifacts dict
            self.artifacts[anchor] = artifact_data

            # Also populate hooks list for HookCards
            if stored.artifact_type == "hook_card":
                self.hooks.append(artifact_data)

            # Restore current brief if active
            if stored.artifact_type == "brief" and stored.status == "active":
                self.current_brief = artifact_data

            count += 1

        logger.info(f"Loaded {count} work artifacts from cold_store")
        self._dirty = False
        return count

    def save_to_cold_store(self, cold_store: ColdStore) -> int:
        """Save persistent work artifacts to cold_store.

        Call this at session end to persist hooks, briefs, and other
        work artifacts for the next session.

        Parameters
        ----------
        cold_store : ColdStore
            The cold store to save to.

        Returns
        -------
        int
            Number of artifacts saved.
        """
        count = 0

        for anchor, artifact in self.artifacts.items():
            # Determine artifact type
            artifact_type = self._get_artifact_type(artifact)
            if artifact_type not in self.PERSISTENT_ARTIFACT_TYPES:
                continue

            # Get status from artifact (default to 'draft')
            status = self._get_artifact_status(artifact)

            # Serialize artifact data
            if hasattr(artifact, "model_dump"):
                data = artifact.model_dump()
            elif isinstance(artifact, dict):
                data = artifact
            else:
                logger.warning(f"Cannot serialize artifact {anchor}: {type(artifact)}")
                continue

            cold_store.save_artifact(
                anchor=anchor,
                artifact_type=artifact_type,
                status=status,
                data=data,
            )
            count += 1

        logger.info(f"Saved {count} work artifacts to cold_store")
        self._dirty = False
        return count

    def _get_artifact_type(self, artifact: Any) -> str:
        """Determine the artifact type from an artifact object or dict."""
        # Check for explicit type field
        if isinstance(artifact, dict):
            if "artifact_type" in artifact:
                return artifact["artifact_type"]
            # Infer from field presence
            if "hook_type" in artifact:
                return "hook_card"
            if "loop_type" in artifact:
                return "brief"
            if "bars_checked" in artifact:
                return "gatecheck_report"
            if "shots" in artifact:
                return "shotlist"
            if "ambient" in artifact or "music_cues" in artifact:
                return "audio_plan"
            if "source_language" in artifact and "target_language" in artifact:
                return "translation_pack"
        elif hasattr(artifact, "__class__"):
            # Use class name
            class_name = artifact.__class__.__name__
            # Convert CamelCase to snake_case
            import re

            return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        return "unknown"

    def _get_artifact_status(self, artifact: Any) -> str:
        """Extract status from an artifact object or dict."""
        if isinstance(artifact, dict):
            return artifact.get("status", "draft")
        elif hasattr(artifact, "status"):
            return artifact.status or "draft"
        return "draft"

    def sync_hooks_to_cold_store(self, cold_store: ColdStore) -> int:
        """Sync just the hooks list to cold_store.

        Convenience method for saving hooks without full artifact sync.

        Parameters
        ----------
        cold_store : ColdStore
            The cold store to save to.

        Returns
        -------
        int
            Number of hooks saved.
        """
        count = 0
        for i, hook in enumerate(self.hooks):
            # Generate anchor if not present
            if isinstance(hook, dict):
                anchor = hook.get("anchor") or f"hook_{i:04d}"
                status = hook.get("status", "proposed")
                data = hook
            elif hasattr(hook, "model_dump"):
                anchor = getattr(hook, "anchor", None) or f"hook_{i:04d}"
                status = getattr(hook, "status", "proposed")
                data = hook.model_dump()
            else:
                continue

            cold_store.save_artifact(
                anchor=anchor,
                artifact_type="hook_card",
                status=status,
                data=data,
            )
            count += 1

        logger.info(f"Synced {count} hooks to cold_store")
        return count
