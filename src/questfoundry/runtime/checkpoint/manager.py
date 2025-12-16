"""
CheckpointManager - Save and restore session checkpoints.

Handles checkpoint lifecycle:
- Create checkpoints (auto and manual)
- Load checkpoints for resumption
- List and delete checkpoints
- Handle schema migrations
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.checkpoint.models import (
    CHECKPOINT_SCHEMA_VERSION,
    Checkpoint,
    CheckpointConfig,
    CheckpointInfo,
    ContextUsage,
    DelegationSnapshot,
)

if TYPE_CHECKING:
    from questfoundry.runtime.delegation.tracker import PlaybookTracker
    from questfoundry.runtime.messaging.broker import AsyncMessageBroker
    from questfoundry.runtime.session import Session
    from questfoundry.runtime.storage import Project

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint lifecycle for a project.

    Responsibilities:
    - Create checkpoints (auto and manual)
    - Load checkpoints for resumption
    - List and delete checkpoints
    - Handle schema migrations
    - Enforce retention policy (rolling window)
    """

    def __init__(
        self,
        project: Project,
        config: CheckpointConfig | None = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            project: Project for checkpoint storage
            config: Checkpoint configuration (uses defaults if None)
        """
        self._project = project
        self._config = config or CheckpointConfig()
        self._checkpoints_dir = project.checkpoints_path

        # Ensure checkpoints directory exists
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> CheckpointConfig:
        """Get checkpoint configuration."""
        return self._config

    async def create_checkpoint(
        self,
        session: Session,
        broker: AsyncMessageBroker,
        tracker: PlaybookTracker | None = None,
        context_usage: dict[str, ContextUsage] | None = None,
        checkpoint_id: str | None = None,
        summary: str | None = None,
    ) -> Checkpoint:
        """
        Create a checkpoint of current session state.

        Args:
            session: Current session
            broker: Message broker with mailbox states
            tracker: Playbook tracker (optional)
            context_usage: Token usage per agent (optional)
            checkpoint_id: Custom ID (auto-generated if None)
            summary: Human-readable summary

        Returns:
            The created Checkpoint
        """
        turn_number = session.turn_count

        # Generate checkpoint ID if not provided
        if checkpoint_id is None:
            checkpoint_id = f"cp_turn_{turn_number:03d}"

        # Capture mailbox states
        mailbox_states: dict[str, list[dict[str, Any]]] = {}
        for agent_id, mailbox in broker._mailboxes.items():
            mailbox_states[agent_id] = [msg.to_dict() for msg in mailbox._pending.values()]

        # Capture active delegations
        active_delegations: list[DelegationSnapshot] = []
        # Note: Would need access to delegation tracker to capture in-flight delegations
        # For now, we capture pending delegation messages from mailboxes

        # Capture playbook instances
        playbook_instances: list[dict[str, Any]] = []
        if tracker:
            for instance in tracker._instances.values():
                playbook_instances.append(instance.to_dict())

        # Create checkpoint
        checkpoint = Checkpoint(
            id=checkpoint_id,
            session_id=session.id,
            turn_number=turn_number,
            created_at=datetime.now(),
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            session_status=session.status,
            entry_agent=session.entry_agent,
            turn_count=session.turn_count,
            mailbox_states=mailbox_states,
            active_delegations=active_delegations,
            playbook_instances=playbook_instances,
            context_usage=context_usage or {},
            summary=summary,
        )

        # Save to disk
        self._save_checkpoint(checkpoint)

        # Enforce retention policy
        self._enforce_retention()

        logger.info(
            "Created checkpoint %s for session %s at turn %d",
            checkpoint.id,
            session.id,
            turn_number,
        )

        return checkpoint

    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to disk."""
        checkpoint_path = self._checkpoints_dir / f"{checkpoint.id}.json"
        checkpoint_data = checkpoint.to_dict()

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.debug("Saved checkpoint to %s", checkpoint_path)

    def load_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """
        Load a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID to load

        Returns:
            Checkpoint or None if not found
        """
        checkpoint_path = self._checkpoints_dir / f"{checkpoint_id}.json"

        if not checkpoint_path.exists():
            logger.warning("Checkpoint not found: %s", checkpoint_id)
            return None

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)

            # Handle schema migration if needed
            data = self._migrate_checkpoint(data)

            checkpoint = Checkpoint.from_dict(data)
            logger.debug("Loaded checkpoint %s", checkpoint_id)
            return checkpoint

        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to load checkpoint %s: %s", checkpoint_id, e)
            return None

    def _migrate_checkpoint(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Migrate checkpoint data to current schema version.

        Args:
            data: Checkpoint data from disk

        Returns:
            Migrated checkpoint data
        """
        schema_version = data.get("schema_version", 1)

        if schema_version == CHECKPOINT_SCHEMA_VERSION:
            return data

        # Future: Add migration logic for schema upgrades
        # For now, just log a warning
        logger.warning(
            "Checkpoint schema version %d differs from current %d",
            schema_version,
            CHECKPOINT_SCHEMA_VERSION,
        )

        return data

    def list_checkpoints(
        self,
        session_id: str | None = None,
    ) -> list[CheckpointInfo]:
        """
        List available checkpoints.

        Args:
            session_id: Optional filter by session ID

        Returns:
            List of CheckpointInfo, sorted by turn number (newest first)
        """
        checkpoints: list[CheckpointInfo] = []

        for checkpoint_file in self._checkpoints_dir.glob("*.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)

                # Filter by session if specified
                if session_id and data.get("session_id") != session_id:
                    continue

                info = CheckpointInfo(
                    id=data["id"],
                    session_id=data["session_id"],
                    turn_number=data["turn_number"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    summary=data.get("summary"),
                )
                checkpoints.append(info)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping invalid checkpoint %s: %s", checkpoint_file, e)

        # Sort by turn number descending (newest first)
        checkpoints.sort(key=lambda c: c.turn_number, reverse=True)

        return checkpoints

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self._checkpoints_dir / f"{checkpoint_id}.json"

        if not checkpoint_path.exists():
            logger.warning("Checkpoint not found for deletion: %s", checkpoint_id)
            return False

        checkpoint_path.unlink()
        logger.info("Deleted checkpoint: %s", checkpoint_id)
        return True

    def _enforce_retention(self) -> None:
        """
        Enforce checkpoint retention policy.

        Deletes oldest checkpoints if we exceed max_checkpoints.
        """
        if self._config.max_checkpoints == 0:
            return  # Unlimited retention

        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= self._config.max_checkpoints:
            return

        # Delete oldest checkpoints
        to_delete = checkpoints[self._config.max_checkpoints :]
        for cp in to_delete:
            self.delete_checkpoint(cp.id)
            logger.debug("Deleted checkpoint %s (retention policy)", cp.id)

    async def restore_from_checkpoint(
        self,
        checkpoint: Checkpoint,
        broker: AsyncMessageBroker,
        tracker: PlaybookTracker | None = None,
    ) -> dict[str, Any]:
        """
        Restore session state from checkpoint.

        This restores:
        - Mailbox states (pending messages)
        - Playbook instances
        - Context usage tracking

        Note: The Session itself is loaded from the database, not from
        the checkpoint. The checkpoint captures transient state that
        isn't persisted elsewhere.

        Args:
            checkpoint: Checkpoint to restore from
            broker: Message broker to restore mailboxes into
            tracker: Playbook tracker to restore instances into

        Returns:
            Dict with restoration details
        """
        mailboxes_restored = 0
        messages_restored = 0
        playbooks_restored = 0

        # Restore mailbox states
        for agent_id, messages in checkpoint.mailbox_states.items():
            if messages:
                mailbox = await broker.get_mailbox(agent_id)
                # Clear existing messages and restore from checkpoint
                await mailbox.clear()

                # Import Message for deserialization
                from questfoundry.runtime.messaging.message import Message

                for msg_data in messages:
                    message = Message.from_dict(msg_data)
                    await mailbox.put(message)
                    messages_restored += 1

                mailboxes_restored += 1

        # Restore playbook instances
        if tracker and checkpoint.playbook_instances:
            from questfoundry.runtime.delegation.tracker import PlaybookInstance

            for instance_data in checkpoint.playbook_instances:
                instance = PlaybookInstance.from_dict(instance_data)
                tracker._instances[instance.instance_id] = instance
                playbooks_restored += 1

        logger.info(
            "Restored from checkpoint %s: %d mailboxes, %d messages, %d playbooks",
            checkpoint.id,
            mailboxes_restored,
            messages_restored,
            playbooks_restored,
        )

        return {
            "checkpoint_id": checkpoint.id,
            "session_id": checkpoint.session_id,
            "turn_number": checkpoint.turn_number,
            "mailboxes_restored": mailboxes_restored,
            "messages_restored": messages_restored,
            "playbooks_restored": playbooks_restored,
        }

    def get_latest_checkpoint(
        self,
        session_id: str | None = None,
    ) -> Checkpoint | None:
        """
        Get the most recent checkpoint.

        Args:
            session_id: Optional filter by session ID

        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints(session_id=session_id)

        if not checkpoints:
            return None

        # First in list is most recent (sorted by turn_number desc)
        return self.load_checkpoint(checkpoints[0].id)
