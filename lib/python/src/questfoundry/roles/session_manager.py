"""Session manager for coordinating multiple active role sessions."""

from pathlib import Path
from typing import Any

from .session import RoleSession


class SessionManager:
    """
    Manages all active role sessions.

    The SessionManager tracks which roles are currently active (awake),
    maintains their conversation history, and handles the wake/dormant
    lifecycle transitions.

    Attributes:
        workspace_path: Path to workspace for session storage
        active_sessions: Dictionary mapping role name to RoleSession
    """

    def __init__(self, workspace_path: Path | None = None):
        """
        Initialize session manager.

        Args:
            workspace_path: Path to workspace. If None, uses current directory
        """
        self.workspace_path = workspace_path or Path.cwd()
        self.active_sessions: dict[str, RoleSession] = {}

    def wake_role(self, role: str, tu_context: str | None = None) -> RoleSession:
        """
        Create new session for role (wake it up).

        If role is already awake, returns existing session.

        Args:
            role: Role name to wake (e.g., "scene_smith", "showrunner")
            tu_context: Optional TU context (e.g., "TU-2024-01-15-TEST01")

        Returns:
            The active session for this role
        """
        if role in self.active_sessions:
            # Role already awake - return existing session
            return self.active_sessions[role]

        # Create new session
        session = RoleSession(
            role=role,
            tu_context=tu_context,
            workspace_path=self.workspace_path,
        )

        self.active_sessions[role] = session
        return session

    def dormant_role(self, role: str) -> dict[str, Any]:
        """
        Archive and clear role session (make it dormant).

        Args:
            role: Role name to make dormant

        Returns:
            Archived session data

        Raises:
            KeyError: If role is not currently awake
        """
        if role not in self.active_sessions:
            raise KeyError(f"Role '{role}' is not currently awake")

        session = self.active_sessions[role]

        # Archive the session
        archive = session.archive()

        # Save to file
        session.save_to_file()

        # Remove from active sessions
        del self.active_sessions[role]

        return archive

    def get_session(self, role: str) -> RoleSession | None:
        """
        Get active session for role.

        Args:
            role: Role name to get session for

        Returns:
            RoleSession if role is awake, None otherwise
        """
        return self.active_sessions.get(role)

    def get_active_roles(self) -> list[str]:
        """
        List currently active roles.

        Returns:
            List of role names that are currently awake
        """
        return list(self.active_sessions.keys())

    def is_role_awake(self, role: str) -> bool:
        """
        Check if a role is currently awake.

        Args:
            role: Role name to check

        Returns:
            True if role is awake, False otherwise
        """
        return role in self.active_sessions

    def archive_all(self) -> dict[str, dict[str, Any]]:
        """
        Archive all active sessions.

        This does NOT make roles dormant - it just captures current state.

        Returns:
            Dictionary mapping role name to archived session data
        """
        return {
            role: session.archive() for role, session in self.active_sessions.items()
        }

    def dormant_all(self) -> dict[str, dict[str, Any]]:
        """
        Archive and make all roles dormant.

        Returns:
            Dictionary mapping role name to archived session data
        """
        archives = {}

        # Make copy of keys since we're modifying the dict
        for role in list(self.active_sessions.keys()):
            archives[role] = self.dormant_role(role)

        return archives

    def get_total_message_count(self) -> int:
        """
        Get total number of messages across all active sessions.

        Returns:
            Total message count
        """
        return sum(
            len(session.conversation_history)
            for session in self.active_sessions.values()
        )

    def clear_dormancy_signals(self, role: str) -> None:
        """
        Clear dormancy signals for a role.

        Useful if you want to keep a role awake despite dormancy signals.

        Args:
            role: Role name to clear signals for

        Raises:
            KeyError: If role is not currently awake
        """
        if role not in self.active_sessions:
            raise KeyError(f"Role '{role}' is not currently awake")

        self.active_sessions[role].dormancy_signals.clear()

    def get_sessions_needing_dormancy(self) -> list[str]:
        """
        Get list of roles that should go dormant based on their signals.

        Returns:
            List of role names that have dormancy signals
        """
        return [
            role
            for role, session in self.active_sessions.items()
            if session.should_dormant()
        ]

    def __repr__(self) -> str:
        """String representation of session manager."""
        active_count = len(self.active_sessions)
        total_messages = self.get_total_message_count()
        return (
            f"SessionManager(active_roles={active_count}, "
            f"total_messages={total_messages})"
        )
