"""Snapshot management for stage-level rollback.

Pre-stage snapshots enable reverting failed or rejected stages without
complex event sourcing.  All snapshots are stored as SQLite ``.db`` files.

See docs/architecture/graph-storage.md for design details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from questfoundry.graph.graph import Graph


def save_snapshot(graph: Graph, project_path: Path, stage_name: str) -> Path:
    """Save pre-stage snapshot.

    Creates a snapshot of the current graph state before a stage runs.
    This allows rolling back if the stage fails or is rejected.

    Note:
        Overwrites any existing snapshot for this stage. If you run the same
        stage multiple times (e.g., after rollback and retry), only the most
        recent snapshot is preserved.

    Args:
        graph: Current graph state.
        project_path: Path to project root directory.
        stage_name: Name of stage about to run.

    Returns:
        Path to snapshot file.
    """
    snapshot_dir = project_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"pre-{stage_name}.db"
    graph.save(snapshot_path)
    return snapshot_path


def rollback_to_snapshot(project_path: Path, stage_name: str) -> Graph:
    """Restore graph from pre-stage snapshot.

    Loads a snapshot and saves it as the current graph, effectively
    reverting all changes made by the stage.

    Args:
        project_path: Path to project root directory.
        stage_name: Name of stage to rollback.

    Returns:
        Restored graph.

    Raises:
        ValueError: If no snapshot exists for the stage.
    """
    # Import here to avoid circular dependency
    from questfoundry.graph.graph import Graph

    snapshot_path = project_path / "snapshots" / f"pre-{stage_name}.db"
    if not snapshot_path.exists():
        raise ValueError(f"No snapshot for stage '{stage_name}'")

    graph = Graph.load_from_file(snapshot_path)
    graph.save(project_path / "graph.db")
    return graph


def list_snapshots(project_path: Path) -> list[str]:
    """List available snapshots.

    Args:
        project_path: Path to project root directory.

    Returns:
        List of stage names that have snapshots.
    """
    snapshot_dir = project_path / "snapshots"
    if not snapshot_dir.exists():
        return []

    stages: set[str] = set()
    for path in snapshot_dir.iterdir():
        if path.name.startswith("pre-") and path.suffix == ".db":
            stage_name = path.stem.removeprefix("pre-")
            stages.add(stage_name)

    return sorted(stages)


def delete_snapshot(project_path: Path, stage_name: str) -> bool:
    """Delete a specific snapshot.

    Args:
        project_path: Path to project root directory.
        stage_name: Name of stage whose snapshot to delete.

    Returns:
        True if the snapshot was deleted, False if it didn't exist.
    """
    path = project_path / "snapshots" / f"pre-{stage_name}.db"
    if path.exists():
        path.unlink()
        return True
    return False


def cleanup_old_snapshots(project_path: Path, keep_count: int = 3) -> list[str]:
    """Remove old snapshots, keeping only the most recent.

    Note:
        Not safe for concurrent snapshot creation. If another process creates
        a snapshot between listing and deletion, more snapshots than intended
        may remain or be deleted.

    Args:
        project_path: Path to project root directory.
        keep_count: Number of most recent snapshots to keep.

    Returns:
        List of stage names whose snapshots were deleted.
    """
    snapshot_dir = project_path / "snapshots"
    if not snapshot_dir.exists():
        return []

    snapshot_files = sorted(
        [p for p in snapshot_dir.iterdir() if p.name.startswith("pre-") and p.suffix == ".db"],
        key=lambda p: p.stat().st_mtime,
    )

    deleted = []
    while len(snapshot_files) > keep_count:
        oldest = snapshot_files.pop(0)
        stage_name = oldest.stem.removeprefix("pre-")
        oldest.unlink()
        deleted.append(stage_name)

    return deleted
