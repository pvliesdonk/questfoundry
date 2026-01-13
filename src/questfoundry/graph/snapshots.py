"""Snapshot management for stage-level rollback.

Pre-stage snapshots enable reverting failed or rejected stages without
complex event sourcing.

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

    Args:
        graph: Current graph state.
        project_path: Path to project root directory.
        stage_name: Name of stage about to run.

    Returns:
        Path to snapshot file.
    """
    snapshot_dir = project_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"pre-{stage_name}.json"
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

    snapshot_path = project_path / "snapshots" / f"pre-{stage_name}.json"
    if not snapshot_path.exists():
        raise ValueError(f"No snapshot for stage '{stage_name}'")

    # Load snapshot
    graph = Graph.load_from_file(snapshot_path)

    # Save as current graph
    graph.save(project_path / "graph.json")

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

    snapshots = []
    for path in sorted(snapshot_dir.glob("pre-*.json")):
        # Extract stage name from "pre-{stage_name}.json"
        stage_name = path.stem.removeprefix("pre-")
        snapshots.append(stage_name)

    return snapshots


def delete_snapshot(project_path: Path, stage_name: str) -> bool:
    """Delete a specific snapshot.

    Args:
        project_path: Path to project root directory.
        stage_name: Name of stage whose snapshot to delete.

    Returns:
        True if snapshot was deleted, False if it didn't exist.
    """
    snapshot_path = project_path / "snapshots" / f"pre-{stage_name}.json"
    if snapshot_path.exists():
        snapshot_path.unlink()
        return True
    return False


def cleanup_old_snapshots(project_path: Path, keep_count: int = 3) -> list[str]:
    """Remove old snapshots, keeping only the most recent.

    Args:
        project_path: Path to project root directory.
        keep_count: Number of most recent snapshots to keep.

    Returns:
        List of stage names whose snapshots were deleted.
    """
    snapshot_dir = project_path / "snapshots"
    if not snapshot_dir.exists():
        return []

    # Get all snapshots sorted by modification time (oldest first)
    snapshot_files = sorted(
        snapshot_dir.glob("pre-*.json"),
        key=lambda p: p.stat().st_mtime,
    )

    deleted = []
    # Remove oldest snapshots, keeping only keep_count
    while len(snapshot_files) > keep_count:
        oldest = snapshot_files.pop(0)
        stage_name = oldest.stem.removeprefix("pre-")
        oldest.unlink()
        deleted.append(stage_name)

    return deleted
