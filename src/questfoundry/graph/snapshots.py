"""Snapshot management for stage-level rollback.

Pre-stage snapshots enable reverting failed or rejected stages without
complex event sourcing. Supports both JSON and SQLite graph files.

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

    For SQLite-backed graphs, saves a ``.db`` snapshot. For dict-backed
    graphs, saves a ``.json`` snapshot.

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
    ext = ".db" if graph.is_sqlite_backed else ".json"
    snapshot_path = snapshot_dir / f"pre-{stage_name}{ext}"
    graph.save(snapshot_path)
    return snapshot_path


def rollback_to_snapshot(project_path: Path, stage_name: str) -> Graph:
    """Restore graph from pre-stage snapshot.

    Loads a snapshot and saves it as the current graph, effectively
    reverting all changes made by the stage.

    Searches for both ``.db`` and ``.json`` snapshots (prefers ``.db``).

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

    snapshot_dir = project_path / "snapshots"
    db_path = snapshot_dir / f"pre-{stage_name}.db"
    json_path = snapshot_dir / f"pre-{stage_name}.json"

    if db_path.exists():
        snapshot_path = db_path
    elif json_path.exists():
        snapshot_path = json_path
    else:
        raise ValueError(f"No snapshot for stage '{stage_name}'")

    graph = Graph.load_from_file(snapshot_path)

    # Save as current graph. Prefer .db if it exists (project already migrated).
    if (project_path / "graph.db").exists():
        graph.save(project_path / "graph.db")
    else:
        graph.save(project_path / "graph.json")

    return graph


def list_snapshots(project_path: Path) -> list[str]:
    """List available snapshots.

    Finds both ``.json`` and ``.db`` snapshot files.

    Args:
        project_path: Path to project root directory.

    Returns:
        List of stage names that have snapshots (deduplicated).
    """
    snapshot_dir = project_path / "snapshots"
    if not snapshot_dir.exists():
        return []

    stages: set[str] = set()
    for path in snapshot_dir.iterdir():
        if path.name.startswith("pre-") and path.suffix in (".json", ".db"):
            stage_name = path.stem.removeprefix("pre-")
            stages.add(stage_name)

    return sorted(stages)


def delete_snapshot(project_path: Path, stage_name: str) -> bool:
    """Delete a specific snapshot.

    Removes both ``.json`` and ``.db`` variants if they exist.

    Args:
        project_path: Path to project root directory.
        stage_name: Name of stage whose snapshot to delete.

    Returns:
        True if any snapshot was deleted, False if none existed.
    """
    snapshot_dir = project_path / "snapshots"
    deleted = False
    for ext in (".json", ".db"):
        path = snapshot_dir / f"pre-{stage_name}{ext}"
        if path.exists():
            path.unlink()
            deleted = True
    return deleted


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
        [
            p
            for p in snapshot_dir.iterdir()
            if p.name.startswith("pre-") and p.suffix in (".json", ".db")
        ],
        key=lambda p: p.stat().st_mtime,
    )

    deleted = []
    while len(snapshot_files) > keep_count:
        oldest = snapshot_files.pop(0)
        stage_name = oldest.stem.removeprefix("pre-")
        oldest.unlink()
        deleted.append(stage_name)

    return deleted
