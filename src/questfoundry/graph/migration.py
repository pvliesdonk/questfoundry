"""JSON-to-SQLite graph migration.

Converts a graph.json file into a graph.db SQLite database.
Used by Graph.load() for automatic migration on first access.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from questfoundry.graph.sqlite_store import SqliteGraphStore

if TYPE_CHECKING:
    from pathlib import Path


def migrate_json_to_sqlite(json_path: Path, db_path: Path) -> SqliteGraphStore:
    """Migrate a graph.json file to a SQLite database.

    Reads the JSON file, bulk-imports all data into a new SqliteGraphStore,
    and returns the open store. The caller is responsible for renaming
    the original JSON file.

    Args:
        json_path: Path to the source graph.json file.
        db_path: Path for the new graph.db file.

    Returns:
        Open SqliteGraphStore populated with the migrated data.

    Raises:
        FileNotFoundError: If json_path doesn't exist.
        json.JSONDecodeError: If JSON is invalid.
    """
    with json_path.open() as f:
        data = json.load(f)
    return SqliteGraphStore.from_dict(data, db_path=db_path)
