-- QuestFoundry Project Database Schema
-- Format: SQLite 3
-- Version: 1

-- Schema version tracking for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial version
INSERT OR IGNORE INTO schema_version (version) VALUES (1);

-- Project metadata
CREATE TABLE IF NOT EXISTS project (
    id INTEGER PRIMARY KEY CHECK (id = 1), -- Singleton table
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '1.0.0',
    author TEXT,
    created TIMESTAMP NOT NULL,
    modified TIMESTAMP NOT NULL,
    metadata JSON NOT NULL DEFAULT '{}'
);

-- Snapshots (cold, immutable state captures)
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    tu_id TEXT NOT NULL,
    created TIMESTAMP NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    metadata JSON NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_snapshots_tu_id ON snapshots(tu_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_created ON snapshots(created DESC);

-- Views (perspectives on data, not SQL views)
CREATE TABLE IF NOT EXISTS views (
    view_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    config JSON NOT NULL,
    created TIMESTAMP NOT NULL,
    modified TIMESTAMP NOT NULL,
    metadata JSON NOT NULL DEFAULT '{}'
);

-- Artifacts (polymorphic, JSON-based storage)
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    created TIMESTAMP NOT NULL,
    modified TIMESTAMP NOT NULL,
    data JSON NOT NULL,
    metadata JSON NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_modified ON artifacts(modified DESC);

-- JSON extraction indexes for common queries
CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(
    json_extract(data, '$.status')
) WHERE json_extract(data, '$.status') IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_artifacts_author ON artifacts(
    json_extract(data, '$.author')
) WHERE json_extract(data, '$.author') IS NOT NULL;

-- Thematic Units (TU state tracking)
CREATE TABLE IF NOT EXISTS tus (
    tu_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    snapshot_id TEXT,
    created TIMESTAMP NOT NULL,
    modified TIMESTAMP NOT NULL,
    data JSON NOT NULL,
    metadata JSON NOT NULL DEFAULT '{}',
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_tus_status ON tus(status);
CREATE INDEX IF NOT EXISTS idx_tus_snapshot ON tus(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_tus_modified ON tus(modified DESC);

-- Audit history (optional logging of state changes)
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    entity_type TEXT NOT NULL, -- 'artifact', 'tu', 'snapshot', etc.
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL, -- 'create', 'update', 'delete'
    changes JSON, -- Diff or full snapshot
    metadata JSON NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_history_entity ON history(entity_type, entity_id);
