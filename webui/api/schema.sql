-- QuestFoundry WebUI Database Schema
-- 
-- This schema defines:
-- 1. Tenancy tables (user_settings, project_ownership)
-- 2. QuestFoundry cold storage tables (artifacts, tus, snapshots, etc.)
--
-- All QuestFoundry tables include project_id for multi-tenancy isolation.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================================
-- TENANCY LAYER TABLES
-- These tables manage users, projects, and tenant-specific configuration.
-- They are NOT part of the QuestFoundry artifact schema.
-- ==============================================================================

-- User settings and BYOK (Bring Your Own Key) storage
CREATE TABLE user_settings (
    user_id TEXT PRIMARY KEY,
    encrypted_keys BYTEA NOT NULL,  -- Fernet-encrypted provider API keys
    encryption_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_user_settings_updated ON user_settings(updated_at);

-- Project ownership and metadata
CREATE TABLE project_ownership (
    project_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL REFERENCES user_settings(user_id) ON DELETE CASCADE,
    project_name TEXT NOT NULL,
    project_description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_project_id CHECK (project_id ~ '^[a-zA-Z0-9_-]+$')
);

CREATE INDEX idx_project_ownership_owner ON project_ownership(owner_user_id);
CREATE INDEX idx_project_ownership_updated ON project_ownership(updated_at);

-- ==============================================================================
-- QUESTFOUNDRY COLD STORAGE TABLES
-- These implement the StateStore protocol with project_id scoping.
-- ==============================================================================

-- Project metadata (per-project configuration)
CREATE TABLE project_info (
    project_id TEXT NOT NULL REFERENCES project_ownership(project_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    version TEXT NOT NULL DEFAULT '1.0.0',
    author TEXT,
    created TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    modified TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (project_id)
);

CREATE INDEX idx_project_info_modified ON project_info(modified);

-- Artifacts (Cold SoT)
CREATE TABLE artifacts (
    project_id TEXT NOT NULL REFERENCES project_ownership(project_id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    modified TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (project_id, artifact_id)
);

CREATE INDEX idx_artifacts_type ON artifacts(project_id, artifact_type);
CREATE INDEX idx_artifacts_modified ON artifacts(project_id, modified);
CREATE INDEX idx_artifacts_data ON artifacts USING GIN(data);

-- Thematic Units (TU state)
CREATE TABLE tus (
    project_id TEXT NOT NULL REFERENCES project_ownership(project_id) ON DELETE CASCADE,
    tu_id TEXT NOT NULL,
    status TEXT NOT NULL,
    snapshot_id TEXT,
    created TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    modified TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    data JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (project_id, tu_id)
);

CREATE INDEX idx_tus_status ON tus(project_id, status);
CREATE INDEX idx_tus_modified ON tus(project_id, modified);

-- Snapshots (immutable checkpoints)
CREATE TABLE snapshots (
    project_id TEXT NOT NULL REFERENCES project_ownership(project_id) ON DELETE CASCADE,
    snapshot_id TEXT NOT NULL,
    tu_id TEXT NOT NULL,
    created TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    description TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (project_id, snapshot_id)
);

CREATE INDEX idx_snapshots_tu ON snapshots(project_id, tu_id);
CREATE INDEX idx_snapshots_created ON snapshots(project_id, created);

-- ==============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ==============================================================================

-- Function to update modified timestamp
CREATE OR REPLACE FUNCTION update_modified_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tenancy tables
CREATE TRIGGER update_user_settings_timestamp
    BEFORE UPDATE ON user_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_timestamp();

CREATE TRIGGER update_project_ownership_timestamp
    BEFORE UPDATE ON project_ownership
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_timestamp();

-- Function for QuestFoundry tables (uses 'modified' column)
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.modified = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to QuestFoundry tables
CREATE TRIGGER update_project_info_modified
    BEFORE UPDATE ON project_info
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_artifacts_modified
    BEFORE UPDATE ON artifacts
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_tus_modified
    BEFORE UPDATE ON tus
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();
