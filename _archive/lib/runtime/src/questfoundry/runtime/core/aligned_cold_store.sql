-- =====================================================
-- QUESTFOUNDRY ALIGNED COLD SOT DATABASE
-- SQLite database storing manifests and metadata
-- External files referenced by path + SHA-256 hash
-- =====================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- =====================================================
-- COLD MANIFEST (from cold_manifest.schema.json)
-- =====================================================

CREATE TABLE IF NOT EXISTS cold_manifest (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Singleton
    schema_version TEXT NOT NULL DEFAULT 'https://questfoundry.liesdonk.nl/schemas/cold_manifest.schema.json',
    version TEXT NOT NULL DEFAULT '1.0.0',
    created_at TEXT NOT NULL,  -- ISO 8601
    snapshot_id TEXT NOT NULL,  -- e.g., 'cold-20251124'

    CHECK (snapshot_id GLOB 'cold-*' OR snapshot_id = 'cold-init')
);

CREATE TABLE IF NOT EXISTS cold_manifest_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest_id INTEGER NOT NULL DEFAULT 1 REFERENCES cold_manifest(id),
    path TEXT NOT NULL,  -- e.g., 'cold/book.json', 'sections/001.md'
    sha256 TEXT NOT NULL,  -- 64-char hex
    size_bytes INTEGER NOT NULL,

    CHECK (length(sha256) = 64),
    CHECK (size_bytes >= 0),
    UNIQUE(path)
);

-- =====================================================
-- COLD BOOK (from cold_book.schema.json)
-- =====================================================

CREATE TABLE IF NOT EXISTS cold_book_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Singleton
    schema_version TEXT NOT NULL DEFAULT 'https://questfoundry.liesdonk.nl/schemas/cold_book.schema.json',
    version TEXT NOT NULL DEFAULT '1.0.0',

    -- Bibliographic metadata
    title TEXT NOT NULL,
    subtitle TEXT,
    language TEXT NOT NULL,  -- ISO 639-1/3
    author TEXT,
    isbn TEXT,
    published_at TEXT,  -- YYYY-MM-DD
    edition TEXT,
    copyright TEXT,
    publisher TEXT,

    -- Start point
    start_section TEXT NOT NULL,  -- anchor reference

    CHECK (language GLOB '[a-z][a-z]' OR language GLOB '[a-z][a-z][a-z]')
);

CREATE TABLE IF NOT EXISTS cold_book_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,  -- e.g., 'anchor001'
    title TEXT NOT NULL,
    text_file TEXT NOT NULL,  -- e.g., 'sections/001.md'
    order_num INTEGER NOT NULL,
    player_safe INTEGER NOT NULL DEFAULT 1,  -- Always true in Cold
    requires_gate INTEGER NOT NULL DEFAULT 0,

    CHECK (anchor GLOB 'anchor[0-9][0-9][0-9]*'),
    CHECK (text_file GLOB 'sections/[0-9][0-9][0-9]*.md'),
    CHECK (player_safe = 1),  -- Cold is always player-safe
    UNIQUE(order_num)
);

-- =====================================================
-- COLD ART MANIFEST (from cold_art_manifest.schema.json)
-- =====================================================

CREATE TABLE IF NOT EXISTS cold_art_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL,  -- e.g., 'anchor001', 'cover', 'icon'
    type TEXT NOT NULL,  -- 'plate', 'cover', 'icon', etc.
    filename TEXT NOT NULL,  -- e.g., 'anchor001__plate__v1.png'

    -- File properties
    sha256 TEXT NOT NULL,  -- 64-char hex
    size_bytes INTEGER NOT NULL,
    width_px INTEGER NOT NULL,
    height_px INTEGER NOT NULL,
    format TEXT NOT NULL,  -- 'PNG', 'JPG', 'SVG', etc.

    -- Approval
    approved_at TEXT NOT NULL,  -- ISO 8601
    approved_by TEXT NOT NULL,  -- Role abbreviation

    -- Provenance (stored as JSON for flexibility)
    provenance TEXT NOT NULL,  -- JSON with role, prompt_snippet, version, policy_notes, source

    CHECK (length(sha256) = 64),
    CHECK (size_bytes >= 1),
    CHECK (width_px >= 1),
    CHECK (height_px >= 1),
    CHECK (format IN ('PNG', 'JPG', 'JPEG', 'SVG', 'WEBP')),
    CHECK (approved_by IN ('SR', 'GK', 'AD', 'IL', 'AuD', 'AuP', 'TR', 'BB', 'PN', 'SS', 'ST', 'LW', 'CC', 'RS', 'PW')),
    UNIQUE(anchor, type)
);

-- =====================================================
-- HOT MANIFEST (from hot_manifest.schema.json)
-- =====================================================

CREATE TABLE IF NOT EXISTS hot_manifest (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Singleton
    schema_version TEXT NOT NULL DEFAULT 'https://questfoundry.liesdonk.nl/schemas/hot_manifest.schema.json',
    version TEXT NOT NULL DEFAULT '1.0.0',
    snapshot_at TEXT NOT NULL,  -- ISO 8601
    snapshot_id TEXT NOT NULL,  -- e.g., 'hot-20251124-143000'
    cold_reference TEXT,  -- Reference to Cold snapshot this is based on

    CHECK (snapshot_id GLOB 'hot-*' OR snapshot_id = 'hot-init')
);

-- Hot artifacts stored as JSON documents
CREATE TABLE IF NOT EXISTS hot_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL UNIQUE,  -- e.g., 'TU-2025-11-24-SR01'
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,  -- e.g., 'hot/tu_briefs/TU-2025-11-24-SR01.json'
    status TEXT NOT NULL,
    content TEXT NOT NULL,  -- JSON document

    -- Metadata
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),

    CHECK (artifact_type IN (
        'tu_brief', 'hook_card', 'research_memo', 'canon_pack',
        'style_addendum', 'art_plan', 'audio_plan', 'gatecheck_report',
        'view_log', 'language_pack', 'shotlist', 'cuelist',
        'edit_notes', 'pn_playtest_notes'
    )),
    CHECK (status IN ('proposed', 'in-progress', 'stabilizing', 'gatecheck', 'resolved'))
);

-- Section references in Hot
CREATE TABLE IF NOT EXISTS hot_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,
    title TEXT,
    text_file TEXT NOT NULL,  -- e.g., 'hot/sections/draft-001.md'
    status TEXT NOT NULL,
    tu_id TEXT,  -- Reference to TU working on it

    CHECK (status IN ('draft', 'revising', 'stabilizing', 'gatecheck', 'approved'))
);

-- Asset references in Hot
CREATE TABLE IF NOT EXISTS hot_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL,
    type TEXT NOT NULL,
    filename TEXT,  -- Proposed filename (may change)
    path TEXT NOT NULL,  -- e.g., 'hot/assets/proposed-cover.png'
    status TEXT NOT NULL,
    art_plan_id TEXT,  -- Reference to art plan

    CHECK (type IN ('plate', 'cover', 'icon', 'logo', 'ornament', 'diagram', 'audio')),
    CHECK (status IN ('proposed', 'in-review', 'approved', 'rejected'))
);

-- =====================================================
-- TRACE UNITS (from tu_brief.schema.json)
-- =====================================================

CREATE TABLE IF NOT EXISTS trace_units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tu_id TEXT NOT NULL UNIQUE,  -- e.g., 'TU-2025-11-24-SR01'

    -- Core fields from schema
    opened TEXT NOT NULL,  -- YYYY-MM-DD
    owner_a TEXT NOT NULL,  -- Accountable role
    responsible_r TEXT NOT NULL,  -- JSON array of responsible roles
    loop TEXT NOT NULL,
    slice TEXT NOT NULL,  -- Player-safe scope description
    snapshot_context TEXT,  -- e.g., 'Cold @ 2025-11-24'

    -- Role management
    awake TEXT NOT NULL,  -- JSON array of active roles
    dormant TEXT NOT NULL,  -- JSON array of inactive roles
    deferral_tags TEXT,  -- JSON array

    -- Quality bars
    press TEXT NOT NULL,  -- JSON array of bars to flip green
    monitor TEXT,  -- JSON array of bars to monitor
    pre_gate_risks TEXT,  -- JSON array

    -- Deliverables
    inputs TEXT,  -- JSON array of prerequisites
    deliverables TEXT NOT NULL,  -- JSON array of exit artifacts
    bars_green TEXT,  -- JSON array of bars that must be green

    -- Execution
    merge_view TEXT,  -- Merge decision notes
    timebox TEXT,  -- e.g., '45 min', '90 min'
    gatecheck TEXT,  -- Pass/fail criteria
    linkage TEXT,  -- Hooks filed, snapshot impact

    -- Lifecycle tracking
    lifecycle_stage TEXT NOT NULL DEFAULT 'hot-proposed',
    stabilized_at TEXT,
    gatecheck_at TEXT,
    merged_at TEXT,

    CHECK (tu_id GLOB 'TU-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-*'),
    CHECK (lifecycle_stage IN ('hot-proposed', 'stabilizing', 'gatecheck', 'cold-merged', 'archived')),
    CHECK (loop IN (
        'Story Spark', 'Hook Harvest', 'Lore Deepening', 'Codex Expansion',
        'Style Tune-up', 'Art Touch-up', 'Audio Pass', 'Translation Pass',
        'Binding Run', 'Narration Dry-Run', 'Gatecheck', 'Post-Mortem',
        'Archive Snapshot'
    ))
);

-- =====================================================
-- QUALITY CHECKS
-- =====================================================

CREATE TABLE IF NOT EXISTS quality_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tu_id TEXT NOT NULL REFERENCES trace_units(tu_id),

    bar_name TEXT NOT NULL,
    status TEXT NOT NULL,
    feedback TEXT,
    checked_by TEXT NOT NULL,
    checked_at TEXT NOT NULL,

    CHECK (bar_name IN (
        'Integrity', 'Reachability', 'Nonlinearity', 'Gateways',
        'Style', 'Determinism', 'Presentation', 'Accessibility'
    )),
    CHECK (status IN ('green', 'yellow', 'red', 'not_checked'))
);

-- =====================================================
-- VIEW LOGS (from view_log.schema.json)
-- =====================================================

CREATE TABLE IF NOT EXISTS view_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Core fields
    title TEXT NOT NULL,
    bound TEXT NOT NULL,  -- YYYY-MM-DD
    binder TEXT NOT NULL,
    tu TEXT NOT NULL,  -- TU reference
    cold_snapshot TEXT NOT NULL,

    -- Export configuration
    targets TEXT NOT NULL,  -- JSON array ['PDF', 'HTML', 'EPUB']
    options_and_coverage TEXT,
    dormancy TEXT,  -- JSON array of deferred items

    -- Quality status
    anchor_map TEXT,  -- Anchor integrity report
    presentation_status TEXT NOT NULL,
    presentation_notes TEXT,
    accessibility_status TEXT NOT NULL,
    accessibility_notes TEXT,

    -- Approval
    gatekeeper TEXT NOT NULL,
    gatecheck_id TEXT,

    -- Export artifacts (stored as JSON)
    export_artifacts TEXT NOT NULL,  -- JSON array of {kind, path, hash, notes}

    CHECK (presentation_status IN ('green', 'yellow', 'red')),
    CHECK (accessibility_status IN ('green', 'yellow', 'red'))
);

-- =====================================================
-- EVENTS (Audit trail)
-- =====================================================

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload TEXT NOT NULL,  -- JSON
    actor_role TEXT NOT NULL,
    occurred_at TEXT NOT NULL DEFAULT (datetime('now')),

    CHECK (event_type IN (
        'tu_opened', 'tu_merged', 'artifact_created', 'artifact_frozen',
        'hook_raised', 'hook_resolved', 'section_approved', 'asset_approved',
        'quality_check', 'cold_snapshot', 'view_exported'
    ))
);

-- =====================================================
-- VIEWS FOR JSON GENERATION
-- =====================================================

-- Generate cold_manifest.json
CREATE VIEW IF NOT EXISTS cold_manifest_json AS
SELECT
    json_object(
        '$schema', 'https://questfoundry.liesdonk.nl/schemas/cold_manifest.schema.json',
        'version', cm.version,
        'created_at', cm.created_at,
        'snapshot_id', cm.snapshot_id,
        'files', (
            SELECT json_group_array(
                json_object(
                    'path', cmf.path,
                    'sha256', cmf.sha256,
                    'size_bytes', cmf.size_bytes
                )
            )
            FROM cold_manifest_files cmf
            WHERE cmf.manifest_id = cm.id
        )
    ) as manifest_json
FROM cold_manifest cm
WHERE cm.id = 1;

-- Generate cold_book.json
CREATE VIEW IF NOT EXISTS cold_book_json AS
SELECT
    json_object(
        '$schema', 'https://questfoundry.liesdonk.nl/schemas/cold_book.schema.json',
        'version', '1.0.0',
        'metadata', json_object(
            'title', cbm.title,
            'subtitle', cbm.subtitle,
            'language', cbm.language,
            'author', cbm.author,
            'isbn', cbm.isbn,
            'published_at', cbm.published_at,
            'edition', cbm.edition,
            'copyright', cbm.copyright,
            'publisher', cbm.publisher
        ),
        'sections', (
            SELECT json_group_array(
                json_object(
                    'anchor', cbs.anchor,
                    'title', cbs.title,
                    'text_file', cbs.text_file,
                    'order', cbs.order_num,
                    'player_safe', cbs.player_safe = 1,
                    'requires_gate', cbs.requires_gate = 1
                )
            )
            FROM cold_book_sections cbs
            ORDER BY cbs.order_num
        ),
        'start_section', cbm.start_section
    ) as book_json
FROM cold_book_metadata cbm
WHERE cbm.id = 1;

-- Generate cold_art_manifest.json
CREATE VIEW IF NOT EXISTS cold_art_manifest_json AS
SELECT
    json_object(
        '$schema', 'https://questfoundry.liesdonk.nl/schemas/cold_art_manifest.schema.json',
        'version', '1.0.0',
        'assets', (
            SELECT json_group_array(
                json_object(
                    'anchor', anchor,
                    'type', type,
                    'filename', filename,
                    'sha256', sha256,
                    'size_bytes', size_bytes,
                    'width_px', width_px,
                    'height_px', height_px,
                    'format', format,
                    'approved_at', approved_at,
                    'approved_by', approved_by,
                    'provenance', json(provenance)
                )
            )
            FROM cold_art_assets
        )
    ) as art_json;

-- Generate hot_manifest.json
CREATE VIEW IF NOT EXISTS hot_manifest_json AS
SELECT
    json_object(
        '$schema', 'https://questfoundry.liesdonk.nl/schemas/hot_manifest.schema.json',
        'version', hm.version,
        'snapshot_at', hm.snapshot_at,
        'snapshot_id', hm.snapshot_id,
        'cold_reference', hm.cold_reference,
        'trace_units', (
            SELECT json_group_array(
                json_object(
                    'id', ha.artifact_id,
                    'path', ha.path,
                    'status', ha.status,
                    'artifact_type', ha.artifact_type
                )
            )
            FROM hot_artifacts ha
            WHERE ha.artifact_type = 'tu_brief'
        ),
        'hooks', (
            SELECT json_group_array(
                json_object(
                    'id', ha.artifact_id,
                    'path', ha.path,
                    'status', ha.status,
                    'artifact_type', ha.artifact_type
                )
            )
            FROM hot_artifacts ha
            WHERE ha.artifact_type = 'hook_card'
        ),
        'sections', (
            SELECT json_group_array(
                json_object(
                    'anchor', hs.anchor,
                    'title', hs.title,
                    'text_file', hs.text_file,
                    'status', hs.status,
                    'tu_id', hs.tu_id
                )
            )
            FROM hot_sections hs
        ),
        'proposed_assets', (
            SELECT json_group_array(
                json_object(
                    'anchor', ha2.anchor,
                    'type', ha2.type,
                    'filename', ha2.filename,
                    'path', ha2.path,
                    'status', ha2.status,
                    'art_plan_id', ha2.art_plan_id
                )
            )
            FROM hot_assets ha2
        )
    ) as manifest_json
FROM hot_manifest hm
WHERE hm.id = 1;

-- =====================================================
-- TRIGGERS FOR IMMUTABILITY
-- =====================================================

-- Prevent modification of Cold manifest (except initial setup)
CREATE TRIGGER IF NOT EXISTS prevent_cold_manifest_update
BEFORE UPDATE ON cold_manifest
WHEN OLD.snapshot_id != 'cold-init'
BEGIN
    SELECT RAISE(ABORT, 'Cold manifest is immutable after initial setup');
END;

-- Prevent deletion of Cold files
CREATE TRIGGER IF NOT EXISTS prevent_cold_file_delete
BEFORE DELETE ON cold_manifest_files
BEGIN
    SELECT RAISE(ABORT, 'Cold manifest files cannot be deleted');
END;

-- Prevent modification of merged TUs
CREATE TRIGGER IF NOT EXISTS prevent_merged_tu_update
BEFORE UPDATE ON trace_units
WHEN OLD.lifecycle_stage = 'cold-merged' AND NEW.lifecycle_stage != 'archived'
BEGIN
    SELECT RAISE(ABORT, 'Cannot modify merged trace unit');
END;

-- Log all TU state changes
CREATE TRIGGER IF NOT EXISTS log_tu_lifecycle
AFTER UPDATE OF lifecycle_stage ON trace_units
BEGIN
    INSERT INTO events (event_type, entity_type, entity_id, payload, actor_role)
    VALUES (
        CASE NEW.lifecycle_stage
            WHEN 'cold-merged' THEN 'tu_merged'
            ELSE 'tu_opened'
        END,
        'trace_unit',
        NEW.tu_id,
        json_object('old_stage', OLD.lifecycle_stage, 'new_stage', NEW.lifecycle_stage),
        'system'
    );
END;

-- Log artifact creation
CREATE TRIGGER IF NOT EXISTS log_artifact_created
AFTER INSERT ON hot_artifacts
BEGIN
    INSERT INTO events (event_type, entity_type, entity_id, payload, actor_role)
    VALUES (
        'artifact_created',
        'hot_artifact',
        NEW.artifact_id,
        json_object('type', NEW.artifact_type, 'status', NEW.status),
        'system'
    );
END;

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_cold_manifest_files_path ON cold_manifest_files(path);
CREATE INDEX IF NOT EXISTS idx_cold_book_sections_anchor ON cold_book_sections(anchor);
CREATE INDEX IF NOT EXISTS idx_cold_book_sections_order ON cold_book_sections(order_num);
CREATE INDEX IF NOT EXISTS idx_cold_art_assets_anchor ON cold_art_assets(anchor);
CREATE INDEX IF NOT EXISTS idx_hot_artifacts_type ON hot_artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_hot_artifacts_status ON hot_artifacts(status);
CREATE INDEX IF NOT EXISTS idx_trace_units_lifecycle ON trace_units(lifecycle_stage);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_entity ON events(entity_type, entity_id);
