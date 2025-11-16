# QuestFoundry WebUI API

Multi-tenant REST API server for QuestFoundry, providing web access to the questfoundry-py library.

## Architecture

This API server acts as a **tenancy layer** that wraps the core questfoundry-py library:

- **Authentication**: Trusts `X-Forwarded-User` header from OIDC reverse proxy
- **Multi-tenancy**: Isolates projects and user data via project-scoped storage backends
- **BYOK**: Stores encrypted provider keys per user
- **Locking**: Prevents concurrent writes to same project via Redis/Valkey

## Key Principles

1. **Stateless API**: No session state in the API server
2. **Tenancy Isolation**: The core `questfoundry-py` library remains user-agnostic
3. **Request Lifecycle**: Every request instantiates fresh library objects with user-specific config
4. **Storage Backends**: New PostgresStore and ValkeyStore replace SQLite and FileStore

## Core Request Lifecycle

All endpoints that interact with QuestFoundry follow this pattern:

1. Extract `user-id` from `X-Forwarded-User` header
2. Acquire Redis lock for the project (e.g., `lock:project-123`)
3. Fetch user's decrypted provider keys from PostgreSQL
4. Build in-memory `ProviderConfig` with user's keys
5. Instantiate storage backends with project_id scoping:
   - `PostgresStore(project_id="project-123")` - Cold storage
   - `ValkeyStore(project_id="project-123")` - Hot storage
6. Instantiate library components:
   - `ProviderRegistry(config=user_provider_config)`
   - `RoleRegistry(provider_registry, spec_path=...)`
   - `Orchestrator(workspace=WorkspaceManager(cold=postgres_store, hot=valkey_store))`
7. Execute library method (e.g., `orchestrator.execute_goal(...)`)
8. Release Redis lock
9. Return JSON response
10. Discard all library instances (no persistence between requests)

## Storage Backends

### PostgresStore (Cold Storage)

Implements the `StateStore` protocol from questfoundry-py.

- Replaces `SQLiteStore` for multi-tenant concurrent access
- Uses `project_id` column to scope all queries
- Stores artifact data in JSONB columns for efficient querying
- Provides ACID transactions and better concurrency

### ValkeyStore (Hot Storage)

Implements the `StateStore` protocol for hot workspace.

- Replaces `FileStore` for multi-tenant concurrent access
- Uses key namespacing: `hot:{project_id}:path/to/artifact.json`
- Ephemeral storage with TTL (e.g., 24h)
- Fast in-memory operations for working artifacts

## Tenancy Schema

The API server maintains its own tables (separate from QuestFoundry artifacts):

### user_settings

```sql
CREATE TABLE user_settings (
    user_id TEXT PRIMARY KEY,
    encrypted_keys BYTEA NOT NULL,  -- Encrypted provider API keys
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### project_ownership

```sql
CREATE TABLE project_ownership (
    project_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL REFERENCES user_settings(user_id),
    project_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## Deployment

See `docker-compose.yml` in the parent directory for a complete deployment example with:

- FastAPI API server
- PostgreSQL database
- Valkey/Redis cache
- OIDC reverse proxy (Traefik + Authelia)
- PWA frontend

## Development

```bash
# Install dependencies
cd webui/api
uv sync

# Run server
uv run uvicorn webui_api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
uv run pytest

# Lint
uv run ruff check .
uv run mypy src
```

## API Endpoints

### Projects

- `POST /projects` - Create new project
- `GET /projects` - List user's projects
- `GET /projects/{project_id}` - Get project details
- `DELETE /projects/{project_id}` - Delete project

### Artifacts

- `POST /projects/{project_id}/artifacts` - Create artifact
- `GET /projects/{project_id}/artifacts` - List artifacts
- `GET /projects/{project_id}/artifacts/{artifact_id}` - Get artifact
- `PUT /projects/{project_id}/artifacts/{artifact_id}` - Update artifact
- `DELETE /projects/{project_id}/artifacts/{artifact_id}` - Delete artifact

### Execution

- `POST /projects/{project_id}/execute` - Execute goal (main orchestrator endpoint)
- `POST /projects/{project_id}/gatecheck` - Run gatekeeper validation

### User Settings

- `GET /user/settings` - Get current user settings
- `PUT /user/settings/keys` - Update provider API keys (BYOK)

## Security Considerations

1. **Provider Keys**: Encrypted at rest using Fernet symmetric encryption
2. **Project Isolation**: All queries include project_id scope check
3. **Locking**: Prevents race conditions on concurrent writes
4. **Header Trust**: Assumes reverse proxy properly validates user identity
5. **No Direct Auth**: API must run behind OIDC proxy (Traefik + Authelia)

## Implementation Status

- [ ] Storage backends (PostgresStore, ValkeyStore)
- [ ] Tenancy schema and migrations
- [ ] Authentication middleware
- [ ] Locking mechanism
- [ ] Core request lifecycle
- [ ] Project endpoints
- [ ] Artifact endpoints
- [ ] Execution endpoints
- [ ] User settings endpoints
- [ ] Tests and documentation
