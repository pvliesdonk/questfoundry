# QuestFoundry WebUI

Multi-tenant web interface for QuestFoundry, enabling browser-based access to the collaborative interactive fiction authoring platform.

## Architecture

The WebUI implements a 3-tier architecture that isolates multi-tenancy concerns from the core questfoundry-py library:

```
┌─────────────────────────────────────────────┐
│  PWA (React)                                │
│  - Mobile-first UI                          │
│  - Hot/Cold SoT visualization               │
│  - Goal-driven workflow                     │
└─────────────┬───────────────────────────────┘
              │ REST API
              │
┌─────────────▼───────────────────────────────┐
│  API Server (FastAPI)                       │
│  - Tenancy layer                            │
│  - BYOK (Bring Your Own Key)                │
│  - Project locking                          │
│  - Wraps questfoundry-py                    │
└─────────────┬───────────────────────────────┘
              │
    ┌─────────┴──────────┐
    │                    │
┌───▼────────┐   ┌──────▼──────┐
│ PostgreSQL │   │ Valkey/Redis│
│ (Cold SoT) │   │ (Hot SoT)   │
└────────────┘   └─────────────┘
```

## Key Architectural Principles

1. **Library Remains User-Agnostic**: The core `questfoundry-py` library knows nothing about users, tenants, or BYOK
2. **Stateless API**: All user context is reconstructed on each request
3. **Request Lifecycle**: Every API call instantiates fresh library objects with user-specific configuration
4. **Project Locking**: Redis-based distributed locking prevents concurrent writes
5. **Multi-Tenant Storage**: PostgresStore and ValkeyStore replace SQLite and FileStore with project_id scoping

## Directory Structure

```
webui/
├── api/                        # FastAPI backend
│   ├── src/
│   │   └── webui_api/
│   │       ├── main.py         # FastAPI app
│   │       ├── config.py       # Settings
│   │       ├── storage/        # Storage backends
│   │       │   ├── postgres_store.py
│   │       │   └── valkey_store.py
│   │       ├── middleware/     # Auth, CORS
│   │       ├── routers/        # API endpoints
│   │       ├── locking.py      # Distributed locking
│   │       └── lifecycle.py    # Request lifecycle
│   ├── tests/
│   ├── schema.sql              # Database schema
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── README.md
│
├── pwa/                        # React PWA
│   ├── src/
│   ├── Dockerfile
│   ├── package.json
│   └── README.md
│
├── docker-compose.yml          # Example deployment
├── IMPLEMENTATION_GUIDE.md     # Detailed implementation steps
└── README.md                   # This file
```

## Core Request Lifecycle

Every API request that interacts with QuestFoundry follows this lifecycle:

1. **Extract User**: Get `user_id` from `X-Forwarded-User` header (set by OIDC proxy)
2. **Acquire Lock**: Atomically acquire Redis lock for project (`lock:project:{id}`)
3. **Fetch Config**: Get user's decrypted provider keys from PostgreSQL
4. **Build Config**: Create in-memory `ProviderConfig` with user's keys
5. **Instantiate Storage**: 
   - `PostgresStore(project_id="...")` for cold storage
   - `ValkeyStore(project_id="...")` for hot storage
6. **Instantiate Library**:
   - `ProviderRegistry(config=user_config)`
   - `RoleRegistry(provider_registry, ...)`
   - `Orchestrator(workspace=WorkspaceManager(cold=postgres, hot=valkey))`
7. **Execute**: Call library method (e.g., `orchestrator.execute_goal(...)`)
8. **Release Lock**: Delete Redis lock
9. **Return Result**: Send JSON response
10. **Discard**: All library objects go out of scope (no persistence)

This ensures:
- Complete isolation between users
- No shared state between requests
- User-specific provider keys per request
- Prevention of concurrent writes to same project

## Storage Backends

### PostgresStore (Cold Storage)

Replaces `SQLiteStore` for multi-tenant concurrent access.

- **Project Scoping**: All queries include `WHERE project_id = $1`
- **JSONB Columns**: Efficient artifact data storage and querying
- **Connection Pooling**: Safe for concurrent access
- **ACID Transactions**: Data integrity guaranteed

### ValkeyStore (Hot Storage)

Replaces `FileStore` for multi-tenant concurrent access.

- **Key Namespacing**: `hot:{project_id}:artifacts:{type}:{id}`
- **Ephemeral Storage**: TTL-based expiration (default 24h)
- **In-Memory**: Fast operations for working artifacts
- **Atomic Operations**: Redis guarantees

## Security

### Authentication

The API server trusts the `X-Forwarded-User` header set by an OIDC reverse proxy (Traefik + Authelia). The API **does not** implement authentication itself.

**Deployment Requirement**: API must run behind OIDC proxy in production.

### BYOK (Bring Your Own Key)

User provider keys (OpenAI, Anthropic, etc.) are:

1. Encrypted using Fernet symmetric encryption
2. Stored in PostgreSQL `user_settings` table
3. Decrypted on each request
4. Never logged or persisted in plaintext

### Project Locking

Redis/Valkey-based distributed locking:

- Prevents race conditions on concurrent writes
- Lock timeout: 5 minutes (configurable)
- Same user can re-acquire own lock
- Returns 423 Locked if another user holds lock

## Deployment

See `docker-compose.yml` for a complete example stack:

- PostgreSQL (cold storage, tenancy tables)
- Valkey/Redis (hot storage, locking)
- FastAPI API server
- React PWA (Nginx)
- Traefik (reverse proxy)
- Authelia (OIDC provider)

### Quick Start

1. Generate encryption key:
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

2. Update `docker-compose.yml` with real passwords and keys

3. Deploy:
   ```bash
   cd webui
   docker-compose up -d
   ```

4. Access:
   - PWA: http://localhost:3000
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Development

### API Server

```bash
cd webui/api
uv sync
uv run uvicorn webui_api.main:app --reload
```

### PWA

```bash
cd webui/pwa
npm install
npm run dev
```

## Implementation Status

This is scaffolding for the full implementation. See `IMPLEMENTATION_GUIDE.md` for detailed step-by-step instructions.

**Phase 1: Storage Backends**
- [ ] PostgresStore implementation
- [ ] ValkeyStore implementation
- [ ] Storage backend tests

**Phase 2: API Server**
- [ ] Authentication middleware
- [ ] Locking mechanism
- [ ] Core request lifecycle
- [ ] Project endpoints
- [ ] Artifact endpoints
- [ ] Execution endpoints
- [ ] User settings endpoints

**Phase 3: Tenancy**
- [ ] Database schema and migrations
- [ ] BYOK encryption/decryption
- [ ] User settings management

**Phase 4: CI/CD**
- [ ] webui-ci.yml workflow
- [ ] publish-webui.yml workflow
- [ ] GHCR image publishing

**Phase 5: PWA**
- [ ] React scaffolding
- [ ] Project management UI
- [ ] Hot/Cold SoT visualization
- [ ] Goal input interface
- [ ] Gatecheck workflow
- [ ] Settings/BYOK UI

## References

- **Specification Gist**: https://gist.github.com/pvliesdonk/785372a19d3bee0fdcb6aceb4998e7ad
- **questfoundry-py**: The core Python library (../lib/python)
- **Implementation Guide**: ./IMPLEMENTATION_GUIDE.md
- **API README**: ./api/README.md
- **PWA README**: ./pwa/README.md
