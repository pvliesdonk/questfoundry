# QuestFoundry WebUI Implementation Guide

This guide provides step-by-step instructions for implementing the multi-tenant WebUI for QuestFoundry based on the specification in [the gist](https://gist.github.com/pvliesdonk/785372a19d3bee0fdcb6aceb4998e7ad).

## Overview

The WebUI consists of three tiers:

1. **Storage Backends** (PostgresStore, ValkeyStore) - Multi-tenant storage
2. **API Server** (FastAPI) - Tenancy layer, BYOK, locking
3. **PWA Frontend** (React) - Mobile-first UI

## Implementation Phases

### Phase 1: Storage Backends (CRITICAL PATH)

The storage backends are the foundation. They must be fully implemented before the API can function.

#### 1.1 PostgresStore Implementation

File: `webui/api/src/webui_api/storage/postgres_store.py`

**Current State**: Skeleton with NotImplementedError stubs

**Implementation Steps**:

1. **Connection Pooling**: Replace single connection with psycopg_pool
   ```python
   from psycopg_pool import ConnectionPool
   
   self.pool = ConnectionPool(
       connection_string,
       min_size=2,
       max_size=10
   )
   ```

2. **Implement get_project_info()**:
   ```python
   def get_project_info(self) -> ProjectInfo:
       with self.pool.connection() as conn:
           with conn.cursor() as cur:
               cur.execute(
                   "SELECT name, description, version, author, created, modified, metadata "
                   "FROM project_info WHERE project_id = %s",
                   (self.project_id,)
               )
               row = cur.fetchone()
               if not row:
                   raise FileNotFoundError(f"Project {self.project_id} not found")
               return ProjectInfo(
                   name=row[0],
                   description=row[1],
                   version=row[2],
                   author=row[3],
                   created=row[4],
                   modified=row[5],
                   metadata=row[6]
               )
   ```

3. **Implement save_artifact()**: Use UPSERT with JSONB
   ```python
   def save_artifact(self, artifact: Artifact) -> None:
       artifact_id = artifact.metadata.get("id")
       if not artifact_id:
           raise ValueError("Artifact must have 'id' in metadata")
       
       with self.pool.connection() as conn:
           with conn.cursor() as cur:
               cur.execute(
                   """
                   INSERT INTO artifacts 
                       (project_id, artifact_id, artifact_type, data, metadata)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (project_id, artifact_id) 
                   DO UPDATE SET 
                       artifact_type = EXCLUDED.artifact_type,
                       data = EXCLUDED.data,
                       metadata = EXCLUDED.metadata,
                       modified = NOW()
                   """,
                   (self.project_id, artifact_id, artifact.type, 
                    Json(artifact.data), Json(artifact.metadata))
               )
               conn.commit()
   ```

4. **Implement list_artifacts()**: Use JSONB querying
   ```python
   def list_artifacts(
       self, artifact_type: str | None = None, filters: dict[str, Any] | None = None
   ) -> list[Artifact]:
       query = """
           SELECT artifact_id, artifact_type, data, metadata
           FROM artifacts
           WHERE project_id = %s
       """
       params: list[Any] = [self.project_id]
       
       if artifact_type:
           query += " AND artifact_type = %s"
           params.append(artifact_type)
       
       # Add JSONB filters
       # Note: JSONB keys should be validated/whitelisted before use
       # to prevent SQL injection. Only accept known safe keys.
       if filters:
           for key, value in filters.items():
               # Use parameterized query for both key and value
               query += " AND data->>%s = %s"
               params.append(key)
               params.append(str(value))
       
       query += " ORDER BY modified DESC"
       
       with self.pool.connection() as conn:
           with conn.cursor() as cur:
               cur.execute(query, params)
               return [
                   Artifact(
                       type=row[1],
                       data=row[2],
                       metadata=row[3]
                   )
                   for row in cur.fetchall()
               ]
   ```

5. **Implement remaining methods**: Follow same pattern for TUs, snapshots

6. **Testing**: Create tests in `webui/api/tests/test_postgres_store.py`

#### 1.2 ValkeyStore Implementation

File: `webui/api/src/webui_api/storage/valkey_store.py`

**Current State**: Skeleton with NotImplementedError stubs

**Implementation Steps**:

1. **Key Design**: Use hierarchical namespacing
   ```python
   def _key(self, *parts: str) -> str:
       """Generate namespaced key: hot:{project_id}:type:id"""
       return f"hot:{self.project_id}:{':'.join(parts)}"
   ```

2. **Implement save_artifact()**:
   ```python
   def save_artifact(self, artifact: Artifact) -> None:
       artifact_id = artifact.metadata.get("id")
       if not artifact_id:
           raise ValueError("Artifact must have 'id' in metadata")
       
       key = self._key("artifacts", artifact.type, artifact_id)
       data = {
           "type": artifact.type,
           "data": artifact.data,
           "metadata": artifact.metadata
       }
       
       # Store as JSON with TTL
       self.client.setex(
           key,
           self.ttl_seconds,
           json.dumps(data, default=str)
       )
   ```

3. **Implement list_artifacts()**: Use SCAN pattern
   ```python
   def list_artifacts(
       self, artifact_type: str | None = None, filters: dict[str, Any] | None = None
   ) -> list[Artifact]:
       pattern = self._key("artifacts", artifact_type or "*", "*")
       artifacts = []
       
       for key in self.client.scan_iter(match=pattern):
           data_str = self.client.get(key)
           if data_str:
               data = json.loads(data_str)
               # Apply filters
               if filters:
                   match = all(
                       data["data"].get(k) == v 
                       for k, v in filters.items()
                   )
                   if not match:
                       continue
               
               artifacts.append(Artifact(
                   type=data["type"],
                   data=data["data"],
                   metadata=data["metadata"]
               ))
       
       return artifacts
   ```

4. **Testing**: Create tests in `webui/api/tests/test_valkey_store.py`

### Phase 2: API Server Core

#### 2.1 Authentication Middleware

File: `webui/api/src/webui_api/middleware/auth.py`

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract user from X-Forwarded-User header
        user_id = request.headers.get("X-Forwarded-User")
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Missing X-Forwarded-User header. Must run behind OIDC proxy."
            )
        
        # Store user_id in request state
        request.state.user_id = user_id
        
        response = await call_next(request)
        return response
```

#### 2.2 Locking Mechanism

File: `webui/api/src/webui_api/locking.py`

```python
import redis
from contextlib import contextmanager
from fastapi import HTTPException

class ProjectLock:
    def __init__(self, redis_client: redis.Redis, timeout: int = 300):
        self.client = redis_client
        self.timeout = timeout
    
    @contextmanager
    def acquire(self, project_id: str, user_id: str):
        """Acquire lock for project"""
        lock_key = f"lock:project:{project_id}"
        
        # Try to acquire lock
        acquired = self.client.set(
            lock_key,
            user_id,
            nx=True,  # Only set if not exists
            ex=self.timeout  # Expire after timeout
        )
        
        if not acquired:
            # Check who owns the lock
            owner = self.client.get(lock_key)
            if owner and owner.decode() == user_id:
                # Same user already has lock, allow
                pass
            else:
                raise HTTPException(
                    status_code=423,
                    detail=f"Project {project_id} is locked by another user"
                )
        
        try:
            yield
        finally:
            # Release lock
            current_owner = self.client.get(lock_key)
            if current_owner and current_owner.decode() == user_id:
                self.client.delete(lock_key)
```

#### 2.3 Core Request Lifecycle

File: `webui/api/src/webui_api/lifecycle.py`

```python
from contextlib import contextmanager
from pathlib import Path

from questfoundry.orchestrator import Orchestrator
from questfoundry.providers.config import ProviderConfig
from questfoundry.providers.registry import ProviderRegistry
from questfoundry.roles.registry import RoleRegistry
from questfoundry.state.workspace import WorkspaceManager

from .storage import PostgresStore, ValkeyStore
from .locking import ProjectLock
from .config import settings

@contextmanager
def orchestrator_context(project_id: str, user_id: str, user_provider_config: ProviderConfig):
    """
    Context manager for orchestrator lifecycle.
    
    Implements the mandated request lifecycle:
    1. Acquire lock
    2. Instantiate storage backends
    3. Instantiate library components
    4. Yield orchestrator
    5. Release lock (automatic via context manager)
    """
    lock = ProjectLock(redis_client, settings.lock_timeout)
    
    with lock.acquire(project_id, user_id):
        # Instantiate storage backends
        cold_store = PostgresStore(settings.postgres_url, project_id)
        hot_store = ValkeyStore(settings.redis_url, project_id)
        
        try:
            # Instantiate library components
            provider_reg = ProviderRegistry(config=user_provider_config)
            role_reg = RoleRegistry(provider_reg, spec_path=Path(settings.spec_path))
            workspace = WorkspaceManager(cold=cold_store, hot=hot_store)
            
            # Create orchestrator
            orchestrator = Orchestrator(
                workspace=workspace,
                provider_registry=provider_reg,
                role_registry=role_reg,
                spec_path=Path(settings.spec_path)
            )
            
            yield orchestrator
            
        finally:
            # Cleanup connections
            cold_store.close()
            hot_store.close()
```

#### 2.4 API Endpoints

File: `webui/api/src/webui_api/routers/execution.py`

```python
from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel

from ..lifecycle import orchestrator_context
from ..user_settings import get_user_provider_config

router = APIRouter(prefix="/projects/{project_id}", tags=["execution"])

class GoalRequest(BaseModel):
    goal: str
    context: dict | None = None

@router.post("/execute")
async def execute_goal(
    project_id: str,
    request: Request,
    goal_request: GoalRequest
):
    """
    Execute a goal using the orchestrator.
    
    This is the main entry point for QuestFoundry operations.
    """
    user_id = request.state.user_id
    
    # Get user's provider config (decrypted BYOK keys)
    provider_config = await get_user_provider_config(user_id)
    
    # Use orchestrator context (handles locking, storage, lifecycle)
    with orchestrator_context(project_id, user_id, provider_config) as orchestrator:
        # Execute goal
        result = orchestrator.execute_goal(
            goal=goal_request.goal,
            context=goal_request.context or {}
        )
        
        return {
            "status": "success",
            "result": result
        }
```

### Phase 3: Tenancy Schema & BYOK

#### 3.1 User Settings Management

File: `webui/api/src/webui_api/user_settings.py`

```python
from cryptography.fernet import Fernet
import psycopg
from questfoundry.providers.config import ProviderConfig

from .config import settings

def encrypt_keys(provider_config: ProviderConfig) -> bytes:
    """Encrypt provider keys using Fernet"""
    f = Fernet(settings.encryption_key.encode())
    data = provider_config.model_dump_json()
    return f.encrypt(data.encode())

def decrypt_keys(encrypted: bytes) -> ProviderConfig:
    """Decrypt provider keys"""
    f = Fernet(settings.encryption_key.encode())
    data = f.decrypt(encrypted).decode()
    return ProviderConfig.model_validate_json(data)

async def get_user_provider_config(user_id: str) -> ProviderConfig:
    """Get user's decrypted provider configuration"""
    with psycopg.connect(settings.postgres_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT encrypted_keys FROM user_settings WHERE user_id = %s",
                (user_id,)
            )
            row = cur.fetchone()
            if not row:
                # Return default config (no BYOK)
                return ProviderConfig()
            
            return decrypt_keys(row[0])

async def save_user_provider_config(user_id: str, config: ProviderConfig) -> None:
    """Save user's encrypted provider configuration"""
    encrypted = encrypt_keys(config)
    
    with psycopg.connect(settings.postgres_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_settings (user_id, encrypted_keys)
                VALUES (%s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET 
                    encrypted_keys = EXCLUDED.encrypted_keys,
                    updated_at = NOW()
                """,
                (user_id, encrypted)
            )
            conn.commit()
```

### Phase 4: CI/CD Workflows

#### 4.1 WebUI CI Workflow

File: `.github/workflows/webui-ci.yml`

```yaml
name: WebUI CI

on:
  push:
    paths:
      - 'webui/**'
  pull_request:
    paths:
      - 'webui/**'

jobs:
  lint-and-test-api:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: webui/api
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: uv sync
      
      - name: Lint
        run: |
          uv run ruff check src tests
          uv run mypy src
      
      - name: Test
        run: uv run pytest

  lint-and-test-pwa:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: webui/pwa
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: webui/pwa/package-lock.json
      
      - name: Install dependencies
        run: npm ci
      
      - name: Lint
        run: npm run lint
      
      - name: Build
        run: npm run build
```

#### 4.2 WebUI Publish Workflow

File: `.github/workflows/publish-webui.yml`

```yaml
name: Publish WebUI

on:
  push:
    tags:
      - 'v*'

env:
  REGISTRY: ghcr.io
  IMAGE_PREFIX: ${{ github.repository }}

jobs:
  publish-api:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/webui-api
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: webui/api
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  publish-pwa:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/webui-pwa
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: webui/pwa
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

### Phase 5: PWA Implementation

The PWA implementation is beyond the scope of this initial scaffolding. Key files to create:

1. `webui/pwa/src/main.tsx` - React entry point
2. `webui/pwa/src/App.tsx` - Main application component
3. `webui/pwa/src/pages/` - Page components
   - `ProjectList.tsx`
   - `ProjectDetail.tsx`
   - `HotWorkspace.tsx`
   - `ColdStorage.tsx`
   - `GoalInput.tsx`
   - `Settings.tsx`
4. `webui/pwa/src/api/` - API client
5. `webui/pwa/vite.config.ts` - Vite configuration with PWA plugin

## Testing Strategy

### Unit Tests

- Storage backends: Test all StateStore methods
- Locking: Test acquire/release scenarios
- Encryption: Test encrypt/decrypt roundtrip
- Lifecycle: Test orchestrator instantiation

### Integration Tests

- API endpoints: Test full request/response cycle
- Docker compose: Test complete stack deployment

### E2E Tests

- PWA workflows: Test user journeys

## Deployment

1. Generate Fernet key:
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

2. Update docker-compose.yml with real passwords and keys

3. Deploy:
   ```bash
   cd webui
   docker-compose up -d
   ```

4. Initialize database:
   ```bash
   docker-compose exec postgres psql -U questfoundry -d questfoundry -f /docker-entrypoint-initdb.d/01-schema.sql
   ```

## Next Steps

1. Implement PostgresStore (Phase 1.1)
2. Implement ValkeyStore (Phase 1.2)
3. Write unit tests for storage backends
4. Implement API authentication and locking (Phase 2.1-2.2)
5. Implement core lifecycle (Phase 2.3)
6. Implement execution endpoint (Phase 2.4)
7. Implement BYOK management (Phase 3)
8. Create CI/CD workflows (Phase 4)
9. Implement PWA (Phase 5)
10. End-to-end testing

## Completion Criteria

- [ ] PostgresStore passes all tests
- [ ] ValkeyStore passes all tests
- [ ] API server starts and handles health check
- [ ] Authentication middleware works
- [ ] Locking prevents concurrent writes
- [ ] Execute endpoint works with test project
- [ ] BYOK encryption/decryption works
- [ ] CI workflows pass
- [ ] Docker images publish to GHCR
- [ ] PWA builds and runs
- [ ] Full stack deploys via docker-compose
