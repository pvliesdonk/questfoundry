# WebUI Implementation Checklist

Use this checklist to track progress on the WebUI implementation. Each checkbox represents a concrete deliverable.

## Phase 1: Storage Backends (Critical Path)

### PostgresStore Implementation
- [ ] Set up connection pooling (psycopg_pool)
- [ ] Implement `get_project_info()`
- [ ] Implement `save_project_info()`
- [ ] Implement `save_artifact()` with UPSERT
- [ ] Implement `get_artifact()`
- [ ] Implement `list_artifacts()` with JSONB filtering
- [ ] Implement `delete_artifact()`
- [ ] Implement `save_tu()`
- [ ] Implement `get_tu()`
- [ ] Implement `list_tus()`
- [ ] Implement `save_snapshot()`
- [ ] Implement `get_snapshot()`
- [ ] Implement `list_snapshots()`
- [ ] Write unit tests for all methods
- [ ] Test with real PostgreSQL database

### ValkeyStore Implementation
- [ ] Implement key namespacing helper
- [ ] Implement `get_project_info()`
- [ ] Implement `save_project_info()` with TTL
- [ ] Implement `save_artifact()` with TTL
- [ ] Implement `get_artifact()`
- [ ] Implement `list_artifacts()` with SCAN
- [ ] Implement `delete_artifact()`
- [ ] Implement `save_tu()` with TTL
- [ ] Implement `get_tu()`
- [ ] Implement `list_tus()`
- [ ] Implement `save_snapshot()` with TTL
- [ ] Implement `get_snapshot()`
- [ ] Implement `list_snapshots()`
- [ ] Write unit tests for all methods
- [ ] Test with real Redis/Valkey

## Phase 2: API Server Core

### Authentication
- [ ] Create `middleware/auth.py`
- [ ] Implement `AuthMiddleware` class
- [ ] Extract `X-Forwarded-User` header
- [ ] Store user_id in request state
- [ ] Handle missing header (401 error)
- [ ] Add middleware to FastAPI app
- [ ] Test authentication flow

### Locking
- [ ] Create `locking.py`
- [ ] Implement `ProjectLock` class
- [ ] Implement `acquire()` context manager
- [ ] Use Redis SET NX EX for atomic lock
- [ ] Handle lock conflicts (423 error)
- [ ] Allow same user to re-acquire lock
- [ ] Implement automatic lock release
- [ ] Test locking scenarios
- [ ] Test lock timeout

### Request Lifecycle
- [ ] Create `lifecycle.py`
- [ ] Implement `orchestrator_context()` context manager
- [ ] Integrate lock acquisition
- [ ] Instantiate storage backends
- [ ] Instantiate provider registry
- [ ] Instantiate role registry
- [ ] Instantiate workspace manager
- [ ] Create orchestrator instance
- [ ] Clean up connections on exit
- [ ] Test full lifecycle

### User Settings (BYOK)
- [ ] Create `user_settings.py`
- [ ] Implement `encrypt_keys()` function
- [ ] Implement `decrypt_keys()` function
- [ ] Implement `get_user_provider_config()`
- [ ] Implement `save_user_provider_config()`
- [ ] Generate Fernet key for deployment
- [ ] Test encryption/decryption roundtrip
- [ ] Test with real database

## Phase 3: API Endpoints

### Project Management
- [ ] Create `routers/projects.py`
- [ ] Implement `POST /projects` (create)
- [ ] Implement `GET /projects` (list)
- [ ] Implement `GET /projects/{id}` (get)
- [ ] Implement `DELETE /projects/{id}` (delete)
- [ ] Add authorization checks (owner only)
- [ ] Test all endpoints
- [ ] Document endpoints

### Artifact Operations
- [ ] Create `routers/artifacts.py`
- [ ] Implement `POST /projects/{id}/artifacts` (create)
- [ ] Implement `GET /projects/{id}/artifacts` (list)
- [ ] Implement `GET /projects/{id}/artifacts/{artifact_id}` (get)
- [ ] Implement `PUT /projects/{id}/artifacts/{artifact_id}` (update)
- [ ] Implement `DELETE /projects/{id}/artifacts/{artifact_id}` (delete)
- [ ] Support hot/cold storage selection
- [ ] Test all endpoints
- [ ] Document endpoints

### Execution
- [ ] Create `routers/execution.py`
- [ ] Implement `POST /projects/{id}/execute` (goal execution)
- [ ] Implement `POST /projects/{id}/gatecheck` (validation)
- [ ] Use orchestrator_context
- [ ] Handle execution errors
- [ ] Return structured results
- [ ] Test execution flow
- [ ] Document endpoints

### User Settings
- [ ] Create `routers/user_settings.py`
- [ ] Implement `GET /user/settings` (get)
- [ ] Implement `PUT /user/settings/keys` (update BYOK)
- [ ] Validate provider configs
- [ ] Test BYOK flow
- [ ] Document endpoints

### Integration
- [ ] Include all routers in main.py
- [ ] Configure CORS appropriately
- [ ] Add error handlers
- [ ] Add request logging
- [ ] Test complete API

## Phase 4: Database & Deployment

### Database
- [ ] Test schema.sql creates all tables
- [ ] Verify indexes are created
- [ ] Verify triggers work
- [ ] Create migration script (if needed)
- [ ] Test with docker-compose

### Docker
- [ ] Test API Dockerfile builds
- [ ] Test PWA Dockerfile builds
- [ ] Test docker-compose stack
- [ ] Verify health checks work
- [ ] Test inter-service communication
- [ ] Document deployment

## Phase 5: PWA Implementation

### Project Setup
- [ ] Initialize React project with Vite
- [ ] Configure TypeScript
- [ ] Set up routing
- [ ] Configure PWA plugin
- [ ] Create manifest.json
- [ ] Test dev server

### API Client
- [ ] Create `src/api/client.ts`
- [ ] Implement fetch wrapper
- [ ] Handle authentication
- [ ] Handle errors
- [ ] Type API responses
- [ ] Test API client

### Core Components
- [ ] Create `App.tsx`
- [ ] Create `Layout.tsx`
- [ ] Create `Navigation.tsx`
- [ ] Create `ErrorBoundary.tsx`
- [ ] Test routing

### Project Management UI
- [ ] Create `ProjectList.tsx`
- [ ] Create `ProjectCard.tsx`
- [ ] Create `ProjectCreate.tsx`
- [ ] Create `ProjectDetail.tsx`
- [ ] Test project workflows

### Hot/Cold SoT UI
- [ ] Create `HotWorkspace.tsx`
- [ ] Create `ColdStorage.tsx`
- [ ] Create `ArtifactList.tsx`
- [ ] Create `ArtifactCard.tsx`
- [ ] Create `ArtifactViewer.tsx`
- [ ] Test artifact viewing

### Goal Execution UI
- [ ] Create `GoalInput.tsx`
- [ ] Create `ExecutionProgress.tsx`
- [ ] Create `ExecutionResult.tsx`
- [ ] Test goal execution flow

### Gatecheck UI
- [ ] Create `GatecheckForm.tsx`
- [ ] Create `GatecheckResults.tsx`
- [ ] Create `QualityBar.tsx`
- [ ] Test gatecheck flow

### Settings UI
- [ ] Create `Settings.tsx`
- [ ] Create `ProviderKeyForm.tsx`
- [ ] Handle key encryption
- [ ] Test BYOK configuration

### Mobile Optimization
- [ ] Test on mobile devices
- [ ] Optimize touch targets
- [ ] Test offline capability
- [ ] Optimize performance
- [ ] Test PWA installation

## Phase 6: CI/CD

### CI Workflow
- [ ] Create `.github/workflows/webui-ci.yml`
- [ ] Configure API linting job
- [ ] Configure API testing job
- [ ] Configure PWA linting job
- [ ] Configure PWA build job
- [ ] Test workflow on push
- [ ] Test workflow on PR

### CD Workflow
- [ ] Create `.github/workflows/publish-webui.yml`
- [ ] Configure API image build
- [ ] Configure API image push to GHCR
- [ ] Configure PWA image build
- [ ] Configure PWA image push to GHCR
- [ ] Test on version tag
- [ ] Document release process

## Phase 7: Testing & Documentation

### Testing
- [ ] Unit tests: Storage backends (100% coverage)
- [ ] Unit tests: API endpoints
- [ ] Unit tests: Authentication
- [ ] Unit tests: Locking
- [ ] Integration tests: Full API workflows
- [ ] Integration tests: Docker stack
- [ ] E2E tests: PWA user flows
- [ ] Load testing: Concurrent requests
- [ ] Security testing: BYOK encryption

### Documentation
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Database schema documentation
- [ ] Deployment guide
- [ ] User guide for PWA
- [ ] Developer setup guide
- [ ] Architecture decision records
- [ ] Security considerations
- [ ] Troubleshooting guide

## Verification

### Smoke Tests
- [ ] API health check responds
- [ ] API docs accessible
- [ ] PWA loads in browser
- [ ] Can create project
- [ ] Can save artifact
- [ ] Can execute goal
- [ ] Can configure BYOK
- [ ] Locking prevents conflicts

### Integration Tests
- [ ] Full workflow: Create project → Add artifacts → Execute goal → Gatecheck
- [ ] Multi-user: Two users can't write to same project simultaneously
- [ ] BYOK: User keys are isolated and encrypted
- [ ] Hot/Cold: Artifacts move between hot and cold correctly

### Production Readiness
- [ ] All tests pass
- [ ] No security vulnerabilities
- [ ] Performance acceptable
- [ ] Documentation complete
- [ ] Docker images published
- [ ] Deployment tested
- [ ] Monitoring configured
- [ ] Logging configured

---

## Progress Tracking

**Current Phase**: Phase 1 - Storage Backends  
**Next Milestone**: PostgresStore fully implemented and tested  
**Blockers**: None  
**Notes**: Scaffolding complete, ready for implementation
