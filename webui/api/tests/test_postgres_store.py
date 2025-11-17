"""Unit tests for PostgresStore implementation

These tests validate the PostgresStore implementation against the StateStore protocol.
They use pytest fixtures to manage database connections and test data.
"""

from __future__ import annotations

import os

import psycopg
import pytest
from questfoundry.models.artifact import Artifact
from questfoundry.state.types import ProjectInfo, SnapshotInfo, TUState

from webui_api.storage.postgres_store import PostgresStore

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason=(
        "PostgreSQL database not available. Set TEST_POSTGRES_URL to run these tests."
    ),
)


@pytest.fixture
def postgres_url() -> str:
    """Get PostgreSQL connection URL from environment"""
    url = os.getenv("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL not set")
    return url


@pytest.fixture
def project_id() -> str:
    """Test project ID"""
    return "test-project-123"


@pytest.fixture
def owner_user_id() -> str:
    """Test owner user ID"""
    return "test-user-123"


def _cleanup_project_data(conn: psycopg.Connection, project_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM snapshots WHERE project_id = %s", (project_id,))
        cur.execute("DELETE FROM tus WHERE project_id = %s", (project_id,))
        cur.execute("DELETE FROM artifacts WHERE project_id = %s", (project_id,))
        cur.execute("DELETE FROM project_info WHERE project_id = %s", (project_id,))


def _upsert_user(conn: psycopg.Connection, user_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_settings (user_id, encrypted_keys)
            VALUES (%s, %s)
            ON CONFLICT (user_id)
            DO UPDATE SET encrypted_keys = EXCLUDED.encrypted_keys
            """,
            (user_id, b"test-key-material"),
        )


def _upsert_project(
    conn: psycopg.Connection,
    project_id: str,
    owner_user_id: str,
    *,
    name: str,
    description: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO project_ownership (
                project_id,
                owner_user_id,
                project_name,
                project_description
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (project_id)
            DO UPDATE SET
                owner_user_id = EXCLUDED.owner_user_id,
                project_name = EXCLUDED.project_name,
                project_description = EXCLUDED.project_description
            """,
            (project_id, owner_user_id, name, description),
        )


def _delete_project(conn: psycopg.Connection, project_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM project_ownership WHERE project_id = %s", (project_id,))


def _delete_user(conn: psycopg.Connection, user_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM user_settings WHERE user_id = %s", (user_id,))


@pytest.fixture(autouse=True)
def ensure_project_ownership(
    postgres_url: str, project_id: str, owner_user_id: str
) -> None:
    """Ensure user and project ownership records exist for FK constraints."""

    with psycopg.connect(postgres_url) as conn:
        _cleanup_project_data(conn, project_id)
        _upsert_user(conn, owner_user_id)
        _upsert_project(
            conn,
            project_id,
            owner_user_id,
            name="Test Project",
            description="PostgresStore test project",
        )
        conn.commit()

    yield

    with psycopg.connect(postgres_url) as conn:
        _cleanup_project_data(conn, project_id)
        _delete_project(conn, project_id)
        _delete_user(conn, owner_user_id)
        conn.commit()


@pytest.fixture
def project_owner_factory(postgres_url: str, owner_user_id: str):
    """Factory fixture to upsert additional project ownership records."""

    created: list[tuple[str, str]] = []

    def _create(
        project_id: str,
        *,
        owner_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        assigned_owner = owner_id or owner_user_id
        with psycopg.connect(postgres_url) as conn:
            _cleanup_project_data(conn, project_id)
            _upsert_user(conn, assigned_owner)
            _upsert_project(
                conn,
                project_id,
                assigned_owner,
                name=name or f"Test Project {project_id}",
                description=description or "Additional PostgresStore test project",
            )
            conn.commit()
        created.append((project_id, assigned_owner))

    yield _create

    with psycopg.connect(postgres_url) as conn:
        for project_id, owner in created:
            _cleanup_project_data(conn, project_id)
            _delete_project(conn, project_id)
            if owner != owner_user_id:
                _delete_user(conn, owner)
        conn.commit()


@pytest.fixture
def store(postgres_url: str, project_id: str) -> PostgresStore:
    """Create PostgresStore instance for testing"""
    store = PostgresStore(postgres_url, project_id)
    yield store
    store.close()


@pytest.fixture
def sample_project_info() -> ProjectInfo:
    """Create sample project info"""
    return ProjectInfo(
        name="Test Project",
        description="A test project",
        version="1.0.0",
        author="Test Author",
        metadata={"key": "value"},
    )


@pytest.fixture
def sample_artifact() -> Artifact:
    """Create sample artifact"""
    return Artifact(
        type="hook_card",
        data={"header": {"short_name": "Test Hook"}},
        metadata={"id": "HOOK-001"},
    )


@pytest.fixture
def sample_tu() -> TUState:
    """Create sample TU"""
    return TUState(
        tu_id="TU-2024-01-01-TEST",
        status="open",
        data={"brief": "Test brief"},
        metadata={"key": "value"},
    )


@pytest.fixture
def sample_snapshot() -> SnapshotInfo:
    """Create sample snapshot"""
    return SnapshotInfo(
        snapshot_id="SNAP-001",
        tu_id="TU-2024-01-01-TEST",
        description="Test snapshot",
        metadata={"key": "value"},
    )


class TestProjectInfo:
    """Test project info operations"""

    def test_save_and_get_project_info(
        self, store: PostgresStore, sample_project_info: ProjectInfo
    ):
        """Test saving and retrieving project info"""
        store.save_project_info(sample_project_info)
        retrieved = store.get_project_info()

        assert retrieved.name == sample_project_info.name
        assert retrieved.description == sample_project_info.description
        assert retrieved.version == sample_project_info.version
        assert retrieved.author == sample_project_info.author
        assert retrieved.metadata == sample_project_info.metadata

    def test_update_project_info(
        self, store: PostgresStore, sample_project_info: ProjectInfo
    ):
        """Test updating existing project info"""
        store.save_project_info(sample_project_info)

        # Update
        sample_project_info.description = "Updated description"
        store.save_project_info(sample_project_info)

        retrieved = store.get_project_info()
        assert retrieved.description == "Updated description"

    def test_get_nonexistent_project_raises_error(self, store: PostgresStore):
        """Test that getting nonexistent project raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            store.get_project_info()


class TestArtifacts:
    """Test artifact operations"""

    def test_save_and_get_artifact(
        self,
        store: PostgresStore,
        sample_project_info: ProjectInfo,
        sample_artifact: Artifact,
    ):
        """Test saving and retrieving artifact"""
        # Need project info first
        store.save_project_info(sample_project_info)

        store.save_artifact(sample_artifact)
        retrieved = store.get_artifact("HOOK-001")

        assert retrieved is not None
        assert retrieved.type == sample_artifact.type
        assert retrieved.data == sample_artifact.data
        assert retrieved.metadata["id"] == "HOOK-001"

    def test_update_artifact(
        self,
        store: PostgresStore,
        sample_project_info: ProjectInfo,
        sample_artifact: Artifact,
    ):
        """Test updating existing artifact"""
        store.save_project_info(sample_project_info)
        store.save_artifact(sample_artifact)

        # Update
        sample_artifact.data["header"]["short_name"] = "Updated Hook"
        store.save_artifact(sample_artifact)

        retrieved = store.get_artifact("HOOK-001")
        assert retrieved is not None
        assert retrieved.data["header"]["short_name"] == "Updated Hook"

    def test_get_nonexistent_artifact_returns_none(
        self, store: PostgresStore, sample_project_info: ProjectInfo
    ):
        """Test that getting nonexistent artifact returns None"""
        store.save_project_info(sample_project_info)
        retrieved = store.get_artifact("NONEXISTENT")
        assert retrieved is None

    def test_list_artifacts(
        self, store: PostgresStore, sample_project_info: ProjectInfo
    ):
        """Test listing artifacts"""
        store.save_project_info(sample_project_info)

        # Create multiple artifacts
        for i in range(3):
            artifact = Artifact(
                type="hook_card",
                data={"header": {"short_name": f"Hook {i}"}},
                metadata={"id": f"HOOK-{i:03d}"},
            )
            store.save_artifact(artifact)

        artifacts = store.list_artifacts()
        assert len(artifacts) == 3

    def test_list_artifacts_with_type_filter(
        self, store: PostgresStore, sample_project_info: ProjectInfo
    ):
        """Test listing artifacts with type filter"""
        store.save_project_info(sample_project_info)

        # Create artifacts of different types
        store.save_artifact(
            Artifact(type="hook_card", data={}, metadata={"id": "HOOK-001"})
        )
        store.save_artifact(
            Artifact(type="canon", data={}, metadata={"id": "CANON-001"})
        )

        hooks = store.list_artifacts(artifact_type="hook_card")
        assert len(hooks) == 1
        assert hooks[0].type == "hook_card"

    def test_delete_artifact(
        self,
        store: PostgresStore,
        sample_project_info: ProjectInfo,
        sample_artifact: Artifact,
    ):
        """Test deleting artifact"""
        store.save_project_info(sample_project_info)
        store.save_artifact(sample_artifact)

        deleted = store.delete_artifact("HOOK-001")
        assert deleted is True

        retrieved = store.get_artifact("HOOK-001")
        assert retrieved is None

    def test_delete_nonexistent_artifact(
        self, store: PostgresStore, sample_project_info: ProjectInfo
    ):
        """Test deleting nonexistent artifact returns False"""
        store.save_project_info(sample_project_info)
        deleted = store.delete_artifact("NONEXISTENT")
        assert deleted is False

    def test_save_artifact_without_id_raises_error(
        self, store: PostgresStore, sample_project_info: ProjectInfo
    ):
        """Test that saving artifact without id raises ValueError"""
        store.save_project_info(sample_project_info)
        artifact = Artifact(type="hook_card", data={}, metadata={})

        with pytest.raises(ValueError, match="must have an 'id'"):
            store.save_artifact(artifact)


class TestTUs:
    """Test TU operations"""

    def test_save_and_get_tu(
        self, store: PostgresStore, sample_project_info: ProjectInfo, sample_tu: TUState
    ):
        """Test saving and retrieving TU"""
        store.save_project_info(sample_project_info)
        store.save_tu(sample_tu)

        retrieved = store.get_tu(sample_tu.tu_id)
        assert retrieved is not None
        assert retrieved.tu_id == sample_tu.tu_id
        assert retrieved.status == sample_tu.status
        assert retrieved.data == sample_tu.data

    def test_update_tu(
        self, store: PostgresStore, sample_project_info: ProjectInfo, sample_tu: TUState
    ):
        """Test updating existing TU"""
        store.save_project_info(sample_project_info)
        store.save_tu(sample_tu)

        # Update
        sample_tu.status = "completed"
        store.save_tu(sample_tu)

        retrieved = store.get_tu(sample_tu.tu_id)
        assert retrieved is not None
        assert retrieved.status == "completed"

    def test_list_tus(self, store: PostgresStore, sample_project_info: ProjectInfo):
        """Test listing TUs"""
        store.save_project_info(sample_project_info)

        # Create multiple TUs
        for i in range(3):
            tu = TUState(
                tu_id=f"TU-2024-01-{i:02d}-TEST",
                status="open",
                data={},
                metadata={},
            )
            store.save_tu(tu)

        tus = store.list_tus()
        assert len(tus) == 3

    def test_list_tus_with_status_filter(
        self, store: PostgresStore, sample_project_info: ProjectInfo
    ):
        """Test listing TUs with status filter"""
        store.save_project_info(sample_project_info)

        store.save_tu(TUState(tu_id="TU-001", status="open", data={}, metadata={}))
        store.save_tu(TUState(tu_id="TU-002", status="completed", data={}, metadata={}))

        open_tus = store.list_tus(filters={"status": "open"})
        assert len(open_tus) == 1
        assert open_tus[0].status == "open"


class TestSnapshots:
    """Test snapshot operations"""

    def test_save_and_get_snapshot(
        self,
        store: PostgresStore,
        sample_project_info: ProjectInfo,
        sample_snapshot: SnapshotInfo,
    ):
        """Test saving and retrieving snapshot"""
        store.save_project_info(sample_project_info)
        store.save_snapshot(sample_snapshot)

        retrieved = store.get_snapshot(sample_snapshot.snapshot_id)
        assert retrieved is not None
        assert retrieved.snapshot_id == sample_snapshot.snapshot_id
        assert retrieved.tu_id == sample_snapshot.tu_id
        assert retrieved.description == sample_snapshot.description

    def test_snapshot_immutability(
        self,
        store: PostgresStore,
        sample_project_info: ProjectInfo,
        sample_snapshot: SnapshotInfo,
    ):
        """Test that snapshots are immutable"""
        store.save_project_info(sample_project_info)
        store.save_snapshot(sample_snapshot)

        # Try to save again - should raise error
        with pytest.raises(ValueError, match="already exists"):
            store.save_snapshot(sample_snapshot)

    def test_list_snapshots(
        self, store: PostgresStore, sample_project_info: ProjectInfo
    ):
        """Test listing snapshots"""
        store.save_project_info(sample_project_info)

        # Create multiple snapshots
        for i in range(3):
            snapshot = SnapshotInfo(
                snapshot_id=f"SNAP-{i:03d}",
                tu_id="TU-2024-01-01-TEST",
                description=f"Snapshot {i}",
                metadata={},
            )
            store.save_snapshot(snapshot)

        snapshots = store.list_snapshots()
        assert len(snapshots) == 3

    def test_list_snapshots_with_tu_filter(
        self, store: PostgresStore, sample_project_info: ProjectInfo
    ):
        """Test listing snapshots with TU filter"""
        store.save_project_info(sample_project_info)

        store.save_snapshot(
            SnapshotInfo(
                snapshot_id="SNAP-001",
                tu_id="TU-001",
                description="Snapshot 1",
                metadata={},
            )
        )
        store.save_snapshot(
            SnapshotInfo(
                snapshot_id="SNAP-002",
                tu_id="TU-002",
                description="Snapshot 2",
                metadata={},
            )
        )

        snapshots = store.list_snapshots(filters={"tu_id": "TU-001"})
        assert len(snapshots) == 1
        assert snapshots[0].tu_id == "TU-001"


class TestProjectIsolation:
    """Test that project_id properly isolates data between projects"""

    def test_artifacts_isolated_by_project(
        self,
        postgres_url: str,
        sample_project_info: ProjectInfo,
        project_owner_factory,
    ):
        """Test that artifacts are isolated by project_id"""
        project_owner_factory("project-1")
        project_owner_factory("project-2")
        store1 = PostgresStore(postgres_url, "project-1")
        store2 = PostgresStore(postgres_url, "project-2")

        try:
            # Setup both projects
            store1.save_project_info(sample_project_info)
            store2.save_project_info(sample_project_info)

            # Save artifact in project 1
            artifact = Artifact(
                type="hook_card",
                data={"name": "Hook 1"},
                metadata={"id": "HOOK-001"},
            )
            store1.save_artifact(artifact)

            # Should not be visible in project 2
            retrieved = store2.get_artifact("HOOK-001")
            assert retrieved is None

            # Should be visible in project 1
            retrieved = store1.get_artifact("HOOK-001")
            assert retrieved is not None

        finally:
            store1.close()
            store2.close()
