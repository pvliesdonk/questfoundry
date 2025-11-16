"""
Test database schema creation and integrity.

This module validates that schema.sql:
1. Creates all required tables
2. Creates all required indexes
3. Creates all required triggers
4. Enforces constraints correctly
5. Works with PostgreSQL
"""

import os
import subprocess
import pytest
import psycopg
from contextlib import contextmanager

# Skip all tests if PostgreSQL is not available
TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not TEST_POSTGRES_URL, reason="TEST_POSTGRES_URL not set"
)


@contextmanager
def get_test_db_connection():
    """Get a connection to the test database."""
    conn = psycopg.connect(TEST_POSTGRES_URL)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def schema_applied():
    """Apply schema.sql to test database once per module."""
    if not TEST_POSTGRES_URL:
        pytest.skip("TEST_POSTGRES_URL not set")
    
    schema_path = os.path.join(
        os.path.dirname(__file__), "..", "schema.sql"
    )
    
    with get_test_db_connection() as conn:
        # Drop all tables first (clean slate)
        with conn.cursor() as cur:
            cur.execute("""
                DROP SCHEMA public CASCADE;
                CREATE SCHEMA public;
                GRANT ALL ON SCHEMA public TO PUBLIC;
            """)
        conn.commit()
        
        # Apply schema
        with open(schema_path) as f:
            schema_sql = f.read()
        
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
    
    yield
    
    # Cleanup (optional, as test database can be ephemeral)


class TestTablesExist:
    """Test that all required tables are created."""
    
    def test_user_settings_table_exists(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'user_settings'
                    );
                """)
                exists = cur.fetchone()[0]
                assert exists, "user_settings table not created"
    
    def test_project_ownership_table_exists(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'project_ownership'
                    );
                """)
                exists = cur.fetchone()[0]
                assert exists, "project_ownership table not created"
    
    def test_project_info_table_exists(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'project_info'
                    );
                """)
                exists = cur.fetchone()[0]
                assert exists, "project_info table not created"
    
    def test_artifacts_table_exists(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'artifacts'
                    );
                """)
                exists = cur.fetchone()[0]
                assert exists, "artifacts table not created"
    
    def test_tus_table_exists(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'tus'
                    );
                """)
                exists = cur.fetchone()[0]
                assert exists, "tus table not created"
    
    def test_snapshots_table_exists(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'snapshots'
                    );
                """)
                exists = cur.fetchone()[0]
                assert exists, "snapshots table not created"


class TestIndexesExist:
    """Test that all required indexes are created."""
    
    def test_user_settings_indexes(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'user_settings';
                """)
                indexes = [row[0] for row in cur.fetchall()]
                assert "idx_user_settings_updated" in indexes
    
    def test_project_ownership_indexes(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'project_ownership';
                """)
                indexes = [row[0] for row in cur.fetchall()]
                assert "idx_project_ownership_owner" in indexes
                assert "idx_project_ownership_updated" in indexes
    
    def test_artifacts_indexes(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'artifacts';
                """)
                indexes = [row[0] for row in cur.fetchall()]
                assert "idx_artifacts_type" in indexes
                assert "idx_artifacts_modified" in indexes
                assert "idx_artifacts_data" in indexes
    
    def test_tus_indexes(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'tus';
                """)
                indexes = [row[0] for row in cur.fetchall()]
                assert "idx_tus_status" in indexes
                assert "idx_tus_modified" in indexes
    
    def test_snapshots_indexes(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'snapshots';
                """)
                indexes = [row[0] for row in cur.fetchall()]
                assert "idx_snapshots_tu" in indexes
                assert "idx_snapshots_created" in indexes


class TestTriggersWork:
    """Test that automatic timestamp triggers work correctly."""
    
    def test_user_settings_trigger(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                # Insert a user
                cur.execute("""
                    INSERT INTO user_settings (user_id, encrypted_keys)
                    VALUES ('test_user', '\\x00');
                """)
                conn.commit()
                
                # Get initial timestamp
                cur.execute("""
                    SELECT updated_at FROM user_settings WHERE user_id = 'test_user';
                """)
                initial_time = cur.fetchone()[0]
                
                # Wait a bit and update
                import time
                time.sleep(0.1)
                
                cur.execute("""
                    UPDATE user_settings 
                    SET encrypted_keys = '\\x01'
                    WHERE user_id = 'test_user';
                """)
                conn.commit()
                
                # Get new timestamp
                cur.execute("""
                    SELECT updated_at FROM user_settings WHERE user_id = 'test_user';
                """)
                new_time = cur.fetchone()[0]
                
                assert new_time > initial_time, "Trigger did not update timestamp"
                
                # Cleanup
                cur.execute("DELETE FROM user_settings WHERE user_id = 'test_user';")
                conn.commit()
    
    def test_artifacts_trigger(self, schema_applied):
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                # Setup: Create user and project
                cur.execute("""
                    INSERT INTO user_settings (user_id, encrypted_keys)
                    VALUES ('test_user2', '\\x00');
                """)
                cur.execute("""
                    INSERT INTO project_ownership (project_id, owner_user_id, project_name)
                    VALUES ('test-project', 'test_user2', 'Test Project');
                """)
                conn.commit()
                
                # Insert artifact
                cur.execute("""
                    INSERT INTO artifacts (project_id, artifact_id, artifact_type, data)
                    VALUES ('test-project', 'TEST-001', 'test', '{}');
                """)
                conn.commit()
                
                # Get initial timestamp
                cur.execute("""
                    SELECT modified FROM artifacts 
                    WHERE project_id = 'test-project' AND artifact_id = 'TEST-001';
                """)
                initial_time = cur.fetchone()[0]
                
                # Wait and update
                import time
                time.sleep(0.1)
                
                cur.execute("""
                    UPDATE artifacts 
                    SET data = '{"updated": true}'
                    WHERE project_id = 'test-project' AND artifact_id = 'TEST-001';
                """)
                conn.commit()
                
                # Get new timestamp
                cur.execute("""
                    SELECT modified FROM artifacts 
                    WHERE project_id = 'test-project' AND artifact_id = 'TEST-001';
                """)
                new_time = cur.fetchone()[0]
                
                assert new_time > initial_time, "Trigger did not update modified timestamp"
                
                # Cleanup
                cur.execute("DELETE FROM project_ownership WHERE project_id = 'test-project';")
                cur.execute("DELETE FROM user_settings WHERE user_id = 'test_user2';")
                conn.commit()


class TestConstraints:
    """Test that constraints are enforced."""
    
    def test_foreign_key_cascade_delete(self, schema_applied):
        """Test that deleting a project cascades to artifacts."""
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                # Setup
                cur.execute("""
                    INSERT INTO user_settings (user_id, encrypted_keys)
                    VALUES ('test_user3', '\\x00');
                """)
                cur.execute("""
                    INSERT INTO project_ownership (project_id, owner_user_id, project_name)
                    VALUES ('test-cascade', 'test_user3', 'Test Cascade');
                """)
                cur.execute("""
                    INSERT INTO artifacts (project_id, artifact_id, artifact_type, data)
                    VALUES ('test-cascade', 'ART-001', 'test', '{}');
                """)
                conn.commit()
                
                # Verify artifact exists
                cur.execute("""
                    SELECT COUNT(*) FROM artifacts WHERE project_id = 'test-cascade';
                """)
                assert cur.fetchone()[0] == 1
                
                # Delete project
                cur.execute("""
                    DELETE FROM project_ownership WHERE project_id = 'test-cascade';
                """)
                conn.commit()
                
                # Verify artifact was deleted
                cur.execute("""
                    SELECT COUNT(*) FROM artifacts WHERE project_id = 'test-cascade';
                """)
                assert cur.fetchone()[0] == 0, "Cascade delete failed"
                
                # Cleanup
                cur.execute("DELETE FROM user_settings WHERE user_id = 'test_user3';")
                conn.commit()
    
    def test_project_id_validation(self, schema_applied):
        """Test that project_id constraint rejects invalid IDs."""
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                # Setup
                cur.execute("""
                    INSERT INTO user_settings (user_id, encrypted_keys)
                    VALUES ('test_user4', '\\x00');
                """)
                conn.commit()
                
                # Try to insert invalid project ID (with spaces)
                with pytest.raises(psycopg.errors.CheckViolation):
                    cur.execute("""
                        INSERT INTO project_ownership (project_id, owner_user_id, project_name)
                        VALUES ('invalid project id', 'test_user4', 'Test');
                    """)
                    conn.commit()
                
                conn.rollback()
                
                # Cleanup
                cur.execute("DELETE FROM user_settings WHERE user_id = 'test_user4';")
                conn.commit()


class TestDataTypes:
    """Test that JSONB columns work correctly."""
    
    def test_jsonb_insert_and_query(self, schema_applied):
        """Test that JSONB data can be inserted and queried."""
        with get_test_db_connection() as conn:
            with conn.cursor() as cur:
                # Setup
                cur.execute("""
                    INSERT INTO user_settings (user_id, encrypted_keys)
                    VALUES ('test_user5', '\\x00');
                """)
                cur.execute("""
                    INSERT INTO project_ownership (project_id, owner_user_id, project_name)
                    VALUES ('test-json', 'test_user5', 'Test JSON');
                """)
                
                # Insert artifact with JSONB data
                cur.execute("""
                    INSERT INTO artifacts (project_id, artifact_id, artifact_type, data, metadata)
                    VALUES ('test-json', 'JSON-001', 'test', 
                            '{"title": "Test", "count": 42}'::jsonb,
                            '{"status": "draft"}'::jsonb);
                """)
                conn.commit()
                
                # Query using JSONB operators
                cur.execute("""
                    SELECT data->>'title', (data->>'count')::int, metadata->>'status'
                    FROM artifacts 
                    WHERE project_id = 'test-json' AND artifact_id = 'JSON-001';
                """)
                result = cur.fetchone()
                assert result[0] == "Test"
                assert result[1] == 42
                assert result[2] == "draft"
                
                # Cleanup
                cur.execute("DELETE FROM project_ownership WHERE project_id = 'test-json';")
                cur.execute("DELETE FROM user_settings WHERE user_id = 'test_user5';")
                conn.commit()
