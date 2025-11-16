"""
Test Docker builds and container functionality.

This module validates that:
1. API Dockerfile builds successfully
2. API container starts and responds
3. Health check works
4. Environment variables are properly configured
"""

import os
import subprocess
import time
import pytest
import requests

# These tests require Docker to be available
pytestmark = pytest.mark.skipif(
    subprocess.run(["which", "docker"], capture_output=True).returncode != 0,
    reason="Docker not available"
)


class TestAPIDockerBuild:
    """Test API Docker image build."""
    
    def test_dockerfile_builds(self):
        """Test that the API Dockerfile builds successfully."""
        api_dir = os.path.join(
            os.path.dirname(__file__), ".."
        )
        
        result = subprocess.run(
            ["docker", "build", "-t", "qf-api-test", "-f", "Dockerfile", "."],
            cwd=api_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for build
        )
        
        assert result.returncode == 0, f"Docker build failed:\n{result.stderr}"
        
        # Verify image was created
        result = subprocess.run(
            ["docker", "images", "qf-api-test", "--format", "{{.Repository}}"],
            capture_output=True,
            text=True
        )
        assert "qf-api-test" in result.stdout
    
    def test_dockerfile_multi_stage(self):
        """Verify that Dockerfile uses multi-stage build pattern."""
        dockerfile_path = os.path.join(
            os.path.dirname(__file__), "..", "Dockerfile"
        )
        
        with open(dockerfile_path) as f:
            content = f.read()
        
        # Check for builder stage
        assert "FROM python:3.11-slim AS builder" in content
        
        # Check for final stage
        assert "FROM python:3.11-slim" in content
        
        # Check for COPY --from=builder
        assert "COPY --from=builder" in content
    
    def test_dockerfile_healthcheck(self):
        """Verify that Dockerfile includes health check."""
        dockerfile_path = os.path.join(
            os.path.dirname(__file__), "..", "Dockerfile"
        )
        
        with open(dockerfile_path) as f:
            content = f.read()
        
        assert "HEALTHCHECK" in content
        assert "/health" in content
    
    def test_dockerfile_security_best_practices(self):
        """Verify Dockerfile follows security best practices."""
        dockerfile_path = os.path.join(
            os.path.dirname(__file__), "..", "Dockerfile"
        )
        
        with open(dockerfile_path) as f:
            content = f.read()
        
        # Should create non-root user
        assert "useradd" in content or "adduser" in content
        
        # Should switch to non-root user
        assert "USER app" in content or "USER " in content
        
        # Should expose port
        assert "EXPOSE 8000" in content


class TestAPIContainer:
    """Test API container runtime."""
    
    @pytest.fixture(scope="class")
    def api_container(self):
        """Start API container for testing."""
        # Build image first
        api_dir = os.path.join(os.path.dirname(__file__), "..")
        subprocess.run(
            ["docker", "build", "-t", "qf-api-test", "-f", "Dockerfile", "."],
            cwd=api_dir,
            check=True,
            capture_output=True
        )
        
        # Start container
        container_name = "qf-api-test-container"
        
        # Stop and remove if exists
        subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True
        )
        subprocess.run(
            ["docker", "rm", container_name],
            capture_output=True
        )
        
        # Start new container
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "-p", "18000:8000",
                "-e", "WEBUI_POSTGRES_HOST=postgres",
                "-e", "WEBUI_REDIS_HOST=redis",
                "-e", "WEBUI_ENCRYPTION_KEY=test_key_for_testing_only_32chars",
                "qf-api-test"
            ],
            check=True,
            capture_output=True
        )
        
        # Wait for container to be ready
        time.sleep(5)
        
        yield container_name
        
        # Cleanup
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
    
    def test_container_starts(self, api_container):
        """Test that container starts successfully."""
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={api_container}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        assert api_container in result.stdout
    
    def test_health_endpoint_responds(self, api_container):
        """Test that health endpoint responds."""
        # Try multiple times as container may take a moment to start
        for i in range(10):
            try:
                response = requests.get("http://localhost:18000/health", timeout=2)
                if response.status_code == 200:
                    assert response.json() == {"status": "healthy"}
                    return
            except requests.exceptions.RequestException:
                if i < 9:
                    time.sleep(2)
                else:
                    raise
        
        pytest.fail("Health endpoint did not respond after 20 seconds")
    
    def test_root_endpoint_responds(self, api_container):
        """Test that root endpoint responds."""
        for i in range(10):
            try:
                response = requests.get("http://localhost:18000/", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    assert "service" in data
                    assert "version" in data
                    return
            except requests.exceptions.RequestException:
                if i < 9:
                    time.sleep(2)
                else:
                    raise
        
        pytest.fail("Root endpoint did not respond after 20 seconds")
    
    def test_docs_endpoint_responds(self, api_container):
        """Test that API docs endpoint is accessible."""
        for i in range(10):
            try:
                response = requests.get("http://localhost:18000/docs", timeout=2)
                if response.status_code == 200:
                    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
                    return
            except requests.exceptions.RequestException:
                if i < 9:
                    time.sleep(2)
                else:
                    raise
        
        pytest.fail("Docs endpoint did not respond after 20 seconds")
    
    def test_container_logs_no_errors(self, api_container):
        """Test that container logs don't show critical errors."""
        result = subprocess.run(
            ["docker", "logs", api_container],
            capture_output=True,
            text=True
        )
        
        logs = result.stdout + result.stderr
        
        # Should not have critical errors
        assert "CRITICAL" not in logs.upper()
        
        # Should show uvicorn started
        assert "uvicorn" in logs.lower() or "application startup complete" in logs.lower()


class TestDockerCompose:
    """Test docker-compose.yml configuration."""
    
    def test_docker_compose_valid(self):
        """Test that docker-compose.yml is valid."""
        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docker-compose.yml"
        )
        
        result = subprocess.run(
            ["docker", "compose", "-f", compose_path, "config"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"docker-compose.yml validation failed:\n{result.stderr}"
    
    def test_docker_compose_includes_required_services(self):
        """Test that docker-compose.yml includes all required services."""
        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docker-compose.yml"
        )
        
        with open(compose_path) as f:
            content = f.read()
        
        # Required services
        assert "postgres:" in content
        assert "valkey:" in content or "redis:" in content
        assert "api:" in content
        assert "pwa:" in content
    
    def test_docker_compose_health_checks(self):
        """Test that services have health checks configured."""
        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docker-compose.yml"
        )
        
        with open(compose_path) as f:
            content = f.read()
        
        # Health checks should be defined
        assert "healthcheck:" in content
        assert content.count("healthcheck:") >= 3  # At least postgres, valkey, api
    
    def test_docker_compose_volumes_defined(self):
        """Test that persistent volumes are defined."""
        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docker-compose.yml"
        )
        
        with open(compose_path) as f:
            content = f.read()
        
        # Should have volume definitions
        assert "volumes:" in content
        assert "postgres_data:" in content
        assert "valkey_data:" in content or "redis_data:" in content
    
    def test_docker_compose_schema_mounted(self):
        """Test that schema.sql is mounted in postgres container."""
        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docker-compose.yml"
        )
        
        with open(compose_path) as f:
            content = f.read()
        
        # Schema should be mounted for init
        assert "schema.sql" in content
        assert "docker-entrypoint-initdb.d" in content
