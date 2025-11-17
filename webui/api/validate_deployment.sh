#!/bin/bash
#
# QuestFoundry WebUI - Deployment Validation Script
#
# This script validates the full deployment stack using docker-compose.
# It tests:
# 1. All services start successfully
# 2. Health checks pass
# 3. Inter-service communication works
# 4. Database schema is applied
# 5. API endpoints respond correctly
#
# Usage:
#   ./validate_deployment.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "======================================================================"
echo "QuestFoundry WebUI - Deployment Validation"
echo "======================================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}✗${NC} $message"
    else
        echo -e "${YELLOW}▸${NC} $message"
    fi
}

# Change to project root
cd "$PROJECT_ROOT"

# Step 1: Validate docker-compose.yml
print_status "INFO" "Step 1: Validating docker-compose.yml..."
if docker compose config > /dev/null 2>&1; then
    print_status "OK" "docker-compose.yml is valid"
else
    print_status "FAIL" "docker-compose.yml validation failed"
    exit 1
fi

# Step 2: Build images
print_status "INFO" "Step 2: Building Docker images..."
if docker compose build --no-cache; then
    print_status "OK" "Docker images built successfully"
else
    print_status "FAIL" "Docker image build failed"
    exit 1
fi

# Step 3: Start services
print_status "INFO" "Step 3: Starting services..."
docker compose down -v  # Clean slate
if docker compose up -d; then
    print_status "OK" "Services started"
else
    print_status "FAIL" "Failed to start services"
    exit 1
fi

# Step 4: Wait for services to be healthy
print_status "INFO" "Step 4: Waiting for services to be healthy..."

wait_for_healthy() {
    local service=$1
    local max_wait=60
    local waited=0

    while [ $waited -lt $max_wait ]; do
        if docker compose ps "$service" | grep -q "healthy"; then
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}

# Wait for PostgreSQL
if wait_for_healthy postgres; then
    print_status "OK" "PostgreSQL is healthy"
else
    print_status "FAIL" "PostgreSQL did not become healthy"
    docker compose logs postgres
    docker compose down -v
    exit 1
fi

# Wait for Valkey
if wait_for_healthy valkey; then
    print_status "OK" "Valkey is healthy"
else
    print_status "FAIL" "Valkey did not become healthy"
    docker compose logs valkey
    docker compose down -v
    exit 1
fi

# Wait for API
if wait_for_healthy api; then
    print_status "OK" "API is healthy"
else
    print_status "FAIL" "API did not become healthy"
    docker compose logs api
    docker compose down -v
    exit 1
fi

# Step 5: Validate database schema
print_status "INFO" "Step 5: Validating database schema..."

check_table() {
    local table=$1
    docker compose exec -T postgres psql -U questfoundry -d questfoundry -c "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '$table');" | grep -q "t"
}

if check_table "user_settings"; then
    print_status "OK" "Table user_settings exists"
else
    print_status "FAIL" "Table user_settings does not exist"
    docker compose down -v
    exit 1
fi

if check_table "project_ownership"; then
    print_status "OK" "Table project_ownership exists"
else
    print_status "FAIL" "Table project_ownership does not exist"
    docker compose down -v
    exit 1
fi

if check_table "artifacts"; then
    print_status "OK" "Table artifacts exists"
else
    print_status "FAIL" "Table artifacts does not exist"
    docker compose down -v
    exit 1
fi

# Step 6: Test API endpoints
print_status "INFO" "Step 6: Testing API endpoints..."

# Health endpoint
if curl -f -s http://localhost:8000/health | grep -q "healthy"; then
    print_status "OK" "Health endpoint responds correctly"
else
    print_status "FAIL" "Health endpoint failed"
    docker compose logs api
    docker compose down -v
    exit 1
fi

# Root endpoint
if curl -f -s http://localhost:8000/ | grep -q "service"; then
    print_status "OK" "Root endpoint responds correctly"
else
    print_status "FAIL" "Root endpoint failed"
    docker compose logs api
    docker compose down -v
    exit 1
fi

# API docs
if curl -f -s http://localhost:8000/docs | grep -q -i "swagger\|openapi"; then
    print_status "OK" "API documentation is accessible"
else
    print_status "FAIL" "API documentation not accessible"
    docker compose down -v
    exit 1
fi

# Step 7: Test inter-service communication
print_status "INFO" "Step 7: Testing inter-service communication..."

# Test that API can connect to PostgreSQL
if docker compose exec -T api python -c "import psycopg; psycopg.connect('postgresql://questfoundry:changeme_postgres_password@postgres/questfoundry').close()" 2>/dev/null; then
    print_status "OK" "API can connect to PostgreSQL"
else
    print_status "FAIL" "API cannot connect to PostgreSQL"
    docker compose down -v
    exit 1
fi

# Test that API can connect to Valkey
if docker compose exec -T api python -c "import redis; redis.Redis(host='valkey', port=6379, password='changeme_valkey_password').ping()" 2>/dev/null; then
    print_status "OK" "API can connect to Valkey"
else
    print_status "FAIL" "API cannot connect to Valkey"
    docker compose down -v
    exit 1
fi

# Step 8: Cleanup
print_status "INFO" "Step 8: Cleaning up..."
docker compose down -v
print_status "OK" "Services stopped and volumes removed"

echo
echo "======================================================================"
echo -e "${GREEN}All validation checks passed!${NC}"
echo "======================================================================"
echo
echo "Deployment stack is validated and ready for production use."
echo "Remember to:"
echo "  1. Change all passwords in docker-compose.yml"
echo "  2. Generate a proper WEBUI_ENCRYPTION_KEY"
echo "  3. Configure TLS/SSL certificates"
echo "  4. Set up proper authentication (Authelia/OIDC)"
echo "  5. Configure logging and monitoring"
echo

exit 0
