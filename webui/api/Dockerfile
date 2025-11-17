# Multi-stage build for QuestFoundry WebUI API

FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy API and shared library sources into the builder image
WORKDIR /app
COPY webui/api /app/webui/api
COPY lib/python /app/lib/python

# Install dependencies from the API project
WORKDIR /app/webui/api
RUN uv pip install --system --no-cache -e .

# Final stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 app

# Set working directory to API project
WORKDIR /app/webui/api

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source (entire API directory for runtime assets)
COPY --from=builder /app/webui/api /app/webui/api

# Switch to app user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "webui_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
