# Multi-stage build for production-ready Python FastAPI app
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python dependencies
COPY requirements.txt .
# Speed-up: install CPU wheels for torch/torchaudio from official index first
# Then install remaining deps. This avoids compiling from source (hours!).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1+cpu torchaudio==2.4.1+cpu && \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        --user -r requirements.txt --prefer-binary


# Production stage
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appgroup /app

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Copy ONLY production application code (exclude tests, test-webapp, docs)
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup services/ ./services/
COPY --chown=appuser:appgroup repositories/ ./repositories/
COPY --chown=appuser:appgroup core/ ./core/

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

# Expose port (Azure App Service will set PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info --no-access-log
