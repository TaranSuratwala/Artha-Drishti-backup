# Hardened Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

# Install runtime dependencies (e.g., libpq5 for psycopg2)
RUN apt-get update && apt-get install -y libpq5 curl && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy only installed packages
COPY --from=builder /install /usr/local

# Copy application code (respects .dockerignore)
COPY . .

# Create directories with correct ownership
RUN mkdir -p saved_models logs unified_models unified_plots unified_metrics data \
    && chown -R appuser:appuser /app

# Drop to non-root
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", \
     "--timeout", "120", "--graceful-timeout", "30", "--max-requests", "1000", \
     "--max-requests-jitter", "50", "application:app"]
