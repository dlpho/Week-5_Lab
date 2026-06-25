# ============================================================
# Multi-stage Dockerfile — targets < 500 MB final image
# Stage 1 (builder): install Python deps into a venv
# Stage 2 (runtime): copy only the venv + app code; no build tools
# ============================================================

# ── Stage 1: builder ────────────────────────────────────────
FROM python:3.12-slim AS builder

# System deps needed only at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create an isolated venv so we can copy it cleanly into stage 2
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Supervisord runs both uvicorn (FastAPI :8000) and streamlit (:8501)
# from the same container — matches the Week 5 architecture diagram.
RUN apt-get update && apt-get install -y --no-install-recommends \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from stage 1
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY core/       ./core/
COPY api.py      .
COPY app.py      .

# Supervisor config — starts both servers; restarts either if it crashes
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Persist ChromaDB and MLFlow DB across container restarts
VOLUME ["/app/chroma_db", "/app/mlflow.db"]

EXPOSE 8000 8501

# Health check against the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]