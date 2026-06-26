# ============================================================
# Multi-stage Dockerfile — targets < 500 MB final image
# Stage 1 (builder): install Python deps into a venv
# Stage 2 (runtime): copy only the venv + app code; no build tools
# ============================================================

# ── Stage 1: builder ────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install PyTorch CPU-only first so pip never pulls the CUDA build from PyPI
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1.0" \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.12-slim AS runtime

# supervisord runs both uvicorn :8000 and streamlit :8501
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Application source
COPY core/                ./core/
COPY api.py               .
COPY app.py               .
COPY supervisord.conf     /etc/supervisor/conf.d/supervisord.conf

# Knowledge base — must be in /app so initialize_handbook() finds it
COPY school_handbook.pdf  .

# Runtime data directories (ChromaDB index + MLFlow traces)
# Declared as a volume so docker-compose can persist them across restarts
RUN mkdir -p /app/chroma_db /app/mlflow_data \
    && mkdir -p /var/log/supervisor

VOLUME ["/app/chroma_db", "/app/mlflow_data"]

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]