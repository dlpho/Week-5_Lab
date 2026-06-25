# ==========================================
# STAGE 1: Builder
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Install dependencies to a local folder to easily copy them later
RUN pip install --user --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 2: Runner
# ==========================================
FROM python:3.12-slim

WORKDIR /app

# Install supervisor (very lightweight) and clean up apt cache to save space
RUN apt-get update && \
    apt-get install -y supervisor && \
    rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from the builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy your application code
COPY . .

# Copy the supervisor config to the system directory
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports (Render will override the external port, but good for local testing)
EXPOSE 7860 8000

# Start supervisor, which will start both FastAPI and Streamlit
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]