FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install PyTorch CPU first, then the rest
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY core/ ./core/
COPY api.py .
COPY school_handbook.pdf .

EXPOSE 8000

# Run the FastAPI server directly
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]