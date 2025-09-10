FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.2 \
    joblib==1.5.1 \
    pandas==2.3.2 \
    numpy==2.3.2 \
    scikit-learn==1.7.1 \
    python-multipart==0.0.6 \
    requests==2.31.0

# Copy application code
COPY app.py .

# Create models directory
RUN mkdir -p /models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:8000/health/'); exit(0 if r.status_code == 200 else 1)" || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]