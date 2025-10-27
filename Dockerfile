FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY ml/ ./ml/
COPY data/ ./data/
COPY .env ./

# Create necessary directories
RUN mkdir -p ml/registry mlruns

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set Python path
ENV PYTHONPATH=/app

# Expose port (if needed for health checks)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from app.main import StudentGradeAPI; api = StudentGradeAPI(); exit(0 if api.model_loaded else 1)"

# Run application
CMD ["python", "app/main.py"]
