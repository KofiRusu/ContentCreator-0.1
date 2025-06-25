# AI Scene-to-Video Pipeline Container
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy dependency files
COPY requirements.txt pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY Makefile ./

# Create assets directory and set permissions
RUN mkdir -p src/assets && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (if needed for web interface in future)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src.main; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.main", "--help"] 